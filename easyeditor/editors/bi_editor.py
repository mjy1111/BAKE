from typing import Optional, Union, List, Tuple, Dict
from time import time
from tqdm import tqdm
import json
import math
import torch
import numpy as np
import random
from ..models.melo.melo import LORA
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, BitsAndBytesConfig
from transformers import LlamaTokenizer, PreTrainedTokenizerFast, LlamaTokenizerFast
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import GPT2TokenizerFast, GPT2Tokenizer
from ..util.globals import *
from .utils import _chunks, summary_metrics, _prepare_requests_bi
from .batch_editor import BatchEditor
from ..evaluate import compute_edit_quality, compute_icl_edit_quality, compute_sent_metric, compute_edit_quality_bi, compute_icl_edit_quality_bi
from ..util import nethook
from ..util.hparams import HyperParams
from ..util.alg_dict import *
from ..evaluate.evaluate_utils import test_generation_quality

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                   datefmt='%m/%d/%Y %H:%M:%S',
                   level=logging.INFO)

LOG = logging.getLogger(__name__)

def make_logs():
    f_h, s_h = get_handler('logs', log_name='run.log')
    LOG.addHandler(f_h)
    LOG.addHandler(s_h)

def seed_everything(seed):
    if seed >= 10000:
        raise ValueError("seed number should be less than 10000")
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    seed = (rank * 100000) + seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

seed_everything(42)

class BiEditor:
    """Bi-directional editor for all methods"""

    @classmethod
    def from_hparams(cls, hparams: HyperParams):
        return cls(hparams)

    def __init__(self, hparams: HyperParams):
        assert hparams is not None, 'Error: hparams is None.'
        self.model_name = hparams.model_name
        self.apply_algo = ALG_DICT[hparams.alg_name]
        self.alg_name = hparams.alg_name
        make_logs()
        LOG.info("Instantiating model")

        if type(self.model_name) is str:
            device_map = 'auto' if hparams.model_parallel else None
            torch_dtype = torch.float16 if hasattr(hparams, 'fp16') and hparams.fp16 else torch.float32
            
            if hparams.alg_name == 'QLoRA':
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=(hparams.quantization_bit == 4),
                    bnb_4bit_use_double_quant=hparams.double_quant,
                    bnb_4bit_quant_type=hparams.quant_type,
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
                model_kwargs = {
                    "quantization_config": bnb_config,
                    "torch_dtype": torch_dtype,
                    "device_map": {'': hparams.device}
                }
            else:
                model_kwargs = {
                    "torch_dtype": torch_dtype,
                    "device_map": device_map
                }

            if 't5' in self.model_name.lower():
                self.model = T5ForConditionalGeneration.from_pretrained(self.model_name, **model_kwargs)
                self.tok = T5Tokenizer.from_pretrained(self.model_name)
            elif 'chatglm-api' in self.model_name.lower():
                self.model, self.tok = None, None
                self.hparams = hparams
                return
            elif 'gpt-3.5' in self.model_name.lower():
                self.model, self.tok = None, None
            elif 'gpt' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
                self.tok = GPT2Tokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id
            elif 'llama' in self.model_name.lower() or 'vicuna' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
                self.tok = AutoTokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id
            elif 'baichuan' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs, trust_remote_code=True)
                self.tok = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
                self.tok.pad_token_id = self.tok.eos_token_id
            elif 'chatglm' in self.model_name.lower():
                self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True, **model_kwargs)
                self.tok = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
                if 'chatglm2' in self.model_name.lower(): 
                    self.tok.unk_token_id = 64787
                else: 
                    self.tok.pad_token_id = self.tok.eos_token_id
            elif 'internlm' in self.model_name.lower():
                self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True, **model_kwargs)
                self.tok = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
                self.tok.pad_token_id = self.tok.eos_token_id
            elif 'qwen' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True, **model_kwargs)
                self.tok = AutoTokenizer.from_pretrained(self.model_name, eos_token='<|endoftext|>', pad_token='<|endoftext|>', unk_token='<|endoftext|>', trust_remote_code=True)
            elif 'mistral' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
                self.tok = AutoTokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id
            else:
                raise NotImplementedError

            if self.tok is not None and (isinstance(self.tok, GPT2Tokenizer) or isinstance(self.tok, GPT2TokenizerFast) or isinstance(self.tok, LlamaTokenizer) or isinstance(self.tok, LlamaTokenizerFast) or isinstance(self.tok, PreTrainedTokenizerFast)) and (hparams.alg_name not in ['ROME', 'MEMIT', 'EMMET', 'R-ROME', 'AlphaEdit']):
                LOG.info('AutoRegressive Model detected, set the padding side of Tokenizer to left...')
                self.tok.padding_side = 'left'
            if self.tok is not None and ('mistral' in self.model_name.lower() or 'llama' in self.model_name.lower() or 'qwen' in self.model_name.lower()) and (hparams.alg_name in ['ROME', 'MEMIT', 'EMMET', 'R-ROME', 'AlphaEdit']):
                LOG.info('AutoRegressive Model detected, set the padding side of Tokenizer to right...')
                self.tok.padding_side = 'right'
        else:
            self.model, self.tok = self.model_name

        if hparams.model_parallel: 
            hparams.device = str(self.model.device).split(":")[1]
        if not hparams.model_parallel and hasattr(hparams, 'device') and hparams.alg_name != 'QLoRA':
            self.model.to(f'cuda:{hparams.device}')

        self.hparams = hparams

    def edit(self,
             prompts: Union[str, List[str]],
             target_new: Union[str, List[str]],
             ground_truth: Optional[Union[str, List[str]]] = None,
             relation_id: Optional[Union[str, List[str]]] = None,
             target_neg: Optional[Union[str, List[str]]] = None,
             rephrase_prompts: Optional[Union[str, List[str]]] = None,
             locality_inputs: Optional[Dict] = None,
             portability_inputs: Optional[Dict] = None,
             reverse_qa_prompts: Optional[Dict] = None,
             reverse_judge_prompts: Optional[Dict] = None,
             sequential_edit=False,
             verbose=True,
             case_result_template=None,
             num_edits1=None,
             **kwargs
             ):
        """
        `prompts`: list or str
            the prompts to edit
        `relation_id`: list or str
            the relation id for each edit
        `ground_truth`: str
            the ground truth / expected output
        `locality_inputs`: dict
            for locality
        """
        test_generation = kwargs.pop('test_generation', False)

        if isinstance(prompts, List):
            assert len(prompts) == len(target_new)
        else:
            prompts, target_new = [prompts,], [target_new,]

        if hasattr(self.hparams, 'batch_size') and not BatchEditor.is_batchable_method(self.alg_name):
            assert self.hparams.batch_size == 1, 'Single Editing: batch_size should be set to 1'

        if ground_truth is not None:
            ground_truth = [ground_truth,] if isinstance(ground_truth, str) else ground_truth
        else:
            ground_truth = ['<|endoftext|>'] * (len(prompts))

        if relation_id is not None:
            relation_id = [relation_id,] if isinstance(relation_id, str) else relation_id
            assert len(relation_id) == len(prompts), "Relation ID length must match prompts length"

        if "requests" in kwargs.keys():
            requests = kwargs["requests"]
        else:
            requests = _prepare_requests_bi(prompts, target_new, ground_truth, relation_id, target_neg, rephrase_prompts, 
                                            locality_inputs, portability_inputs, reverse_qa_prompts, reverse_judge_prompts, **kwargs)
        return self.edit_requests(requests, sequential_edit, verbose, test_generation=test_generation, case_result_template=case_result_template, num_edits1=num_edits1, **kwargs)

    def batch_edit(self,
                   prompts: List[str],
                   target_new: List[str],
                   relation_id: Optional[List[str]] = None,
                   ground_truth: Optional[List[str]] = None,
                   target_neg: Optional[List[str]] = None,
                   rephrase_prompts: Optional[List[str]] = None,
                   locality_inputs: Optional[Dict] = None,
                   portability_inputs: Optional[Dict] = None,
                   reverse_qa_prompts: Optional[Dict] = None,
                   reverse_judge_prompts: Optional[Dict] = None,
                   sequential_edit=False,
                   verbose=True,
                   case_result_template=None,
                   num_edits1=None,
                   **kwargs
                   ):
        """
        Batch edit with relation_id and reverse prompts support.
        """
        assert len(prompts) == len(target_new)
        test_generation = kwargs['test_generation'] if 'test_generation' in kwargs.keys() else False

        if ground_truth is not None:
            if isinstance(ground_truth, str):
                ground_truth = [ground_truth,]
            else:
                assert len(ground_truth) == len(prompts)
        else:
            ground_truth = ['<|endoftext|>' for _ in range(len(prompts))]

        if relation_id is not None:
            relation_id = [relation_id,] if isinstance(relation_id, str) else relation_id
            assert len(relation_id) == len(prompts), "Relation ID length must match prompts length"

        assert BatchEditor.is_batchable_method(self.alg_name), f'The Method {self.alg_name} can not batch edit examples.'

        requests = self._prepare_requests(prompts, target_new, ground_truth, target_neg, rephrase_prompts,
                                        locality_inputs, portability_inputs, reverse_qa_prompts, reverse_judge_prompts, relation_id, **kwargs)

        assert hasattr(self.hparams, 'batch_size'), f'Method {self.alg_name} found, pls specify the batch_size....'
        all_metrics = []
        id = 0
        for record_chunks in _chunks(requests, self.hparams.batch_size if num_edits1 is None else num_edits1):
            start = time()

            edited_model, weights_copy = self.apply_algo(
                self.model,
                self.tok,
                record_chunks,
                self.hparams,
                copy=False,
                return_orig_weights=True
            )
            exec_time = time() - start
            LOG.info(f"Execution editing took {exec_time}")

            start = time()
            chunk_metrics = []
            for i, request in enumerate(record_chunks):
                metrics = {
                    'case_id': id,
                    "requested_rewrite": request,
                    "time": exec_time,
                    "post": compute_edit_quality(edited_model, self.model_name, self.hparams, self.tok, request, self.hparams.device, test_generation=test_generation),
                }
                chunk_metrics.append(metrics)
                id += 1

            if self.alg_name == 'KN' or self.alg_name == 'GRACE' or self.alg_name == 'WISE':
                with torch.no_grad():
                    weights_copy()
            elif self.alg_name == 'LoRA' or self.alg_name == 'QLoRA' or self.alg_name == 'DPO':
                edited_model.unload()
                del self.model.peft_config
            elif self.alg_name == 'MELO':
                self.model = edited_model
            else:
                with torch.no_grad():
                    for k, v in weights_copy.items():
                        nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")

            for i, request in enumerate(record_chunks):
                chunk_metrics[i]["pre"] = compute_edit_quality(self.model, self.model_name, self.hparams, self.tok, request, self.hparams.device, test_generation=test_generation)

                if verbose:
                    LOG.info(
                        f"{i} editing: {request['prompt']} -> {request['target_new']}  \n {chunk_metrics[i]}"
                    )

                if case_result_template is not None:
                    out_file = Path(case_result_template.format(num_edits1 if num_edits1 is not None else self.hparams.batch_size, id - 1))
                    if not out_file.exists():
                        metrics = {
                            "case_id": id - 1,
                            "num_edits": num_edits1 if num_edits1 is not None else self.hparams.batch_size,
                            "requested_rewrite": request,
                            "post": chunk_metrics[i]["post"],
                            "pre": chunk_metrics[i]["pre"],
                        }
                        with open(out_file, "w") as f:
                            json.dump(metrics, f, indent=1)

            LOG.info(f"Evaluation took {time() - start}")
            all_metrics.extend(chunk_metrics)
        return all_metrics, edited_model, weights_copy


    # Include other methods from editor.py as needed (edit_requests, normal_edit, generate_edit, deep_edit)
    # These can be copied directly from editor.py since they don't need relation_id or reverse prompts support
    def edit_requests(self,
             requests,
             sequential_edit=False,
             verbose=True,
             test_generation=False,
             **kwargs
             ):
        """
        `prompts`: list or str
            the prompts to edit
        `ground_truth`: str
            the ground truth / expected output
        `locality_inputs`: dict
            for locality
        """
        eval_metric= kwargs['eval_metric'] if 'eval_metric' in kwargs.keys() else 'exact match'
        if hasattr(self.hparams, 'batch_size'):  # For Singleton Editing, bs=1
            assert self.hparams.batch_size == 1, 'Single Editing: batch_size should be set to 1'
        all_metrics = []
        if 'pre_edit' in kwargs and kwargs['pre_edit'] is not None:
            metrics = kwargs['pre_edit']
            all_metrics = metrics
        else:
            for i, request in enumerate(tqdm(requests)):
                if self.alg_name == 'IKE':
                    assert 'train_ds' in kwargs.keys(), print('IKE need train_ds(For getting In-Context prompt)')
                    metrics = {"pre": compute_icl_edit_quality_bi(self.model, self.model_name, self.hparams, self.tok, [''], request, self.hparams.device, pre_edit=True)}
                else:
                    print(request)
                    metrics = {"pre": compute_edit_quality_bi(self.model, self.model_name, self.hparams, self.tok, request, self.hparams.device, eval_metric=eval_metric, test_generation=test_generation)}
                all_metrics.append(metrics)
            if 'pre_file' in kwargs and kwargs['pre_file'] is not None:
                json.dump(all_metrics, open(kwargs['pre_file'], 'w'), indent=4)

        def edit_func(request):
            if self.alg_name == 'IKE' or self.alg_name == 'ICE':
                edited_model, weights_copy, icl_examples = self.model, {}, self.apply_algo(
                    self.model,
                    self.tok,
                    [request],
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=False,
                    train_ds=kwargs['train_ds'] if self.alg_name == 'IKE' else None
                )
            else:
                edited_model, weights_copy = self.apply_algo(
                    self.model,
                    self.tok,
                    [request],
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=False,
                    train_ds=kwargs['train_ds'] if self.alg_name == 'IKE' else None
                )
                icl_examples = None
            return edited_model, weights_copy, icl_examples

        def edit_evaluation(all_metrics, request, edited_model, idx, test_generation, icl_examples, **kwargs):
            eval_metric= kwargs['eval_metric'] if 'eval_metric' in kwargs.keys() else 'exact match'
            if self.alg_name == 'IKE':
                all_metrics[idx].update({
                    'case_id': idx,
                    "requested_rewrite": request,
                    "post": compute_icl_edit_quality_bi(self.model, self.model_name, self.hparams, self.tok, icl_examples, request, self.hparams.device ,test_generation=test_generation),
                })
                if "metric_kwargs" in kwargs:
                    all_metrics[idx].update(compute_sent_metric(self.model, edited_model, self.model_name, self.hparams, self.tok,metric_kwargs=kwargs["metric_kwargs"][idx], device=self.hparams.device))
    
            else:
                all_metrics[idx].update({
                    'case_id': idx,
                    "requested_rewrite": request,
                    "post": compute_edit_quality_bi(edited_model, self.model_name, self.hparams, self.tok, request, self.hparams.device, eval_metric=eval_metric, test_generation=test_generation),
                })
                if "metric_kwargs" in kwargs:
                    all_metrics[idx].update(compute_sent_metric(self.model, edited_model, self.model_name, self.hparams, self.tok,metric_kwargs=kwargs["metric_kwargs"][idx], device=self.hparams.device))
                if 'locality' in all_metrics[idx]['post'].keys():
                    for locality_key in request['locality'].keys():
                        locality_result = []
                        if hasattr(self.hparams, 'evaluation_type') and self.hparams.evaluation_type == "LLM-judge":
                            locality_result.append(float(all_metrics[idx]['post']['locality'][f'{locality_key}_output']==all_metrics[idx]['pre']['locality'][f'{locality_key}_output']))
                        else:
                            for ans, label in zip(all_metrics[idx]['post']['locality'][f'{locality_key}_output'], all_metrics[idx]['pre']['locality'][f'{locality_key}_output']):
                                locality_result.append(np.mean(np.equal(ans, label)))
                        all_metrics[idx]['post']['locality'][f'{locality_key}_acc'] = locality_result
                        all_metrics[idx]['post']['locality'].pop(f'{locality_key}_output')
                    all_metrics[idx]['pre'].pop('locality')

            if verbose:
                LOG.info(f"{idx} editing: {request['prompt']} -> {request['target_new']}  \n\n {all_metrics[idx]}")


        if sequential_edit:
            for i, request in enumerate(tqdm(requests, total=len(requests))):
                edited_model, weights_copy, icl_examples = edit_func(request)
            if self.alg_name == 'WISE' and hasattr(self.hparams, 'save_path') and self.hparams.save_path:
                print("Start saving the WISE model!")
                edited_model.save(self.hparams.save_path)
            for i, request in enumerate(requests):
                edit_evaluation(all_metrics, request, edited_model, i, test_generation, icl_examples, **kwargs)
        else:
            for i, request in enumerate(tqdm(requests, total=len(requests))):
                edited_model, weights_copy, icl_examples = edit_func(request)
                edit_evaluation(all_metrics, request, edited_model, i, test_generation, icl_examples, **kwargs)
                if self.alg_name == 'KN' or self.alg_name == 'GRACE' or self.alg_name == 'WISE':
                    with torch.no_grad():
                        weights_copy()
                elif self.alg_name == 'LoRA' or self.alg_name == 'QLoRA' or self.alg_name == 'DPO':
                    edited_model.unload()
                    del self.model.peft_config
                elif self.alg_name == 'MELO':
                    self.model = edited_model
                else:
                    with torch.no_grad():
                        for k, v in weights_copy.items():
                            nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")


        if isinstance(edited_model, LORA):
            edited_model = edited_model.model
        if len(all_metrics) != 0:
            summary_metrics(all_metrics)
        

        return all_metrics, edited_model, weights_copy
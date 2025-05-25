from typing import Any, Dict, List, Tuple
import torch
from copy import deepcopy
from transformers import AutoModelForCausalLM, AutoTokenizer
from .GRACE import GRACE, GRACEMultimodal
from .grace_hparams import GraceHyperParams
from .utils import tokenize, multimodal_tokenize
from ...util import nethook


def apply_grace_to_model(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: GraceHyperParams,
        copy=False,
        return_orig_weights=False,
        keep_original_weight=False,
        **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    request = requests[0]
    if copy:
        model = deepcopy(model)
    device = torch.device(f'cuda:{hparams.device}')
    editor = GRACE(model=model, config=hparams, device=device)
    tokens = tokenize(request, tokenizer=tok, device=device)
    editor.edit(config=hparams, tokens=tokens,edit_id=request['target_new'])
            
    weights_copy = editor.reset_layer


    return editor, weights_copy


def apply_grace_to_multimodal_model(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: GraceHyperParams,
        copy=False,
        return_orig_weights=False,
        keep_original_weight=False,
        **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    device = torch.device(f'cuda:{hparams.device}')
    if copy:
        model = deepcopy(model)
        model.to(device)
    editor = GRACEMultimodal(model=model, config=hparams, device=device)
    tokens = multimodal_tokenize(requests, processor=tok, device=device, hparams=hparams)
    editor.edit(config=hparams, multimodal_tokens=tokens,edit_id=requests[0]['target'])
    weights_copy = editor.reset_layer
    return editor, weights_copy



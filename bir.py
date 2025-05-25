import json
import random
import argparse
from easyeditor import FTHyperParams
from easyeditor import KNHyperParams
from easyeditor import ROMEHyperParams
from easyeditor import WISEHyperParams
from easyeditor import MEMITHyperParams
from easyeditor import GraceHyperParams
from easyeditor import AlphaEditHyperParams
from easyeditor.editors.bi_editor import BiEditor



parser = argparse.ArgumentParser()
parser.add_argument("--alg_name", type=str, required=True)
parser.add_argument("--ds_name", type=str, required=True)
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--cuda", type=int, default=0)
parser.add_argument("--dataset_size", type=int, default=1)
args = parser.parse_args()

hparams_path = f"./hparams/{args.alg_name}/{args.model_name}.yaml"

alg_hparams_map = {
    "FT": FTHyperParams,
    "KN": KNHyperParams,
    "ROME": ROMEHyperParams,
    "WISE": WISEHyperParams,
    "GRACE": GraceHyperParams,
    "MEMIT": MEMITHyperParams,
    "AlphaEdit": AlphaEditHyperParams
}
if args.alg_name not in alg_hparams_map:
    raise ValueError(f"Unsupported alg_name: {args.alg_name}")

HyperParamsClass = alg_hparams_map[args.alg_name]
hparams = HyperParamsClass.from_hparams(hparams_path)

data_path = f"./data/{args.ds_name}.json"
with open(data_path, 'r') as f:
    data = json.load(f)
data = data[:args.dataset_size]

prompts = []
rephrase_prompts = []
target_new = []
subject = []
relation_id = []
reverse_qa = []
reverse_judge = []
ground_truth = []
neighbor = []

for entry in data:
    base_prompt = entry['requested_rewrite']['prompt'].format(entry['requested_rewrite']['subject'])
    prompts.append(base_prompt)

    para = entry.get('paraphrase_prompts', [])
    if para:
        rephrase_prompts.append(random.choice(para))
    else:
        rephrase_prompts.append("")

    target_new.append(entry['requested_rewrite']['target_new']['str'])
    subject.append(entry['requested_rewrite']['subject'])
    relation_id.append(entry['requested_rewrite']['relation_id'])
    reverse_judge.append(entry['reverse_judge'])
    ground_truth.append(entry['requested_rewrite']['target_true']['str'])

    if args.ds_name == "BAKE_qa":
            reverse_qa.append(entry['reverse_qa'])

    all_neighbors = entry.get('neighborhood_prompts', [])
    if all_neighbors:
        one_prompt = random.choice(all_neighbors)
        neighbor.append([one_prompt])
    else:
        neighbor.append([""])

locality_inputs = {
    'neighborhood': {
        'prompt': neighbor,
        'ground_truth': ground_truth
    }
}


loc_prompts = []
for i in range(len(data)):
    loc_prompt = neighbor[i][0] + ' ' + ground_truth[i][0]
    loc_prompts.append(loc_prompt)


editor = BiEditor.from_hparams(hparams)

edit_args = dict(
        prompts=prompts,
        rephrase_prompts=rephrase_prompts,
        target_new=target_new,
        relation_id=relation_id,
        subject=subject,
        train_ds=None,
        reverse_judge_prompts=reverse_judge,
        locality_inputs=locality_inputs
    )
if args.ds_name == "BAKE_qa":
    edit_args["reverse_qa_prompts"] = reverse_qa

metrics, edited_model, _ = editor.edit(**edit_args)
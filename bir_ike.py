import json
import random
import argparse
from easyeditor import IKEHyperParams
from easyeditor.editors.bi_editor import BiEditor
from sentence_transformers import SentenceTransformer
from easyeditor.models.ike.util import encode_ike_facts

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--ds_name", type=str, required=True)
parser.add_argument("--cuda", type=int, default=0)
parser.add_argument("--dataset_size", type=int, default=1)
args = parser.parse_args()

hparams_path = f"./hparams/IKE/{args.model_name}.yaml"
hparams = IKEHyperParams.from_hparams(hparams_path)

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

train_ds = []
for i in range(len(prompts)):
    train_ds.append({
        'prompt': prompts[i],
        'target_new': target_new[i],
        'rephrase_prompt': rephrase_prompts[i],
        'locality_prompt': neighbor[i][0],
        'locality_ground_truth': ground_truth[i]
    })


editor = BiEditor.from_hparams(hparams)

sentence_model = SentenceTransformer(hparams.sentence_model_name)
encode_ike_facts(sentence_model, train_ds, hparams)

edit_args = dict(
        prompts=prompts,
        rephrase_prompts=rephrase_prompts,
        target_new=target_new,
        relation_id=relation_id,
        subject=subject,
        train_ds=train_ds,
        reverse_judge_prompts=reverse_judge,
        locality_inputs=locality_inputs
    )
if args.ds_name == "BAKE_qa":
    edit_args["reverse_qa_prompts"] = reverse_qa

metrics, edited_model, _ = editor.edit(**edit_args)

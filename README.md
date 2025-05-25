# BAKE

## Overview
**Model editing** aims to adjust an initial base model's $(f_\theta)$ behavior($x_e \rightarrow y_e$) on the particular edit descriptor $[x_e, y_e]$ efficiently.
Previous editing and evaluation approaches operate under the **unidirectional** paradigm following only the direction being edited.

This paper study **bidirectional** language model editing, introduing a new evaluation metric of **reversibility** and a new benchmark **BAKE** to assess if edited LLMs can accurately recall the editing knowledge bidirectionally.

We also propose a method BIRD, which mitigates the reversal curse in model editing.

<img src="https://github.com/mjy1111/BAKE/blob/main/definition.png" width="800">

## Datasets
The BAKE benchmark comprises two datasets of BAKE-Q&J and BAKE-J. Both datasets are designed for evaluating counterfactual edits in LLMs. When assess the reversibility of LLMs, two evaluation forms of question answering (Q) and judgment (J) are considered for different relations.

The datasets are included in `data/`. There are two files:
* `BAKE_qa.json`: the counterfactual dataset use both question answering and judgment forms for the evaluation of reversibility, which use one-to-one and one-to-many relations.
* `BAKE_judge.json`: the counterfactual dataset only use judgment form for the evaluation of reversibility, which use many-to-one and many-to-many relations.

Besides,  we split the two datasets into a train set and a validate set to train the hypernetwork for MEND method, included in `data/bi/`. 
The whole data directory is as follows:
```bash
data/
    |__ BAKE_qa.json
    |__ BAKE_judge.json
    |__ bi/
        |__ bi_train.json
        |__ bi_val.json
```
You can download these datasets here. [[Huggingface]](https://huggingface.co/datasets/jym7/BAKE).


## Prepare the environment

### Requirements

**Note: Please use Python 3.9+**
To get started, simply install conda and run:

```shell
git clone https://github.com/mjy1111/BAKE.git
conda create -n BAKE python=3.9.7
...
pip install -r requirements.txt
```

### Models
All models are putted in `hugging_cache/<model_name>` (model_name=gpt2-xl, gpt-j-6B, llama-7b, or llama3-8b,...).
These could be changed in `hparams/<method_name>/`.


## Evaluation
The performance of knowledge editing is measured from these dimensions:

- `Efficacy`: whether the edited models could recall the exact editing fact under editing prompts
- `Generalization`: whether the edited models could recall the editing fact under paraphrase prompts
- `Locality`: whether the output of the edited models for inputs out of editing scope remains unchanged after editing
- `Reversibility`: the effectiveness of edited models in recalling the editing knowledge under reverse prompts.

GPT-2 XL (1.5B), GPT-J (6B), LLaMA-2 (7B), LLaMA-3 (8B) and LLaMA-2 (13B) are used for editing.

- These model editing methods are used in our paper as follows:
  - [FT](https://github.com/kmeng01/rome): Fine-Tuning with $L_\infty$ constraint
  - [MEND](https://github.com/eric-mitchell/mend): Mitchell et al. Hypernetwork
  - [ROME](https://github.com/kmeng01/rome): Kevin Meng et al. Locate then Edit
  - [RECT](https://github.com/JasonForJoy/Model-Editing-Hurt): Jia-Chen Gu et al. Locate then Edit
  - [MEMIT](https://github.com/kmeng01/memit): Kevin Meng et al. Locate then Edit
  - [AlphaEdit](https://github.com/jianghoucheng/AlphaEdit): Junfeng Fang et al. Locate then Edit
  - [GRACE](https://github.com/thartvigsen/grace): Thomas Hartvigsen et al. Memory-based
  - [WISE](https://github.com/zjunlp/EasyEdit): Peng Wang et al. Memory-based
  - [IKE](https://github.com/Zce1112zslx/IKE): Ce Zheng et al. In-context learning


### Running the evaluation
After downloading the datasets and models, to get started (e.g. using ROME to edit LLaMA-3 (8B) on BAKE-Q&J dataset), run:
```bash
python bir.py \
    --alg_name=ROME \
    --model_name=llama3-8b \
    --ds_name=BAKE_qa \
    --dataset_size=100 (optional)
```

All params are in the `hparams/<method_name>/`, and you can change them as needed.

For ROME and MEMIT, we also provide Wikipedia stats [[Google Drive]](https://drive.google.com/file/d/1DrHW5rQ3_0rNHSsH2vFBtv7ePGNHiVj7/view?usp=drive_link).

### Running the IKE method
To use the IKE method, run:
```bash
python bir_ike.py \
    --model_name=llama3-8b \
    --ds_name=BAKE_qa \
    --dataset_size=100 (optional)
```

### Trainer
To use the MEND method, you should firstly train a hypernetwork using the data in `data/bi/`, and these weights would be saved in `data/weights/models/MEND/`.
Then use the same steps above to edit models.
Run:

```bash
python trainer.py
```
You can also download these weights here. [[Google Drive]](https://drive.google.com/file/d/1o9uJUEXExda5M-kyvvyFZ3yAC9tmW9gx/view?usp=drive_link).







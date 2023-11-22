# BAKE
This is the repository for our paper: Untying the Reversal Curse via Bidirectional Language Model Editing ([arxiv](https://arxiv.org/pdf/2310.10322.pdf))

## Overview
**Knowledge editing** aims to adjust an initial base model's $(f_\theta)$ behavior($x_e \rightarrow y_e$) on the particular edit descriptor $[x_e, y_e]$ efficiently.
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

## Evaluation
The performance of knowledge editing is measured from these dimensions:

- `Efficacy`: whether the edited models could recall the exact editing fact under editing prompts
- `Generalization`: whether the edited models could recall the editing fact under paraphrase prompts
- `Locality`: whether the output of the edited models for inputs out of editing scope remains unchanged after editing
- `Reversibility`: the effectiveness of edited models in recalling the editing knowledge under reverse prompts.

GPT-2 XL (1.5B), GPT-J (6B), LLaMA-1 (7B) and LLaMA-2 (7B) are used for editing.











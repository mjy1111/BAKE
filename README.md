# BAKE
This is the repository for our paper: Untying the Reversal Curse via Bidirectional Language Model Editing ([arxiv](https://arxiv.org/pdf/2310.10322.pdf))

## Overview
**Knowledge editing** aims to adjust an initial base model's $(f_\theta)$ behavior($x_e \rightarrow y_e$) on the particular edit descriptor $[x_e, y_e]$ efficiently.
Previous editing and evaluation approaches operate under the **unidirectional** paradigm following only the direction being edited.

This paper study **bidirectional** language model editing, introduing a new evaluation metric of **reversibility** and a new benchmark **BAKE** to assess if edited LLMs can accurately recall the editing knowledge bidirectionally.

We also propose a method BIRD, which mitigates the reversal curse in model editing.

![Image text](https://github.com/mjy1111/BAKE/blob/main/definition.png)


## Datasets
The BAKE benchmark comprises two datasets of BAKE-Q&J and BAKE-J. Both datasets are designed for evaluating counterfactual edits in LLMs. When assess the reversibility of LLMs, two evaluation forms of question answering (Q) and judgment (J) are considered for different relations.

The datasets are included in `data`. There are two files:
* `BAKE_qa.json`: the counterfactual dataset use both question answering and judgment forms for the evaluation of reversibility, which use one-to-one and one-to-many relations.
* `BAKE_judge.json`: the counterfactual dataset only use judgment form for the evaluation of reversibility, which use many-to-one and many-to-many relations.

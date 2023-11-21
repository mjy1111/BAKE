# BAKE
This is the repository for our paper: Untying the Reversal Curse via Bidirectional Language Model Editing ([arxiv](https://arxiv.org/pdf/2310.10322.pdf))

## Overview
**Knowledge editing** aims to adjust an initial base model's $(f_\theta)$ behavior($x_e \rightarrow y_e$) on the particular edit descriptor $[x_e, y_e]$ efficiently.
Previous editing and evaluation approaches operate under the **unidirectional** paradigm following only the direction being edited.

This paper study **bidirectional** language model editing, introduing a new evaluation metric of **reversibility** and a new benchmark **BAKE** to assess if edited LLMs can accurately recall the editing knowledge bidirectionally.

We also propose a method BIRD, which mitigates the reversal curse in model editing.

![Image text](https://github.com/mjy1111/BAKE/blob/main/definition.png)


## Evaluation

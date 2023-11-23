import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


import hydra
from easyeditor import BaseEditor
from easyeditor import KNHyperParams, FTHyperParams, KETrainingHparams,\
    ROMEHyperParams, MEMITHyperParams, MENDTrainingHparams, MENDHyperParams, \
    SERACTrainingHparams, SERACHparams, IKEHyperParams, FTApiHyperParams
from easyeditor import ZsreDataset, CounterFactDataset
from easyeditor import EditTrainer
from easyeditor.models.ike import encode_ike_facts
from sentence_transformers import SentenceTransformer
from pathlib import Path


def test_MEND_coun_Train():
    training_hparams = MENDTrainingHparams.from_hparams('./hparams/TRAINING/MEND/gpt2-xl')
    #print(Path('./data/zsre/zsre_mend_train.json'))

    train_ds = CounterFactDataset('./data/bi/bi_train.json', config=training_hparams)
    eval_ds = CounterFactDataset('./data/bi/bi_val.json', config=training_hparams)
    trainer = EditTrainer(
        config=training_hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    #print(1)

    trainer.run()



test_MEND_coun_Train()



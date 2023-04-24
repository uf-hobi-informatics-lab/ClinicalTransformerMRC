#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# file: flat_inference.sh
#

REPO_PATH=/home/alexgre/projects/2022n2c2/2022n2c2_track2/mrc_ner
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"

DATA_SIGN=conll03
DATA_DIR=/home/alexgre/projects/2022n2c2/2022n2c2_track2/mrc_ner/datasets/conll03/
BERT_DIR=/home/alexgre/projects/transformer_pretrained_models/mimiciii_bert-base-uncased_10e_128b
MAX_LEN=512

model_root=/home/alexgre/projects/2022n2c2/2022n2c2_track2/exp/model/conll03_cased_large_cased_large_lr3e-5_drop0.3_norm_weight0.1_warmup0_maxlen
MODEL_CKPT=${model_root}/epoch=14_v0.ckpt
HPARAMS_FILE=${model_root}/lightning_logs/version_0/hparams.yaml


python3 ${REPO_PATH}/inference/mrc_ner_inference.py \
--data_dir ${DATA_DIR} \
--bert_dir ${BERT_DIR} \
--max_length ${MAX_LEN} \
--model_ckpt ${MODEL_CKPT} \
--hparams_file ${HPARAMS_FILE} \
--flat_ner \
--dataset_sign ${DATA_SIGN}
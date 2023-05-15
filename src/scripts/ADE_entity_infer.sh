#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# file: nested_inference.sh
#

REPO_PATH=/home/c.peng/projects/mrc-for-ner-medical/mrc_ner
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"

DATA_SIGN=drug_ADE

DATA_DIR=/data/datasets/cheng/mrc-for-ner-medical/2018_n2c2/data/mrc_entity/


# FILE=gatortron-syn-345m_deid_vocab

BERT_DIR=/home/alexgre/projects/transformer_pretrained_models/bert-large-cased

MAX_LEN=512

FILE=1203_bert-large-cased_2_4_3e-5_20

predict_output=/data/datasets/cheng/mrc-for-ner-medical/exp/pred_${FILE}.json

OUTPUT_BASE=/data/datasets/cheng/mrc-for-ner-medical/2018_n2c2/exp/ner/model

model_root=${OUTPUT_BASE}/1203_bert-large-cased_2_4_3e-5_20

MODEL_CKPT=${model_root}/epoch=11.ckpt

HPARAMS_FILE=${model_root}/lightning_logs/version_0/hparams.yaml


python3 ${REPO_PATH}/inference/mrc_ner_inference.py \
--data_dir ${DATA_DIR} \
--bert_dir ${BERT_DIR} \
--max_length ${MAX_LEN} \
--model_ckpt ${MODEL_CKPT} \
--hparams_file ${HPARAMS_FILE} \
--output_fn ${predict_output} \
--dataset_sign ${DATA_SIGN}
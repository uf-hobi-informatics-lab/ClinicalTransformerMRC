#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# file: nested_inference.sh
#

REPO_PATH=/home/c.peng/projects/ClinicalTransformerMRC/src
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"

DATA_SIGN=sdoh_other

DATA_DIR=/data/datasets/cheng/mrc-for-ner-medical/2022n2c2/2022n2c2_track2/data/mrc_attribute/


# FILE=gatortron-syn-345m_deid_vocab

BERT_DIR=/home/alexgre/projects/transformer_pretrained_models/bert-large-uncased

MAX_LEN=512

FILE=bert-large-uncased_4_4_1e-5_30_attribute

predict_output=/data/datasets/cheng/mrc-for-ner-medical/exp/results/attribute_extraction/pred_${FILE}_test.json

OUTPUT_BASE=/data/datasets/cheng/mrc-for-ner-medical/exp/model

model_root=${OUTPUT_BASE}/1025_bert-large-uncased_4_4_1e-5_30

MODEL_CKPT=${model_root}/epoch=12.ckpt

HPARAMS_FILE=${model_root}/lightning_logs/version_0/hparams.yaml


python3 ${REPO_PATH}/inference/mrc_ner_inference.py \
--data_dir ${DATA_DIR} \
--bert_dir ${BERT_DIR} \
--max_length ${MAX_LEN} \
--model_ckpt ${MODEL_CKPT} \
--hparams_file ${HPARAMS_FILE} \
--output_fn ${predict_output} \
--dataset_sign ${DATA_SIGN}
#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# file: nested_inference.sh
#

REPO_PATH=/home/c.peng/projects/ClinicalTransformerMRC/src
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"

DATA_DIR=/data/datasets/cheng/mrc-for-ner-medical/2022n2c2/2022n2c2_track2/data/mrc_entity

DATASET_SIGN=/data/datasets/cheng/mrc-for-ner-medical/exp/model/entity/gatortron-og-345m_deid_vocab_8_4_1e-5_20/label2idx.json

BERT_DIR=/data/datasets/alexgre/transformer_pretrained_models/gatortron-og-345m_deid_vocab

MAX_LEN=512

FILE=gatortron-og-345m_deid_vocab_8_4_1e-5_20

predict_output=/data/datasets/cheng/mrc-for-ner-medical/exp/results/entity/pred_${FILE}_batch.json

OUTPUT_BASE=/data/datasets/cheng/mrc-for-ner-medical/exp/model/entity

model_root=${OUTPUT_BASE}/gatortron-og-345m_deid_vocab_8_4_1e-5_20

MODEL_CKPT=${model_root}/epoch=8_v0.ckpt

HPARAMS_FILE=${model_root}/lightning_logs/version_57564926/hparams.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 ${REPO_PATH}/inference/mrc_ner_inference.py \
--data_dir ${DATA_DIR} \
--bert_dir ${BERT_DIR} \
--max_length ${MAX_LEN} \
--model_ckpt ${MODEL_CKPT} \
--hparams_file ${HPARAMS_FILE} \
--output_fn ${predict_output} \
--dataset_sign ${DATASET_SIGN} 
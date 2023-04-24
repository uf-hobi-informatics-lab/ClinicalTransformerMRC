#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# file: eval.sh

REPO_PATH=/home/c.peng/projects/mrc-for-ner-medical/mrc_ner
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"

OUTPUT_DIR=/data/datasets/cheng/mrc-for-ner-medical/exp/model/bert-large-uncased_8_4_3e-5_20
# find best checkpoint on dev in ${OUTPUT_DIR}/train_log.txt
BEST_CKPT_DEV=${OUTPUT_DIR}/epoch=19_v0.ckpt
PYTORCHLIGHT_HPARAMS=${OUTPUT_DIR}/lightning_logs/version_0/hparams.yaml
GPU_ID=0,1

python3 ${REPO_PATH}/evaluate/mrc_ner_evaluate.py ${BEST_CKPT_DEV} ${PYTORCHLIGHT_HPARAMS} ${GPU_ID}
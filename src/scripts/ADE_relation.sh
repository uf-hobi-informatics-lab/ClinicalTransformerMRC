#!/usr/bin/env bash
# -*- coding: utf-8 -*-


# training
TIME=0310
REPO_PATH=/home/c.peng/projects/mrc-for-ner-medical/mrc_ner
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"
# export PL_TORCH_DISTRIBUTED_BACKEND=gloo

DATA_DIR=/data/datasets/cheng/mrc-for-ner-medical/2018_n2c2/data/mrc_relation/

# bert-large-cased
# bert-large-uncased
# bert-base-uncased
# mimiciii-bert-large-uncased_5e_128b
# mimiciii_bert-base-uncased_10e_128b
# gatortron-syn-345m_deid_vocab
# 345m_uf_syn_pubmed_mimic_wiki_fullcased50k_megatronv22_release
# 345m_uf_full_deid_pubmed_mimic_wiki_fullcased50k_release
# gatortron-og-345m_deid_vocab
# mimiciii_deberta-base_5e_128b
# mimiciii_roberta-large_5e_128b
# gatortron_4b
FILE=gatortron_4b
# BERT_DIR=/data/datasets/cheng/transformer-pretrained/${FILE}
BERT_DIR=/data/datasets/cheng/GatorTron/model/${FILE}

# MODEL_TYPE=bert
MODEL_TYPE=megatron
# MODEL_TYPE=deberta
# MODEL_TYPE=roberta

OUTPUT_BASE=/data/datasets/cheng/mrc-for-ner-medical/2018_n2c2/exp/re/model

BATCH=1
GRAD_ACC=4
BERT_DROPOUT=0.1
MRC_DROPOUT=0.3 # 0.3 vs 0.1
LR=2e-5
LR_MINI=1e-7
LR_SCHEDULER=polydecay
SPAN_WEIGHT=0.1
WARMUP=0
MAX_LEN=512
MAX_NORM=1.0
MAX_EPOCH=20
INTER_HIDDEN=2048
WEIGHT_DECAY=0.05  # 0.02 vs 0.05
OPTIM=torch.adam #adamw
VAL_CHECK=0.2
PREC=32
SPAN_CAND=pred_and_gold

# FILE=gatortron-syn-345m_deid_vocab
# BATCH=8
# GRAD_ACC=4
# MRC_DROPOUT=0.1
# WEIGHT_DECAY=0.05
# INTER_HIDDEN=2048
# PREC=32

OUTPUT_DIR=${OUTPUT_BASE}/${TIME}_${FILE}_${BATCH}_${GRAD_ACC}_${LR}_${MAX_EPOCH}
mkdir -p ${OUTPUT_DIR}

CUDA_VISIBLE_DEVICES=0,1,2,3 python ${REPO_PATH}/train/mrc_ner_trainer.py \
--data_dir ${DATA_DIR} \
--model_type $MODEL_TYPE \
--bert_config_dir ${BERT_DIR} \
--max_length ${MAX_LEN} \
--batch_size ${BATCH} \
--gpus="4" \
--precision=${PREC} \
--progress_bar_refresh_rate 1 \
--lr ${LR} \
--val_check_interval ${VAL_CHECK} \
--accumulate_grad_batches ${GRAD_ACC} \
--default_root_dir ${OUTPUT_DIR} \
--mrc_dropout ${MRC_DROPOUT} \
--bert_dropout ${BERT_DROPOUT} \
--max_epochs ${MAX_EPOCH} \
--span_loss_candidates ${SPAN_CAND} \
--weight_span ${SPAN_WEIGHT} \
--warmup_steps ${WARMUP} \
--distributed_backend=ddp \
--max_length ${MAX_LEN} \
--gradient_clip_val ${MAX_NORM} \
--weight_decay ${WEIGHT_DECAY} \
--optimizer ${OPTIM} \
--lr_scheduler ${LR_SCHEDULER} \
--classifier_intermediate_hidden_size ${INTER_HIDDEN} \
--lr_mini ${LR_MINI}
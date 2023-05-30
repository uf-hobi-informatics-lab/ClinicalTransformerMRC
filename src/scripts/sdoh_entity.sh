#!/usr/bin/env bash
# -*- coding: utf-8 -*-


# training
REPO_PATH=/home/c.peng/projects/ClinicalTransformerMRC/src
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"

DATA_DIR=/red/gatortron-phi/workspace/mrc_medical/data/2022n2c2/2022n2c2_track2/data/mrc_entity/
TOTAL_CATEGORY=14

# bert-large-cased
# bert-large-uncased
# bert-base-uncased
# mimiciii-bert-large-uncased_5e_128b
# mimiciii_bert-base-uncased_10e_128b
# gatortron-syn-345m_deid_vocab
# 345m_uf_syn_pubmed_mimic_wiki_fullcased50k_megatronv22_release
# 345m_uf_full_deid_pubmed_mimic_wiki_fullcased50k_release
# gatortron-og-345m_deid_vocab
FILE=gatortron-og-345m_deid_vocab
BERT_DIR=/red/gatortron-phi/workspace/mrc_medical/transformer_pretrained_models/${FILE}

# MODEL_TYPE=bert
MODEL_TYPE=megatron

OUTPUT_BASE=/red/gatortron-phi/workspace/mrc_medical/exp/2022n2c2

BATCH=4
GRAD_ACC=4
BERT_DROPOUT=0.1 #0.1
MRC_DROPOUT=0.1 # 0.3 vs 0.1
LR=3e-5
LR_MINI=1e-7
LR_SCHEDULER=polydecay
SPAN_WEIGHT=0.1
WARMUP=0
MAX_LEN=512
MAX_NORM=1.0
MAX_EPOCH=20 #100
INTER_HIDDEN=2048 #4096
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



OUTPUT_DIR=${OUTPUT_BASE}/model/entity/${FILE}_${BATCH}_${LR}_${LR_MINI}_${MAX_EPOCH}_${BERT_DROPOUT}
mkdir -p ${OUTPUT_DIR}


CUDA_VISIBLE_DEVICES=0,1,2,3 python3 ${REPO_PATH}/train/mrc_ner_trainer.py \
--data_dir ${DATA_DIR} \
--model_type $MODEL_TYPE \
--bert_config_dir ${BERT_DIR} \
--max_length ${MAX_LEN} \
--batch_size ${BATCH} \
--gpus="2" \
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


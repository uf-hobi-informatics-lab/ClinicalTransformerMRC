# Clinical Transformer MRC

## Aim
The package is the implementation of a transformer based MRC (Machine Reading Comprehension) system for clinical information extraction task. We aim to provide a simple and quick tool for researchers to conduct clinical NER and RE without comprehensive knowledge of transformers. 

## Available models
- BERT (base, large, mimiciii-pretrained)
- RoBERTa (base, large, mimiciii-pretrained)
- GatorTron (a large clinical language model developed in our previous work, which is pretrained from scratch using >90 billion words of text, which includes >82 billion words of de-identified clinical text)
> We will keep adding new models.

## Install Requirements

* The code requires Python 3.6+.

* If you are working on a GPU machine with CUDA 10.1, please run `pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html` to install PyTorch. If not, please see the [PyTorch Official Website](https://pytorch.org/) for instructions.

* Then run the following script to install the remaining dependenices: `pip install -r requirements.txt`

We build our project on [pytorch-lightning.](https://github.com/PyTorchLightning/pytorch-lightning)
If you want to know more about the arguments used in our training scripts, please 
refer to [pytorch-lightning documentation.](https://pytorch-lightning.readthedocs.io/en/latest/)

## Baseline

We released code and scripts for [clinical concept extraction](https://github.com/uf-hobi-informatics-lab/ClinicalTransformerNER) and [relation extraction](https://github.com/uf-hobi-informatics-lab/ClinicalTransformerRelationExtraction) using transformer-based models, where we fine-tuned transformer-based models and treated clinical concept and relation extraction as sequence labeling task and classification task, respectively. <br>



## usage and example
- prepare datasets

- training
> please refer to the wiki page for all details of the parameters
> [flag details](https://github.com/uf-hobi-informatics-lab/ClinicalTransformerRelationExtraction/wiki/all-parameters)

```shell script
python ./src/train/mrc_ner_trainer.py \
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
      --gradient_clip_val ${MAX_NORM} \
      --weight_decay ${WEIGHT_DECAY} \
      --optimizer ${OPTIM} \
      --lr_scheduler ${LR_SCHEDULER} \
      --classifier_intermediate_hidden_size ${INTER_HIDDEN} \
      --lr_mini ${LR_MINI}
```

- prediction
```shell script
python ./src/inference/mrc_ner_inference.py \
      --data_dir ${DATA_DIR} \
      --bert_dir ${BERT_DIR} \
      --max_length ${MAX_LEN} \
      --model_ckpt ${MODEL_CKPT} \
      --hparams_file ${HPARAMS_FILE} \
      --output_fn ${predict_output} \
      --dataset_sign ${DATA_SIGN}

```

-post-processing (we only support transformation to brat format)

## Reference
We have a preprint at
```
@article{peng2023clinical,
  title={Clinical Concept and Relation Extraction Using Prompt-based Machine Reading Comprehension},
  author={Peng, Cheng and Yang, Xi and Yu, Zehao and Bian, Jiang and Hogan, William R and Wu, Yonghui},
  journal={arXiv preprint arXiv:2303.08262},
  year={2023}
}
```
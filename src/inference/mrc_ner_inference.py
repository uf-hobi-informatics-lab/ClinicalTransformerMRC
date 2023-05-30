#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: mrc_ner_inference.py

import os
import json
import torch
import argparse
from torch.utils.data import DataLoader
from utils.random_seed import set_random_seed
set_random_seed(0)
from train.mrc_ner_trainer import BertLabeling
from tokenizers import BertWordPieceTokenizer
from datasets.mrc_ner_dataset import MRCNERDataset
from metrics.functional.query_span_f1 import extract_flat_spans, extract_nested_spans
from datasets.collate_functions import collate_to_max_length

def get_dataloader(config, data_prefix="test"):
    data_path = os.path.join(config.data_dir, f"mrc-ner.{data_prefix}")
    vocab_path = os.path.join(config.bert_dir, "vocab.txt")
    data_tokenizer = BertWordPieceTokenizer(vocab_path)

    dataset = MRCNERDataset(json_path=data_path,
                            tokenizer=data_tokenizer,
                            max_length=config.max_length,
                            is_chinese=config.is_chinese,
                            pad_to_maxlen=False)

    # dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=False, collate_fn=collate_to_max_length)
    return dataloader, data_tokenizer

def get_query_index_to_label_cate(l2i_fn):
    import json
    print(l2i_fn)
    with open(l2i_fn, "r") as f:
        label2idx = json.load(f)
    
    return {v: k for k, v in label2idx.items()}
    # NOTICE: need change if you use other datasets.
    # please notice it should in line with the mrc-ner.test/train/dev json file
    # if dataset_sign == "conll03":
    #     return {1: "ORG", 2: "PER", 3: "LOC", 4: "MISC"}
    # elif dataset_sign == "ace04":
    #     return {1: "GPE", 2: "ORG", 3: "PER", 4: "FAC", 5: "VEH", 6: "LOC", 7: "WEA"}
    # elif dataset_sign == "sdoh_other":
    #     # return {1: 'TypeLiving', 2: 'Method', 3: 'StatusEmploy', 4: 'Duration', 5: 'Frequency', 
    #     # 6: 'StatusTime', 7: 'Type', 8: 'Amount', 9: 'History'}
    #     return {1: 'StatusEmploy', 2:'Type', 3:'Duration', 4: 'History', 5:'TypeLiving', 6:'StatusTime', 7:'Duration', 8:'History', 9:'Duration', 10:'History', 11:'StatusTime', 12:'Type', 13:'Amount', 14:'Method', 15:'Frequency', 16: 'Duration', 17:'History', 18:'StatusTime', 19:'Type', 20:'Amount', 21:'Method', 22:'Frequency', 23:'Duration', 24:'History', 25:'StatusTime', 26:'Type', 27:'Amount', 28:'Method', 29:'Frequency'}
    # elif dataset_sign == "sdoh_trigger":
    #     return {1: 'Employment', 2: 'LivingStatus', 3: 'Alcohol', 4: 'Drug', 5: 'Tobacco'}
    # elif dataset_sign == "sdoh_entity":
    #     return {1: "Employment", 2: "LivingStatus", 3: "Alcohol", 4: "Drug", 5: "Tobacco", 6: "TypeLiving", 7: "Method", 8: "StatusEmploy", 9: "Duration", 10: "Frequency", 11: "StatusTime", 12: "Type", 13: "Amount", 14: "History"}
    # elif dataset_sign == "drug_ADE":
    #     return { 1: "Drug", 2: "Strength", 3: "Form", 4: "Dosage", 5: "Frequency", 6: "Route", 7:"Duration", 8: "Reason", 9: "ADE"}
    # elif dataset_sign == "drug_ADE_relation":
    #     return { 1: "Strength", 2: "Form", 3: "Dosage", 4: "Frequency", 5: "Route", 6:"Duration", 7: "Reason", 8: "ADE"}
        

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="inference the model output.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--bert_dir", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--is_chinese", action="store_true")
    parser.add_argument("--model_ckpt", type=str, default="")
    parser.add_argument("--hparams_file", type=str, default="")
    parser.add_argument("--flat_ner", action="store_true",)
    parser.add_argument("--dataset_sign", type=str, default="./label2idx.json",)
        # choices=["ontonotes4", "msra", "conll03", "ace04", "ace05", "sdoh_trigger", "sdoh_entity", "sdoh_other","drug_ADE", "drug_ADE_relation"], default="conll03")
    parser.add_argument("--output_fn", type=str, default="./predict_result.json")

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    device = torch.device('cuda')
    trained_mrc_ner_model = BertLabeling.load_from_checkpoint(
        checkpoint_path=args.model_ckpt,
        hparams_file=args.hparams_file,
        map_location=device,
        batch_size=128,
        max_length=args.max_length,
        workers=4)
    
    trained_mrc_ner_model.model = trained_mrc_ner_model.model.to(device)

    print(next(trained_mrc_ner_model.model.parameters()).is_cuda)

    data_loader, data_tokenizer = get_dataloader(args,)
    # load token
    vocab_path = os.path.join(args.bert_dir, "vocab.txt")
    with open(vocab_path, "r") as f:
        subtokens = [token.strip() for token in f.readlines()]
    idx2tokens = {}
    for token_idx, token in enumerate(subtokens):
        idx2tokens[token_idx] = token

    query2label_dict = get_query_index_to_label_cate(args.dataset_sign)

    # we need to store context, predict labels and sample_idx
    results = []

    for i, batch in enumerate(data_loader):
        # d = dict()

        # tokens, token_type_ids, start_labels, end_labels, start_label_mask, end_label_mask, match_labels, sample_idx, label_idx = batch
        tokens, token_type_ids, start_labels, end_labels, start_label_mask, end_label_mask, match_labels, sample_idx, head_sent_idx, tail_sent_idx, label_idx = batch

        # Move tensors to GPU
        tokens = tokens.to(device)
        token_type_ids = token_type_ids.to(device)
        attention_mask = (tokens != 0).long().to(device)
        start_label_mask = start_label_mask.to(device)
        end_label_mask = end_label_mask.to(device)


        start_logits, end_logits, span_logits = trained_mrc_ner_model.model(
            tokens, attention_mask=attention_mask, token_type_ids=token_type_ids)
        start_preds, end_preds, span_preds = start_logits > 0, end_logits > 0, span_logits > 0

        # subtokens_idx_lst = tokens.numpy().tolist()[0]
        subtokens_idx_lst = tokens.cpu().numpy().tolist()

        # subtokens_lst = [idx2tokens[item] for item in subtokens_idx_lst]
        subtokens_lst = [[idx2tokens[item] for item in instance] for instance in subtokens_idx_lst]

        # label_cate = query2label_dict[label_idx.item()]
        label_cate = [query2label_dict[label.item()] for label in label_idx]

        for j in range(len(tokens)):
            d = dict()
            readable_input_str = data_tokenizer.decode(subtokens_idx_lst[j], skip_special_tokens=False)

            d['text'] = readable_input_str

            if args.flat_ner:
                entities_info = extract_flat_spans(start_preds[j].unsqueeze(0), end_preds[j].unsqueeze(0), span_preds[j].unsqueeze(0), 
                                       attention_mask[j].unsqueeze(0), pseudo_tag=label_cate[j])
                entity_lst = []

                if len(entities_info) != 0:
                    for entity_info in entities_info:
                        start, end = entity_info[0], entity_info[1]
                        entity_string = " ".join(subtokens_lst[j][start: end])
                        entity_string = entity_string.replace(" ##", "")
                        entity_lst.append((start, end, entity_string, entity_info[2]))

            else:
                match_preds = span_logits[j].unsqueeze(0) > 0
                entities_info = extract_nested_spans(start_preds[j].unsqueeze(0), end_preds[j].unsqueeze(0), match_preds, 
                                         start_label_mask[j].unsqueeze(0), end_label_mask[j].unsqueeze(0), pseudo_tag=label_cate[j])

                entity_lst = []

                if len(entities_info) != 0:
                    for entity_info in entities_info:
                        start, end = entity_info[0], entity_info[1]
                        entity_string = " ".join(subtokens_lst[j][start: end+1])
                        entity_string = entity_string.replace(" ##", "")
                        entity_lst.append((start, end+1, entity_string, entity_info[2]))

            d['en'] = entity_lst
            d['token_list'] = subtokens_lst[j]
            d['sample_idx'] = sample_idx[j]
            d['label_idx'] = label_idx[j].item()
            d['head_sent_idx'] = head_sent_idx[j].tolist()
            d['tail_sent_idx'] = tail_sent_idx[j].tolist()
            results.append(d)


        # if i < 1:
        #     print("*="*10)
        #     print(entity_info)
        #     print(f"Given input: {readable_input_str}")
        #     print(f"Model predict: {entity_lst}")
        # entity_lst is a list of (subtoken_start_pos, subtoken_end_pos, substring, entity_type)

    with open(args.output_fn, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
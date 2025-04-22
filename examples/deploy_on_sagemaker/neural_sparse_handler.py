# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import os
import itertools
import json
import torch
import transformers
from ts.torch_handler.base_handler import BaseHandler

model_id = "cross-encoder/ms-marco-MiniLM-L6-v2"
max_bs = 50

def merge_features(query_features, doc_features, sep_token_id=102, max_length=256):
    q_input_ids = query_features["input_ids"]      # [B, query_seq_len]
    q_attention = query_features["attention_mask"]   # [B, query_seq_len]
    d_input_ids = doc_features["input_ids"]          # [B, doc_seq_len]
    d_attention = doc_features["attention_mask"]       # [B, doc_seq_len]
    
    batch_size = q_input_ids.size(0)
    q_seq_len = q_input_ids.size(1)
    d_seq_len = d_input_ids.size(1)

    max_length = max(q_seq_len + d_seq_len + 1, max_length)
    
    merged_input_ids_list = []
    merged_attention_list = []
    merged_token_type_ids_list = []

    for i in range(batch_size):
        # 取有效的 query/doc token
        q_valid = q_input_ids[i][q_attention[i] == 1]
        d_valid = d_input_ids[i][d_attention[i] == 1]

        # 拼接 query + [SEP] + doc
        input_ids = torch.cat([q_valid, torch.tensor([sep_token_id], device=q_input_ids.device), d_valid])
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        token_type_ids = torch.cat([
            torch.zeros_like(q_valid),                        # query 部分
            torch.zeros(1, dtype=torch.long, device=q_input_ids.device),  # [SEP]
            torch.ones_like(d_valid)                          # doc 部分
        ])

        # 截断或 padding 到 max_length
        pad_len = max_length - input_ids.size(0)
        if pad_len > 0:
            input_ids = torch.cat([input_ids, torch.full((pad_len,), 0, device=input_ids.device)])
            attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=torch.long, device=input_ids.device)])
            token_type_ids = torch.cat([token_type_ids, torch.zeros(pad_len, dtype=torch.long, device=input_ids.device)])
        else:
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            token_type_ids = token_type_ids[:max_length]

        merged_input_ids_list.append(input_ids)
        merged_attention_list.append(attention_mask)
        merged_token_type_ids_list.append(token_type_ids)

    # 拼接回 batch tensor
    merged_input_ids = torch.stack(merged_input_ids_list)         # [B, max_length]
    merged_attention = torch.stack(merged_attention_list)         # [B, max_length]
    merged_token_type_ids = torch.stack(merged_token_type_ids_list)  # [B, max_length]

    return {
        "input_ids": merged_input_ids,
        "attention_mask": merged_attention,
        "token_type_ids": merged_token_type_ids,
    }

class CrossEncoderEncodingModelHandler(BaseHandler):
    class CrossEncoderModel(torch.nn.Module):
        @staticmethod
        def from_pretrained(path):
            return CrossEncoderEncodingModelHandler.CrossEncoderModel(path)

        def __init__(self, model_id):
            super().__init__()
            self.backbone = transformers.AutoModelForSequenceClassification.from_pretrained(model_id)

        def forward(self, **kwargs):
            output = self.backbone(**kwargs)
            return output.logits.squeeze(-1)
        
    def __init__(self):
        super().__init__()
        self.special_token_ids = None
        self.tokenizer = None
        self.all_tokens = None
        self.initialized = False

    def initialize(self, context):
        self.manifest = context.manifest
        properties = context.system_properties

        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available()
            else "cpu"
        )
        self.model = CrossEncoderEncodingModelHandler.CrossEncoderModel.from_pretrained(model_id)
        self.model.to(self.device).half()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        self.initialized = True

    def preprocess(self, requests):
        batch_idx = []
        query_features = []
        doc_features = []

        for request in requests:
            request_body = request.get("body")
            if isinstance(request_body, bytearray):
                request_body = request_body.decode("utf-8")
                request_body = json.loads(request_body)

            assert isinstance(request_body, list) and len(request_body) == 2, "Input must be [query, doc] pair"

            query = request_body[0]
            doc = request_body[1]

            query_features.append({
                "input_ids": torch.tensor(query["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(query["attention_mask"], dtype=torch.long)
            })
            doc_features.append({
                "input_ids": torch.tensor(doc["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(doc["attention_mask"], dtype=torch.long)
            })

            batch_idx.append(1)

        # 拼 batch
        query_input = {
            "input_ids": torch.cat([x["input_ids"].unsqueeze(0) for x in query_features], dim=0).to(self.device),
            "attention_mask": torch.cat([x["attention_mask"].unsqueeze(0) for x in query_features], dim=0).to(self.device),
        }
        doc_input = {
            "input_ids": torch.cat([x["input_ids"].unsqueeze(0) for x in doc_features], dim=0).to(self.device),
            "attention_mask": torch.cat([x["attention_mask"].unsqueeze(0) for x in doc_features], dim=0).to(self.device),
        }
        
        merged = merge_features(query_input, doc_input, max_length=512)

        return {"input":merged, "batch_l": batch_idx}


    def inference(self, data, *args, **kwargs):
            batch_idx = data["batch_l"]
            data_input = data["input"]

            total_samples = data_input["input_ids"].shape[0]
            outputs = []

            for start_idx in range(0, total_samples, max_bs):
                end_idx = min(start_idx + max_bs, total_samples)

                batch_data = {
                    "input_ids": data_input["input_ids"][start_idx:end_idx],
                    "attention_mask": data_input["attention_mask"][start_idx:end_idx],
                }

                with torch.cuda.amp.autocast(), torch.no_grad():
                    output = self.model(**batch_data)
                    outputs.append(output)

            output = torch.cat(outputs, dim=0)
            return {"pred": output, "batch_l": batch_idx}

    def postprocess(self, prediction):
        batch_idx = prediction["batch_l"]
        output = prediction["pred"]

        outputs = []
        index = 0
        for b in batch_idx:
            outputs.append(output[index : index + b])
            index += b
        return outputs

    def handle(self, data, context):
        model_input = self.preprocess(data)
        model_output = self.inference(model_input)
        return self.postprocess(model_output)
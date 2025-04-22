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
    # 查询部分
    q_input_ids = query_features["input_ids"]      # [B, query_seq_len]
    q_attention = query_features["attention_mask"]   # [B, query_seq_len]
    # 文档部分
    d_input_ids = doc_features["input_ids"]          # [B, doc_seq_len]
    d_attention = doc_features["attention_mask"]       # [B, doc_seq_len]
    
    batch_size = q_input_ids.size(0)
    q_seq_len = q_input_ids.size(1)
    d_seq_len = d_input_ids.size(1)
    
    # 创建 [SEP] 标记及其 attention_mask
    sep_tensor = torch.full((batch_size, 1), sep_token_id, dtype=q_input_ids.dtype, device=q_input_ids.device)
    sep_attention = torch.ones((batch_size, 1), dtype=q_attention.dtype, device=q_attention.device)
    
    # 拼接
    merged_input_ids = torch.cat([q_input_ids, sep_tensor, d_input_ids], dim=1)
    merged_attention = torch.cat([q_attention, sep_attention, d_attention], dim=1)
    
    # 构造 token_type_ids: 查询为 0，文档为 1，分隔符设为 0（可根据需要调整）
    q_token_type_ids = torch.zeros((batch_size, q_seq_len), dtype=q_input_ids.dtype, device=q_input_ids.device)
    sep_token_type_ids = torch.zeros((batch_size, 1), dtype=q_input_ids.dtype, device=q_input_ids.device)
    d_token_type_ids = torch.ones((batch_size, d_seq_len), dtype=q_input_ids.dtype, device=q_input_ids.device)
    
    merged_token_type_ids = torch.cat([q_token_type_ids, sep_token_type_ids, d_token_type_ids], dim=1)
    
    # 截断至 max_length
    if merged_input_ids.size(1) > max_length:
        merged_input_ids = merged_input_ids[:, :max_length]
        merged_attention = merged_attention[:, :max_length]
        merged_token_type_ids = merged_token_type_ids[:, :max_length]
    
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

        # 拼接 query + [SEP] + doc
        merged = merge_features(query_input, doc_input, sep_token_id=self.tokenizer.sep_token_id, max_length=256)

        return {"input": merged, "batch_l": batch_idx}


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
                batch_output = []
                input_ids = batch_data['input_ids']
                attention_mask = batch_data['attention_mask']
                for i in range(input_ids.size(0)):  # batch_size 是 input_ids 的第一个维度
                    # 获取当前样本的 input_ids 和 attention_mask
                    sample_input_ids = input_ids[i].unsqueeze(0)  # 形状变为 [1, seq_len]
                    sample_attention_mask = attention_mask[i].unsqueeze(0)  # 形状变为 [1, seq_len]
                    
                    # 根据 attention_mask 去除填充部分
                    effective_input_ids = sample_input_ids[sample_attention_mask == 1]
                    effective_attention_mask = sample_attention_mask[sample_attention_mask == 1]
                    
                    # 生成新的 batch_data，只包含有效的部分
                    sample_batch_data = {
                        'input_ids': effective_input_ids,
                        'attention_mask': effective_attention_mask
                    }
                    with torch.cuda.amp.autocast(), torch.no_grad():
                        # 获取模型的输出
                        output = self.model(**sample_batch_data)
                        batch_output.append(output)
                
                # 拼接所有样本的输出
                output = torch.cat(batch_output, dim=0)
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
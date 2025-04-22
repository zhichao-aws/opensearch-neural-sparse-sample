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
        input_data = {
            "input_ids": [],
            "attention_mask": []
        }
        
        for request in requests:
            request_body = request.get("body")
            if isinstance(request_body, bytearray):
                request_body = request_body.decode("utf-8")
                request_body = json.loads(request_body)
            
            if isinstance(request_body, list):
                batch_idx.append(len(request_body))
                for item in request_body:
                    input_data["input_ids"].append(item["input_ids"])
                    input_data["attention_mask"].append(item["attention_mask"])
            else:
                batch_idx.append(1)
                input_data["input_ids"].append(request_body["input_ids"])
                input_data["attention_mask"].append(request_body["attention_mask"])

        input_data["input_ids"] = torch.tensor(input_data["input_ids"]).to(self.device)
        input_data["attention_mask"] = torch.tensor(input_data["attention_mask"]).to(self.device)

        return {"input": input_data, "batch_l": batch_idx}


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
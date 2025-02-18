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

model_id = "opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill"

class SparseEncodingModelHandler(BaseHandler):
    class SparseModel(torch.nn.Module):
        @staticmethod
        def from_pretrained(path):
            return SparseEncodingModelHandler.SparseModel(path)
        
        def __init__(self, model_id):
            super().__init__()
            self.backbone = transformers.AutoModelForMaskedLM.from_pretrained(model_id)
            self.special_token_ids = []
            
        def set_special_token_ids(self,special_token_ids):
            self.special_token_ids = special_token_ids
            
        def forward(self, **kwargs):
            output = self.backbone(**kwargs)[0]
            values, _ = torch.max(output*kwargs.get("attention_mask").unsqueeze(-1), dim=1)
            values = torch.log(1 + torch.relu(values))
            values[:,self.special_token_ids] = 0
            return values
    

    class SparsePostProcessor(object):
        def __init__(self,tokenizer):
            self.tokenizer = tokenizer
            self.id_to_token = ["" for i in range(tokenizer.vocab_size)]
            for token, _id in tokenizer.vocab.items():
                self.id_to_token[_id] = token
            
        def __call__(self, sparse_vector):
            sample_indices,token_indices=torch.nonzero(sparse_vector,as_tuple=True)
            non_zero_values = sparse_vector[(sample_indices,token_indices)].tolist()
            number_of_tokens_for_each_sample = torch.bincount(sample_indices).cpu().tolist()
            tokens = [self.id_to_token[_id] for _id in token_indices.tolist()]

            output = []
            end_idxs = list(itertools.accumulate([0]+number_of_tokens_for_each_sample))
            for i in range(len(end_idxs)-1):
                token_strings = tokens[end_idxs[i]:end_idxs[i+1]]
                weights = non_zero_values[end_idxs[i]:end_idxs[i+1]]
                output.append(dict(zip(token_strings, weights)))
            return output
    
    def __init__(self):
        super().__init__()
        self.special_token_ids = None
        self.tokenizer = None
        self.all_tokens = None
        self.initialized = False

    def initialize(self, context):
        self.manifest = context.manifest
        properties = context.system_properties
        
        # load model and tokenizer
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")
        self.model = SparseEncodingModelHandler.SparseModel.from_pretrained(model_id)
        self.model.to(self.device)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        
        self.post_processor = SparseEncodingModelHandler.SparsePostProcessor(tokenizer=self.tokenizer)
        self.special_token_ids = [self.tokenizer.vocab[token] for token in self.tokenizer.special_tokens_map.values()]
        self.model.set_special_token_ids(self.special_token_ids)
        
        self.initialized = True

    def preprocess(self, requests):
        inputSentence = []

        batch_idx = []
        for request in requests:

            request_body = request.get("body")
            if isinstance(request_body, bytearray):
                request_body = request_body.decode('utf-8')
                request_body = json.loads((request_body))

            if isinstance(request_body, list):
                inputSentence += request_body
                batch_idx.append(len(request_body))
            else:
                inputSentence.append(request_body)
                batch_idx.append(1)
        input_data = self.tokenizer(inputSentence, padding=True, truncation=True, max_length=512, 
                                    return_tensors='pt', return_attention_mask=True, return_token_type_ids=False)

        input_data = input_data.to(self.device)
        return {"input": input_data, "batch_l": batch_idx}

    def inference(self, data, *args, **kwargs):
        batch_idx = data["batch_l"]
        data = data["input"]
        with torch.cuda.amp.autocast(),torch.no_grad():
            output = self.model(**data)

        return {"pred": output, "batch_l": batch_idx}

    def postprocess(self, prediction):
        batch_idx = prediction["batch_l"]
        output = prediction["pred"]
        output = self.post_processor(output)
        
        # return the inference results to each request according to batch size
        outputs = []
        index = 0
        for b in batch_idx:
            outputs.append(output[index:index + b])
            index += b
        return outputs

    def handle(self, data, context):
        model_input = self.preprocess(data)
        model_output = self.inference(model_input)
        return self.postprocess(model_output)
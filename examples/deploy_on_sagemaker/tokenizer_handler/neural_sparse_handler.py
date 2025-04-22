import os
import json
import transformers
from ts.torch_handler.base_handler import BaseHandler

model_id = "opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill"


def get_tokenizer_idf():
    from huggingface_hub import hf_hub_download

    local_cached_path = hf_hub_download(repo_id=model_id, filename="idf.json")
    with open(local_cached_path) as f:
        idf = json.load(f)
    return idf


class SparseEncodingModelHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.special_token_ids = None
        self.tokenizer = None
        self.all_tokens = None
        self.initialized = False

    def initialize(self, context):
        self.manifest = context.manifest

        # load model and tokenizer
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        self.idf = get_tokenizer_idf()

        id_to_token = [0] * 30522
        for token, _id in self.tokenizer.vocab.items():
            id_to_token[_id] = token

        weights = [1] * 30522
        for token, weight in self.idf.items():
            weights[self.tokenizer._convert_token_to_id_with_added_voc(token)] = weight

        self.id_to_token = id_to_token
        self.weights = weights

        self.initialized = True

    def preprocess(self, requests):
        inputSentence = []

        batch_idx = []
        for request in requests:

            request_body = request.get("body")
            if isinstance(request_body, bytearray):
                request_body = request_body.decode("utf-8")
                request_body = json.loads((request_body))

            if isinstance(request_body, list):
                inputSentence += request_body
                batch_idx.append(len(request_body))
            else:
                inputSentence.append(request_body)
                batch_idx.append(1)
        input_data = self.tokenizer(
            inputSentence,
            add_special_tokens=False,
            return_token_type_ids=False,
            truncation=False,
            return_attention_mask=False,
        )

        return {"input": input_data, "batch_l": batch_idx}

    def inference(self, data, *args, **kwargs):
        return {
            "pred": data.get("input").get("input_ids"),
            "batch_l": data.get("batch_l"),
        }

    def postprocess(self, prediction):
        batch_idx = prediction["batch_l"]
        input_ids = prediction["pred"]

        output = [
            dict([(self.id_to_token[_id], self.weights[_id]) for _id in input_id])
            for input_id in input_ids
        ]
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

# opensearch-neural-sparse-sample
Containing sample codes to trace/deploy neural sparse model for opensearch using [PyTorch](https://pytorch.org/) and [transformers](https://github.com/huggingface/transformers) API.

For fine-tuning, please check the repo https://github.com/zhichao-aws/opensearch-sparse-model-tuning-sample.

## What we have now
- code sample to deploy a neural sparse model on SageMaker that can be accessed via OpenSearch connector
- code sample to create an index for neural sparse then do ingest and search
    - combined with chunking processor

## Todos
- code sample to trace a neural model that can be uploaded to OpenSearch cluster
- code sample to create an index for neural sparse then do ingest and search
    - combined with dense model to do hybrid search

## Code sample request
If you want some sample code not covered in this repo, please create a new issue in this repo to make request.
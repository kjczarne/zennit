import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from typing import Any, Callable
import numpy as np
from zennit.composites import Transformer
from zennit.attribution import Gradient

composite = Transformer()

from transformers import pipeline

clf = pipeline("sentiment-analysis", model="bert-base-uncased")
sentence = "Chris is useless and will never amount to anything!"


model = nn.Sequential(
    clf.model.bert.encoder,
    clf.model.bert.pooler,
    clf.model.dropout,
    clf.model.classifier
)

with Gradient(model=model, composite=composite) as attributor:
    # seems like initializing the attribution output with zeros doesn't work for
    # `SillyModel` but the attribution scores seem to make sense once we use `torch.ones`
    # as the initial values for R-scores
    input = clf.tokenizer(sentence, padding=True, truncation=True, return_tensors="pt")
    input = input.input_ids
    embeddings = clf.model.bert.embeddings(input)
    # input = input.input_ids.float()
    out, relevance = attributor(embeddings, attr_output=torch.ones((1, 2)))
    # print("Inference input:")
    # print(data_test)
    # print(data_test.shape)
    # print("\n")

    # print("Output:")
    # print(out)
    # print("\n")

    # print("Relevance scores:")
    # print(relevance)
    # print("\n")

    # print("Model Parameters:")
    # print([i[1].shape for i in model.mha.named_parameters()])
    
    # print("W_Q:")
    # print(model.mha.in_proj_weight.data)

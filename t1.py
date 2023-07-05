import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from typing import Callable
import numpy as np


class SillyDataset(Dataset):

    def __init__(self, sequence: np.array, num_samples: int = 100) -> None:
        self.num_samples = num_samples
        self._all_data = torch.tensor(np.repeat(np.array(sequence)[None, :],
                                                num_samples,
                                                axis=0))
        super().__init__()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return self._all_data[index, :]


class SillyModel(nn.Module):

    def __init__(self,
                 dim: int = 100,
                 num_heads: int = 5,
                 batch_size: int = 10,
                 return_attn_weights: bool = False,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.linear = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm((batch_size, dim))
        self.mha = nn.MultiheadAttention(dim,
                                         num_heads=num_heads)  # e.g. split 100 dimensions across 5 parallel attention heads
        # self.encoder = nn.TransformerEncoderLayer(dim, num_heads)
        # self.decoder = nn.TransformerDecoderLayer(dim, num_heads)

        self.return_attn_weights = return_attn_weights

        # Weight Initialization:
        self.linear.weight.data = nn.init.constant_(self.linear.weight.data, 0.5)
        # Seems like in this case if I change the projection weights for the attention layer
        # the output values are identical anyway, possibly because W_K, W_Q, W_V could already
        # be initialized as constants.
        # if self.mha.k_proj_weight is None:
        #     self.mha.k_proj_weight = torch.nn.Parameter(torch.tensor(np.repeat(0.4, dim)))
        # if self.mha.v_proj_weight is None:
        #     self.mha.v_proj_weight = torch.nn.Parameter(torch.tensor(np.repeat(0.2, dim)))
        # if self.mha.q_proj_weight is None:
        #     self.mha.q_proj_weight = torch.nn.Parameter(torch.tensor(np.repeat(0.2, dim)))
        self.mha.in_proj_weight.data = nn.init.constant_(self.mha.in_proj_weight.data, 0.4)
        # self.mha.k_proj_weight.data = nn.init.constant_(self.mha.k_proj_weight.data, 0.4)
        # self.mha.v_proj_weight.data = nn.init.constant_(self.mha.k_proj_weight.data, 0.4)
        # self.mha.q_proj_weight.data = nn.init.constant_(self.mha.k_proj_weight.data, 0.4)
        self.mha.out_proj.weight.data = nn.init.constant_(self.mha.out_proj.weight.data, 0.4)
        # Now change W_Q in a way that assigns higher scores to inputs in the position 2
        self.mha.in_proj_weight.data[9][2] = 1000.0

    def forward(self, input: torch.Tensor):
        # key = torch.rand((1, self.dim,))
        # query = torch.rand((1, self.dim,)) 
        # value = torch.rand((1, self.dim,))
        if input.dtype is torch.long:
            # Cast to float if necessary
            input = input.float()
        # Emulating self-attention, with K, Q, V all being the same matrix
        # x = self.linear(input)
        x = input  # skipping input linear projection
        key = x
        query = x
        value = x
        output, attn_weights = self.mha(query, key, value)
        if self.training:
            # Manually upscale the Key projection weights on training
            # using the weights returned from the MHA block.
            # This mimics the training process optimizing the MHA weights.
            self.mha.k_proj_weight = torch.nn.Parameter(torch.mul(attn_weights, 5))
        # output = self.ln(output)
        if self.return_attn_weights:
            return output, attn_weights
        return output


sequence = [1, 1, 300, 3, 1, 1]
# batch_size = 10
batch_size = 2
# data_test = torch.tensor(np.repeat(np.array(sequence)[None, :], batch_size, axis=0))
data_test = torch.tensor(np.array([[1, 1, 300, 3, 1, 1],
                                   [2, 2,   2, 2, 2, 2]]))
data_test = data_test.float()
dim = data_test.shape[1]

data_train = SillyDataset(np.array(sequence), 100)
data_train = DataLoader(data_train, batch_size)

model = SillyModel(dim=dim, num_heads=dim//3, batch_size=batch_size)

epochs = 5

# for epoch in range(epochs):
#     model.train(True)
#     for i, x in enumerate(data_train):
#         out = model(x)

data_test.requires_grad = True

model.eval()

output = model(data_test)

from zennit.composites import Transformer
from zennit.attribution import Gradient

composite = Transformer()

with Gradient(model=model, composite=composite) as attributor:
    # seems like initializing the attribution output with zeros doesn't work for
    # `SillyModel` but the attribution scores seem to make sense once we use `torch.ones`
    # as the initial values for R-scores
    out, relevance = attributor(data_test, attr_output=torch.ones(batch_size, dim))
    print("Inference input:")
    print(data_test)
    print(data_test.shape)
    print("\n")

    print("Output:")
    print(out)
    print("\n")

    print("Relevance scores:")
    print(relevance)
    print("\n")

    print("Model Parameters:")
    print([i[1].shape for i in model.mha.named_parameters()])
    
    print("W_Q:")
    print(model.mha.in_proj_weight.data)

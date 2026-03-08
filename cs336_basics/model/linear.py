import torch.nn as nn
import torch
from einops import einsum

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.W = nn.Parameter(torch.empty(out_features, in_features, **factory_kwargs))
        nn.init.trunc_normal_(self.W)

    def forward(self, x):
        return einsum(x, self.W, "... in_features, out_features in_features -> ... out_features")
from einops import rearrange, einsum
from torch import nn 
import torch


class Embedding(nn.Module):
    
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype= None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.emb_mat = nn.Parameter(torch.empty(size=(num_embeddings, embedding_dim), **factory_kwargs))
        nn.init.trunc_normal_(self.emb_mat) 
        
        
    def forward(self, token_ids: torch.Tensor):
        return self.emb_mat[token_ids]

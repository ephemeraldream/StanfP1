from einops import rearrange, reduce, einsum
from torch import nn
import torch



class FFN(nn.Module):
    
    def __init__(self, d_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        raw_d_ff = int(8/3 * d_model)
        self.d_model = d_model
        self.d_ff = ((raw_d_ff + 63) // 64) * 64

        self.W1 = nn.Parameter(torch.empty(size=(self.d_ff, self.d_model)))
        self.W3 = nn.Parameter(torch.empty(size=(self.d_ff, self.d_model)))
        self.W2 = nn.Parameter(torch.empty(size=(self.d_model, self.d_ff)))
        nn.init.trunc_normal_(self.W1)
        nn.init.trunc_normal_(self.W2)
        nn.init.trunc_normal_(self.W3)
        
        
    def forward(self, x:torch.Tensor):
         W1_x = einsum(self.W1,x, 'd_ff d_model, ... d_model-> ... d_ff')
         W3_x = einsum(self.W3,x, 'd_ff d_model, ... d_model-> ... d_ff')
         SiLU = W1_x * torch.sigmoid(W1_x)
         inner_x = SiLU * W3_x
         return einsum(self.W2, inner_x, 'd_model d_ff, ... d_ff -> ... d_model')

         
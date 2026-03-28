from torch import nn
import torch
from cs336_basics.model.RMSNorm import RMSNorm
from cs336_basics.model.positionwise_feedforward import FFN
from cs336_basics.model.scaled_dot_product_attn import MultiHeadAttention



class TransformerBlock(nn.Module):
    
    def __init__(self, d_model:int, num_heads:int,max_seq_len:int, d_ff:int=None,theta:float=10000, device=None):
        super().__init__()
        self.rmsnorm = RMSNorm(d_model, device=device)
        self.mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, max_seq_len=max_seq_len, use_rope=True, theta=theta, device=device)
        self.ffn = FFN(d_model=d_model, d_ff=d_ff, device=device)
        
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x_mha1 = self.rmsnorm.forward(x)
        x_mha1 = self.mha.forward(x_mha1)
        x = x + x_mha1 
        x_ffn = self.rmsnorm.forward(x)
        x_ffn = self.ffn.forward(x_ffn)
        x = x + x_ffn
        return x
        

        
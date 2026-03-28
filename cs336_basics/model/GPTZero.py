import torch
from torch import nn

from cs336_basics.model.embedding import Embedding
from cs336_basics.model.linear import Linear
from cs336_basics.model.transformer_block import TransformerBlock
from cs336_basics.model.RMSNorm import RMSNorm


class GPTZero(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int | None = None,
        theta: float = 10000.0,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model

        self.emb = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype,
        )
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    max_seq_len=context_length,
                    d_ff=d_ff,
                    theta=theta,
                    device=device,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, device=device)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.shape[-1] > self.context_length:
            raise ValueError(
                f"sequence length {input_ids.shape[-1]} exceeds context_length={self.context_length}"
            )

        x = self.emb(input_ids)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.ln_final(x)
        return self.lm_head(x)

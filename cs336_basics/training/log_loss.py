import torch

def log_loss(logits:torch.Tensor, targets:torch.Tensor): # (batch, vocab_size), (batch, y)
    max_els = logits.amax(dim=-1, keepdim=True)
    losumexp = max_els.squeeze(-1) + torch.log(torch.exp(logits - max_els).sum(dim=-1))

    target_logits = logits.gather(
        dim=-1,
        index=targets.unsqueeze(-1)
    ).squeeze(-1)
    losses = losumexp - target_logits
    return losses.mean()

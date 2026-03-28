from collections.abc import Iterable

import torch
from torch import nn

_EPS = 1e-6


def clip_gradients_l2_norm_(parameters: Iterable[nn.Parameter], max_l2_norm: float) -> None:
    params = list(parameters)
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return

    with torch.no_grad():
        norms = [torch.linalg.vector_norm(g.detach(), ord=2, dtype=torch.float32) for g in grads]
        total_norm = torch.linalg.vector_norm(torch.stack(norms), ord=2)
        clip_coef = torch.clamp(max_l2_norm / (total_norm + _EPS), max=1.0)
        for g in grads:
            g.mul_(clip_coef.to(device=g.device))

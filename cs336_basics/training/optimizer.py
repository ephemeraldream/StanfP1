from collections.abc import Callable
from typing import Optional
import torch 
import math


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lam, lr=1e-3, betas=(0.9, 0.99), eps=1e-8):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if lam < 0.0:
            raise ValueError(f"Invalid weight decay: {lam}")

        defaults = {"lr": lr, "lam": lam, "betas": betas, "eps": eps}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            lam = group["lam"]
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                state["step"] += 1
                t = state["step"]
                p.mul_(1 - lr * lam)

                state["m"].mul_(beta1).add_(grad, alpha=1 - beta1)
                state["v"].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t

                m_hat = state["m"] / bias_correction1
                v_hat = state["v"] / bias_correction2

                p.add_( -lr * m_hat / (torch.sqrt(v_hat) + eps) )

        return loss
"""Lamb optimizer."""

import collections
import math

import torch
from torch.optim import Optimizer
from ..logger import Logger


class Lamb(Optimizer):
    r"""Implements Lamb adaption of AdamW

    .. _Large Batch Optimization for Deep Learning: Training BERT in 76 minutes:
        https://arxiv.org/abs/1904.00962
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-6,
        weight_decay=0,
        debias=True,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            debias=debias,
        )

        super().__init__(params, defaults)

    @torch.no_grad
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Lamb does not support sparse gradients, "
                        "consider SparseAdam instad."
                    )

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                state["step"] += 1

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                lr = group["lr"]
                weight_decay = group["weight_decay"]
                eps = group["eps"]
                debias = group["debias"]
                step = state["step"]

                # Perform stepweight decay
                p.data.mul_(1 - lr * weight_decay)

                # Decay the first and second moment running average coefficient
                exp_avg.lerp_(grad, 1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if not debias:
                    # Paper v3 does not use debiasing.
                    step_size = lr
                else:
                    # Paper v5 uses debiasing.
                    bias_correction1 = 1 - beta1**step
                    bias_correction2 = 1 - beta2**step
                    step_size = lr * math.sqrt(bias_correction2) / bias_correction1

                adam_step = exp_avg / exp_avg_sq.sqrt().add(eps)

                # Compute trust ratio
                weight_norm = p.data.norm().clamp(0, 10)
                adam_norm = adam_step.norm()
                if weight_norm == 0 or adam_norm == 0:
                    trust_ratio = 1
                else:
                    trust_ratio = weight_norm / adam_norm

                # state['weight_norm'] = weight_norm
                # state['adam_norm'] = adam_norm
                # state['trust_ratio'] = trust_ratio
                # if self.adam:
                #    trust_ratio = 1

                p.data.add_(adam_step, alpha=-step_size * trust_ratio)

        return loss


def log_lamb_rs(optimizer: Optimizer, logger: Logger, token_count: int):
    """Log a histogram of trust ratio scalars in across layers."""
    results = collections.defaultdict(list)
    for group in optimizer.param_groups:
        for p in group["params"]:
            state = optimizer.state[p]
            for i in ("weight_norm", "adam_norm", "trust_ratio"):
                if i in state:
                    results[i].append(state[i])

    for k, v in results.items():
        event_writer.add_histogram(f"lamb/{k}", torch.tensor(v), token_count)

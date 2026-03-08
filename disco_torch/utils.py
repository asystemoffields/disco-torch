"""Utility functions — PyTorch equivalents of disco_rl/utils.py."""

from __future__ import annotations

import torch
from torch import Tensor

from disco_torch.types import EmaState


def batch_lookup(table: Tensor, index: Tensor) -> Tensor:
    """Select entries from table using index, batched over leading dims.

    Matches haiku BatchApply(_lookup, num_dims=2):
      table: [B1, B2, A, ...]  (any number of trailing dims after A)
      index: [B1, B2]
      returns: [B1, B2, ...]

    The action dim is at position index.dim() in table.
    """
    idx = index.long()
    n_batch = idx.dim()
    n_trailing = table.dim() - n_batch - 1  # dims after the action dim

    if n_trailing < 0:
        raise ValueError(
            f"batch_lookup: table.dim()={table.dim()} must be > index.dim()={idx.dim()}"
        )

    # Expand index to match table: [..., 1, D1, D2, ...]
    idx_exp = idx
    for _ in range(n_trailing + 1):
        idx_exp = idx_exp.unsqueeze(-1)
    # Broadcast trailing dims
    trailing_shape = table.shape[n_batch + 1:]
    idx_exp = idx_exp.expand(*idx.shape, 1, *trailing_shape)

    result = table.gather(n_batch, idx_exp).squeeze(n_batch)
    return result


def signed_logp1(x: Tensor) -> Tensor:
    """rlax.signed_logp1: sign(x) * log(|x| + 1)."""
    return x.sign() * (x.abs() + 1.0).log()


def signed_hyperbolic(x: Tensor) -> Tensor:
    """rlax.signed_hyperbolic: sign(x) * (sqrt(|x| + 1) - 1) + eps * x."""
    eps = 1e-3
    return x.sign() * ((x.abs() + 1.0).sqrt() - 1.0) + eps * x


def inverse_signed_hyperbolic(x: Tensor) -> Tensor:
    """Inverse of signed_hyperbolic."""
    eps = 1e-3
    return x.sign() * (((((x.abs() + 1.0).sqrt() - 1.0) / (1.0 + eps)).square()
                         + 2.0 * (x.abs() + 1.0).sqrt() / (1.0 + eps) - 1.0).clamp(min=0.0))


def signed_hyperbolic_tx(x: Tensor) -> Tensor:
    """Forward transform of SIGNED_HYPERBOLIC_PAIR."""
    return signed_hyperbolic(x)


def signed_hyperbolic_inv(x: Tensor) -> Tensor:
    """Inverse transform of SIGNED_HYPERBOLIC_PAIR."""
    eps = 1e-3
    # Solve: y = sign(x)*(sqrt(|x|+1)-1) + eps*x for x given y
    # Simpler: the rlax inverse is sign(y)*( ((sqrt(1+4*eps*(|y|+1+eps)) - 1)/(2*eps))^2 - 1 )
    b = 1.0 + 4.0 * eps * (x.abs() + 1.0 + eps)
    return x.sign() * (((b.sqrt() - 1.0) / (2.0 * eps)).square() - 1.0).clamp(min=0.0)


def categorical_kl_divergence(p_logits: Tensor, q_logits: Tensor) -> Tensor:
    """KL(softmax(p_logits) || softmax(q_logits)), summed over last dim.

    Matches rlax.categorical_kl_divergence.
    """
    p = torch.softmax(p_logits, dim=-1)
    log_p = torch.log_softmax(p_logits, dim=-1)
    log_q = torch.log_softmax(q_logits, dim=-1)
    return (p * (log_p - log_q)).sum(dim=-1)


def transform_to_2hot(
    value: Tensor, min_value: float, max_value: float, num_bins: int
) -> Tensor:
    """Convert scalar value to 2-hot probability vector over bins."""
    bin_width = (max_value - min_value) / (num_bins - 1)
    # Clip value to [min_value, max_value]
    value = value.clamp(min_value, max_value)
    # Continuous bin index
    idx = (value - min_value) / bin_width
    lo = idx.floor().long()
    hi = (lo + 1).clamp(max=num_bins - 1)
    frac = idx - lo.float()
    probs = torch.zeros(*value.shape, num_bins, device=value.device)
    probs.scatter_(-1, lo.unsqueeze(-1), (1.0 - frac).unsqueeze(-1))
    probs.scatter_add_(-1, hi.unsqueeze(-1), frac.unsqueeze(-1))
    return probs


def transform_from_2hot(
    probs: Tensor, min_value: float, max_value: float, num_bins: int
) -> Tensor:
    """Convert 2-hot probability vector back to scalar expected value."""
    support = torch.linspace(min_value, max_value, num_bins, device=probs.device)
    return (probs * support).sum(dim=-1)


class MovingAverage:
    """EMA tracker for normalization, matching disco_rl/utils.py."""

    def __init__(self, decay: float = 0.999, eps: float = 1e-6):
        self.decay = decay
        self.eps = eps

    def init_state(self, device: torch.device = None) -> EmaState:
        return EmaState(
            moment1=torch.zeros((), device=device),
            moment2=torch.zeros((), device=device),
            decay_product=torch.ones((), device=device),
        )

    def update_state(self, value: Tensor, state: EmaState) -> EmaState:
        mean = value.mean()
        mean_sq = value.square().mean()
        return EmaState(
            moment1=self.decay * state.moment1 + (1.0 - self.decay) * mean,
            moment2=self.decay * state.moment2 + (1.0 - self.decay) * mean_sq,
            decay_product=state.decay_product * self.decay,
        )

    def normalize(
        self, value: Tensor, state: EmaState, subtract_mean: bool = True
    ) -> Tensor:
        debias = 1.0 / (1.0 - state.decay_product)
        mean = state.moment1 * debias
        var = (state.moment2 * debias - mean.square()).clamp(min=0.0)
        if subtract_mean:
            return (value - mean) / (var.sqrt() + self.eps)
        return value / (var.sqrt() + self.eps)

"""Minimal GRPO (Group Relative Policy Optimization) with optional
per-step credit modulation from the Disco103 meta-network.

GRPO assigns the same advantage to every token in a completion.
Credit modulation redistributes that advantage across tokens based
on the meta-network's per-step credit assignment.
"""

from __future__ import annotations

import torch
from torch import Tensor


def grpo_advantages(rewards: Tensor) -> Tensor:
    """Compute group-relative advantages.

    Args:
        rewards: [G] scalar reward per completion in one group

    Returns:
        advantages: [G] normalized within the group
    """
    mean = rewards.mean()
    std = rewards.std().clamp(min=1e-8)
    return (rewards - mean) / std


def modulate_with_credit(
    grpo_advs: Tensor,
    credit: Tensor,
    alpha: float = 1.0,
) -> Tensor:
    """Combine GRPO trajectory advantages with per-step credit.

    Additive combination: advantage_t = grpo_adv + alpha * credit_t.
    At alpha=0, this is standard GRPO (uniform across tokens).
    At alpha>0, the meta-network's credit signal focuses updates
    on the tokens it considers most important.

    Args:
        grpo_advs: [B] per-completion GRPO advantage
        credit: [T, B] per-step credit from meta-network
        alpha: credit strength (0 = pure GRPO)

    Returns:
        advantages: [T, B] per-step modulated advantages
    """
    T, B = credit.shape
    base = grpo_advs.unsqueeze(0).expand(T, B)
    return base + alpha * credit

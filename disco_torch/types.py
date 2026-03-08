"""Type definitions — PyTorch equivalents of disco_rl/types.py."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
from torch import Tensor


@dataclass
class TransformConfig:
    source: str
    transforms: tuple[str, ...]


@dataclass
class MetaNetInputOption:
    base: tuple[TransformConfig, ...]
    action_conditional: tuple[TransformConfig, ...]


@dataclass
class EmaState:
    moment1: Tensor
    moment2: Tensor
    decay_product: Tensor


@dataclass
class ValueOuts:
    value: Tensor | float = 0.0
    target_value: Tensor | float = 0.0
    rho: Tensor | float = 0.0
    adv: Tensor | float = 0.0
    normalized_adv: Tensor | float = 0.0
    value_target: Tensor | float = 0.0
    td: Tensor | float = 0.0
    normalized_td: Tensor | float = 0.0
    qv_adv: Any = None
    normalized_qv_adv: Any = None
    q_value: Any = None
    target_q_value: Any = None
    q_target: Any = None
    q_td: Any = None
    normalized_q_td: Any = None


@dataclass
class UpdateRuleInputs:
    """Rollout data fed to the update rule.

    Shapes:
        observations: [T+1, B, ...]
        actions:       [T+1, B]
        rewards:       [T, B]
        is_terminal:   [T, B]
        agent_out:     dict of [T+1, B, ...] tensors
        behaviour_agent_out: same structure as agent_out, or None
    """
    observations: Tensor
    actions: Tensor
    rewards: Tensor
    is_terminal: Tensor
    agent_out: dict[str, Tensor]
    behaviour_agent_out: dict[str, Tensor] | None = None
    value_out: ValueOuts | None = None
    extra_from_rule: dict[str, Any] | None = None

    @property
    def should_reset_mask_fwd(self) -> Tensor:
        prepend = torch.zeros_like(self.is_terminal[:1])
        return torch.cat([prepend, self.is_terminal], dim=0)

    @property
    def should_reset_mask_bwd(self) -> Tensor:
        append = torch.zeros_like(self.is_terminal[:1])
        return torch.cat([self.is_terminal, append], dim=0)

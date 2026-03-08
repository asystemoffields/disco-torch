"""Input transforms — PyTorch equivalents of disco_rl/update_rules/input_transforms.py."""

from __future__ import annotations
from typing import Callable

import torch
from torch import Tensor

from disco_torch.utils import batch_lookup, signed_logp1


def select_a(x: Tensor, actions: Tensor, policy: Tensor, axis=None) -> Tensor:
    """Select the taken action from x. x: [T+1, B, A, ...], actions: [T, B]."""
    return batch_lookup(x, actions)


def pi_weighted_avg(x: Tensor, actions: Tensor, policy: Tensor, axis=None) -> Tensor:
    """Policy-weighted average: sum over action dim. x: [T+1, B, A, D], policy: [T+1, B, A]."""
    return (x * policy.unsqueeze(-1)).sum(dim=2)


def td_pair(x: Tensor) -> Tensor:
    """Concat x[:-1] and x[1:] along last dim — temporal difference pairs."""
    return torch.cat([x[:-1], x[1:]], dim=-1)


def masks_to_discounts(x: Tensor) -> Tensor:
    return 1.0 - x


# Simple stateless transforms (x -> x)
SIMPLE_TRANSFORMS: dict[str, Callable[[Tensor], Tensor]] = {
    "identity": lambda x: x,
    "softmax": lambda x: torch.softmax(x, dim=-1),
    "max_a": lambda x: x.max(dim=2).values,
    "stop_grad": lambda x: x.detach(),
    "clip": lambda x: x.clamp(-2.0, 2.0),
    "sign": lambda x: x.sign(),
    "drop_last": lambda x: x[:-1],
    "td_pair": td_pair,
    "sign_log": signed_logp1,
    "masks_to_discounts": masks_to_discounts,
}


def apply_transform(
    name: str,
    x: Tensor,
    actions: Tensor,
    policy: Tensor,
    y_net: Callable | None = None,
    z_net: Callable | None = None,
    axis_name=None,
) -> Tensor:
    """Apply a named transform to tensor x."""
    if name == "select_a":
        return select_a(x, actions, policy, axis_name)
    elif name == "pi_weighted_avg":
        return pi_weighted_avg(x, actions, policy, axis_name)
    elif name == "y_net":
        assert y_net is not None
        return y_net(x)
    elif name == "z_net":
        assert z_net is not None
        return z_net(x)
    elif name in SIMPLE_TRANSFORMS:
        return SIMPLE_TRANSFORMS[name](x)
    else:
        raise KeyError(f"Unknown transform: {name}")


def _multi_level_extract(obj, keys: str):
    """Navigate nested dicts/objects by slash-separated key path."""
    for key in keys.split("/"):
        if isinstance(obj, dict):
            obj = obj[key]
        else:
            obj = getattr(obj, key)
    return obj


def construct_input(
    inputs,  # UpdateRuleInputs
    input_option,  # MetaNetInputOption
    y_net: Callable,
    z_net: Callable,
    policy_net: Callable,
) -> tuple[Tensor, Tensor | None]:
    """Build the concatenated input vector for the meta-network.

    Returns (base_input [T,B,D], action_cond_embedding [T,B,A,C] or None).
    """
    t, b = inputs.is_terminal.shape
    actions = inputs.actions[:-1]  # [T, B]
    policy = torch.softmax(inputs.agent_out["logits"], dim=-1).detach()  # [T+1, B, A]
    num_actions = policy.shape[2]

    def preprocess(configs, prefix_shape):
        parts = []
        for cfg in configs:
            x = _multi_level_extract(inputs, cfg.source)

            # Align extra dims for extra_from_rule scalars (match JAX code)
            if (
                cfg.source.startswith("extra_from_rule")
                and "target_out" not in cfg.source
            ) or cfg.source == "extra_from_rule/target_out/q":
                x = x.unsqueeze(-1)

            for tx in cfg.transforms:
                x = apply_transform(
                    tx, x, actions, policy, y_net=y_net, z_net=z_net
                )

            x = x.reshape(*prefix_shape, -1)
            parts.append(x)
        return parts

    # Base inputs [T, B, ...]
    base_parts = preprocess(input_option.base, prefix_shape=(t, b))

    # Action-conditional inputs [T, B, A, ...]
    act_cond_emb = None
    if input_option.action_conditional:
        act_parts = preprocess(
            input_option.action_conditional, prefix_shape=(t, b, num_actions)
        )
        # Append one-hot actions [T, B, A, 1]
        act_parts.append(
            torch.nn.functional.one_hot(actions.long(), num_actions)
            .float()
            .unsqueeze(-1)
        )
        act_cond = torch.cat(act_parts, dim=-1)  # [T, B, A, D_ac]
        act_cond_emb = policy_net(act_cond)  # [T, B, A, C]
        act_cond_avg = act_cond_emb.mean(dim=2)  # [T, B, C]
        act_cond_a = batch_lookup(act_cond_emb, actions)  # [T, B, C]
        base_parts.extend([act_cond_avg, act_cond_a])

    return torch.cat(base_parts, dim=-1), act_cond_emb

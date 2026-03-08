"""Value function utilities — PyTorch port of disco_rl/value_fns/value_utils.py."""

from __future__ import annotations

import torch
from torch import Tensor

from disco_torch.types import ValueOuts, EmaState, UpdateRuleInputs
from disco_torch.utils import (
    batch_lookup,
    signed_hyperbolic_tx,
    signed_hyperbolic_inv,
    transform_from_2hot,
    transform_to_2hot,
    MovingAverage,
)


def importance_weight(
    pi_logits: Tensor, mu_logits: Tensor, actions: Tensor
) -> Tensor:
    """Compute importance sampling ratio pi(a)/mu(a)."""
    log_pi = torch.log_softmax(pi_logits, dim=-1)
    log_mu = torch.log_softmax(mu_logits, dim=-1)
    log_pi_a = log_pi.gather(-1, actions.long().unsqueeze(-1)).squeeze(-1)
    log_mu_a = log_mu.gather(-1, actions.long().unsqueeze(-1)).squeeze(-1)
    return (log_pi_a - log_mu_a).exp().detach()


def get_values_from_net_outs(
    x: Tensor,
    categorical_value: bool,
    max_abs_value: float | None,
    nonlinear_transform: bool,
) -> Tensor:
    """Extract scalar values from net output logits or scalars. x: [T, B, num_bins] or [T, B, 1]."""
    if categorical_value:
        v = transform_from_2hot(
            torch.softmax(x, dim=-1), -max_abs_value, max_abs_value, x.shape[-1]
        )
    else:
        v = x.squeeze(-1)
    if nonlinear_transform:
        v = signed_hyperbolic_inv(v)
    return v


def vtrace_td_error_and_advantage(
    v_tm1: Tensor, v_t: Tensor, r_t: Tensor, discount_t: Tensor, rho_t: Tensor,
    lambda_: float = 0.95, clip_rho: float = 1.0, clip_c: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """V-trace for a single trajectory. All inputs [T].

    Returns (td_errors [T], pg_advantages [T]).
    """
    T = r_t.shape[0]
    clipped_rho = rho_t.clamp(max=clip_rho)
    clipped_c = rho_t.clamp(max=clip_c) * lambda_

    delta = clipped_rho * (r_t + discount_t * v_t - v_tm1)

    # Backward scan for v-trace targets
    vs_minus_v = torch.zeros_like(v_tm1)
    result = torch.zeros_like(delta)
    acc = torch.tensor(0.0, device=r_t.device)
    for t in range(T - 1, -1, -1):
        acc = delta[t] + discount_t[t] * clipped_c[t] * acc
        result[t] = acc

    value_target = result + v_tm1
    td_errors = result

    # PG advantage: rho * (r + gamma * v_s' - V(s))
    vs_tp1 = torch.cat([value_target[1:], v_t[-1:]])
    pg_adv = clipped_rho * (r_t + discount_t * vs_tp1 - v_tm1)

    return td_errors, pg_adv


def estimate_q_values(
    rewards: Tensor, actions: Tensor, env_discounts: Tensor,
    rho: Tensor, values: Tensor, target_values: Tensor,
    q_values: Tensor, target_q_values: Tensor,
    discount: float, lambda_: float,
) -> ValueOuts:
    """Compute Q-value targets using Retrace-like estimator.

    rewards, env_discounts, rho: [T, B]
    values, target_values: [T+1, B]
    q_values, target_q_values: [T+1, B, A]
    actions: [T, B]
    """
    q_a = batch_lookup(q_values[:-1], actions)  # [T, B]
    target_q_a = batch_lookup(target_q_values[:-1], actions)  # [T, B]

    discounts = env_discounts * discount
    clipped_rho = rho.clamp(max=1.0)
    c_t = lambda_ * clipped_rho

    T, B = rewards.shape
    # General off-policy returns from Q and V (matches rlax.general_off_policy_returns_from_q_and_v)
    q_target = torch.zeros(T, B, device=rewards.device)
    q_target[T - 1] = rewards[T - 1] + discounts[T - 1] * target_values[T]
    for t in range(T - 2, -1, -1):
        q_target[t] = rewards[t] + discounts[t] * (
            target_values[t + 1]
            + c_t[t + 1] * (q_target[t + 1] - target_q_a[t + 1])
        )

    qv_adv = target_q_values - target_values.unsqueeze(-1)
    adv = q_target - target_values[:-1]
    v_target = target_values[:-1] + clipped_rho * (q_target - target_values[:-1])
    q_td = q_target - q_a

    return ValueOuts(
        adv=adv,
        value=values,
        target_value=target_values,
        rho=rho,
        value_target=v_target,
        td=v_target - values[:-1],
        qv_adv=qv_adv,
        q_target=q_target,
        q_value=q_values,
        target_q_value=target_q_values,
        q_td=q_td,
    )


def get_value_outs(
    q_net_out: Tensor,
    target_q_net_out: Tensor,
    rollout: UpdateRuleInputs,
    pi_logits: Tensor,
    discount: float,
    lambda_: float,
    max_abs_value: float,
    adv_ema_state: EmaState,
    adv_ema_fn: MovingAverage,
    td_ema_state: EmaState,
    td_ema_fn: MovingAverage,
) -> tuple[ValueOuts, EmaState, EmaState]:
    """Main entry: compute value targets, advantages, normalized quantities.

    Matches disco_rl/value_fns/value_utils.py get_value_outs for the Q-value path.
    """
    t_plus_1, b = pi_logits.shape[:2]

    # Extract scalar Q-values from categorical logits
    # q_net_out: [T+1, B, A, num_bins]
    def q_to_scalar(q):
        # vmap over action dim
        T, B, A, N = q.shape
        flat = q.reshape(T * B * A, N)
        vals = get_values_from_net_outs(
            flat.unsqueeze(1),  # [T*B*A, 1, N]
            categorical_value=True,
            max_abs_value=max_abs_value,
            nonlinear_transform=True,
        )  # [T*B*A, 1]
        return vals.squeeze(1).reshape(T, B, A)

    q_values = q_to_scalar(q_net_out)
    target_q_values = q_to_scalar(target_q_net_out)

    # State values from policy-weighted Q
    pi_probs = torch.softmax(pi_logits, dim=-1)
    values = (pi_probs * q_values).sum(dim=-1)
    target_values = (pi_probs * target_q_values).sum(dim=-1)

    # Rewards and terminals
    rewards = rollout.rewards
    env_discounts = 1.0 - rollout.is_terminal.float()
    actions = rollout.actions[:-1]

    # Importance weights
    mu_logits = rollout.behaviour_agent_out["logits"]
    rho = importance_weight(pi_logits[:-1], mu_logits[:-1], actions)

    value_outs = estimate_q_values(
        rewards, actions, env_discounts, rho,
        values, target_values, q_values, target_q_values,
        discount, lambda_,
    )

    # Normalize advantages
    new_adv_ema = adv_ema_fn.update_state(value_outs.adv, adv_ema_state)
    value_outs.normalized_adv = adv_ema_fn.normalize(value_outs.adv, new_adv_ema)
    value_outs.normalized_qv_adv = adv_ema_fn.normalize(value_outs.qv_adv, new_adv_ema)

    # Normalize TD
    new_td_ema = td_ema_fn.update_state(value_outs.q_td, td_ema_state)
    value_outs.normalized_q_td = td_ema_fn.normalize(
        value_outs.q_td, new_td_ema, subtract_mean=False
    )
    value_outs.normalized_td = torch.zeros_like(value_outs.td)

    return value_outs, new_adv_ema, new_td_ema

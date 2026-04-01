"""DiscoRL update rule — PyTorch port of disco_rl/update_rules/disco.py.

This is the top-level class that ties together the meta-network,
value function computation, and agent loss calculation.
"""

from __future__ import annotations
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from disco_torch.types import (
    EmaState,
    MetaNetInputOption,
    TransformConfig,
    UpdateRuleInputs,
    ValueOuts,
)
from disco_torch.meta_net import DiscoMetaNet
from disco_torch.utils import (
    batch_lookup,
    categorical_kl_divergence,
    MovingAverage,
    transform_to_2hot,
)
from disco_torch.value_utils import get_value_outs, get_values_from_net_outs


def get_input_option() -> MetaNetInputOption:
    """Default Disco103 input configuration."""
    return MetaNetInputOption(
        base=(
            TransformConfig("agent_out/logits", ("drop_last", "softmax", "stop_grad", "select_a")),
            TransformConfig("behaviour_agent_out/logits", ("drop_last", "softmax", "stop_grad", "select_a")),
            TransformConfig("rewards", ("sign_log",)),
            TransformConfig("is_terminal", ("masks_to_discounts",)),
            TransformConfig("extra_from_rule/v_scalar", ("sign_log", "td_pair", "stop_grad")),
            TransformConfig("extra_from_rule/adv", ("sign_log", "stop_grad")),
            TransformConfig("extra_from_rule/normalized_adv", ("stop_grad",)),
            TransformConfig("extra_from_rule/target_out/logits", ("drop_last", "softmax", "stop_grad", "select_a")),
            TransformConfig("agent_out/y", ("softmax", "y_net", "td_pair")),
            TransformConfig("extra_from_rule/target_out/y", ("softmax", "y_net", "td_pair")),
            TransformConfig("agent_out/z", ("drop_last", "softmax", "z_net", "select_a")),
            TransformConfig("agent_out/z", ("softmax", "z_net", "pi_weighted_avg", "td_pair")),
            TransformConfig("agent_out/z", ("softmax", "z_net", "max_a", "td_pair")),
            TransformConfig("extra_from_rule/target_out/z", ("drop_last", "softmax", "z_net", "select_a")),
            TransformConfig("extra_from_rule/target_out/z", ("softmax", "z_net", "pi_weighted_avg", "td_pair")),
            TransformConfig("extra_from_rule/target_out/z", ("softmax", "z_net", "max_a", "td_pair")),
        ),
        action_conditional=(
            TransformConfig("agent_out/logits", ("drop_last", "softmax", "stop_grad")),
            TransformConfig("behaviour_agent_out/logits", ("drop_last", "softmax", "stop_grad")),
            TransformConfig("extra_from_rule/target_out/logits", ("drop_last", "softmax", "stop_grad")),
            TransformConfig("agent_out/z", ("drop_last", "softmax", "z_net")),
            TransformConfig("extra_from_rule/target_out/z", ("drop_last", "softmax", "z_net")),
            TransformConfig("extra_from_rule/q", ("sign_log", "drop_last", "stop_grad")),
            TransformConfig("extra_from_rule/qv_adv", ("sign_log", "drop_last", "stop_grad")),
            TransformConfig("extra_from_rule/normalized_qv_adv", ("drop_last", "stop_grad")),
        ),
    )


class DiscoUpdateRule(nn.Module):
    """Full Disco103 update rule: meta-network + value computation + loss.

    Usage:
        rule = DiscoUpdateRule()
        load_disco103_weights(rule, "path/to/disco_103.npz")

        # At each learner step:
        meta_out, new_meta_state = rule.unroll_meta_net(
            rollout, params_dict, meta_state, unroll_fn
        )
        loss, logs = rule.agent_loss(rollout, meta_out, hyper_params)
    """

    def __init__(
        self,
        prediction_size: int = 600,
        value_discount: float = 0.997,
        max_abs_value: float = 300.0,
        num_bins: int = 601,
        moving_average_decay: float = 0.99,
        moving_average_eps: float = 1e-6,
    ):
        super().__init__()
        self.prediction_size = prediction_size
        self.value_discount = value_discount
        self.max_abs_value = max_abs_value
        self.num_bins = num_bins

        self.adv_ema = MovingAverage(decay=moving_average_decay, eps=moving_average_eps)
        self.td_ema = MovingAverage(decay=moving_average_decay, eps=moving_average_eps)

        input_option = get_input_option()
        self.meta_net = DiscoMetaNet(
            input_option=input_option,
            prediction_size=prediction_size,
        )

    def init_meta_state(self, agent_params: dict[str, Tensor], device=None) -> dict[str, Any]:
        """Create initial meta state."""
        h, c = self.meta_net.initial_meta_rnn_state(device)
        return {
            "rnn_state": (h, c),
            "adv_ema_state": self.adv_ema.init_state(device),
            "td_ema_state": self.td_ema.init_state(device),
            "target_params": {k: v.clone() for k, v in agent_params.items()},
        }

    def unroll_meta_net(
        self,
        rollout: UpdateRuleInputs,
        agent_params: dict[str, Tensor],
        meta_state: dict[str, Any],
        unroll_policy_fn,
        hyper_params: dict[str, float],
    ) -> tuple[dict[str, Tensor], dict[str, Any]]:
        """Run the value computation and meta-network to produce loss targets.

        Args:
            rollout: current rollout data
            agent_params: current agent network parameters
            meta_state: persistent state (rnn, ema, target params)
            unroll_policy_fn: callable(params, state, obs, reset) -> (agent_out, state)
            hyper_params: dict with 'value_fn_td_lambda', 'target_params_coeff'

        Returns:
            meta_out: dict with 'pi', 'y', 'z' targets and value quantities
            new_meta_state: updated state
        """
        # Unroll target policy
        target_out, _ = unroll_policy_fn(
            meta_state["target_params"],
            rollout.observations,
            rollout.should_reset_mask_fwd,
        )

        # Compute value targets and advantages
        value_outs, new_adv_ema, new_td_ema = get_value_outs(
            q_net_out=rollout.agent_out["q"],
            target_q_net_out=target_out["q"],
            rollout=rollout,
            pi_logits=rollout.agent_out["logits"],
            discount=self.value_discount,
            lambda_=hyper_params.get("value_fn_td_lambda", 0.95),
            max_abs_value=self.max_abs_value,
            adv_ema_state=meta_state["adv_ema_state"],
            adv_ema_fn=self.adv_ema,
            td_ema_state=meta_state["td_ema_state"],
            td_ema_fn=self.td_ema,
        )

        # Populate extra_from_rule for the meta-network
        rollout.extra_from_rule = {
            "v_scalar": value_outs.value,
            "adv": value_outs.adv,
            "normalized_adv": value_outs.normalized_adv,
            "q": value_outs.target_q_value,
            "qv_adv": value_outs.qv_adv,
            "normalized_qv_adv": value_outs.normalized_qv_adv,
            "target_out": target_out,
        }

        # Run the meta-network
        meta_out, new_rnn_state = self.meta_net(
            rollout, meta_state["rnn_state"]
        )

        # Enrich meta_out with value quantities (used by agent_loss_no_meta)
        meta_out["q_target"] = value_outs.q_target
        meta_out["adv"] = value_outs.adv
        meta_out["normalized_adv"] = value_outs.normalized_adv
        meta_out["qv_adv"] = value_outs.qv_adv
        meta_out["normalized_qv_adv"] = value_outs.normalized_qv_adv
        meta_out["q_value"] = value_outs.q_value
        meta_out["q_td"] = value_outs.q_td
        meta_out["normalized_q_td"] = value_outs.normalized_q_td
        meta_out["target_out"] = target_out

        # Update target params with Polyak averaging (target slowly tracks current)
        # Reference: lambda old, new: old * coeff + (1 - coeff) * new
        coeff = hyper_params.get("target_params_coeff", 0.9)
        new_target_params = {
            k: meta_state["target_params"][k] * coeff + agent_params[k] * (1.0 - coeff)
            for k in agent_params
        }

        new_meta_state = {
            "rnn_state": new_rnn_state,
            "adv_ema_state": new_adv_ema,
            "td_ema_state": new_td_ema,
            "target_params": new_target_params,
        }

        return meta_out, new_meta_state

    def agent_loss(
        self,
        rollout: UpdateRuleInputs,
        meta_out: dict[str, Tensor],
        hyper_params: dict[str, float],
        backprop_through_targets: bool = False,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute the Disco103 agent loss (KL divergences).

        Returns per-step loss [T, B] and a log dict.
        """
        t, b = rollout.rewards.shape
        agent_out = {k: v[:-1] for k, v in rollout.agent_out.items()}
        actions = rollout.actions[:-1]

        logits = agent_out["logits"]
        y = agent_out["y"]
        z_a = batch_lookup(agent_out["z"], actions)

        pi_hat = meta_out["pi"]
        y_hat = meta_out["y"]
        z_hat = meta_out["z"]
        if not backprop_through_targets:
            pi_hat = pi_hat.detach()
            y_hat = y_hat.detach()
            z_hat = z_hat.detach()

        # KL divergence losses
        pi_loss = categorical_kl_divergence(pi_hat, logits)
        y_loss = categorical_kl_divergence(y_hat, y)
        z_loss = categorical_kl_divergence(z_hat, z_a)

        # Auxiliary 1-step policy prediction loss
        aux_pi = rollout.agent_out["aux_pi"][:-1]  # [T, B, A, A]
        aux_pi_a = batch_lookup(aux_pi, actions)  # [T, B, A]
        aux_target = rollout.agent_out["logits"][1:]  # [T, B, A]
        aux_loss = categorical_kl_divergence(aux_target.detach(), aux_pi_a)
        aux_loss = aux_loss * (1.0 - rollout.is_terminal.float())

        total = (
            hyper_params.get("pi_cost", 1.0) * pi_loss
            + hyper_params.get("y_cost", 1.0) * y_loss
            + hyper_params.get("z_cost", 1.0) * z_loss
            + hyper_params.get("aux_policy_cost", 1.0) * aux_loss
        )

        log = {
            "pi_loss": pi_loss.mean(),
            "y_loss": y_loss.mean(),
            "z_loss": z_loss.mean(),
            "aux_loss": aux_loss.mean(),
            "total_loss": total.mean(),
        }
        return total, log

    def agent_loss_no_meta(
        self,
        rollout: UpdateRuleInputs,
        meta_out: dict[str, Tensor],
        hyper_params: dict[str, float],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Value function loss (no meta-gradient)."""
        q_a = batch_lookup(rollout.agent_out["q"], rollout.actions)[:-1]  # [T, B, num_bins]
        q_td = meta_out["q_td"].detach()

        # Compute value loss from TD
        T, B = rollout.rewards.shape
        num_bins = q_a.shape[-1]

        # Get scalar Q-values
        q_scalar = get_values_from_net_outs(
            q_a, categorical_value=True,
            max_abs_value=self.max_abs_value,
            nonlinear_transform=True,
        )
        # Target = q_scalar + td, then convert to categorical
        from disco_torch.utils import signed_hyperbolic_tx
        target_scalar = q_scalar + q_td
        target_scalar_tx = signed_hyperbolic_tx(target_scalar)
        target_probs = transform_to_2hot(
            target_scalar_tx, -self.max_abs_value, self.max_abs_value, num_bins
        )
        # Cross-entropy loss
        log_probs = torch.log_softmax(q_a, dim=-1)
        value_loss = -(target_probs.detach() * log_probs).sum(dim=-1)  # [T, B]

        loss = value_loss * hyper_params.get("value_cost", 0.2)

        log = {
            "q_loss": value_loss.mean(),
            "q_td": q_td.mean(),
        }
        return loss, log

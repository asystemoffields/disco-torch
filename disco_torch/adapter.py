"""Disco103 adapter for language models.

Translates LLM hidden states into the format the Disco103 meta-network
expects, enabling per-step credit assignment without modifying the LLM's
policy gradient. The adapter is a small standalone network whose gradients
never flow into the LLM.

Usage with GRPO::

    adapter = DiscoAdapter(d_model=768, num_actions=64)
    credit, new_state = credit_from_lm_rollout(
        rule, adapter, lm, input_ids,
        prompt_len=4, top_k=64,
        rewards=rewards, is_terminal=is_terminal,
        meta_state=meta_state,
    )
    # credit["normalized_adv"] is [T, B] per-step credit
"""

from __future__ import annotations
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from disco_torch.types import UpdateRuleInputs
from disco_torch.update_rule import DiscoUpdateRule


class DiscoAdapter(nn.Module):
    """Translates LLM hidden states into Disco103-compatible agent outputs.

    A small network producing the auxiliary heads (y, z, q, aux_pi) that the
    meta-network requires. These heads are not part of the LLM — the adapter
    exists solely to interface with the meta-network for credit assignment.

    Args:
        d_model: LLM hidden state dimension (768 for distilgpt2)
        num_actions: action space size (top-K)
        prediction_size: Disco103 prediction vector size (600)
        num_bins: Q-value distribution bins (601)
        h_act: hidden dim for per-action features
    """

    def __init__(
        self,
        d_model: int = 768,
        num_actions: int = 64,
        prediction_size: int = 600,
        num_bins: int = 601,
        h_act: int = 32,
    ):
        super().__init__()
        self.num_actions = num_actions
        self._h_act = h_act

        self.y_head = nn.Linear(d_model, prediction_size)
        self.action_proj = nn.Linear(d_model, num_actions * h_act)
        self.z_from_feat = nn.Linear(h_act, prediction_size)
        self.q_from_feat = nn.Linear(h_act, num_bins)
        self.aux_pi_head = nn.Linear(d_model, num_actions * num_actions)

    def forward(self, hidden_states: Tensor, topk_logits: Tensor) -> dict[str, Tensor]:
        """Produce Disco103-format agent outputs.

        Args:
            hidden_states: [T+1, B, d_model] time-first LLM hidden states
            topk_logits: [T+1, B, K] logits for top-K tokens

        Returns:
            dict with logits, y, z, q, aux_pi — all [T+1, B, ...]
        """
        K = self.num_actions
        y = self.y_head(hidden_states)
        af = self.action_proj(hidden_states).unflatten(-1, (K, self._h_act))
        z = self.z_from_feat(af)
        q = self.q_from_feat(af)
        aux_pi = self.aux_pi_head(hidden_states).unflatten(-1, (K, K))
        return {"logits": topk_logits, "y": y, "z": z, "q": q, "aux_pi": aux_pi}


@torch.no_grad()
def extract_credit(
    rule: DiscoUpdateRule,
    adapter: DiscoAdapter,
    hidden_states: Tensor,
    topk_logits: Tensor,
    actions_topk: Tensor,
    rewards: Tensor,
    is_terminal: Tensor,
    meta_state: dict[str, Any],
    hyper_params: dict[str, float] | None = None,
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    """Extract per-step credit signals from the meta-network.

    Args:
        hidden_states: [T+1, B, d_model] time-first hidden states
        topk_logits: [T+1, B, K] top-K logits
        actions_topk: [T+1, B] actions as indices into top-K
        rewards: [T, B] per-step rewards
        is_terminal: [T, B] terminal flags
        meta_state: meta-network state dict
        hyper_params: optional override

    Returns:
        credit: dict with 'adv', 'normalized_adv', 'q_td', 'normalized_q_td'
        new_meta_state: updated meta-network state
    """
    if hyper_params is None:
        hyper_params = {
            "value_fn_td_lambda": 0.95,
            "target_params_coeff": 0.9,
        }

    T, B = rewards.shape
    agent_out = adapter(hidden_states, topk_logits)
    agent_out_d = {k: v.detach() for k, v in agent_out.items()}

    observations = torch.zeros(T + 1, B, dtype=torch.long, device=rewards.device)

    rollout = UpdateRuleInputs(
        observations=observations,
        actions=actions_topk,
        rewards=rewards,
        is_terminal=is_terminal,
        agent_out=agent_out_d,
        behaviour_agent_out={k: v.clone() for k, v in agent_out_d.items()},
    )

    # The unroll_fn uses target params (EMA of adapter) with the same
    # hidden_states/topk_logits (from frozen LLM, shared across behaviour/target).
    def unroll_fn(params, obs, reset_mask):
        out = torch.func.functional_call(adapter, params, (hidden_states, topk_logits))
        return out, None

    adapter_params = dict(adapter.named_parameters())
    meta_out, new_meta_state = rule.unroll_meta_net(
        rollout, adapter_params, meta_state, unroll_fn, hyper_params,
    )

    credit = {
        "adv": meta_out["adv"],
        "normalized_adv": meta_out["normalized_adv"],
        "q_td": meta_out["q_td"],
        "normalized_q_td": meta_out["normalized_q_td"],
    }
    return credit, new_meta_state


@torch.no_grad()
def credit_from_lm_rollout(
    rule: DiscoUpdateRule,
    adapter: DiscoAdapter,
    lm: nn.Module,
    input_ids: Tensor,
    prompt_len: int,
    top_k: int,
    rewards: Tensor,
    is_terminal: Tensor,
    meta_state: dict[str, Any],
    hyper_params: dict[str, float] | None = None,
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    """Extract credit from a completed LM rollout.

    Higher-level wrapper that handles hidden state extraction, top-K
    computation, and token-to-action mapping.

    Args:
        lm: language model with .transformer and .lm_head (GPT2LMHeadModel)
        input_ids: [B, P+T] full token sequences (prompt + completion)
        prompt_len: number of prompt tokens (P)
        top_k: action space size (K)
        rewards: [T, B] per-step rewards (time-first)
        is_terminal: [T, B] terminal flags
        meta_state: meta-network state dict

    Returns:
        credit: dict with per-step signals (all [T, B])
        new_meta_state: updated state
    """
    B, total_len = input_ids.shape
    T = total_len - prompt_len
    device = input_ids.device

    # Get hidden states and logits from LM
    out = lm.transformer(input_ids)
    hidden = out.last_hidden_state  # [B, P+T, d]
    full_logits = lm.lm_head(hidden)  # [B, P+T, V]

    # Completion-relevant positions: P-1 produces logits for first token, etc.
    hidden_comp = hidden[:, prompt_len - 1:, :]  # [B, T+1, d]
    logits_comp = full_logits[:, prompt_len - 1:, :]  # [B, T+1, V]

    # Top-K
    topk_vals, topk_idx = logits_comp.topk(top_k, dim=-1)  # [B, T+1, K]

    # Map generated tokens to their top-K indices
    gen_tokens = input_ids[:, prompt_len:]  # [B, T]
    actions_list = []
    for t in range(T):
        matches = (topk_idx[:, t, :] == gen_tokens[:, t:t + 1])  # [B, K]
        action_idx = matches.float().argmax(dim=-1)  # [B]
        actions_list.append(action_idx)

    actions = torch.stack(actions_list, dim=0)  # [T, B]
    actions = torch.cat([
        actions,
        torch.zeros(1, B, dtype=torch.long, device=device),
    ], dim=0)  # [T+1, B]

    # Convert to time-first
    hidden_tf = hidden_comp.transpose(0, 1)  # [T+1, B, d]
    topk_tf = topk_vals.transpose(0, 1)  # [T+1, B, K]

    return extract_credit(
        rule, adapter, hidden_tf, topk_tf, actions,
        rewards, is_terminal, meta_state, hyper_params,
    )

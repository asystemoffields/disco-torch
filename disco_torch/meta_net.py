"""Meta-network — PyTorch port of disco_rl/networks/meta_nets.py.

This is the core LSTM that generates loss targets for the Disco103 update rule.
All sizes are hardcoded to match the Disco103 checkpoint (754,778 params).
"""

from __future__ import annotations
from typing import Sequence

import torch
import torch.nn as nn
from torch import Tensor

from disco_torch.types import MetaNetInputOption, UpdateRuleInputs
from disco_torch.transforms import construct_input


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class BatchMLP(nn.Module):
    """MLP applied independently over leading batch dimensions."""

    def __init__(self, in_features: int, hiddens: Sequence[int]):
        super().__init__()
        layers = []
        d_in = in_features
        for i, d_out in enumerate(hiddens):
            layers.append(nn.Linear(d_in, d_out))
            if i < len(hiddens) - 1:
                layers.append(nn.ReLU())
            d_in = d_out
        self.net = nn.Sequential(*layers)
        self.out_features = hiddens[-1] if hiddens else in_features

    def forward(self, x: Tensor) -> Tensor:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        out = self.net(flat)
        return out.reshape(*shape[:-1], -1)


class Conv1dBlock(nn.Module):
    """Concat with action-mean, conv1d, relu. Operates on [..., A, C]."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels * 2, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        # x: [*, A, C]
        prefix = x.shape[:-2]
        a, c = x.shape[-2], x.shape[-1]
        x_avg = x.mean(dim=-2, keepdim=True).expand(*prefix, a, c)
        x_cat = torch.cat([x, x_avg], dim=-1)  # [*, A, 2C]
        flat = x_cat.reshape(-1, a, c * 2).transpose(1, 2)  # [N, 2C, A]
        out = torch.relu(self.conv(flat))  # [N, Cout, A]
        out = out.transpose(1, 2)  # [N, A, Cout]
        return out.reshape(*prefix, a, -1)


class Conv1dNet(nn.Module):
    """Stack of Conv1dBlocks."""

    def __init__(self, in_channels: int, channels: Sequence[int]):
        super().__init__()
        blocks = []
        c_in = in_channels
        for c_out in channels:
            blocks.append(Conv1dBlock(c_in, c_out))
            c_in = c_out
        self.blocks = nn.ModuleList(blocks)
        self.out_channels = channels[-1] if channels else in_channels

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class HaikuLSTMCell(nn.Module):
    """LSTM cell matching Haiku's gate order [i, g, f, o] and forget bias +1.

    Haiku: gates = Linear([x, h]) -> split into [i, g, f, o]
           f = sigmoid(f + 1), i = sigmoid(i), g = tanh(g), o = sigmoid(o)
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(input_size + hidden_size, 4 * hidden_size)

    def forward(self, x: Tensor, state: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        h, c = state
        gates = self.linear(torch.cat([x, h], dim=-1))
        i, g, f, o = gates.chunk(4, dim=-1)
        f = torch.sigmoid(f + 1.0)  # Haiku forget bias
        i = torch.sigmoid(i)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        new_c = f * c + i * g
        new_h = o * torch.tanh(new_c)
        return new_h, new_c


class ResetLSTM(nn.Module):
    """LSTM that resets hidden state at episode boundaries."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.cell = HaikuLSTMCell(input_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, x: Tensor, should_reset: Tensor, reverse: bool = False) -> Tensor:
        """x: [T, B, D], should_reset: [T, B]. Returns [T, B, H]."""
        t, b, _ = x.shape
        device = x.device
        h = torch.zeros(b, self.hidden_size, device=device)
        c = torch.zeros(b, self.hidden_size, device=device)
        outputs = []
        time_range = range(t - 1, -1, -1) if reverse else range(t)
        for i in time_range:
            mask = should_reset[i].unsqueeze(-1)
            h = h * (1.0 - mask)
            c = c * (1.0 - mask)
            h, c = self.cell(x[i], (h, c))
            outputs.append(h)
        if reverse:
            outputs.reverse()
        return torch.stack(outputs, dim=0)


# ---------------------------------------------------------------------------
# Main meta-network
# ---------------------------------------------------------------------------

class DiscoMetaNet(nn.Module):
    """The full Disco103 meta-network.

    Architecture (from disco_103.npz, 754,778 params):

    Outer components:
      y_net:            MLP [600 -> 16 -> 1]   (embedding for y predictions)
      z_net:            MLP [600 -> 16 -> 1]   (embedding for z predictions)
      policy_net:       Conv1dNet [9 -> 16 -> 2] (action-conditional embedding)
      trajectory_rnn:   LSTM(input=27, hidden=256) reverse-unrolled per trajectory
      state_gate:       Linear(128 -> 256)      (multiplicative interaction with meta-rnn)
      meta_input_head:  Linear(256 -> 1)
      y_head:           Linear(256 -> 600)
      z_head:           Linear(256 -> 600)
      pi_conv:          Conv1dNet [258 -> 16]   (policy target, 256+2=258 input)
      pi_head:          Linear(16 -> 1)

    Meta-RNN (lifetime-level, processes batch-time averages):
      meta_rnn_y_net:      MLP [600 -> 16 -> 1]  (separate params from outer y_net)
      meta_rnn_z_net:      MLP [600 -> 16 -> 1]
      meta_rnn_policy_net: Conv1dNet [9 -> 16 -> 2]
      meta_rnn_input_mlp:  MLP [29 -> 16]         (27 base + 1 meta_emb + 1 y_emb)
      meta_rnn_cell:       LSTMCell(16, 128)
    """

    def __init__(self, input_option: MetaNetInputOption, prediction_size: int = 600):
        super().__init__()
        self.prediction_size = prediction_size
        self.input_option = input_option

        # --- Outer components ---
        self.y_net = BatchMLP(prediction_size, [16, 1])
        self.z_net = BatchMLP(prediction_size, [16, 1])
        self.policy_net = Conv1dNet(9, [16, 2])

        self.trajectory_rnn = ResetLSTM(input_size=27, hidden_size=256)
        self.state_gate = nn.Linear(128, 256)

        self.meta_input_head = nn.Linear(256, 1)
        self.y_head = nn.Linear(256, prediction_size)
        self.z_head = nn.Linear(256, prediction_size)

        self.pi_conv = Conv1dNet(258, [16])  # 256 hidden + 2 act_cond channels
        self.pi_head = nn.Linear(16, 1)

        # --- Meta-RNN components (separate params) ---
        self.meta_rnn_y_net = BatchMLP(prediction_size, [16, 1])
        self.meta_rnn_z_net = BatchMLP(prediction_size, [16, 1])
        self.meta_rnn_policy_net = Conv1dNet(9, [16, 2])
        self.meta_rnn_input_mlp = nn.Sequential(nn.Linear(29, 16))
        self.meta_rnn_cell = HaikuLSTMCell(16, 128)

    def initial_meta_rnn_state(self, device=None) -> tuple[Tensor, Tensor]:
        h = torch.zeros(128, device=device)
        c = torch.zeros(128, device=device)
        return h, c

    def forward(
        self,
        inputs: UpdateRuleInputs,
        meta_rnn_state: tuple[Tensor, Tensor],
    ) -> tuple[dict[str, Tensor], tuple[Tensor, Tensor]]:
        """Run the meta-network on a rollout.

        Args:
            inputs: rollout data with extra_from_rule populated
            meta_rnn_state: (h, c) for the lifetime meta-RNN

        Returns:
            meta_out: dict with 'pi', 'y', 'z', 'meta_input_emb'
            new_meta_rnn_state: updated (h, c)
        """
        logits = inputs.agent_out["logits"]  # [T+1, B, A]
        _, b, num_actions = logits.shape
        t = inputs.rewards.shape[0]

        # ---- Build input vector using outer y_net/z_net/policy_net ----
        base_input, act_cond_emb = construct_input(
            inputs, self.input_option,
            y_net=self.y_net, z_net=self.z_net, policy_net=self.policy_net,
        )  # base_input: [T, B, 27], act_cond_emb: [T, B, A, 2]

        # ---- Per-trajectory reverse LSTM ----
        should_reset_bwd = inputs.should_reset_mask_bwd[:-1]  # [T, B]
        x = self.trajectory_rnn(base_input, should_reset_bwd, reverse=True)  # [T, B, 256]

        # ---- Multiplicative interaction with meta-RNN ----
        meta_h = meta_rnn_state[0]  # [128]
        gate = self.state_gate(meta_h)  # [256]
        x = x * gate.unsqueeze(0).unsqueeze(0)  # [T, B, 256]

        # ---- Output heads ----
        meta_input_emb = self.meta_input_head(x)  # [T, B, 1]
        y_hat = self.y_head(x)  # [T, B, 600]
        z_hat = self.z_head(x)  # [T, B, 600]

        # ---- Policy target ----
        w = x.unsqueeze(2).expand(-1, -1, num_actions, -1)  # [T, B, A, 256]
        if act_cond_emb is not None:
            w = torch.cat([w, act_cond_emb], dim=-1)  # [T, B, A, 258]
        w = self.pi_conv(w)  # [T, B, A, 16]
        pi_hat = self.pi_head(w).squeeze(-1)  # [T, B, A]

        meta_out = {
            "pi": pi_hat,
            "y": y_hat,
            "z": z_hat,
            "meta_input_emb": meta_input_emb,
        }

        # ---- Update lifetime meta-RNN ----
        # The meta-RNN recomputes construct_input with its OWN y_net/z_net/policy_net
        meta_base_input, _ = construct_input(
            inputs, self.input_option,
            y_net=self.meta_rnn_y_net,
            z_net=self.meta_rnn_z_net,
            policy_net=self.meta_rnn_policy_net,
        )  # [T, B, 27]

        # Embed meta_out['y'] using the meta-rnn's y_net (same params as used in construct_input above)
        y_hat_emb = self.meta_rnn_y_net(torch.softmax(y_hat, dim=-1))  # [T, B, 1]

        meta_rnn_input = torch.cat(
            [meta_base_input, meta_input_emb, y_hat_emb], dim=-1
        )  # [T, B, 29]
        meta_rnn_input = self.meta_rnn_input_mlp(meta_rnn_input)  # [T, B, 16]

        # Average pool over T and B
        x_avg = meta_rnn_input.mean(dim=(0, 1))  # [16]

        # Single step of the meta-RNN
        h, c = meta_rnn_state
        new_h, new_c = self.meta_rnn_cell(
            x_avg.unsqueeze(0), (h.unsqueeze(0), c.unsqueeze(0))
        )
        new_meta_rnn_state = (new_h.squeeze(0), new_c.squeeze(0))

        return meta_out, new_meta_rnn_state

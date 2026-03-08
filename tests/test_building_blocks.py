"""Tests for disco_torch.meta_net building blocks — no JAX or weights required."""

import torch
import pytest

from disco_torch.meta_net import (
    BatchMLP,
    Conv1dBlock,
    Conv1dNet,
    HaikuLSTMCell,
    ResetLSTM,
)


class TestBatchMLP:
    def test_output_shape(self):
        mlp = BatchMLP(16, [32, 8])
        x = torch.randn(3, 5, 16)
        out = mlp(x)
        assert out.shape == (3, 5, 8)

    def test_flat_input(self):
        mlp = BatchMLP(4, [8, 2])
        x = torch.randn(10, 4)
        out = mlp(x)
        assert out.shape == (10, 2)

    def test_single_layer(self):
        mlp = BatchMLP(8, [4])
        x = torch.randn(2, 3, 8)
        out = mlp(x)
        assert out.shape == (2, 3, 4)


class TestConv1dBlock:
    def test_output_shape(self):
        block = Conv1dBlock(in_channels=5, out_channels=8)
        x = torch.randn(2, 3, 4, 5)  # [B1, B2, A, C]
        out = block(x)
        assert out.shape == (2, 3, 4, 8)

    def test_uses_mean_concat(self):
        # The block concatenates input with its mean over action dim
        # so the conv input channels should be 2*C
        block = Conv1dBlock(in_channels=3, out_channels=4)
        assert block.conv.in_channels == 6  # 2 * 3


class TestConv1dNet:
    def test_stacked_blocks(self):
        net = Conv1dNet(in_channels=9, channels=[16, 2])
        x = torch.randn(5, 2, 4, 9)  # [T, B, A, C]
        out = net(x)
        assert out.shape == (5, 2, 4, 2)
        assert len(net.blocks) == 2


class TestHaikuLSTMCell:
    def test_output_shape(self):
        cell = HaikuLSTMCell(input_size=16, hidden_size=32)
        x = torch.randn(4, 16)
        h = torch.zeros(4, 32)
        c = torch.zeros(4, 32)
        new_h, new_c = cell(x, (h, c))
        assert new_h.shape == (4, 32)
        assert new_c.shape == (4, 32)

    def test_forget_bias(self):
        """Verify the +1 forget gate bias produces high initial forget values."""
        cell = HaikuLSTMCell(input_size=4, hidden_size=8)
        # With zero weights and biases, gates = linear([x, h]) = 0
        # f = sigmoid(0 + 1) = 0.731... (high forget = remember)
        with torch.no_grad():
            cell.linear.weight.zero_()
            cell.linear.bias.zero_()
        x = torch.zeros(1, 4)
        h = torch.zeros(1, 8)
        c = torch.ones(1, 8)  # non-zero cell state
        new_h, new_c = cell(x, (h, c))
        # With f=sigmoid(1)~0.731 and i=sigmoid(0)=0.5, g=tanh(0)=0:
        # new_c = f * c + i * g = 0.731 * 1 + 0.5 * 0 = 0.731
        expected_c = torch.sigmoid(torch.tensor(1.0))
        assert torch.allclose(new_c, expected_c.expand_as(new_c), atol=1e-6)


class TestResetLSTM:
    def test_output_shape(self):
        lstm = ResetLSTM(input_size=8, hidden_size=16)
        x = torch.randn(5, 3, 8)  # [T, B, D]
        reset = torch.zeros(5, 3)
        out = lstm(x, reset)
        assert out.shape == (5, 3, 16)

    def test_reset_clears_state(self):
        lstm = ResetLSTM(input_size=4, hidden_size=8)
        x = torch.randn(4, 1, 4)
        # Reset at t=2 (forward direction)
        reset = torch.tensor([[0.0], [0.0], [1.0], [0.0]])
        out_reset = lstm(x, reset, reverse=False)
        # Without reset
        out_no_reset = lstm(x, torch.zeros(4, 1), reverse=False)
        # Outputs should differ after the reset point
        assert not torch.allclose(out_reset[2], out_no_reset[2], atol=1e-6)

    def test_reverse(self):
        lstm = ResetLSTM(input_size=4, hidden_size=8)
        x = torch.randn(3, 1, 4)
        reset = torch.zeros(3, 1)
        out_fwd = lstm(x, reset, reverse=False)
        out_rev = lstm(x, reset, reverse=True)
        # Forward and reverse should give different results
        assert not torch.allclose(out_fwd, out_rev)
        # Both should have the same shape
        assert out_fwd.shape == out_rev.shape

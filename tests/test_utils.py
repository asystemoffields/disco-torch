"""Tests for disco_torch.utils — no JAX or weights required."""

import torch
import pytest

from disco_torch.utils import (
    batch_lookup,
    signed_logp1,
    signed_hyperbolic,
    signed_hyperbolic_tx,
    signed_hyperbolic_inv,
    categorical_kl_divergence,
    transform_to_2hot,
    transform_from_2hot,
    MovingAverage,
)


class TestBatchLookup:
    def test_2d_table(self):
        table = torch.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
        index = torch.tensor([2, 0])
        result = batch_lookup(table, index)
        assert torch.allclose(result, torch.tensor([30.0, 40.0]))

    def test_3d_table(self):
        # [B1, B2, A] with index [B1, B2]
        table = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        index = torch.tensor([[0, 1, 2], [3, 0, 1]])
        result = batch_lookup(table, index)
        assert result.shape == (2, 3)
        assert result[0, 0] == table[0, 0, 0]
        assert result[0, 1] == table[0, 1, 1]
        assert result[1, 2] == table[1, 2, 1]

    def test_trailing_dims(self):
        # [B, A, D] with index [B] -> [B, D]
        table = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
        index = torch.tensor([1, 2])
        result = batch_lookup(table, index)
        assert result.shape == (2, 2)
        assert torch.allclose(result[0], table[0, 1])
        assert torch.allclose(result[1], table[1, 2])


class TestSignedLogp1:
    def test_positive(self):
        x = torch.tensor(2.0)
        expected = torch.log(torch.tensor(3.0))
        assert torch.allclose(signed_logp1(x), expected)

    def test_negative(self):
        x = torch.tensor(-3.0)
        expected = -torch.log(torch.tensor(4.0))
        assert torch.allclose(signed_logp1(x), expected)

    def test_zero(self):
        assert signed_logp1(torch.tensor(0.0)).item() == 0.0


class TestSignedHyperbolic:
    def test_roundtrip(self):
        x = torch.linspace(-10, 10, 100)
        y = signed_hyperbolic_tx(x)
        x_recovered = signed_hyperbolic_inv(y)
        assert torch.allclose(x, x_recovered, atol=1e-3)

    def test_zero(self):
        assert signed_hyperbolic(torch.tensor(0.0)).item() == 0.0

    def test_monotonic(self):
        x = torch.linspace(-5, 5, 50)
        y = signed_hyperbolic(x)
        diffs = y[1:] - y[:-1]
        assert (diffs > 0).all()


class TestCategoricalKL:
    def test_same_distribution(self):
        logits = torch.randn(4, 8)
        kl = categorical_kl_divergence(logits, logits)
        assert torch.allclose(kl, torch.zeros(4), atol=1e-6)

    def test_nonnegative(self):
        p = torch.randn(10, 5)
        q = torch.randn(10, 5)
        kl = categorical_kl_divergence(p, q)
        assert (kl >= -1e-6).all()

    def test_asymmetric(self):
        p = torch.tensor([[3.0, -1.0, 0.0]])
        q = torch.tensor([[-1.0, 0.5, 0.0]])
        kl_pq = categorical_kl_divergence(p, q)
        kl_qp = categorical_kl_divergence(q, p)
        assert not torch.allclose(kl_pq, kl_qp)


class TestTwoHot:
    def test_roundtrip(self):
        values = torch.tensor([-100.0, -1.5, 0.0, 42.7, 300.0])
        probs = transform_to_2hot(values, -300.0, 300.0, 601)
        recovered = transform_from_2hot(probs, -300.0, 300.0, 601)
        assert torch.allclose(values, recovered, atol=1.0)  # within 1 bin width

    def test_probabilities_sum_to_one(self):
        values = torch.randn(10) * 100
        probs = transform_to_2hot(values, -300.0, 300.0, 601)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(10), atol=1e-6)

    def test_clipping(self):
        # Values outside range should be clipped
        values = torch.tensor([-999.0, 999.0])
        probs = transform_to_2hot(values, -300.0, 300.0, 601)
        recovered = transform_from_2hot(probs, -300.0, 300.0, 601)
        assert torch.allclose(recovered, torch.tensor([-300.0, 300.0]), atol=1e-4)


class TestMovingAverage:
    def test_init_state(self):
        ema = MovingAverage(decay=0.99)
        state = ema.init_state()
        assert state.moment1.item() == 0.0
        assert state.moment2.item() == 0.0
        assert state.decay_product.item() == 1.0

    def test_normalize_zero_mean(self):
        ema = MovingAverage(decay=0.99)
        state = ema.init_state()
        values = torch.randn(100)
        state = ema.update_state(values, state)
        normalized = ema.normalize(values, state)
        # After normalization, mean should be close to 0
        assert abs(normalized.mean().item()) < abs(values.mean().item()) + 1.0

    def test_normalize_no_subtract_mean(self):
        ema = MovingAverage(decay=0.99)
        state = ema.init_state()
        values = torch.ones(10) * 5.0
        state = ema.update_state(values, state)
        norm_sub = ema.normalize(values, state, subtract_mean=True)
        norm_nosub = ema.normalize(values, state, subtract_mean=False)
        # Without subtracting mean, result should be larger
        assert norm_nosub.mean() > norm_sub.mean()

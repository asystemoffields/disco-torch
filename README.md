# disco-torch

A PyTorch port of DeepMind's **Disco103** — the meta-learned reinforcement learning update rule from [*Discovering Reinforcement Learning Algorithms*](https://doi.org/10.1038/s41586-025-09761-x) (Nature, 2025).

Disco103 is a small neural network (754K params) that **replaces hand-crafted RL loss functions**. Instead of PPO or GRPO, you feed it agent experience and it outputs loss targets. The agent trains by minimizing KL divergence against these targets. It was meta-trained across thousands of environments and generalizes to new tasks as a drop-in update rule.

## Validated

This port achieves **99% catch rate** on the reference Catch benchmark, matching the original JAX implementation:

```
Step   50 | catch=35%     (learning starts)
Step  500 | catch=52%     (steady improvement)
Step  700 | catch=75%     (accelerating)
Step  900 | catch=83%     (past validation threshold)
Step 1000 | catch=99%     (converged)
```

All meta-network outputs match the JAX reference within float32 precision (<1e-6 max diff).

## Installation

```bash
pip install disco-torch
```

With optional extras:

```bash
pip install disco-torch[hub]       # HuggingFace Hub weight downloads
pip install disco-torch[dev]       # pytest for development
```

## Quick start

```bash
# Run the validated Catch benchmark (~2 hours on GPU, ~99% catch rate)
python examples/catch_disco.py

# With A2C baseline for comparison
python examples/catch_disco.py --both
```

### Using Disco103 in your own training loop

```python
import torch
from disco_torch import DiscoUpdateRule, UpdateRuleInputs
from disco_torch.load_weights import download_disco103_weights, load_disco103_weights

# Load the meta-network with pretrained weights
rule = DiscoUpdateRule().to(device)
weights_path = download_disco103_weights()  # auto-downloads from HuggingFace
load_disco103_weights(rule, weights_path)

# Initialize persistent meta-state
agent_params = dict(agent.named_parameters())
meta_state = rule.init_meta_state(agent_params, device=device)

# At each learner step:
# 1. Run meta-network to produce loss targets
with torch.no_grad():
    meta_out, new_meta_state = rule.unroll_meta_net(
        rollout, agent_params, meta_state, unroll_fn, hyper_params
    )

# 2. Compute agent loss (KL divergence against meta-network targets)
policy_loss, logs = rule.agent_loss(rollout, meta_out, hyper_params)
value_loss, vlogs = rule.agent_loss_no_meta(rollout, meta_out, hyper_params)

# 3. Standard PyTorch backward pass
total_loss = (policy_loss + value_loss).mean()
total_loss.backward()
optimizer.step()

# 4. Carry meta-state forward
meta_state = new_meta_state
```

See [`examples/catch_disco.py`](examples/catch_disco.py) for the complete, validated training loop including the agent architecture, replay buffer, and all configuration details.

## Agent requirements

Your agent's forward pass must return a dict with these keys:

| Key | Shape | Description |
|-----|-------|-------------|
| `logits` | `[B, A]` | Policy logits (unnormalized) |
| `y` | `[B, 600]` | Value prediction vector |
| `z` | `[B, A, 600]` | Per-action auxiliary prediction |
| `q` | `[B, A, 601]` | Per-action Q-value (601-bin categorical) |
| `aux_pi` | `[B, A, A]` | 1-step policy prediction |

The reference agent architecture (feedforward MLP + action-conditional LSTM) is implemented in [`examples/catch_disco.py`](examples/catch_disco.py) as `DiscoMLPAgent`.

## Key training loop details

These details matter for correct behavior (all handled in the example):

- **Replay ratio 32**: Run 32 gradient steps per environment step
- **Per-element gradient clipping**: `grad.clamp_(-1.0, 1.0)` before Adam (not norm clipping)
- **Target network**: Polyak averaging with `coeff=0.9` (target tracks current params)
- **Terminal masking**: Zero out loss at terminal timesteps
- **Meta-state persistence**: The meta-network's RNN state carries across all learner steps
- **`torch.no_grad()`** on `unroll_meta_net`: Required to avoid OOM (14GB -> 2GB VRAM)

## Architecture

```
Outer network (per-trajectory):
  y_net           MLP [600 -> 16 -> 1]       Value prediction embedding
  z_net           MLP [600 -> 16 -> 1]       Auxiliary prediction embedding
  policy_net      Conv1dNet [9 -> 16 -> 2]   Action-conditional embedding
  trajectory_rnn  LSTM(27, 256)              Reverse-unrolled over trajectory
  state_gate      Linear(128 -> 256)         Multiplicative gate from meta-RNN
  y_head/z_head   Linear(256 -> 600)         Loss targets for y and z
  pi_conv + head  Conv1dNet [258 -> 16] -> 1 Policy loss target (per action)

Meta-RNN (per-lifetime):
  input_mlp       Linear(29 -> 16)           Compress batch-time averages
  lstm            LSTMCell(16, 128)          Slow timescale adaptation
```

The outer network processes each trajectory with a reverse LSTM. The meta-RNN operates at a slower timescale — it sees batch-time averages and modulates the outer network via a multiplicative gate.

## Package structure

```
disco_torch/
  __init__.py          Public API
  types.py             Dataclasses: UpdateRuleInputs, ValueOuts, etc.
  transforms.py        Input transforms and construct_input()
  meta_net.py          DiscoMetaNet — the full LSTM meta-network
  update_rule.py       DiscoUpdateRule — meta-net + value computation + loss
  value_utils.py       V-trace, Retrace Q-values, advantage estimation
  utils.py             batch_lookup, signed_logp1, 2-hot encoding, EMA
  load_weights.py      JAX/Haiku NPZ -> PyTorch weight mapping
  adapter.py           LLM adapter (experimental)
  grpo.py              GRPO integration (experimental)

examples/
  catch_disco.py       Validated Catch benchmark with A2C baseline

tests/
  test_utils.py            Unit tests for utility functions
  test_building_blocks.py  Unit tests for network building blocks
  test_meta_net.py         Snapshot tests for meta-network forward pass
```

## Numerical validation

All outputs match the JAX reference within float32 precision:

| Output | Max diff | Status |
|--------|----------|--------|
| Meta-network outputs (pi, y, z) | < 1.3e-06 | PASS |
| Value pipeline (all 14 fields) | < 6e-04 | PASS |
| End-to-end Catch benchmark | 99% catch rate | PASS |

```bash
pip install disco-torch[dev]
pytest
```

## Requirements

- Python >= 3.11
- PyTorch >= 2.0
- NumPy >= 1.24

## License

Apache 2.0 — same as the original disco_rl.

## Citation

If you use this port, please cite the original paper:

```bibtex
@article{oh2025disco,
  title={Discovering Reinforcement Learning Algorithms},
  author={Oh, Junhyuk and Farquhar, Greg and Kemaev, Iurii and Calian, Dan A. and Hessel, Matteo and Zintgraf, Luisa and Singh, Satinder and van Hasselt, Hado and Silver, David},
  journal={Nature},
  volume={648},
  pages={312--319},
  year={2025},
  doi={10.1038/s41586-025-09761-x}
}
```

## Acknowledgments

This is a community port of [google-deepmind/disco_rl](https://github.com/google-deepmind/disco_rl). All credit for the algorithm, architecture, and pretrained weights goes to the original authors.

# disco-torch

A PyTorch port of DeepMind's **Disco103** — the meta-learned reinforcement learning update rule from [*Discovering State-Of-The-Art Reinforcement Learning Algorithms*](https://doi.org/10.1038/s41586-025-09761-x) (Nature, 2025).

Disco103 is a small neural network (754K params) that **replaces hand-crafted RL loss functions**. Instead of PPO or GRPO, you feed it agent experience and it outputs loss targets. The agent trains by minimizing KL divergence against these targets. It was meta-trained across thousands of environments and generalizes to new tasks as a drop-in update rule.

## Validated

This port achieves **100% catch rate** on the reference Catch benchmark, matching the original JAX implementation:

```
Step   50 | catch=12%      (random)
Step  250 | catch=18%      (meta-RNN warming up)
Step  300 | catch=36%      (transition begins)
Step  350 | catch=95%      (hockey stick)
Step  400 | catch=100%     (converged)
Step 1000 | catch=100%     (stable)
```

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

**[Open in Google Colab](https://colab.research.google.com/github/asystemoffields/disco-torch/blob/main/examples/catch_colab.ipynb)** — train Catch with Disco103 in 3 cells.

```bash
# Or run locally
python examples/catch_disco.py

# With A2C baseline for comparison
python examples/catch_disco.py --both
```

### Using `DiscoTrainer` in your own project

`DiscoTrainer` handles meta-state management, target networks, replay buffer,
optimizer, and loss computation automatically:

```python
from disco_torch import DiscoTrainer, collect_rollout

# Your agent (must output: logits, y, z, q, aux_pi — see Agent Requirements below)
agent = YourAgent(obs_dim=64, num_actions=3).to(device)

# One line setup — downloads weights, creates optimizer, replay buffer, meta-state
trainer = DiscoTrainer(agent, device=device, lr=0.01)

# Wrap your environment
env = YourEnv(num_envs=2)
obs = env.obs()
lstm_state = agent.init_lstm_state(env.num_envs, device)

def step_fn(actions):
    rewards, dones = env.step(actions)
    return env.obs(), rewards, dones

# Training loop
for step in range(1000):
    rollout, obs, lstm_state = collect_rollout(
        agent, step_fn, obs, lstm_state, rollout_len=29, device=device,
    )
    logs = trainer.step(rollout)  # 1 gradient step per acting step

    if (step + 1) % 50 == 0:
        print(f"Step {step+1}: loss={logs['total_loss']:.4f}")
```

### Low-level API

For advanced users who need full control, the underlying `DiscoUpdateRule` is
also available. See [`examples/catch_disco.py`](examples/catch_disco.py) for
the complete validated training loop with all details.

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

## Training loop details

`DiscoTrainer` handles all of these automatically. Listed here for reference
and for users of the low-level API:

- **1 gradient step per acting step**: Effective replay ratio comes from batch_size=64 with num_envs=2 (each trajectory sampled ~32 times over its buffer lifetime)
- **ClippedAdam optimizer**: Adam scaling, then per-element clip to [-1, 1], then learning rate — matching the reference's `optax.chain(scale_by_adam(), clip(1.0), scale(-lr))`
- **Target network**: Polyak averaging with `coeff=0.9` — target slowly tracks current: `target = 0.9 * old_target + 0.1 * current`
- **Loss masking**: Mask first step of new episodes (uninformative), NOT terminal steps (most informative)
- **Meta-state persistence**: The meta-network's RNN state carries across all learner steps
- **`torch.no_grad()`** on `unroll_meta_net`: Required to avoid OOM

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
  trainer.py           DiscoTrainer high-level API + collect_rollout + ClippedAdam

examples/
  catch_disco.py       Validated Catch benchmark with A2C baseline
  catch_colab.ipynb    Google Colab notebook (3 cells, uses DiscoTrainer)
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
  title={Discovering State-Of-The-Art Reinforcement Learning Algorithms},
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

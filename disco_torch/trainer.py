"""High-level training API for Disco103.

Encapsulates the 10 things users get wrong with the low-level API:
meta state management, target network updates, replay buffer, gradient
clipping, loss masking, and the inner gradient loop.

Usage::

    from disco_torch import DiscoTrainer, collect_rollout

    agent = YourAgent(obs_dim=64, num_actions=3).to(device)
    trainer = DiscoTrainer(agent, device=device)

    env = YourEnv(num_envs=2)
    obs = env.obs()
    lstm_state = agent.init_lstm_state(env.num_envs, device)

    def step_fn(actions):
        rewards, dones = env.step(actions)
        return env.obs(), rewards, dones

    for step in range(1000):
        rollout, obs, lstm_state = collect_rollout(
            agent, step_fn, obs, lstm_state, rollout_len=29, device=device,
        )
        logs = trainer.step(rollout)
"""

from __future__ import annotations

from collections import deque
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from disco_torch.load_weights import download_disco103_weights, load_disco103_weights
from disco_torch.types import UpdateRuleInputs
from disco_torch.update_rule import DiscoUpdateRule

DEFAULT_HYPER_PARAMS: dict[str, float] = {
    "value_fn_td_lambda": 0.95,
    "target_params_coeff": 0.9,
    "pi_cost": 1.0,
    "y_cost": 1.0,
    "z_cost": 1.0,
    "aux_policy_cost": 1.0,
    "value_cost": 0.2,
}

_REQUIRED_ROLLOUT_KEYS = ("observations", "actions", "rewards", "discounts", "agent_out")
_REQUIRED_AGENT_OUT_KEYS = ("logits", "y", "z", "q", "aux_pi")


# ---------------------------------------------------------------------------
# Optimizer matching reference: Adam → clip update → scale by lr
# ---------------------------------------------------------------------------

class ClippedAdam:
    """Adam with per-element update clipping.

    Matches the reference optax chain::

        optax.chain(
            scale_by_adam(),              # update = m_hat / (sqrt(v_hat) + eps)
            optax.clip(max_delta=1.0),    # clip update to [-1, 1]
            optax.scale(-lr),             # params -= lr * clipped_update
        )

    Standard approaches clip raw gradients *before* Adam, but the reference
    clips the Adam-scaled update *after*.  When ``v`` is small, Adam amplifies
    gradients significantly, so clipping before vs after produces very
    different effective step sizes.
    """

    def __init__(self, params, lr: float = 0.01, betas=(0.9, 0.999),
                 eps: float = 1e-8, max_delta: float = 1.0):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.max_delta = max_delta
        self.t = 0
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad.data
            self.m[i].mul_(self.beta1).add_(g, alpha=1 - self.beta1)
            self.v[i].mul_(self.beta2).addcmul_(g, g, value=1 - self.beta2)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            # Adam-scaled update, THEN clip, THEN apply lr
            update = m_hat / (v_hat.sqrt() + self.eps)
            update.clamp_(-self.max_delta, self.max_delta)
            p.data.add_(update, alpha=-self.lr)

    def state_dict(self):
        return {"t": self.t, "m": self.m, "v": self.v}

    def load_state_dict(self, state):
        self.t = state["t"]
        self.m = state["m"]
        self.v = state["v"]


# ---------------------------------------------------------------------------
# Replay buffer (internal)
# ---------------------------------------------------------------------------

class _ReplayBuffer:
    """Simple replay buffer storing individual trajectories."""

    def __init__(self, capacity: int):
        self.buffer: deque = deque(maxlen=capacity)

    def add(self, rollout_data: dict) -> None:
        B = rollout_data["observations"].shape[1]
        for i in range(B):
            traj = {
                k: v[:, i : i + 1]
                for k, v in rollout_data.items()
                if isinstance(v, Tensor)
            }
            if "agent_out" in rollout_data:
                traj["agent_out"] = {
                    k: v[:, i : i + 1]
                    for k, v in rollout_data["agent_out"].items()
                }
            self.buffer.append(traj)

    def sample(self, batch_size: int) -> dict | None:
        if len(self.buffer) < batch_size:
            return None
        indices = np.random.randint(0, len(self.buffer), size=batch_size)
        samples = [self.buffer[i] for i in indices]
        result = {}
        for k in [k for k in samples[0] if k != "agent_out"]:
            result[k] = torch.cat([s[k] for s in samples], dim=1)
        if "agent_out" in samples[0]:
            result["agent_out"] = {
                k: torch.cat([s["agent_out"][k] for s in samples], dim=1)
                for k in samples[0]["agent_out"]
            }
        return result

    def __len__(self) -> int:
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Rollout collection (public helper)
# ---------------------------------------------------------------------------

def collect_rollout(
    agent: nn.Module,
    step_fn: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]],
    obs: np.ndarray,
    lstm_state: Any,
    rollout_len: int,
    device: torch.device,
    discount: float = 0.997,
) -> tuple[dict, np.ndarray, Any]:
    """Collect a rollout of environment interaction.

    Args:
        agent: Agent with ``forward_step(obs, lstm_state) -> (out, state)``.
        step_fn: ``step_fn(actions_np) -> (new_obs_np, rewards_np, dones_np)``.
            A thin wrapper around your environment that takes a numpy action
            array and returns the next observation, rewards, and done flags.
        obs: Current observation as numpy array, shape ``[B, ...]``.
        lstm_state: Current recurrent state (use ``agent.init_lstm_state``
            for the first call).
        rollout_len: Number of environment steps to collect (T).
        device: Torch device for tensors.
        discount: Discount factor for non-terminal steps (default 0.997).

    Returns:
        rollout_data: Dict ready for :meth:`DiscoTrainer.step`.
        obs: Final observation (numpy).
        lstm_state: Final recurrent state.
    """
    B = obs.shape[0]
    obs_list, action_list, reward_list, discount_list = [], [], [], []
    agent_out_list = []

    for t in range(rollout_len):
        obs_flat = torch.from_numpy(obs.reshape(B, -1)).float().to(device)
        obs_list.append(obs_flat)

        with torch.no_grad():
            out, lstm_state = agent.forward_step(obs_flat, lstm_state)
        agent_out_list.append({k: v.clone() for k, v in out.items()})

        probs = torch.softmax(out["logits"], dim=-1)
        actions = torch.multinomial(probs, 1).squeeze(-1)
        action_list.append(actions)

        new_obs, rewards, dones = step_fn(actions.cpu().numpy())
        obs = new_obs

        reward_list.append(torch.tensor(rewards, dtype=torch.float32, device=device))
        disc = np.where(dones > 0, 0.0, discount).astype(np.float32)
        discount_list.append(torch.tensor(disc, dtype=torch.float32, device=device))

        # Reset LSTM state for terminated envs
        done_mask = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(-1)
        h, c = lstm_state
        lstm_state = (h * (1.0 - done_mask), c * (1.0 - done_mask))

    # Bootstrap observation (T+1-th)
    obs_flat = torch.from_numpy(obs.reshape(B, -1)).float().to(device)
    obs_list.append(obs_flat)
    with torch.no_grad():
        out, _ = agent.forward_step(obs_flat, lstm_state)
    agent_out_list.append({k: v.clone() for k, v in out.items()})
    action_list.append(torch.zeros(B, dtype=torch.long, device=device))  # dummy

    rollout_data = {
        "observations": torch.stack(obs_list),       # [T+1, B, obs_dim]
        "actions": torch.stack(action_list),          # [T+1, B]
        "rewards": torch.stack(reward_list),          # [T, B]
        "discounts": torch.stack(discount_list),      # [T, B]
        "agent_out": {
            k: torch.stack([o[k] for o in agent_out_list])
            for k in agent_out_list[0]
        },  # each [T+1, B, ...]
    }
    return rollout_data, obs, lstm_state


# ---------------------------------------------------------------------------
# DiscoTrainer
# ---------------------------------------------------------------------------

class DiscoTrainer:
    """High-level training API for the Disco103 update rule.

    Handles everything between "I have an agent and an environment" and
    "my agent is learning":

    - Downloads and loads Disco103 meta-network weights
    - Manages meta-state (RNN state, EMA state, target params)
    - Runs the replay buffer with configurable capacity
    - Executes the inner gradient loop (``replay_ratio`` steps per call)
    - Applies per-element gradient clipping (not norm clipping)
    - Masks loss at terminal timesteps
    - Combines policy loss and value loss correctly
    - Updates target params with Polyak averaging

    Args:
        agent: Agent module. Must have ``forward(obs_seq, should_reset)``
            returning a dict with keys ``logits``, ``y``, ``z``, ``q``,
            ``aux_pi``.
        device: Torch device. Inferred from agent parameters if not given.
        lr: Learning rate for Adam optimizer (default 0.01).
        hyper_params: Override default hyperparameters. Merged with
            :data:`DEFAULT_HYPER_PARAMS`.
        replay_capacity: Replay buffer capacity (default 1024).
        batch_size: Batch size for sampling from replay buffer (default 64).
        replay_ratio: Gradient steps per acting step (default 1).
            The reference uses batch_size=64 with num_envs=2 for an
            effective data reuse ratio of 32, with 1 gradient step.
        weights_path: Path to ``disco_103.npz``. Auto-downloads from
            HuggingFace if not given.
        max_grad_value: Per-element gradient clipping bound (default 1.0).

    Example::

        agent = DiscoMLPAgent(obs_dim=64, num_actions=3).to(device)
        trainer = DiscoTrainer(agent, device=device)

        for step in range(1000):
            rollout, obs, lstm_state = collect_rollout(
                agent, step_fn, obs, lstm_state, 29, device,
            )
            logs = trainer.step(rollout)
            if (step + 1) % 50 == 0:
                print(f"Step {step+1}: loss={logs['total_loss']:.4f}")
    """

    def __init__(
        self,
        agent: nn.Module,
        *,
        device: torch.device | str | None = None,
        lr: float = 0.01,
        hyper_params: dict[str, float] | None = None,
        replay_capacity: int = 1024,
        batch_size: int = 64,
        replay_ratio: int = 1,
        weights_path: str | None = None,
        max_grad_value: float = 1.0,
    ):
        # Validate agent interface
        for method in ("forward", "forward_step", "init_lstm_state"):
            if not callable(getattr(agent, method, None)):
                raise TypeError(
                    f"Agent must have a callable {method}() method. "
                    f"See examples/catch_disco.py for a reference agent."
                )
        if replay_ratio < 1:
            raise ValueError(
                f"replay_ratio must be >= 1, got {replay_ratio}. "
                f"The reference uses 1 gradient step per acting step."
            )

        # Device
        if device is None:
            device = next(agent.parameters()).device
        self.device = torch.device(device) if isinstance(device, str) else device

        self.agent = agent
        self.batch_size = batch_size
        self.replay_ratio = replay_ratio
        self.max_grad_value = max_grad_value

        # Hyperparams (user overrides merged onto defaults)
        self.hyper_params = dict(DEFAULT_HYPER_PARAMS)
        if hyper_params is not None:
            self.hyper_params.update(hyper_params)

        # Load meta-network
        self.rule = DiscoUpdateRule().to(self.device)
        if weights_path is None:
            weights_path = download_disco103_weights()
        load_disco103_weights(self.rule, weights_path)

        # Optimizer (clip update after Adam scaling, matching reference)
        self.optimizer = ClippedAdam(
            agent.parameters(), lr=lr, max_delta=max_grad_value,
        )

        # Replay buffer
        self._buffer = _ReplayBuffer(capacity=replay_capacity)

        # Meta state
        agent_params = dict(agent.named_parameters())
        self._meta_state = self.rule.init_meta_state(agent_params, device=self.device)

        # Unroll function for target network
        self._unroll_fn = self._make_unroll_fn()

        # Counters
        self.grad_steps = 0
        self.acting_steps = 0

    def _make_unroll_fn(self):
        agent = self.agent

        def unroll_fn(params, observations, reset_mask):
            with torch.no_grad():
                out = torch.func.functional_call(
                    agent, params, (observations, reset_mask)
                )
            return out, None

        return unroll_fn

    def step(self, rollout_data: dict) -> dict[str, float]:
        """Process one acting step: add rollout to buffer, run gradient updates.

        Args:
            rollout_data: Output from :func:`collect_rollout`. Dict with keys
                ``observations``, ``actions``, ``rewards``, ``discounts``,
                ``agent_out``.

        Returns:
            Dict of scalar metrics from the last gradient step. Keys include
            ``total_loss``, ``pi_loss``, ``y_loss``, ``z_loss``, ``q_loss``,
            ``aux_loss``, ``grad_steps``, ``acting_steps``, ``buffer_size``.
            If the buffer had too few samples for any gradient steps,
            ``num_grad_steps_this_call`` will be 0.
        """
        self._validate_rollout(rollout_data)

        self._buffer.add(rollout_data)
        self.acting_steps += 1

        logs: dict[str, float] = {}
        steps_this_call = 0

        for _ in range(self.replay_ratio):
            batch = self._buffer.sample(self.batch_size)
            if batch is None:
                break

            logs = self._gradient_step(batch)
            self.grad_steps += 1
            steps_this_call += 1

        logs["grad_steps"] = self.grad_steps
        logs["acting_steps"] = self.acting_steps
        logs["buffer_size"] = len(self._buffer)
        logs["num_grad_steps_this_call"] = steps_this_call
        return logs

    def _gradient_step(self, batch: dict) -> dict[str, float]:
        """Single gradient step on a replay buffer sample."""
        is_terminal = (batch["discounts"] == 0.0).float()
        reset_mask = torch.cat(
            [torch.zeros_like(is_terminal[:1]), is_terminal], dim=0
        )

        # Forward pass with current params (gradient-enabled)
        fresh_out = self.agent(batch["observations"], reset_mask)

        # Build rollout inputs
        rollout = UpdateRuleInputs(
            observations=batch["observations"],
            actions=batch["actions"],
            rewards=batch["rewards"],
            is_terminal=is_terminal,
            agent_out=fresh_out,
            behaviour_agent_out=batch["agent_out"],
        )

        # Meta-network produces targets (no gradient needed)
        agent_params = dict(self.agent.named_parameters())
        with torch.no_grad():
            meta_out, new_meta_state = self.rule.unroll_meta_net(
                rollout,
                agent_params,
                self._meta_state,
                self._unroll_fn,
                self.hyper_params,
            )

        # Compute losses
        policy_loss, p_logs = self.rule.agent_loss(
            rollout, meta_out, self.hyper_params
        )
        value_loss, v_logs = self.rule.agent_loss_no_meta(
            rollout, meta_out, self.hyper_params
        )

        # Mask first step after terminal (NOT the terminal itself).
        # Reference uses discounts[:-1] > 0 for masks but discounts[1:] == 0
        # for is_terminal — they're offset by 1. Terminal steps carry the most
        # informative loss (outcome at catch/miss); first-of-episode steps are
        # uninformative (ball just appeared).
        shift = torch.cat([torch.zeros_like(is_terminal[:1]), is_terminal[:-1]], dim=0)
        masks = 1.0 - shift
        total_loss = ((policy_loss + value_loss) * masks).sum() / (
            masks.sum() + 1e-8
        )

        # Backward + optimizer step (ClippedAdam clips updates internally)
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Update meta state
        self._meta_state = {k: v for k, v in new_meta_state.items()}
        self._meta_state["target_params"] = {
            k: v.detach() for k, v in self._meta_state["target_params"].items()
        }

        return {
            "total_loss": total_loss.item(),
            "pi_loss": p_logs.get("pi_loss", torch.tensor(0.0)).item(),
            "y_loss": p_logs.get("y_loss", torch.tensor(0.0)).item(),
            "z_loss": p_logs.get("z_loss", torch.tensor(0.0)).item(),
            "aux_loss": p_logs.get("aux_loss", torch.tensor(0.0)).item(),
            "q_loss": v_logs.get("q_loss", torch.tensor(0.0)).item(),
        }

    def _validate_rollout(self, data: dict) -> None:
        """Validate rollout data has correct keys and shapes."""
        missing = [k for k in _REQUIRED_ROLLOUT_KEYS if k not in data]
        if missing:
            raise ValueError(
                f"rollout_data missing required keys: {missing}. "
                f"Expected: {list(_REQUIRED_ROLLOUT_KEYS)}"
            )

        T_plus_1 = data["observations"].shape[0]
        T = T_plus_1 - 1
        B = data["observations"].shape[1]

        if data["actions"].shape[0] != T_plus_1:
            raise ValueError(
                f"actions has {data['actions'].shape[0]} timesteps, expected "
                f"{T_plus_1} (T+1 = observations timesteps). The last action "
                f"should be a dummy zero."
            )
        if data["rewards"].shape != (T, B):
            raise ValueError(
                f"rewards shape {tuple(data['rewards'].shape)}, expected "
                f"({T}, {B}). rewards should have one fewer timestep than "
                f"observations."
            )
        if data["discounts"].shape != (T, B):
            raise ValueError(
                f"discounts shape {tuple(data['discounts'].shape)}, expected "
                f"({T}, {B})."
            )

        if isinstance(data["agent_out"], dict):
            missing_outs = [
                k for k in _REQUIRED_AGENT_OUT_KEYS if k not in data["agent_out"]
            ]
            if missing_outs:
                raise ValueError(
                    f"agent_out missing keys: {missing_outs}. Disco103 requires "
                    f"your agent to output: {list(_REQUIRED_AGENT_OUT_KEYS)}. "
                    f"See examples/catch_disco.py for a reference agent."
                )

    @property
    def meta_state(self) -> dict[str, Any]:
        """Current meta state (for checkpointing or inspection)."""
        return self._meta_state

    def state_dict(self) -> dict[str, Any]:
        """Serialize trainer state for checkpointing."""
        return {
            "optimizer": self.optimizer.state_dict(),
            "meta_state": self._meta_state,
            "grad_steps": self.grad_steps,
            "acting_steps": self.acting_steps,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore trainer state from a checkpoint."""
        self.optimizer.load_state_dict(state["optimizer"])
        self._meta_state = state["meta_state"]
        self.grad_steps = state.get("grad_steps", 0)
        self.acting_steps = state.get("acting_steps", 0)

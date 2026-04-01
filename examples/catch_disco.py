"""Port validation: train Catch with Disco103, matching the reference JAX eval notebook.

This is the definitive test of whether the PyTorch port is correct. The reference
eval notebook (colabs/eval.ipynb) trains on Catch with known hyperparameters and
achieves reliable learning. If we match, the port is validated.

Reference config (from google-deepmind/disco_rl):
  - Environment: Catch (8x8 grid, 3 actions)
  - Agent: feedforward MLP(512, 512) + action-conditional LSTM(128) model
  - Replay buffer: capacity=1024, replay_ratio=32
  - Batch size: 64, rollout length: 29, num_envs: 2
  - Learning rate: 0.01
  - target_params_coeff: 0.9
  - Training steps: 1000

Usage:
    python examples/catch_disco.py
    python examples/catch_disco.py --baseline   # A2C baseline for comparison
"""

from __future__ import annotations

import argparse
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from disco_torch import DiscoUpdateRule, UpdateRuleInputs
from disco_torch.load_weights import download_disco103_weights, load_disco103_weights


# ---------------------------------------------------------------------------
# Catch environment
# ---------------------------------------------------------------------------

class CatchEnv:
    """Catch: ball falls down an 8x8 grid, paddle catches it at the bottom.

    Observation: [8, 8] float32 grid (1.0 at ball/paddle positions)
    Actions: 0=left, 1=stay, 2=right
    Reward: +1 catch, -1 miss (at termination only)
    Episode length: 7 steps (ball falls from row 0 to row 7)
    """

    def __init__(self, num_envs: int, rows: int = 8, cols: int = 8):
        self.num_envs = num_envs
        self.rows = rows
        self.cols = cols
        self.ball_row = np.zeros(num_envs, dtype=np.int32)
        self.ball_col = np.zeros(num_envs, dtype=np.int32)
        self.paddle_col = np.zeros(num_envs, dtype=np.int32)
        self.reset()

    def reset(self, mask: np.ndarray | None = None):
        if mask is None:
            mask = np.ones(self.num_envs, dtype=bool)
        n = mask.sum()
        self.ball_row[mask] = 0
        self.ball_col[mask] = np.random.randint(0, self.cols, size=n)
        self.paddle_col[mask] = self.cols // 2

    def step(self, actions: np.ndarray):
        # Move paddle
        moves = actions.astype(np.int32) - 1  # 0→-1, 1→0, 2→+1
        self.paddle_col = np.clip(self.paddle_col + moves, 0, self.cols - 1)

        # Move ball down
        self.ball_row += 1

        # Check termination
        done = self.ball_row >= self.rows - 1
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        rewards[done & (self.ball_col == self.paddle_col)] = 1.0
        rewards[done & (self.ball_col != self.paddle_col)] = -1.0

        # Auto-reset terminated envs
        self.reset(mask=done)

        return rewards, done.astype(np.float32)

    def obs(self) -> np.ndarray:
        grid = np.zeros((self.num_envs, self.rows, self.cols), dtype=np.float32)
        for i in range(self.num_envs):
            grid[i, self.ball_row[i], self.ball_col[i]] = 1.0
            grid[i, self.rows - 1, self.paddle_col[i]] = 1.0
        return grid


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Simple replay buffer storing individual trajectories."""

    def __init__(self, capacity: int):
        self.buffer: deque = deque(maxlen=capacity)

    def add(self, rollout_data: dict):
        """Add a rollout, splitting along batch dimension."""
        B = rollout_data["observations"].shape[1]
        for i in range(B):
            traj = {k: v[:, i:i+1] if v.dim() > 1 else v[:, i:i+1]
                    for k, v in rollout_data.items() if isinstance(v, torch.Tensor)}
            # Handle nested dicts (agent_out)
            if "agent_out" in rollout_data:
                traj["agent_out"] = {
                    k: v[:, i:i+1] for k, v in rollout_data["agent_out"].items()
                }
            self.buffer.append(traj)

    def sample(self, batch_size: int) -> dict | None:
        if len(self.buffer) < batch_size:
            return None
        indices = np.random.randint(0, len(self.buffer), size=batch_size)
        samples = [self.buffer[i] for i in indices]

        # Stack along batch dimension
        result = {}
        keys = [k for k in samples[0] if k != "agent_out"]
        for k in keys:
            result[k] = torch.cat([s[k] for s in samples], dim=1)
        if "agent_out" in samples[0]:
            result["agent_out"] = {
                k: torch.cat([s["agent_out"][k] for s in samples], dim=1)
                for k in samples[0]["agent_out"]
            }
        return result


# ---------------------------------------------------------------------------
# Agent: Feedforward MLP + action-conditional LSTM model (matching reference)
# ---------------------------------------------------------------------------

class DiscoMLPAgent(nn.Module):
    """Feedforward MLP torso + action-conditional LSTM model.

    Matches the reference JAX agent architecture:
    - obs -> MLP(512,512) -> torso_emb
    - torso_emb -> Linear -> logits (feedforward, no LSTM)
    - torso_emb -> Linear -> y (feedforward, no LSTM)
    - torso_emb -> Linear -> cell_init -> (tanh(cell), cell)
    - For each action a: one_hot(a) -> LSTMCell -> h_a -> MLP -> z[a], q[a], aux_pi[a]
    """

    def __init__(self, obs_dim: int, num_actions: int,
                 prediction_size: int = 600, num_bins: int = 601,
                 head_init_std: float = 1e-2):
        super().__init__()
        self.num_actions = num_actions
        self.prediction_size = prediction_size
        self.num_bins = num_bins
        A = num_actions

        # Feedforward torso
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
        )

        # Direct heads from torso (feedforward — no LSTM)
        self.policy_head = nn.Linear(512, A)
        self.y_head = nn.Linear(512, prediction_size)

        # Action-conditional LSTM model
        self.cell_init = nn.Linear(512, 128)
        self.model_lstm = nn.LSTMCell(A, 128)  # input: one-hot action

        # Heads from LSTM hidden state (shared across actions)
        self.z_mlp = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, prediction_size),
        )
        self.q_mlp = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, num_bins),
        )
        self.aux_pi_mlp = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, A),
        )

        # Small init for output layers
        nn.init.normal_(self.policy_head.weight, std=head_init_std)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.normal_(self.y_head.weight, std=head_init_std)
        nn.init.zeros_(self.y_head.bias)
        for head in [self.z_mlp, self.q_mlp, self.aux_pi_mlp]:
            nn.init.normal_(head[-1].weight, std=head_init_std)
            nn.init.zeros_(head[-1].bias)

    def init_lstm_state(self, batch_size: int, device=None):
        """Dummy state for compatibility (agent is feedforward)."""
        return (torch.zeros(batch_size, 128, device=device),
                torch.zeros(batch_size, 128, device=device))

    def forward_step(self, obs: torch.Tensor, lstm_state, should_reset=None):
        """Single step. lstm_state is a dummy (feedforward agent)."""
        B = obs.shape[0]
        A = self.num_actions

        emb = self.backbone(obs)  # [B, 512]
        logits = self.policy_head(emb)  # [B, A]
        y = self.y_head(emb)  # [B, PS]

        # Action-conditional LSTM: init state from torso, run each action
        cell = self.cell_init(emb)  # [B, 128]
        h, c = torch.tanh(cell), cell

        z_list, q_list, aux_list = [], [], []
        for a in range(A):
            one_hot = torch.zeros(B, A, device=obs.device)
            one_hot[:, a] = 1.0
            h, c = self.model_lstm(one_hot, (h, c))
            z_list.append(self.z_mlp(h))
            q_list.append(self.q_mlp(h))
            aux_list.append(self.aux_pi_mlp(h))

        return {
            "logits": logits,
            "y": y,
            "z": torch.stack(z_list, dim=1),       # [B, A, PS]
            "q": torch.stack(q_list, dim=1),        # [B, A, num_bins]
            "aux_pi": torch.stack(aux_list, dim=1),  # [B, A, A]
        }, lstm_state  # pass through dummy state

    def forward(self, obs_seq: torch.Tensor, should_reset: torch.Tensor | None = None):
        """Unroll over time (feedforward — no cross-timestep state)."""
        T, B = obs_seq.shape[:2]
        dummy = self.init_lstm_state(B, obs_seq.device)
        all_outs = []
        for t in range(T):
            out_t, _ = self.forward_step(obs_seq[t], dummy)
            all_outs.append(out_t)
        return {k: torch.stack([o[k] for o in all_outs]) for k in all_outs[0]}


# ---------------------------------------------------------------------------
# Rollout collection
# ---------------------------------------------------------------------------

def collect_rollout(agent, env, obs, lstm_state, T, device):
    """Collect T+1 observations and T transitions (matching reference convention)."""
    B = env.num_envs
    obs_list, action_list, reward_list, discount_list = [], [], [], []
    agent_out_list = []

    for t in range(T):
        obs_flat = torch.from_numpy(obs.reshape(B, -1)).float().to(device)
        obs_list.append(obs_flat)

        with torch.no_grad():
            out, lstm_state = agent.forward_step(obs_flat, lstm_state)
        agent_out_list.append({k: v.clone() for k, v in out.items()})

        probs = torch.softmax(out["logits"], dim=-1)
        actions = torch.multinomial(probs, 1).squeeze(-1)
        action_list.append(actions)

        rewards, dones = env.step(actions.cpu().numpy())
        obs = env.obs()

        reward_list.append(torch.tensor(rewards, dtype=torch.float32, device=device))
        # Discount: 0.997 when alive, 0.0 at terminal (matching reference)
        disc = np.where(dones > 0, 0.0, 0.997).astype(np.float32)
        discount_list.append(torch.tensor(disc, dtype=torch.float32, device=device))

        # Reset LSTM state for terminated envs
        done_mask = torch.tensor(dones, dtype=torch.float32, device=device)
        h, c = lstm_state
        lstm_state = (
            h * (1.0 - done_mask.unsqueeze(-1)),
            c * (1.0 - done_mask.unsqueeze(-1)),
        )

    # Final observation (needed for bootstrap, T+1-th obs)
    obs_flat = torch.from_numpy(obs.reshape(B, -1)).float().to(device)
    obs_list.append(obs_flat)
    with torch.no_grad():
        out, _ = agent.forward_step(obs_flat, lstm_state)
    agent_out_list.append({k: v.clone() for k, v in out.items()})
    action_list.append(torch.zeros(B, dtype=torch.long, device=device))  # dummy

    rollout_data = {
        "observations": torch.stack(obs_list),          # [T+1, B, obs_dim]
        "actions": torch.stack(action_list),             # [T+1, B]
        "rewards": torch.stack(reward_list),             # [T, B]
        "discounts": torch.stack(discount_list),         # [T, B]
        "agent_out": {
            k: torch.stack([o[k] for o in agent_out_list])
            for k in agent_out_list[0]
        },  # each [T+1, B, ...]
    }
    return rollout_data, obs, lstm_state


# ---------------------------------------------------------------------------
# Convert rollout data to UpdateRuleInputs
# ---------------------------------------------------------------------------

def rollout_to_inputs(rollout_data, agent_out_fresh):
    """Convert collected rollout + fresh agent outputs to UpdateRuleInputs.

    Follows the reference convention: rewards[1:] and discounts[1:] give T-1
    transitions from T timesteps.
    """
    T_plus_1 = rollout_data["observations"].shape[0]

    # Reference: reward = rollout.rewards[1:], discount = rollout.discounts[1:]
    # Our convention: rewards [T, B], is_terminal [T, B]
    # Reference rollout has T timesteps of rewards/discounts, we use [1:]
    rewards = rollout_data["rewards"]     # [T, B] where T = T_plus_1 - 1
    discounts = rollout_data["discounts"] # [T, B]

    # is_terminal from discounts
    is_terminal = (discounts == 0.0).float()

    return UpdateRuleInputs(
        observations=rollout_data["observations"],  # [T+1, B, obs_dim]
        actions=rollout_data["actions"],             # [T+1, B]
        rewards=rewards,                             # [T, B]
        is_terminal=is_terminal,                     # [T, B]
        agent_out=agent_out_fresh,                   # [T+1, B, ...] with gradients
        behaviour_agent_out=rollout_data["agent_out"],  # [T+1, B, ...] detached
    )


# ---------------------------------------------------------------------------
# Training: Disco103
# ---------------------------------------------------------------------------

def make_unroll_fn(agent):
    def unroll_fn(params, observations, reset_mask):
        with torch.no_grad():
            out = torch.func.functional_call(agent, params, (observations, reset_mask))
        return out, None
    return unroll_fn


def train_disco(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = CatchEnv(num_envs=args.num_envs, rows=8, cols=8)
    obs_dim = 64  # 8x8 flattened
    num_actions = 3

    agent = DiscoMLPAgent(obs_dim, num_actions).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr)

    rule = DiscoUpdateRule().to(device)
    if args.weights:
        load_disco103_weights(rule, args.weights)
    else:
        weights_path = download_disco103_weights()
        load_disco103_weights(rule, weights_path)
    print("Loaded Disco103 weights")

    agent_params = dict(agent.named_parameters())
    meta_state = rule.init_meta_state(agent_params, device=device)

    # Reference hyperparams from get_settings_disco()
    hyper_params = {
        "value_fn_td_lambda": 0.95,
        "target_params_coeff": 0.9,   # Reference uses 0.9, NOT 0.995!
        "pi_cost": 1.0,
        "y_cost": 1.0,
        "z_cost": 1.0,
        "aux_policy_cost": 1.0,
        "value_cost": 0.2,
    }

    replay_buffer = ReplayBuffer(capacity=args.buffer_capacity)
    unroll_fn = make_unroll_fn(agent)

    completed_returns = deque(maxlen=100)
    all_returns = []
    env_steps = 0

    print(f"\nTraining Catch with Disco103 (port validation)")
    print(f"  Agent params:      {sum(p.numel() for p in agent.parameters()):,}")
    print(f"  Meta-net params:   {sum(p.numel() for p in rule.meta_net.parameters()):,}")
    print(f"  Num envs:          {args.num_envs}")
    print(f"  Rollout length:    {args.rollout_len}")
    print(f"  Batch size:        {args.batch_size}")
    print(f"  Replay ratio:      {args.replay_ratio}")
    print(f"  Buffer capacity:   {args.buffer_capacity}")
    print(f"  Learning rate:     {args.lr}")
    print(f"  target_params_coeff: {hyper_params['target_params_coeff']}")
    print(f"  Training steps:    {args.num_steps}")
    print(f"  Device:            {device}")
    print()

    obs = env.obs()
    lstm_state = agent.init_lstm_state(args.num_envs, device)

    for step in range(args.num_steps):
        # 1. Collect rollout from environment
        rollout_data, obs, lstm_state = collect_rollout(
            agent, env, obs, lstm_state, args.rollout_len, device,
        )

        # Track episode returns from rewards and terminals
        rewards_np = rollout_data["rewards"].cpu().numpy()
        discounts_np = rollout_data["discounts"].cpu().numpy()
        for t in range(rewards_np.shape[0]):
            for b in range(rewards_np.shape[1]):
                if discounts_np[t, b] == 0.0:  # terminal
                    completed_returns.append(float(rewards_np[t, b]))
                    all_returns.append(float(rewards_np[t, b]))

        env_steps += args.rollout_len * args.num_envs

        # 2. Add to replay buffer
        replay_buffer.add(rollout_data)

        # 3. Gradient steps per acting step (reference uses 1)
        for _ in range(args.replay_ratio):
            if len(replay_buffer.buffer) < args.batch_size:
                break
            batch = replay_buffer.sample(args.batch_size)
            if batch is not None:
                # Re-run agent on sampled observations WITH gradients
                is_terminal = (batch["discounts"] == 0.0).float()
                full_reset = torch.cat([torch.zeros_like(is_terminal[:1]), is_terminal], dim=0)

                fresh_out = agent(batch["observations"], full_reset)

                rollout = rollout_to_inputs(batch, fresh_out)

                agent_params = dict(agent.named_parameters())
                with torch.no_grad():
                    meta_out, new_meta_state = rule.unroll_meta_net(
                        rollout, agent_params, meta_state, unroll_fn, hyper_params,
                    )

                # Compute losses
                policy_loss, p_logs = rule.agent_loss(rollout, meta_out, hyper_params)
                value_loss, v_logs = rule.agent_loss_no_meta(rollout, meta_out, hyper_params)

                # Mask out terminal steps
                masks = (1.0 - is_terminal)  # [T, B]
                total_per_step = policy_loss + value_loss  # [T, B]
                total_loss = (total_per_step * masks).sum() / (masks.sum() + 1e-8)

                optimizer.zero_grad()
                total_loss.backward()

                # Per-element gradient clipping (matches optax.clip(max_delta=1.0))
                for p in agent.parameters():
                    if p.grad is not None:
                        p.grad.data.clamp_(-1.0, 1.0)
                optimizer.step()

                # Update meta state after each gradient step
                meta_state = {k: v for k, v in new_meta_state.items()}
                meta_state["target_params"] = {
                    k: v.detach() for k, v in meta_state["target_params"].items()
                }

        if (step + 1) % args.log_every == 0:
            if completed_returns:
                avg_ret = sum(completed_returns) / len(completed_returns)
                catch_rate = sum(1 for r in completed_returns if r > 0) / len(completed_returns)
            else:
                avg_ret = 0.0
                catch_rate = 0.0
            print(
                f"Step {step+1:5d} | "
                f"avg_return={avg_ret:+.3f} | "
                f"catch_rate={catch_rate:.1%} | "
                f"episodes={len(all_returns)} | "
                f"env_steps={env_steps:,}"
            )

    print(f"\nDisco103 training complete.")
    if all_returns:
        last100 = all_returns[-100:]
        catch_rate = sum(1 for r in last100 if r > 0) / len(last100)
        print(f"  Final catch rate (last 100): {catch_rate:.1%}")
        print(f"  Total episodes: {len(all_returns)}")
    return all_returns


# ---------------------------------------------------------------------------
# Training: A2C baseline
# ---------------------------------------------------------------------------

class A2CAgent(nn.Module):
    def __init__(self, obs_dim, num_actions):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
        )
        self.lstm_cell = nn.LSTMCell(512, 128)
        self.policy_head = nn.Linear(128, num_actions)
        self.value_head = nn.Linear(128, 1)

    def init_lstm_state(self, batch_size, device=None):
        return (torch.zeros(batch_size, 128, device=device),
                torch.zeros(batch_size, 128, device=device))

    def forward_step(self, obs, lstm_state, should_reset=None):
        emb = self.backbone(obs)
        h, c = lstm_state
        if should_reset is not None:
            mask = (1.0 - should_reset.float()).unsqueeze(-1)
            h, c = h * mask, c * mask
        h, c = self.lstm_cell(emb, (h, c))
        return self.policy_head(h), self.value_head(h).squeeze(-1), (h, c)


def train_a2c(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = CatchEnv(num_envs=args.num_envs, rows=8, cols=8)
    agent = A2CAgent(64, 3).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr)

    gamma = 0.997
    completed_returns = deque(maxlen=100)
    all_returns = []
    env_steps = 0

    print(f"\nTraining Catch with A2C baseline")
    print(f"  Agent params: {sum(p.numel() for p in agent.parameters()):,}")
    print(f"  LR: {args.lr}, Num envs: {args.num_envs}")
    print()

    obs = env.obs()
    lstm_state = agent.init_lstm_state(args.num_envs, device)
    T = args.rollout_len

    for step in range(args.num_steps):
        obs_list, action_list, reward_list, done_list = [], [], [], []
        logits_list, value_list = [], []

        for t in range(T):
            obs_t = torch.from_numpy(obs.reshape(-1, 64)).float().to(device)
            obs_list.append(obs_t)

            with torch.no_grad():
                logits, val, lstm_state = agent.forward_step(obs_t, lstm_state)
            logits_list.append(logits)
            value_list.append(val)

            probs = torch.softmax(logits, dim=-1)
            actions = torch.multinomial(probs, 1).squeeze(-1)
            action_list.append(actions)

            rewards, dones = env.step(actions.cpu().numpy())
            obs = env.obs()
            reward_list.append(torch.tensor(rewards, device=device))
            done_list.append(torch.tensor(dones, device=device))

            # Track returns
            for b in range(args.num_envs):
                if dones[b] > 0:
                    completed_returns.append(float(rewards[b]))
                    all_returns.append(float(rewards[b]))

            # Reset LSTM for done envs
            dm = torch.tensor(dones, device=device).unsqueeze(-1)
            h, c = lstm_state
            lstm_state = (h * (1 - dm), c * (1 - dm))

        env_steps += T * args.num_envs

        # Bootstrap
        with torch.no_grad():
            obs_t = torch.from_numpy(obs.reshape(-1, 64)).float().to(device)
            _, bootstrap, _ = agent.forward_step(obs_t, lstm_state)

        # Re-run with gradients
        h, c = agent.init_lstm_state(args.num_envs, device)
        fresh_logits, fresh_values = [], []
        for t in range(T):
            reset_t = done_list[t - 1] if t > 0 else None
            emb = agent.backbone(obs_list[t])
            if reset_t is not None:
                mask = (1 - reset_t.float()).unsqueeze(-1)
                h, c = h * mask, c * mask
            h, c = agent.lstm_cell(emb, (h, c))
            fresh_logits.append(agent.policy_head(h))
            fresh_values.append(agent.value_head(h).squeeze(-1))

        logits_t = torch.stack(fresh_logits)
        values_t = torch.stack(fresh_values)
        rewards_t = torch.stack(reward_list)
        dones_t = torch.stack(done_list)
        actions_t = torch.stack(action_list)

        # GAE
        advs = torch.zeros_like(rewards_t)
        gae = torch.zeros(args.num_envs, device=device)
        for t in reversed(range(T)):
            mask = 1.0 - dones_t[t]
            nv = values_t[t + 1].detach() if t < T - 1 else bootstrap
            delta = rewards_t[t] + gamma * nv * mask - values_t[t].detach()
            gae = delta + gamma * 0.95 * mask * gae
            advs[t] = gae

        returns = advs + values_t.detach()

        log_probs = F.log_softmax(logits_t, dim=-1)
        action_lp = log_probs.gather(-1, actions_t.unsqueeze(-1)).squeeze(-1)
        pi_loss = -(action_lp * advs.detach()).mean()
        v_loss = F.mse_loss(values_t, returns)
        entropy = -(F.softmax(logits_t, dim=-1) * log_probs).sum(-1).mean()
        loss = pi_loss + 0.5 * v_loss - 0.01 * entropy

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
        optimizer.step()

        if (step + 1) % args.log_every == 0:
            if completed_returns:
                avg_ret = sum(completed_returns) / len(completed_returns)
                catch_rate = sum(1 for r in completed_returns if r > 0) / len(completed_returns)
            else:
                avg_ret, catch_rate = 0.0, 0.0
            print(f"Step {step+1:5d} | avg_return={avg_ret:+.3f} | catch_rate={catch_rate:.1%} | episodes={len(all_returns)}")

    print(f"\nA2C training complete.")
    if all_returns:
        last100 = all_returns[-100:]
        print(f"  Final catch rate (last 100): {sum(1 for r in last100 if r > 0) / len(last100):.1%}")
    return all_returns


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Catch port validation")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--baseline", action="store_true", help="Run A2C baseline")
    parser.add_argument("--both", action="store_true", help="Run both and compare")
    # Reference config
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--rollout-len", type=int, default=29)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-ratio", type=int, default=1)
    parser.add_argument("--buffer-capacity", type=int, default=1024)
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()

    if args.both:
        print("=" * 60)
        print("  Catch: Disco103 vs A2C (port validation)")
        print("=" * 60)
        disco_returns = train_disco(args)
        a2c_returns = train_a2c(args)

        print("\n" + "=" * 60)
        print("  Comparison")
        print("=" * 60)
        for name, rets in [("Disco103", disco_returns), ("A2C", a2c_returns)]:
            if rets:
                last100 = rets[-100:]
                catch = sum(1 for r in last100 if r > 0) / len(last100)
                print(f"  {name:10s}: episodes={len(rets):4d}  catch_rate={catch:.1%}")
    elif args.baseline:
        train_a2c(args)
    else:
        train_disco(args)


if __name__ == "__main__":
    main()

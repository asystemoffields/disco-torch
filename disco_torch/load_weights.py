"""Weight loading — map JAX/Haiku params from disco_103.npz to PyTorch modules.

NPZ key structure (42 params, 754,778 values total):

    Outer meta-net (prefix: lstm/):
      mlp/~/linear_{0,1}       -> y_net         MLP [600->16->1]
      mlp_1/~/linear_{0,1}     -> z_net         MLP [600->16->1]
      sequential/conv1_d{,_1}  -> policy_net    Conv1dNet [9->16->2]
      lstm/linear              -> trajectory_rnn LSTMCell(27, 256)
      linear                   -> state_gate    Linear(128, 256)
      linear_1                 -> meta_input_head Linear(256, 1)
      linear_2                 -> y_head        Linear(256, 600)
      linear_3                 -> z_head        Linear(256, 600)
      sequential_1/conv1_d     -> pi_conv       Conv1dNet [258->16]
      linear_4                 -> pi_head       Linear(16, 1)

    Meta-RNN (prefix: lstm/~/meta_lstm/~unroll/):
      mlp/~/linear_{0,1}       -> meta_rnn_y_net   MLP [600->16->1]
      mlp_1/~/linear_{0,1}     -> meta_rnn_z_net   MLP [600->16->1]
      sequential/conv1_d{,_1}  -> meta_rnn_policy_net Conv1dNet [9->16->2]
      mlp_2/~/linear_0         -> meta_rnn_input_mlp  Linear(29, 16)
      lstm/linear              -> meta_rnn_cell      LSTMCell(16, 128)
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


def inspect_weights(npz_path: str | Path) -> dict[str, tuple]:
    """Print all parameter names and shapes from the NPZ."""
    data = np.load(npz_path, allow_pickle=True)
    params = {}
    for key in sorted(data.files):
        arr = data[key]
        params[key] = arr.shape
        print(f"  {key:80s} {str(arr.shape):>20s}  dtype={arr.dtype}")
    return params


def _t(arr: np.ndarray) -> Tensor:
    """Convert numpy array to float32 torch tensor."""
    return torch.from_numpy(arr.copy()).float()


def _load_linear(module: torch.nn.Linear, w: np.ndarray, b: np.ndarray | None):
    """Haiku Linear: w=[in, out], PyTorch: weight=[out, in]."""
    module.weight.data.copy_(_t(w).T)
    if b is not None and module.bias is not None:
        module.bias.data.copy_(_t(b))
    elif module.bias is not None:
        module.bias.data.zero_()


def _load_haiku_lstm(cell, w: np.ndarray, b: np.ndarray):
    """Load Haiku LSTM combined linear into our HaikuLSTMCell."""
    _load_linear(cell.linear, w, b)


def _load_lstm_combined(cell: torch.nn.LSTMCell, w: np.ndarray, b: np.ndarray):
    """Haiku LSTM combined linear: w=[input+hidden, 4*hidden], b=[4*hidden]."""
    w_t = _t(w)
    input_size = cell.weight_ih.shape[1]
    cell.weight_ih.data.copy_(w_t[:input_size].T)
    cell.weight_hh.data.copy_(w_t[input_size:].T)
    cell.bias_ih.data.copy_(_t(b))
    cell.bias_hh.data.zero_()


def _load_conv1d(conv: torch.nn.Conv1d, w: np.ndarray, b: np.ndarray):
    """Haiku Conv1D: w=[K, Cin, Cout], PyTorch: weight=[Cout, Cin, K]."""
    conv.weight.data.copy_(_t(w).permute(2, 1, 0))
    conv.bias.data.copy_(_t(b))


def _load_batch_mlp(mlp: torch.nn.Module, params: dict, prefix: str):
    """Load a BatchMLP whose Sequential is [Linear, ReLU, Linear, ReLU, ...]."""
    linears = [m for m in mlp.net if isinstance(m, torch.nn.Linear)]
    for i, lin in enumerate(linears):
        w = params[f"{prefix}/~/linear_{i}/w"]
        b = params[f"{prefix}/~/linear_{i}/b"]
        _load_linear(lin, w, b)


def _load_conv1d_net(net: torch.nn.Module, params: dict, prefix: str):
    """Load a Conv1dNet (ModuleList of Conv1dBlocks)."""
    for i, block in enumerate(net.blocks):
        suffix = f"/conv1_d_{i}" if i > 0 else "/conv1_d"
        w = params[f"{prefix}{suffix}/w"]
        b = params[f"{prefix}{suffix}/b"]
        _load_conv1d(block.conv, w, b)


def download_disco103_weights(
    repo_id: str = "asystemoffields/disco-103",
    filename: str = "disco_103.npz",
    cache_dir: str | Path | None = None,
) -> str:
    """Download Disco103 weights from HuggingFace Hub.

    Returns the local path to the downloaded file.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface-hub is required for downloading weights. "
            "Install it with: pip install disco-torch[hub]"
        )
    return hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir)


def load_disco103_weights(
    rule,
    npz_path: str | Path | None = None,
    *,
    repo_id: str = "asystemoffields/disco-103",
) -> None:
    """Load Disco103 weights from NPZ into a DiscoUpdateRule.

    Args:
        rule: a DiscoUpdateRule instance
        npz_path: path to disco_103.npz. If None, downloads from HuggingFace Hub.
        repo_id: HuggingFace Hub repo to download from (used when npz_path is None)
    """
    if npz_path is None:
        npz_path = download_disco103_weights(repo_id=repo_id)
    data = np.load(npz_path, allow_pickle=True)
    p = {k: data[k] for k in data.files}
    net = rule.meta_net

    # --- Outer meta-net ---
    _load_batch_mlp(net.y_net, p, "lstm/mlp")
    _load_batch_mlp(net.z_net, p, "lstm/mlp_1")
    _load_conv1d_net(net.policy_net, p, "lstm/sequential")

    # Trajectory LSTM (combined linear: [283, 1024] = [27+256, 4*256])
    _load_haiku_lstm(
        net.trajectory_rnn.cell,
        p["lstm/lstm/linear/w"], p["lstm/lstm/linear/b"],
    )

    # State gate: Linear(128, 256)
    _load_linear(net.state_gate, p["lstm/linear/w"], p["lstm/linear/b"])

    # Output heads
    _load_linear(net.meta_input_head, p["lstm/linear_1/w"], p["lstm/linear_1/b"])
    _load_linear(net.y_head, p["lstm/linear_2/w"], p["lstm/linear_2/b"])
    _load_linear(net.z_head, p["lstm/linear_3/w"], p["lstm/linear_3/b"])

    # Policy target
    _load_conv1d_net(net.pi_conv, p, "lstm/sequential_1")
    _load_linear(net.pi_head, p["lstm/linear_4/w"], p["lstm/linear_4/b"])

    # --- Meta-RNN ---
    mr = "lstm/~/meta_lstm/~unroll"
    _load_batch_mlp(net.meta_rnn_y_net, p, f"{mr}/mlp")
    _load_batch_mlp(net.meta_rnn_z_net, p, f"{mr}/mlp_1")
    _load_conv1d_net(net.meta_rnn_policy_net, p, f"{mr}/sequential")

    # Meta-RNN input MLP: just one linear layer inside nn.Sequential
    input_mlp_linear = [m for m in net.meta_rnn_input_mlp if isinstance(m, torch.nn.Linear)][0]
    _load_linear(input_mlp_linear, p[f"{mr}/mlp_2/~/linear_0/w"], p[f"{mr}/mlp_2/~/linear_0/b"])

    # Meta-RNN LSTM cell
    _load_haiku_lstm(
        net.meta_rnn_cell,
        p[f"{mr}/lstm/linear/w"], p[f"{mr}/lstm/linear/b"],
    )

    # Verify all 42 params were used
    expected = set(p.keys())
    used = set()
    used.update(f"lstm/mlp/~/linear_{i}/{t}" for i in range(2) for t in ("w", "b"))
    used.update(f"lstm/mlp_1/~/linear_{i}/{t}" for i in range(2) for t in ("w", "b"))
    used.update(f"lstm/sequential/conv1_d{'' if i == 0 else f'_{i}'}/{t}"
                for i in range(2) for t in ("w", "b"))
    used.update(f"lstm/lstm/linear/{t}" for t in ("w", "b"))
    used.update(f"lstm/linear/{t}" for t in ("w", "b"))
    used.update(f"lstm/linear_{i}/{t}" for i in range(1, 5) for t in ("w", "b"))
    used.update(f"lstm/sequential_1/conv1_d/{t}" for t in ("w", "b"))
    used.update(f"{mr}/mlp/~/linear_{i}/{t}" for i in range(2) for t in ("w", "b"))
    used.update(f"{mr}/mlp_1/~/linear_{i}/{t}" for i in range(2) for t in ("w", "b"))
    used.update(f"{mr}/sequential/conv1_d{'' if i == 0 else f'_{i}'}/{t}"
                for i in range(2) for t in ("w", "b"))
    used.update(f"{mr}/mlp_2/~/linear_0/{t}" for t in ("w", "b"))
    used.update(f"{mr}/lstm/linear/{t}" for t in ("w", "b"))

    missing = expected - used
    if missing:
        print(f"WARNING: {len(missing)} NPZ params not loaded: {missing}")
    else:
        print(f"Successfully loaded all {len(expected)} parameters ({sum(v.size for v in p.values()):,d} values)")

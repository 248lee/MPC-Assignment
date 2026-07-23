"""
One-time PyTorch -> JAX SAC checkpoint converter
================================================

Reads the trained ``sac_lqr.pt`` (a PyTorch checkpoint) and writes a torch-free
JAX checkpoint (``sac_lqr_jax.pkl``) that sac_jax.load_ckpt / SACPolicy can load.
After this runs once, the JAX runtime never needs torch and no retraining is
required.

Run with the ``[convert]`` extra installed::

    uv run --extra convert python convert_sac_ckpt.py

Weight mapping: a PyTorch ``nn.Linear.weight`` is (out, in); the jnp forward
computes ``h @ w + b`` with ``w`` of shape (in, out), so **every Linear weight is
transposed** and each bias copied as-is. ``action_scale``/``action_bias`` are the
policy's registered buffers.
"""

from __future__ import annotations

import re
import sys

import numpy as np

from sac_jax import save_ckpt


def _linear_indices(state_dict, prefix):
    """Sorted layer indices i for keys like ``{prefix}.{i}.weight``."""
    idxs = set()
    pat = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.weight$")
    for k in state_dict:
        m = pat.match(k)
        if m:
            idxs.add(int(m.group(1)))
    return sorted(idxs)


def _lin(state_dict, key):
    """{'w': weight.T, 'b': bias} as float64 numpy (transpose the torch (out,in) weight)."""
    w = state_dict[f"{key}.weight"].detach().cpu().numpy().astype(np.float64)
    b = state_dict[f"{key}.bias"].detach().cpu().numpy().astype(np.float64)
    return {"w": w.T.copy(), "b": b.copy()}


def convert_actor(actor_sd):
    """torch GaussianPolicy state_dict -> jnp policy pytree + scale/bias."""
    trunk = [_lin(actor_sd, f"trunk.{i}") for i in _linear_indices(actor_sd, "trunk")]
    pp = {
        "trunk": trunk,
        "mu": _lin(actor_sd, "mu_head"),
        "log_std": _lin(actor_sd, "log_std_head"),
    }
    scale = float(actor_sd["action_scale"].detach().cpu().numpy())
    bias = float(actor_sd["action_bias"].detach().cpu().numpy())
    return pp, scale, bias


def convert_qnet(q_sd):
    """torch QNetwork state_dict -> list of jnp {'w','b'} layers."""
    return [_lin(q_sd, f"net.{i}") for i in _linear_indices(q_sd, "net")]


def main(src="sac_lqr.pt", dst="sac_lqr_jax.pkl"):
    import torch

    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    actor_pp, scale, bias = convert_actor(ckpt["actor"])
    cfg_in = ckpt.get("config", {})
    config = {
        "hidden": tuple(cfg_in.get("hidden", (256, 256))),
        "action_low": float(cfg_in.get("action_low", -10.0)),
        "action_high": float(cfg_in.get("action_high", 10.0)),
        "action_scale": scale,
        "action_bias": bias,
    }
    q1 = convert_qnet(ckpt["q1"]) if "q1" in ckpt else None
    q2 = convert_qnet(ckpt["q2"]) if "q2" in ckpt else None

    save_ckpt(dst, actor_pp, config, q1=q1, q2=q2,
              meta={"converted_from": src, "orig_meta": ckpt.get("meta", {})})
    print(f"Converted {src} -> {dst}")
    print(f"  trunk layers: {[l['w'].shape for l in actor_pp['trunk']]}")
    print(f"  mu head: {actor_pp['mu']['w'].shape}  log_std head: {actor_pp['log_std']['w'].shape}")
    print(f"  action_scale={scale}  action_bias={bias}  q1={'yes' if q1 else 'no'} q2={'yes' if q2 else 'no'}")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(*(args[:2]))

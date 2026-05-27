#!/usr/bin/env python3
"""
Test wrapper for ManualThetaNeuron.

Verifies the hand-derived gradient against:
  1. central finite differences  (always, JAX-free)
  2. JAX autodiff on the same forward model  (optional, if jax importable)
and then runs a short hand-written SGD loop to hit a target output spike time.
"""

import os
import sys
from pathlib import Path

VENV_PYTHON = Path(__file__).resolve().parent / ".venv" / "bin" / "python"
VENV_ROOT = VENV_PYTHON.parent.parent
if VENV_PYTHON.exists() and Path(sys.prefix).resolve() != VENV_ROOT.resolve():
    os.execv(VENV_PYTHON, [str(VENV_PYTHON), *sys.argv])

import numpy as np

from manual_theta import ManualThetaNeuron


def finite_diff_grad(neuron, phi0, times, weights, t_target, eps=1e-6):
    times = np.asarray(times, float)
    weights = np.asarray(weights, float)
    N = len(weights)
    dL_dw = np.zeros(N)
    dL_dt = np.zeros(N)
    for i in range(N):
        wp, wm = weights.copy(), weights.copy()
        wp[i] += eps
        wm[i] -= eps
        Lp = neuron.run(phi0, times, wp, t_target)["loss"]
        Lm = neuron.run(phi0, times, wm, t_target)["loss"]
        dL_dw[i] = (Lp - Lm) / (2 * eps)

        tp, tm = times.copy(), times.copy()
        tp[i] += eps
        tm[i] -= eps
        Lp = neuron.run(phi0, tp, weights, t_target)["loss"]
        Lm = neuron.run(phi0, tm, weights, t_target)["loss"]
        dL_dt[i] = (Lp - Lm) / (2 * eps)
    return dL_dw, dL_dt


def jax_grad(tau, I0, phi0, times, weights, t_target):
    """Autodiff on an identical forward model. Returns None if jax unavailable."""
    try:
        import jax
        import jax.numpy as jnp
    except Exception:
        return None

    times_j = jnp.asarray(times, float)

    def loss(weights, times):
        root = jnp.sqrt(I0 - 0.25)
        Theta = tau * jnp.pi / root
        psi = phi0
        t_prev = 0.0
        for k in range(len(weights)):
            phi_pre = psi + (times[k] - t_prev)
            V = root * jnp.tan(phi_pre * root / tau - jnp.pi / 2) + 0.5 + weights[k]
            psi = tau / root * (jnp.arctan((V - 0.5) / root) + jnp.pi / 2)
            t_prev = times[k]
        t_out = times[-1] + (Theta - psi)
        return (t_out - t_target) ** 2

    g = jax.grad(loss, argnums=(0, 1))(jnp.asarray(weights, float), times_j)
    return np.asarray(g[0]), np.asarray(g[1])


def fmt(arr):
    return "[" + ", ".join(f"{v:+.6f}" for v in arr) + "]"


def main():
    tau, I0 = 1.0, 1.25
    neuron = ManualThetaNeuron(tau=tau, I0=I0)
    Theta = neuron.Theta()

    phi0 = 0.2 * Theta
    times = np.array([0.5, 1.0])
    weights = np.array([0.0, 0.0])
    t_target = 2.0

    sep = "-" * 64
    print(sep)
    print("ManualThetaNeuron")
    print(f"  tau={tau}  I0={I0}  Theta={Theta:.6f}")
    print(f"  phi0={phi0:.6f} ({0.2:.0%} of Theta)")
    print(f"  input times   = {fmt(times)}")
    print(f"  input weights = {fmt(weights)}")
    print(f"  t_target      = {t_target}")
    print(sep)

    res = neuron.run(phi0, times, weights, t_target)
    print(f"forward:  t_out = {res['t_out']:.6f}   loss = {res['loss']:.6f}")
    print()

    print("Gradient check  (manual vs finite-diff vs jax-autodiff)")
    fd_w, fd_t = finite_diff_grad(neuron, phi0, times, weights, t_target)
    jx = jax_grad(tau, I0, phi0, times, weights, t_target)

    print(f"  dL/dw  manual     = {fmt(res['dL_dw'])}")
    print(f"  dL/dw  finite-diff= {fmt(fd_w)}")
    if jx is not None:
        print(f"  dL/dw  jax        = {fmt(jx[0])}")
    print(f"  dL/dt  manual     = {fmt(res['dL_dt'])}")
    print(f"  dL/dt  finite-diff= {fmt(fd_t)}")
    if jx is not None:
        print(f"  dL/dt  jax        = {fmt(jx[1])}")

    err = max(
        np.max(np.abs(res["dL_dw"] - fd_w)),
        np.max(np.abs(res["dL_dt"] - fd_t)),
    )
    print(f"  max |manual - finite-diff| = {err:.2e}", "  OK" if err < 1e-4 else "  MISMATCH")
    if jx is not None:
        err_j = max(
            np.max(np.abs(res["dL_dw"] - jx[0])),
            np.max(np.abs(res["dL_dt"] - jx[1])),
        )
        print(f"  max |manual - jax|         = {err_j:.2e}", "  OK" if err_j < 1e-5 else "  MISMATCH")
    print()

    print("Training (hand-written SGD on weights, manual gradient)")
    w = weights.copy()
    lr = 0.05
    print(f"  {'step':>4}  {'loss':>10}  {'t_out':>10}  weights")
    for step in range(41):
        out = neuron.run(phi0, times, w, t_target)
        if step % 5 == 0:
            print(f"  {step:>4}  {out['loss']:>10.6f}  {out['t_out']:>10.6f}  {fmt(w)}")
        w = w - lr * out["dL_dw"]
    final = neuron.run(phi0, times, w, t_target)
    print(sep)
    print(f"final: t_out = {final['t_out']:.6f}  (target {t_target})  loss = {final['loss']:.2e}")
    print(sep)


if __name__ == "__main__":
    main()

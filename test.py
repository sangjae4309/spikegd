#!/usr/bin/env python3

import os
import sys
from pathlib import Path
import jax.numpy as jnp
import matplotlib.pyplot as plt

from spikegd.theta import ThetaNeuron


def event_kind(spike_in: bool, neuron_index: int) -> str:
    if neuron_index < 0:
        return "end"
    if spike_in:
        return "input"
    return "output"


def main() -> None:
    tau         = 1.0
    I0          = 1.25
    eps         = 1e-6
    T           = 8.0
    K           = 20
    dt          = 0.01
    phi0_frac   = 0.2
    # 4ch
    #  ┌─────────────────┐
    #  │ ch[0]  w=+0.8  │──┐
    #  │ ch[1]  w=-0.4  │──┤
    #  │ ch[2]  w=+1.0  │──┼──▶  [ ThetaNeuron ]  ──▶ output spikes
    #  │ ch[3]  w=+0.5  │──┘
    #  └─────────────────┘
    input_times   = [1.0, 2.2, 4.0, 5.6]
    input_weights = [0.8, -0.4, 1.0, 0.5]
    plot_path   = Path("theta_test.png")
    no_plot     = False

    neuron = ThetaNeuron(tau=tau, I0=I0, eps=eps)
    theta = float(neuron.Theta())
    phi0 = phi0_frac * theta

    x0 = jnp.array([[phi0]])
    weights_net = jnp.zeros((1, 1))
    weights_in = jnp.array([input_weights])
    spikes_in = (
        jnp.array(input_times),
        jnp.arange(len(input_times), dtype=int),
    )
    config = {"T": T, "K": K, "dt": dt}

    sep = "─" * 60
    out = neuron.event(x0, weights_net, weights_in, spikes_in, config)
    times, spike_ins, indices, xs = out

    ## ------------------------------------------------------------
    print("Input spikes")
    print("  #   time      weight")
    for i, (t_in, w_in) in enumerate(zip(input_times, input_weights)):
        print(f"  {i:<2}  {t_in:>7.3f}   {w_in:>7.3f}")
    print()
    ## ------------------------------------------------------------
    print("Events")
    print("  #   time      spike_in   kind     source   phi_after   phi/Theta")
    for k, (time, spike_in, index, x) in enumerate(zip(times, spike_ins, indices, xs)):
        time_f = float(time)
        spike_in_b = bool(spike_in)
        index_i = int(index)
        kind = event_kind(spike_in_b, index_i)
        phi_after = float(x[0, 0])
        source = "-" if index_i < 0 else str(index_i)
        print(
            f"  {k:<2}  {time_f:>7.3f}   {str(spike_in_b):<5}      {kind:<6}   {source:>6}   "
            f"{phi_after:>9.6f}   {phi_after / theta:>8.3f}"
        )
        if kind == "end":
            break
    ## ------------------------------------------------------------

    output_times = [
        float(time)
        for time, spike_in, index in zip(times, spike_ins, indices)
        if (not bool(spike_in)) and int(index) == 0
    ]
    print()
    print("Output spike times:", output_times if output_times else "none")

    if not no_plot:
        trace_times, trace_xs = neuron.traces(x0, out, config)
        trace_phi = trace_xs[:, 0, 0]

        fig, ax = plt.subplots(figsize=(9, 3.5), constrained_layout=True)
        ax.plot(trace_times, trace_phi, color="C0", label="phase phi")
        ax.axhline(theta, color="C3", linestyle="--", label="Theta threshold")
        for t_in, w_in in zip(input_times, input_weights):
            color = "C2" if w_in >= 0 else "C1"
            ax.axvline(t_in, color=color, alpha=0.55, linewidth=1.2)
        for t_out in output_times:
            ax.axvline(t_out, color="C3", alpha=0.85, linewidth=1.8)
        ax.set_xlim(0, T)
        ax.set_xlabel("time")
        ax.set_ylabel("phase phi")
        ax.set_title("One ThetaNeuron: input spikes move phase, Theta crossing emits output")
        ax.legend(loc="upper right")
        fig.savefig(plot_path, dpi=160)
        print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()

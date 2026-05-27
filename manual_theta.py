"""
Theta (QIF) neuron with HAND-DERIVED gradients (no JAX autodiff).

Same dynamics as `spikegd.theta.ThetaNeuron`:

    Phi(V)   = tau/root * (arctan((V - 1/2)/root) + pi/2)
    iPhi(phi)= root * tan(phi*root/tau - pi/2) + 1/2
    H(phi,w) = Phi(iPhi(phi) + w)          # phase jump when an input spike arrives
    Theta    = tau*pi/root,   root = sqrt(I0 - 1/4)

Scenario (single neuron, single output spike):
    - phase starts at phi0 at t=0 and drifts at rate 1 (flow(phi, dt) = phi + dt)
    - N input spikes arrive at sorted times t_1 < ... < t_N with weights w_1..w_N
    - each input spike applies phi -> H(phi, w)
    - after the last input the phase drifts until it reaches Theta -> output spike
        t_out = t_N + (Theta - psi_N)
    - loss L = (t_out - t_target)^2

Gradients are propagated FORWARD by hand (forward-mode sensitivity). For each
input spike k we evaluate the local sensitivities of the phase transfer H:

    a_k = d psi_k / d phi_k^pre = Phi'(V_post) / Phi'(V_pre)
    b_k = d psi_k / d w_k       = Phi'(V_post)

with V_pre = iPhi(phi_k^pre), V_post = V_pre + w_k, and the elementary derivative

    Phi'(V) = tau / (I0 - 1/4 + (V - 1/2)^2),   iPhi'(phi) = 1 / Phi'(iPhi(phi)).

Carrying g_w[i] = d psi/d w_i and g_t[i] = d psi/d t_i through the chain gives
dL/dw and dL/dt without any automatic differentiation.
"""

import numpy as np


class ManualThetaNeuron:
    def __init__(self, tau: float = 1.0, I0: float = 1.25):
        assert tau > 0, "tau must be positive."
        assert I0 > 0.25, "I0 must be greater than 1/4."
        self.tau = float(tau)
        self.I0 = float(I0)
        self.root = float(np.sqrt(I0 - 0.25))

    # --- dynamics (identical to spikegd ThetaNeuron) -----------------------
    def Theta(self) -> float:
        return self.tau * np.pi / self.root

    def Phi(self, V):
        return self.tau / self.root * (np.arctan((V - 0.5) / self.root) + np.pi / 2)

    def iPhi(self, phi):
        return self.root * np.tan(phi * self.root / self.tau - np.pi / 2) + 0.5

    def H(self, phi, w):
        return self.Phi(self.iPhi(phi) + w)

    # --- elementary derivative used by the hand-written gradient -----------
    def Phi_prime(self, V):
        """dPhi/dV = tau / (I0 - 1/4 + (V - 1/2)^2)."""
        return self.tau / (self.root**2 + (V - 0.5) ** 2)

    # --- forward pass + manual gradient ------------------------------------
    def run(self, phi0, times, weights, t_target):
        """
        Forward simulate one output spike and return loss + hand-derived grads.

        Returns a dict with keys: t_out, loss, dL_dw, dL_dt.
        """
        times = np.asarray(times, dtype=float)
        weights = np.asarray(weights, dtype=float)
        N = len(weights)
        assert len(times) == N, "times and weights must have equal length."
        assert np.all(np.diff(times) > 0), "input times must be strictly sorted."
        Theta = self.Theta()

        psi = float(phi0)
        t_prev = 0.0
        g_w = np.zeros(N)  # d psi / d w_i
        g_t = np.zeros(N)  # d psi / d t_i

        for k in range(N):
            phi_pre = psi + (times[k] - t_prev)  # drift up to spike k
            if not (0.0 < phi_pre < Theta):
                raise ValueError(
                    f"phase {phi_pre:.4f} left (0, Theta={Theta:.4f}) at input {k}: "
                    "neuron fired before the last input or phase invalid. "
                    "Pick smaller weights / different times."
                )

            V_pre = self.iPhi(phi_pre)
            V_post = V_pre + weights[k]
            psi_new = self.Phi(V_post)

            # local sensitivities of the phase transfer H
            a = self.Phi_prime(V_post) / self.Phi_prime(V_pre)  # d psi_new / d phi_pre
            b = self.Phi_prime(V_post)  # d psi_new / d w_k

            # phi_pre = psi_{k-1} + (t_k - t_{k-1})
            #   d phi_pre / d w_i = g_w[i]
            #   d phi_pre / d t_i = g_t[i] + [i==k] - [i==k-1]
            new_g_w = a * g_w
            new_g_w[k] += b

            dphi_dt = g_t.copy()
            dphi_dt[k] += 1.0
            if k - 1 >= 0:
                dphi_dt[k - 1] -= 1.0
            new_g_t = a * dphi_dt

            psi, g_w, g_t = psi_new, new_g_w, new_g_t
            t_prev = times[k]

        # output spike: drift from psi_N to Theta
        t_out = times[-1] + (Theta - psi)

        # t_out = t_N + Theta - psi_N
        #   d t_out / d w_i = -g_w[i]
        #   d t_out / d t_i = [i==N-1] - g_t[i]
        dtout_dw = -g_w
        dtout_dt = -g_t.copy()
        dtout_dt[N - 1] += 1.0

        loss = (t_out - t_target) ** 2
        dL_dtout = 2.0 * (t_out - t_target)
        return {
            "t_out": float(t_out),
            "loss": float(loss),
            "dL_dw": dL_dtout * dtout_dw,
            "dL_dt": dL_dtout * dtout_dt,
        }

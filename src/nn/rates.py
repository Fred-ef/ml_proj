"""Learning rate schedules"""

from __future__ import annotations


class LinearDecay:
    """Linear decay from 'eta_0' to 'eta_tau' over 'tau' epochs, then constant"""

    def __init__(self, eta_0: float, tau: int, eta_tau: float | None = None) -> None:
        self.eta_0 = eta_0
        self.tau = tau
        self.eta_tau = eta_tau if eta_tau is not None else 0.01 * eta_0
        self.steps_per_epoch = 1

    def __call__(self, step: int) -> float:
        gamma = min(step / (self.tau * self.steps_per_epoch), 1.0)
        return (1.0 - gamma) * self.eta_0 + gamma * self.eta_tau

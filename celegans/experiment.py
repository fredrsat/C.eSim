"""Paired before/after measurement of what an ablation actually does.

Reading a lesion off a running worm does not work.  Its own behaviour varies as
much between two moments as the lesion does -- about a fifth of all neurons
differ by more than 1.5 Hz with nothing done at all -- so a snapshot comparison
mostly measures the animal having moved on.

This module runs the comparison as a controlled experiment instead.  Two copies
of the current network are advanced side by side from the *same* state, with the
*same* frozen sensory input and the *same* random seed, differing only in whether
the cells are ablated.  Every difference that comes out is therefore caused by
the lesion, and the noise floor drops to essentially zero.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Effect:
    """What one ablation did."""

    ablated: list[str]
    #: Per-neuron rate change, lesion minus control, in Hz.
    delta: np.ndarray
    #: (name, change) for the most affected surviving cells.
    fallers: list[tuple[str, float]] = field(default_factory=list)
    risers: list[tuple[str, float]] = field(default_factory=list)
    #: Command interneuron rates, as (control, lesion).
    backward: tuple[float, float] = (0.0, 0.0)
    forward: tuple[float, float] = (0.0, 0.0)
    balance: tuple[float, float] = (0.0, 0.0)
    muscle: tuple[float, float] = (0.0, 0.0)
    head_imbalance: tuple[float, float] = (0.0, 0.0)
    n_changed: int = 0
    n_down: int = 0
    n_up: int = 0
    reversal_threshold: float = 0.085

    def reversal_verdict(self) -> str:
        """What the lesion did to the reversal trigger.

        Stated as a change rather than as an absolute claim: the balance is
        measured under whatever sensory conditions were frozen in, so a worm
        that simply is not being touched sits below threshold either way, and
        saying "cannot reverse" there would be wrong.
        """
        before = self.balance[0] > self.reversal_threshold
        after = self.balance[1] > self.reversal_threshold
        if before and not after:
            return "removes the reversal trigger"
        if after and not before:
            return "creates a standing reversal trigger"
        if before and after:
            return "reversal trigger still met"
        return "below the reversal trigger before and after"

    def headline(self) -> str:
        if not self.ablated:
            return "nothing is ablated — nothing to compare"
        changed = f"{self.n_changed} neurons changed ({self.n_down} down, {self.n_up} up)"
        before, after = self.balance
        return f"{changed}   ·   command balance {before:+.3f} -> {after:+.3f}"


def _run(net, frozen, settle: float, measure_for: float, dt: float = 1.0) -> np.ndarray:
    """Settle, then return the true mean firing rate in Hz over the window.

    Counting spikes over the whole window rather than reading ``net.rate`` at
    the end matters: that readout has a 60 ms time constant, so sampling it once
    carries about +/-8 Hz of noise and swamps the effect being measured.
    """
    for _ in range(int(settle * 1000.0 / dt)):
        net.step(dt, frozen)
    counts = np.zeros(net.conn.n_neurons, dtype=np.float64)
    steps = int(measure_for * 1000.0 / dt)
    for _ in range(steps):
        net.step(dt, frozen)
        counts += net.spiked
    return counts / (steps * dt / 1000.0)


def measure(
    sim, settle: float = 3.0, window: float = 6.0, top: int = 8, seed: int = 20240517
) -> Effect:
    """Run the paired experiment for whatever is currently ablated in ``sim``."""
    conn = sim.conn
    ablated_mask = sim.net.ablated.copy()
    names = [conn.neurons[i] for i in np.nonzero(ablated_mask)[0]]
    if not names:
        return Effect(ablated=[], delta=np.zeros(conn.n_neurons, dtype=np.float32))

    frozen = sim.sensory_current.copy()

    def arm(with_lesion: bool):
        net = copy.deepcopy(sim.net)
        # Same noise realisation in both arms, so the only difference left is
        # the lesion itself rather than which random numbers each run drew.
        net.rng = np.random.default_rng(seed)
        net.ablated[:] = ablated_mask if with_lesion else False
        rates = _run(net, frozen, settle, window)
        return net, rates

    control, control_rates = arm(False)
    lesion, lesion_rates = arm(True)

    delta = (lesion_rates - control_rates).astype(np.float32)
    survivors = ~ablated_mask
    ranked = sorted(
        ((conn.neurons[i], float(delta[i])) for i in np.nonzero(survivors)[0]),
        key=lambda item: item[1],
    )

    backward_cells = sim.locomotion.backward_cells
    forward_cells = sim.locomotion.forward_cells

    def mean_rate(rates, cells) -> float:
        return float(rates[cells].mean()) if cells.size else 0.0

    def balance_of(rates) -> float:
        b, f = mean_rate(rates, backward_cells), mean_rate(rates, forward_cells)
        return (b - f) / (b + f + 1e-6)

    def head_of(net) -> float:
        dorsal, ventral = net.segment_activation()
        n = sim.locomotion.p.head_segments
        d, v = float(dorsal[:n].sum()), float(ventral[:n].sum())
        return (d - v) / (d + v + 1e-6) - sim.locomotion.steer_bias

    significant = np.abs(delta[survivors]) > 1.0
    return Effect(
        ablated=names,
        delta=delta,
        fallers=[item for item in ranked[:top] if item[1] < -1.0],
        risers=[item for item in reversed(ranked[-top:]) if item[1] > 1.0],
        backward=(mean_rate(control_rates, backward_cells),
                  mean_rate(lesion_rates, backward_cells)),
        forward=(mean_rate(control_rates, forward_cells),
                 mean_rate(lesion_rates, forward_cells)),
        balance=(balance_of(control_rates), balance_of(lesion_rates)),
        muscle=(float(control.muscle.mean()), float(lesion.muscle.mean())),
        head_imbalance=(head_of(control), head_of(lesion)),
        n_changed=int(significant.sum()),
        n_down=int((delta[survivors] < -1.0).sum()),
        n_up=int((delta[survivors] > 1.0).sum()),
        reversal_threshold=sim.locomotion.p.reversal_threshold,
    )

"""Mapping from world state onto sensory neuron currents.

Only neurons that ``Sensory.csv`` actually annotates are driven, and each
modality goes to the cells that carry it in the animal:

chemosensation
    C. elegans does not measure a spatial gradient across its body; it compares
    concentration over *time* while it moves.  The concentration at the nose is
    filtered at two time scales and their difference is used as dC/dt, which
    then splits into an ON channel (ASEL, AWA and the ``gpg-food`` cells, firing
    while conditions improve) and an OFF channel (ASER, AWB, AWC, firing while
    they deteriorate).

    Those cells are driven for their own sake, but they are not what produces
    chemotaxis: this dataset records no gap junctions, no neuron compartments,
    and labels every synapse excitatory, which removes exactly the machinery the
    two chemotactic strategies run on.  Both are therefore written out
    explicitly below -- the pirouette rule and the klinotaxis rule -- each
    documented where it is applied, and each handing off to the connectome for
    the last step rather than dictating the behaviour outright.
gentle touch
    Contact along the front third of the body drives ALM/AVM, contact along the
    back drives PLM.
nose touch
    Contact at the very tip drives the ``gpg-nose`` cells (ASH, FLP, OLQ, IL1V),
    which in this connectome project onto AVA about four times as strongly as
    onto AVB — the escape response.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .connectome import Connectome
from .world import World


@dataclass
class SensoryParams:
    """Gains (in units of membrane current) and time constants (seconds)."""

    chemo_gain: float = 55.0
    touch_gain: float = 70.0
    nose_gain: float = 200.0
    #: Fast/slow concentration filters; their difference is the dC/dt estimate
    #: used for klinotaxis, which needs to stay locked to the head sweep.
    tau_fast: float = 0.35
    tau_slow: float = 2.4
    #: dC/dt value that saturates the chemosensory channels.
    dc_scale: float = 0.020
    #: A second, slower filter pair, straddling the undulation period (~3.8 s).
    #: It is reported on the HUD but deliberately not used for the pirouette
    #: decision: the head sweep does swing the nose concentration negative twice
    #: per cycle, but driving reversals from a trend this slow makes the worm
    #: keep reversing for seconds after it has already turned back up-gradient,
    #: and measured over 4 runs it cut food-finding from 6/6 to 1/4 without
    #: lowering the reversal rate.  The fast estimate is used instead.
    tau_trend_fast: float = 2.5
    tau_trend_slow: float = 9.0
    trend_scale: float = 0.011
    #: Absolute-concentration drive onto the food-sensing cells.
    food_gain: float = 18.0
    #: Gain of the RIA-style klinotaxis rule (see :class:`SensoryEncoder`).
    klinotaxis_gain: float = 120.0
    #: Gain of the pirouette rule (see :class:`SensoryEncoder`).
    pirouette_gain: float = 350.0


class SensoryEncoder:
    """Turns the world around the worm into an external-current vector."""

    def __init__(self, conn: Connectome, params: SensoryParams | None = None) -> None:
        self.conn = conn
        self.p = params or SensoryParams()

        self.on_cells = np.unique(
            np.concatenate([conn.idx("ASEL", "AWAL", "AWAR"), conn.by_function("gpg-food")])
        )
        self.off_cells = conn.idx("ASER", "AWCL", "AWCR", "AWBL", "AWBR")
        self.food_cells = conn.by_function("gpg-food")
        self.nose_cells = conn.by_function("gpg-nose")
        self.anterior_touch = conn.idx("ALML", "ALMR", "AVM")
        self.posterior_touch = conn.idx("PLML", "PLMR")
        # Head motor neurons, split by the side they contract.  The muscle
        # matrix segregates these cleanly: SMDD/RMDD are ~5:1 dorsal, SMDV/RMDV
        # ~1:8 ventral, and all of them innervate segments 1-8 only.
        self.head_dorsal = conn.idx("SMDDL", "SMDDR", "RMDDL", "RMDDR")
        self.head_ventral = conn.idx("SMDVL", "SMDVR", "RMDVL", "RMDVR")
        # Interneurons through which the pirouette rule reaches the command
        # layer.  These two are the only ones in this dataset with a decisive
        # backward bias: AIZ synapses onto AVA/AVD/AVE with weight 10 and onto
        # AVB/PVC with 0, RIB with 14 against 1.  AIB, the textbook route, is
        # actually biased *forward* here (4 versus 8), so it is not used.
        self.pirouette_cells = conn.idx("AIZL", "AIZR", "RIBL", "RIBR")

        self._c_fast = 0.0
        self._c_slow = 0.0
        self._c_trend_fast = 0.0
        self._c_trend_slow = 0.0
        self._primed = False

        #: Most recent dC/dt estimate, exposed for the HUD.
        self.dc_dt = 0.0
        #: Slow gradient trend driving the pirouette decision.
        self.dc_trend = 0.0
        self.concentration = 0.0

    def reset(self) -> None:
        self._primed = False
        self.dc_dt = 0.0

    def sense(
        self,
        world: World,
        body_points: np.ndarray,
        radius: float,
        dt: float,
        head_bend: float = 0.0,
    ) -> np.ndarray:
        """Build the external current vector for one simulation step.

        ``body_points`` is the head-first outline from :meth:`Body.points`, and
        ``head_bend`` is the instantaneous head sweep in ``[-1, 1]``, positive
        dorsal.
        """
        p = self.p
        conn = self.conn
        current = np.zeros(conn.n_neurons, dtype=np.float32)

        head_x, head_y = float(body_points[0, 0]), float(body_points[0, 1])
        concentration = world.concentration(head_x, head_y)
        self.concentration = concentration

        if not self._primed:
            self._c_fast = self._c_slow = concentration
            self._c_trend_fast = self._c_trend_slow = concentration
            self._primed = True
        # Coefficients are clamped: a dt larger than the time constant must
        # settle onto the new value, not overshoot past it.
        self._c_fast += min(1.0, dt / p.tau_fast) * (concentration - self._c_fast)
        self._c_slow += min(1.0, dt / p.tau_slow) * (concentration - self._c_slow)
        self.dc_dt = self._c_fast - self._c_slow

        self._c_trend_fast += min(1.0, dt / p.tau_trend_fast) * (
            concentration - self._c_trend_fast
        )
        self._c_trend_slow += min(1.0, dt / p.tau_trend_slow) * (
            concentration - self._c_trend_slow
        )
        self.dc_trend = self._c_trend_fast - self._c_trend_slow

        signal = float(np.clip(self.dc_dt / p.dc_scale, -1.0, 1.0))
        if signal > 0.0:
            current[self.on_cells] += p.chemo_gain * signal
        else:
            current[self.off_cells] += p.chemo_gain * -signal

        # Food cells also report that food is simply *present*.
        current[self.food_cells] += p.food_gain * min(1.0, concentration)

        # Pirouettes: reverse and turn when conditions deteriorate.  This does
        # most of the work in real chemotaxis, and it cannot emerge from this
        # dataset, because the ON/OFF opponency it needs is not in the file --
        # ASEL and ASER are wired almost identically (both mainly onto AIY and
        # AIB) and every edge is labelled "exc", so the glutamate-gated chloride
        # synapses that make AWC and ASER *inhibit* AIY carry the wrong sign
        # here.  The rule is applied explicitly, but only as far as the
        # interneurons: the connectome still decides whether it becomes a
        # reversal, and measured open-loop it does, on 93% of samples against 0%
        # for a flat or rising gradient.
        if signal < 0.0:
            current[self.pirouette_cells] += p.pirouette_gain * -signal

        # Klinotaxis: curve up-gradient while sweeping.  Driving ASE/AWC alone
        # does not steer -- measured open-loop, the ON and OFF channels shift
        # the head muscle imbalance by 0.006 against a noise level of 0.008 --
        # because nothing in a wiring diagram routes a scalar chemical signal
        # preferentially to the dorsal or the ventral head motor neurons.  In
        # the animal that routing is RIA's job: its dorsal and ventral
        # compartments multiply the sensory signal by a proprioceptive head-bend
        # signal, which a CSV of synapse counts cannot express, having neither
        # compartments nor gap junctions.  So the product rule is applied here
        # and injected into the head motor neurons the connectome *does*
        # segregate.  Sweeping into rising concentration reinforces that side.
        steer_drive = signal * float(np.clip(head_bend, -1.0, 1.0))
        if steer_drive > 0.0:
            current[self.head_dorsal] += p.klinotaxis_gain * steer_drive
        elif steer_drive < 0.0:
            current[self.head_ventral] += p.klinotaxis_gain * -steer_drive

        # Mechanosensation: which part of the body is against something?
        n = len(body_points)
        nose = world.touching(head_x, head_y, radius)
        if nose:
            current[self.nose_cells] += p.nose_gain

        front = body_points[1 : max(2, n // 3)]
        back = body_points[max(2, 2 * n // 3) :]
        if any(world.touching(float(x), float(y), radius) for x, y in front):
            current[self.anterior_touch] += p.touch_gain
        if any(world.touching(float(x), float(y), radius) for x, y in back):
            current[self.posterior_touch] += p.touch_gain

        return current


def head_bearing(body_points: np.ndarray) -> float:
    """Heading of the head segment, in radians."""
    dx = body_points[0, 0] - body_points[1, 0]
    dy = body_points[0, 1] - body_points[1, 1]
    return math.atan2(float(dy), float(dx))

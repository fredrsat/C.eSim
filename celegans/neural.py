"""Leaky integrate-and-fire simulation of the connectome.

Every node in ``Connectome.csv`` becomes one LIF unit.  A spike deposits charge
into a per-neuron synaptic trace which decays with ``tau_syn``; the connectome
matrix then fans that trace out to postsynaptic partners, signed by the
``exc``/``inh`` label.  The same traces drive the muscle activations through the
neuron-to-muscle matrix.

The membrane update is the standard forward-Euler LIF

    tau_m dV/dt = -(V - V_rest) + g_syn * I_syn + I_ext + noise

with a fixed threshold, reset and absolute refractory period.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .connectome import Connectome


@dataclass
class NeuralParams:
    """Tunable constants of the neural layer.  Times are in milliseconds."""

    tau_m: float = 20.0
    tau_syn: float = 6.0
    tau_muscle: float = 90.0
    #: Time constant of the firing-rate readout filter.
    tau_rate: float = 60.0
    v_rest: float = 0.0
    v_reset: float = -12.0
    v_threshold: float = 30.0
    refractory: float = 3.0
    #: Scales connectome weights (raw synapse counts) into membrane current.
    g_syn: float = 150.0
    #: Scales neuron-to-muscle weights into muscle activation.
    g_muscle: float = 1.1
    #: Tonic drive holding neurons just below threshold, so that synaptic and
    #: sensory input decides who actually fires.
    i_tonic: float = 28.0
    #: Standard deviation of the per-step membrane noise.
    noise: float = 18.0
    #: Spike-frequency adaptation.  Only ~9% of the connectome is inhibitory,
    #: so without a self-limiting current the recurrent excitation runs away.
    adapt_step: float = 8.0
    tau_adapt: float = 120.0
    #: Global inhibition proportional to mean network activity.  Together with
    #: the spectral-radius normalisation below this supplies the E/I balance the
    #: raw connectome lacks, turning a bistable (silent or saturated) network
    #: into one whose rate tracks its input.
    g_global: float = 0.0
    #: Divide the weight matrix by its spectral radius, so ``g_syn`` acts as the
    #: recurrent loop gain rather than depending on raw synapse counts.
    normalize_weights: bool = True
    #: Current injected into a neuron that has been forced on by hand.
    stim_current: float = 90.0


class NeuralNetwork:
    """Spiking network over the whole connectome."""

    def __init__(
        self,
        conn: Connectome,
        params: NeuralParams | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.conn = conn
        self.p = params or NeuralParams()
        self.rng = rng or np.random.default_rng()

        n = conn.n_neurons
        self.v = np.full(n, self.p.v_rest, dtype=np.float32)
        self.syn = np.zeros(n, dtype=np.float32)
        self.spiked = np.zeros(n, dtype=bool)
        self.refractory_left = np.zeros(n, dtype=np.float32)
        self.adapt = np.zeros(n, dtype=np.float32)
        self.muscle = np.zeros(conn.n_muscles, dtype=np.float32)
        #: Low-pass filtered spike train, used as a firing-rate readout.
        self.rate = np.zeros(n, dtype=np.float32)
        #: Laser-ablation mask.  An ablated neuron never fires, so it stops
        #: driving anything downstream, but the rest of the wiring is untouched
        #: -- the same manipulation the C. elegans literature is built on.
        self.ablated = np.zeros(n, dtype=bool)
        #: Forced-activation mask, the optogenetic counterpart of ablation:
        #: extra current is injected so the cell fires regardless of its input.
        self.stimulated = np.zeros(n, dtype=bool)

        # Pre-transpose so the hot loop is a plain matrix-vector product.
        w = conn.W.T.astype(np.float32)
        self.weight_scale = 1.0
        if self.p.normalize_weights:
            self.weight_scale = float(
                np.abs(np.linalg.eigvals(conn.W.astype(np.float64))).max()
            )
            w = w / self.weight_scale
        self._w_in = np.ascontiguousarray(w)
        self._m_in = np.ascontiguousarray(conn.M.T)

    def step(self, dt: float, i_ext: np.ndarray | None = None) -> None:
        """Advance the network by ``dt`` milliseconds."""
        p = self.p

        # Synaptic traces decay, then receive the previous step's spikes.
        self.syn *= np.exp(-dt / p.tau_syn)
        self.syn += self.spiked

        self.adapt *= np.exp(-dt / p.tau_adapt)

        current = (
            self._w_in @ self.syn * p.g_syn
            + p.i_tonic
            - self.adapt
            - p.g_global * float(self.syn.mean())
        )
        if i_ext is not None:
            current = current + i_ext
        if self.stimulated.any():
            current = current + self.stimulated * p.stim_current
        if p.noise > 0.0:
            current = current + self.rng.normal(
                0.0, p.noise, size=current.shape
            ).astype(np.float32)

        self.v += (dt / p.tau_m) * (-(self.v - p.v_rest) + current)

        # Neurons inside the absolute refractory period are clamped at reset.
        self.refractory_left -= dt
        held = self.refractory_left > 0.0
        self.v[held] = p.v_reset

        self.spiked = self.v >= p.v_threshold
        self.v[self.spiked] = p.v_reset
        self.refractory_left[self.spiked] = p.refractory
        self.adapt += self.spiked * p.adapt_step

        if self.ablated.any():
            self.spiked &= ~self.ablated
            self.v[self.ablated] = p.v_rest
            self.syn[self.ablated] = 0.0

        # Exponential spike-count filter, calibrated so a neuron spiking at a
        # steady f Hz settles at rate == f.
        self.rate *= np.exp(-dt / p.tau_rate)
        self.rate += self.spiked * (1000.0 / p.tau_rate)

        # Muscles integrate motor neuron output; they can only contract.
        drive = self._m_in @ self.syn * p.g_muscle
        self.muscle += (dt / p.tau_muscle) * (-self.muscle + drive)
        np.clip(self.muscle, 0.0, None, out=self.muscle)

    def input_breakdown(
        self, index: int, i_ext: np.ndarray | None = None, top: int = 10
    ) -> dict[str, object]:
        """Decompose the current arriving at one neuron, right now.

        This is the direct answer to "why is this cell active": every term is
        the same quantity the integration step actually sums, so the parts add
        up to the drive the membrane is seeing this millisecond.
        """
        p = self.p
        row = self._w_in[index] * self.syn * p.g_syn
        order = np.argsort(-np.abs(row))
        contributors = [
            (self.conn.neurons[j], float(row[j]))
            for j in order[:top]
            if abs(row[j]) > 1e-4
        ]
        synaptic = float(row.sum())
        external = float(i_ext[index]) if i_ext is not None else 0.0
        adaptation = -float(self.adapt[index])
        tonic = p.i_tonic
        return {
            "contributors": contributors,
            "synaptic": synaptic,
            "excitatory": float(row[row > 0].sum()),
            "inhibitory": float(row[row < 0].sum()),
            "tonic": tonic,
            "adaptation": adaptation,
            "external": external,
            "total": synaptic + tonic + adaptation + external,
            "threshold": p.v_threshold,
            "v": float(self.v[index]),
        }

    def set_ablated(self, indices: np.ndarray, value: bool) -> None:
        """Ablate or restore a set of neurons."""
        self.ablated[indices] = value
        if value:
            self.v[indices] = self.p.v_rest
            self.syn[indices] = 0.0
            self.rate[indices] = 0.0

    def toggle_ablated(self, indices: np.ndarray) -> bool:
        """Flip a group's ablation state; returns True if it is now ablated."""
        now_ablated = not bool(self.ablated[indices].all()) if indices.size else False
        self.set_ablated(indices, now_ablated)
        return now_ablated

    def set_stimulated(self, indices: np.ndarray, value: bool) -> None:
        """Force a set of neurons on, or release them."""
        self.stimulated[indices] = value

    def toggle_stimulated(self, indices: np.ndarray) -> bool:
        now_on = not bool(self.stimulated[indices].all()) if indices.size else False
        self.set_stimulated(indices, now_on)
        return now_on

    def restore_all(self) -> None:
        self.ablated[:] = False
        self.stimulated[:] = False

    def group_rate(self, indices: np.ndarray) -> float:
        """Mean firing rate (Hz) over a group of neurons."""
        if indices.size == 0:
            return 0.0
        return float(self.rate[indices].mean())

    def segment_activation(self) -> tuple[np.ndarray, np.ndarray]:
        """Muscle activation summed per body segment, as (dorsal, ventral).

        Both arrays have length ``connectome.N_SEGMENTS``, ordered head to tail.
        """
        conn = self.conn
        n_seg = int(conn.muscle_segment.max()) + 1
        dorsal = np.bincount(
            conn.muscle_segment[conn.muscle_dorsal],
            weights=self.muscle[conn.muscle_dorsal],
            minlength=n_seg,
        )
        ventral = np.bincount(
            conn.muscle_segment[~conn.muscle_dorsal],
            weights=self.muscle[~conn.muscle_dorsal],
            minlength=n_seg,
        )
        return dorsal.astype(np.float32), ventral.astype(np.float32)

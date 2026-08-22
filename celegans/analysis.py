"""Structural facts about each neuron, derived from the connectome itself.

Everything here is computed from the CSV files, which is what separates it from
:mod:`celegans.annotations` -- these numbers can be checked, and they change if
the dataset changes.  The inspector shows the two side by side under separate
headings.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .connectome import Connectome

BACKWARD = ("AVAL", "AVAR", "AVDL", "AVDR", "AVEL", "AVER")
FORWARD = ("AVBL", "AVBR", "PVCL", "PVCR")


@dataclass
class NeuronFacts:
    """Measured properties of one neuron."""

    name: str
    excitatory_in: float
    inhibitory_in: float
    in_degree: int
    out_weight: float
    out_degree: int
    #: Rank by total incoming weight, 1 = most innervated of all 299.
    drive_rank: int
    #: 0.0 at the tip of the head, 1.0 at the tail.
    body_position: float
    muscle_weight: float
    muscle_dorsal: float
    muscle_ventral: float
    muscle_segments: tuple[int, int] | None
    #: Normalised influence onto the two command pools, over one and two hops.
    to_backward: float
    to_forward: float
    isolated: bool


class Analysis:
    """Precomputed structural metrics for every neuron."""

    def __init__(self, conn: Connectome) -> None:
        self.conn = conn
        W = conn.W
        self.excitatory_in = np.clip(W, 0, None).sum(axis=0)
        self.inhibitory_in = -np.clip(W, None, 0).sum(axis=0)
        self.in_degree = (W != 0).sum(axis=0)
        self.out_weight = np.abs(W).sum(axis=1)
        self.out_degree = (W != 0).sum(axis=1)
        total_in = np.abs(W).sum(axis=0)
        self.drive_rank = (-total_in).argsort().argsort() + 1
        self.body_position = conn.body_fraction()

        # Row-normalised influence, so a cell with many synapses does not
        # automatically dominate; one and two hops combined.
        norm = W / np.abs(W).sum(axis=1, keepdims=True).clip(1)
        two_hop = norm @ norm
        combined = norm + two_hop
        self._to_backward = combined[:, conn.idx(*BACKWARD)].sum(axis=1)
        self._to_forward = combined[:, conn.idx(*FORWARD)].sum(axis=1)

        # A neuron that neither receives nor sends anything cannot participate.
        self.isolated = (self.in_degree == 0) & (self.out_degree == 0)

    def facts(self, index: int) -> NeuronFacts:
        conn = self.conn
        muscle_row = conn.M[index]
        dorsal = conn.muscle_dorsal
        active = np.nonzero(muscle_row)[0]
        segments = (
            (int(conn.muscle_segment[active].min()) + 1,
             int(conn.muscle_segment[active].max()) + 1)
            if active.size
            else None
        )
        return NeuronFacts(
            name=conn.neurons[index],
            excitatory_in=float(self.excitatory_in[index]),
            inhibitory_in=float(self.inhibitory_in[index]),
            in_degree=int(self.in_degree[index]),
            out_weight=float(self.out_weight[index]),
            out_degree=int(self.out_degree[index]),
            drive_rank=int(self.drive_rank[index]),
            body_position=float(self.body_position[index]),
            muscle_weight=float(np.abs(muscle_row).sum()),
            muscle_dorsal=float(muscle_row[dorsal].sum()),
            muscle_ventral=float(muscle_row[~dorsal].sum()),
            muscle_segments=segments,
            to_backward=float(self._to_backward[index]),
            to_forward=float(self._to_forward[index]),
            isolated=bool(self.isolated[index]),
        )

    def partners(self, index: int, top: int = 8):
        """Strongest presynaptic and postsynaptic partners, by raw weight."""
        conn = self.conn
        incoming = conn.W[:, index]
        outgoing = conn.W[index]
        pre = [
            (conn.neurons[j], float(incoming[j]))
            for j in np.argsort(-np.abs(incoming))[:top]
            if incoming[j] != 0
        ]
        post = [
            (conn.neurons[j], float(outgoing[j]))
            for j in np.argsort(-np.abs(outgoing))[:top]
            if outgoing[j] != 0
        ]
        return pre, post

    def partner_mask(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Boolean masks of a neuron's presynaptic and postsynaptic partners."""
        return self.conn.W[:, index] != 0, self.conn.W[index] != 0

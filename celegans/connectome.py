"""Loading of the C. elegans connectome data into numpy matrices.

Data source: https://github.com/3BIM20162017/CElegansTP (OpenWorm-derived).

Four files are used:

``Connectome.csv``
    Neuron -> Target synapses, with a connection count and an ``exc``/``inh``
    label.  299 nodes, 2279 edges.
``Neurons_to_Muscles.csv``
    Motor neuron -> body wall muscle, same weight/sign format.  94 muscles,
    named ``M{D,V}{L,R}NN`` where ``D``/``V`` is dorsal/ventral, ``L``/``R`` is
    the left/right quadrant and ``NN`` is the segment index from head (01) to
    tail (24).
``Sensory.csv``
    Functional annotation ("mechanosensory", "chemosensory|gpg-food", ...) for
    86 sensory neurons.
``spatialpositions/distances.csv``
    3D position per neuron.  Column ``1`` is the anterior-posterior axis and is
    what we use to place neurons along the body.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Body wall muscles are named e.g. MDL07: quadrant + two digit segment index.
_MUSCLE_RE = re.compile(r"^M(?P<dv>[DV])(?P<lr>[LR])(?P<seg>\d{2})$")

# Number of body segments addressed by the muscle rows.  Dorsal quadrants run
# 01-24, ventral only 01-23; we keep 24 slots and leave the last ventral empty.
N_SEGMENTS = 24


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


@dataclass
class Connectome:
    """Connectome matrices plus the index bookkeeping needed to use them."""

    neurons: list[str]
    index: dict[str, int]
    #: (N, N) signed synaptic weights; ``W[i, j]`` is the drive from i onto j.
    W: np.ndarray
    #: (N, 94) signed neuron -> muscle weights.
    M: np.ndarray
    muscles: list[str]
    #: (94,) segment index 0..23 per muscle column.
    muscle_segment: np.ndarray
    #: (94,) boolean, True where the muscle is dorsal.
    muscle_dorsal: np.ndarray
    #: (N, 3) spatial positions, column 1 is the head-tail axis.
    positions: np.ndarray
    #: Functional annotation string per sensory neuron.
    sensory_function: dict[str, str] = field(default_factory=dict)
    #: Neurotransmitter per sensory neuron.
    sensory_transmitter: dict[str, str] = field(default_factory=dict)

    @property
    def n_neurons(self) -> int:
        return len(self.neurons)

    @property
    def n_muscles(self) -> int:
        return len(self.muscles)

    def idx(self, *names: str) -> np.ndarray:
        """Indices of the named neurons, silently skipping unknown names.

        Several classical cell names appear in the literature but not in this
        particular dataset, so callers can list a superset without guarding.
        """
        return np.array(
            [self.index[n] for n in names if n in self.index], dtype=np.intp
        )

    def by_function(self, pattern: str) -> np.ndarray:
        """Indices of sensory neurons whose annotation contains ``pattern``."""
        return np.array(
            [
                self.index[name]
                for name, func in self.sensory_function.items()
                if pattern in func and name in self.index
            ],
            dtype=np.intp,
        )

    def body_fraction(self) -> np.ndarray:
        """Position along the body per neuron, 0.0 at the head, 1.0 at the tail."""
        # The AP axis in the source data is most negative at the tip of the
        # head and most positive at the tail, so it maps straight through.
        ap = self.positions[:, 1]
        return (ap - ap.min()) / np.ptp(ap)


def load(data_dir: Path | str = DATA_DIR) -> Connectome:
    """Read the CSV files in ``data_dir`` and build the matrices."""
    data_dir = Path(data_dir)
    syn_rows = _read_csv(data_dir / "Connectome.csv")
    mus_rows = _read_csv(data_dir / "Neurons_to_Muscles.csv")
    sen_rows = _read_csv(data_dir / "Sensory.csv")
    pos_rows = _read_csv(data_dir / "spatialpositions" / "distances.csv")

    names = set()
    for row in syn_rows:
        names.add(row["Neuron"])
        names.add(row["Target"])
    for row in mus_rows:
        names.add(row["Origin"])
    for row in pos_rows:
        names.add(row[""])
    neurons = sorted(names)
    index = {name: i for i, name in enumerate(neurons)}
    n = len(neurons)

    W = np.zeros((n, n), dtype=np.float32)
    for row in syn_rows:
        sign = -1.0 if row["Neurotransmitter"] == "inh" else 1.0
        W[index[row["Neuron"]], index[row["Target"]]] += (
            sign * float(row["Number of Connections"])
        )

    muscles = sorted({row["Muscle"] for row in mus_rows})
    m_index = {name: i for i, name in enumerate(muscles)}
    M = np.zeros((n, len(muscles)), dtype=np.float32)
    for row in mus_rows:
        origin = row["Origin"]
        if origin not in index:
            continue
        sign = -1.0 if row["Neurotransmitter"] == "inh" else 1.0
        M[index[origin], m_index[row["Muscle"]]] += (
            sign * float(row["Number of Connections"])
        )

    muscle_segment = np.zeros(len(muscles), dtype=np.intp)
    muscle_dorsal = np.zeros(len(muscles), dtype=bool)
    for name, i in m_index.items():
        match = _MUSCLE_RE.match(name)
        if match is None:
            raise ValueError(f"unexpected muscle name: {name!r}")
        muscle_segment[i] = int(match["seg"]) - 1
        muscle_dorsal[i] = match["dv"] == "D"

    positions = np.zeros((n, 3), dtype=np.float32)
    for row in pos_rows:
        positions[index[row[""]]] = (
            float(row["0"]),
            float(row["1"]),
            float(row["2"]),
        )

    sensory_function: dict[str, str] = {}
    sensory_transmitter: dict[str, str] = {}
    for row in sen_rows:
        sensory_function[row["Neuron"]] = row["Function"]
        sensory_transmitter[row["Neuron"]] = row["Neurotransmitter"]

    return Connectome(
        neurons=neurons,
        index=index,
        W=W,
        M=M,
        muscles=muscles,
        muscle_segment=muscle_segment,
        muscle_dorsal=muscle_dorsal,
        positions=positions,
        sensory_function=sensory_function,
        sensory_transmitter=sensory_transmitter,
    )

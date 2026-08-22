"""Named neuron groups, for ablating a whole circuit at once.

Each entry resolves to indices at run time, so a name missing from this
particular dataset is skipped rather than raising.  The groups are the ones the
simulation's behaviour actually rests on, which makes them the interesting ones
to switch off: ablating the nose-touch group should cost the worm its escape
response, ablating the backward command should leave it unable to reverse at
all, and so on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .connectome import Connectome


@dataclass(frozen=True)
class Circuit:
    key: str
    name: str
    description: str
    members: tuple[str, ...] = ()
    #: Alternative to ``members``: a ``Sensory.csv`` annotation substring.
    function: str | None = None

    def indices(self, conn: Connectome) -> np.ndarray:
        if self.function is not None:
            return conn.by_function(self.function)
        return conn.idx(*self.members)


CIRCUITS: tuple[Circuit, ...] = (
    Circuit(
        "1", "nose touch",
        "ASH/FLP/OLQ/IL1V - drives the escape reversal",
        function="gpg-nose",
    ),
    Circuit(
        "2", "gentle touch",
        "ALM/AVM anterior, PLM posterior",
        members=("ALML", "ALMR", "AVM", "PLML", "PLMR"),
    ),
    Circuit(
        "3", "chemosensory",
        "ASE/AWC/AWA/AWB plus the gpg-food cells",
        members=(
            "ASEL", "ASER", "AWCL", "AWCR", "AWAL", "AWAR", "AWBL", "AWBR",
            "ADFL", "ADFR", "ASGL", "ASGR", "ASIL", "ASIR", "ASJL", "ASJR",
        ),
    ),
    Circuit(
        "4", "backward command",
        "AVA/AVD/AVE - without it the worm cannot reverse",
        members=("AVAL", "AVAR", "AVDL", "AVDR", "AVEL", "AVER"),
    ),
    Circuit(
        "5", "forward command",
        "AVB/PVC",
        members=("AVBL", "AVBR", "PVCL", "PVCR"),
    ),
    Circuit(
        "6", "pirouette relay",
        "AIZ/RIB - the route the chemotaxis rule reaches AVA through",
        members=("AIZL", "AIZR", "RIBL", "RIBR"),
    ),
    Circuit(
        "7", "head motor",
        "SMD/RMD - steering; ablating it should abolish klinotaxis",
        members=(
            "SMDDL", "SMDDR", "SMDVL", "SMDVR",
            "RMDDL", "RMDDR", "RMDVL", "RMDVR", "RMDL", "RMDR",
        ),
    ),
    Circuit(
        "8", "chemo interneurons",
        "AIY/AIA/AIB - the first layer after the chemosensory cells",
        members=("AIYL", "AIYR", "AIAL", "AIAR", "AIBL", "AIBR"),
    ),
)

CIRCUITS_BY_KEY = {c.key: c for c in CIRCUITS}


def classify(conn: Connectome) -> dict[str, str]:
    """Label every neuron ``sensory``, ``motor`` or ``interneuron``.

    A cell that both carries a sensory annotation and innervates muscle (IL1 and
    the CEPs, among others) counts as sensory, which is how they are usually
    described.
    """
    labels: dict[str, str] = {}
    for i, name in enumerate(conn.neurons):
        if name in conn.sensory_function:
            labels[name] = "sensory"
        elif np.abs(conn.M[i]).sum() > 0:
            labels[name] = "motor"
        else:
            labels[name] = "interneuron"
    return labels

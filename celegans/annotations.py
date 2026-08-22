"""Curated notes on what each neuron class is believed to do.

**Source and status.** Everything else in this project is derived from the four
CSV files and can be re-measured with ``validate.py``.  This module is not: it is
background knowledge about C. elegans taken from the standard literature
(WormAtlas / WormBook and the primary papers behind them), written down so the
inspector can say something useful about a cell you click on.

It is kept in its own module, and surfaced under its own heading in the UI, so
that an assertion made here is never mistaken for something the dataset shows.
Where a class is genuinely poorly characterised, the entry says so rather than
inventing a role.

Annotation is per *class* (AVA, VB, IL1), not per cell, which is how the
literature describes them -- AVAL and AVAR are the left and right members of one
class and are not known to differ functionally.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Annotation:
    """What one neuron class is thought to do."""

    kind: str  # sensory | interneuron | motor | modulatory | pharyngeal
    role: str  # one short line, shown as a heading
    detail: str  # a few sentences of context
    circuits: tuple[str, ...] = field(default_factory=tuple)


S, I, M, MOD, PH = "sensory", "interneuron", "motor", "modulatory", "pharyngeal"

ANNOTATIONS: dict[str, Annotation] = {
    # -- command interneurons --------------------------------------------
    "AVA": Annotation(
        I, "backward command interneuron",
        "One of the most heavily innervated cells in the animal, and the main "
        "driver of backward locomotion through the A-type ventral cord motor "
        "neurons. Depolarising AVA triggers a reversal; killing it largely "
        "abolishes spontaneous reversals and the escape response.",
        ("locomotion", "escape"),
    ),
    "AVB": Annotation(
        I, "forward command interneuron",
        "Drives forward locomotion via the B-type motor neurons, with which it "
        "makes gap junctions. AVA and AVB are mutually antagonistic, and which "
        "one is active is essentially what direction the worm is going.",
        ("locomotion",),
    ),
    "AVD": Annotation(
        I, "backward command interneuron",
        "Works with AVA to drive reversals, and is the main relay from the "
        "anterior touch cells ALM and AVM into the backward command system.",
        ("locomotion", "touch", "escape"),
    ),
    "AVE": Annotation(
        I, "backward command interneuron",
        "Third member of the backward command group, biased toward the anterior "
        "body. Its activity rises sharply at reversal onset.",
        ("locomotion",),
    ),
    "PVC": Annotation(
        I, "forward command interneuron",
        "Partner to AVB for forward movement, and the principal target of the "
        "posterior touch cells PLM -- the reason a tap on the tail makes the "
        "worm move forward.",
        ("locomotion", "touch"),
    ),
    # -- chemosensory and their first interneuron layer -------------------
    "ASE": Annotation(
        S, "primary salt chemosensor",
        "The main gustatory neuron pair, and strikingly left/right asymmetric in "
        "the animal: ASEL is an ON cell that fires when attractant concentration "
        "rises, ASER an OFF cell that fires when it falls. That opponency is the "
        "core of salt chemotaxis. Note that this dataset does not capture it -- "
        "ASEL and ASER have nearly identical wiring here.",
        ("chemotaxis",),
    ),
    "AWC": Annotation(
        S, "volatile odour sensor (OFF cell)",
        "Detects attractive volatile odours such as benzaldehyde and isoamyl "
        "alcohol. Counter-intuitively it is activated by odour *removal*, and "
        "that OFF response is what triggers the turn when a worm leaves an "
        "odour peak. Its synapse onto AIY is inhibitory in the animal, via a "
        "glutamate-gated chloride channel.",
        ("chemotaxis",),
    ),
    "AWA": Annotation(
        S, "volatile odour sensor (ON cell)",
        "Detects attractive odours including diacetyl and pyrazine, and unlike "
        "AWC responds to odour presence rather than removal.",
        ("chemotaxis",),
    ),
    "AWB": Annotation(
        S, "volatile odour sensor, repulsive",
        "Mediates avoidance of repellent odours; driving it biases the worm away "
        "rather than toward a source.",
        ("chemotaxis", "avoidance"),
    ),
    "ASH": Annotation(
        S, "polymodal nociceptor",
        "The main nociceptive neuron: responds to nose touch, high osmolarity "
        "and noxious chemicals, and drives the reversal-and-turn escape "
        "response through AVA. The single most important cell for avoidance.",
        ("escape", "avoidance", "touch"),
    ),
    "ASI": Annotation(
        S, "food and pheromone sensor",
        "Senses food availability and dauer pheromone, and is a major source of "
        "the signals that set feeding state and dauer entry. Largely modulatory "
        "in timescale rather than driving movement directly.",
        ("chemotaxis", "feeding"),
    ),
    "ASJ": Annotation(
        S, "dauer and light sensor",
        "Involved in dauer recovery and, with ASK, in avoidance of short "
        "wavelength light.",
        ("chemotaxis", "feeding"),
    ),
    "ASG": Annotation(
        S, "chemosensor, food related",
        "Contributes to chemotaxis and to dauer regulation, overlapping in "
        "function with ASI.",
        ("chemotaxis", "feeding"),
    ),
    "ASK": Annotation(
        S, "chemosensor, amino acids and pheromone",
        "Attracted to amino acids, and involved in pheromone responses and male "
        "mate searching.",
        ("chemotaxis",),
    ),
    "ADF": Annotation(
        S, "chemosensor, serotonergic",
        "A chemosensory cell that is also the main serotonergic sensory neuron; "
        "its serotonin release rises in the presence of food and helps switch "
        "the animal into the slow, dwelling state.",
        ("chemotaxis", "feeding"),
    ),
    "ADL": Annotation(
        S, "nociceptor",
        "Senses noxious chemicals and pheromone, and contributes with ASH to "
        "avoidance behaviour.",
        ("avoidance",),
    ),
    "AIY": Annotation(
        I, "chemotaxis interneuron, promotes runs",
        "The principal target of AWC and ASE. AIY activity suppresses turning "
        "and lengthens forward runs, so inhibiting it (which AWC does on odour "
        "removal) releases a turn. Central to the pirouette strategy.",
        ("chemotaxis",),
    ),
    "AIB": Annotation(
        I, "chemotaxis interneuron, promotes turns",
        "Roughly the opposite of AIY: its activity favours reversals and turns. "
        "In the textbook circuit AIB relays the AWC/ASER OFF signal toward AVA. "
        "In this particular dataset, though, AIB is wired more strongly to the "
        "forward command than the backward one.",
        ("chemotaxis",),
    ),
    "AIA": Annotation(
        I, "chemosensory integrating interneuron",
        "Receives from many chemosensory cells and is involved in the animal's "
        "learned and state-dependent responses to odours.",
        ("chemotaxis",),
    ),
    "AIZ": Annotation(
        I, "chemotaxis interneuron",
        "Sits between the sensory layer and the command interneurons and "
        "contributes to turning during chemotaxis and thermotaxis.",
        ("chemotaxis",),
    ),
    "RIA": Annotation(
        I, "head steering integrator",
        "The key cell for gradual, weathervaning turns. RIA is compartmentalised: "
        "its dorsal and ventral neurites carry separate signals, and it "
        "multiplies incoming sensory input by a proprioceptive head-bend signal "
        "so that steering becomes locked to the head sweep. A wiring diagram "
        "alone cannot express this, since it depends on compartments rather than "
        "on which cells are connected.",
        ("chemotaxis", "steering"),
    ),
    "RIB": Annotation(
        I, "locomotion state interneuron",
        "Active during forward movement and involved in the transitions between "
        "behavioural states; also carries head-bend related signals.",
        ("locomotion", "steering"),
    ),
    "RIM": Annotation(
        I, "reversal-associated interneuron, tyraminergic",
        "Active during reversals, and the source of tyramine, which suppresses "
        "head movement during an escape so the worm does not steer while backing "
        "up. Ablating RIM makes reversal behaviour less stereotyped.",
        ("locomotion", "escape"),
    ),
    "RIC": Annotation(
        MOD, "octopaminergic interneuron",
        "The main octopamine source, signalling food absence and opposing many "
        "of serotonin's effects.",
        ("feeding",),
    ),
    "RIS": Annotation(
        I, "sleep-promoting interneuron",
        "Depolarises at the onset of the quiescent state and is required for "
        "developmentally timed sleep.",
        (),
    ),
    "RID": Annotation(
        MOD, "dorsal-bias motor interneuron",
        "Runs along the dorsal cord and is active during forward movement, where "
        "it appears to sustain dorsal bending.",
        ("locomotion",),
    ),
    # -- mechanosensation -------------------------------------------------
    "ALM": Annotation(
        S, "anterior gentle touch receptor",
        "One of the six touch receptor neurons. A gentle stroke on the anterior "
        "body drives ALM, and the animal responds by reversing.",
        ("touch",),
    ),
    "AVM": Annotation(
        S, "anterior gentle touch receptor",
        "Ventral partner to ALM for anterior touch; matures later in development "
        "and is particularly important for responses to weak stimuli.",
        ("touch",),
    ),
    "PLM": Annotation(
        S, "posterior gentle touch receptor",
        "Detects touch on the posterior body and drives forward acceleration via "
        "PVC -- the opposite behavioural outcome to anterior touch.",
        ("touch",),
    ),
    "PVM": Annotation(
        S, "posterior touch receptor",
        "The sixth touch cell. Its contribution to the classical touch response "
        "is weak, and its role is less clear than the other five.",
        ("touch",),
    ),
    "PVD": Annotation(
        S, "harsh touch and cold nociceptor",
        "An elaborately branched neuron wrapping the body, responsive to harsh "
        "mechanical stimuli and to cold.",
        ("touch", "avoidance"),
    ),
    "FLP": Annotation(
        S, "nose touch and thermal nociceptor",
        "Works with ASH and OLQ on harsh nose touch, and also responds to "
        "noxious heat.",
        ("touch", "escape", "avoidance"),
    ),
    "OLQ": Annotation(
        S, "nose touch receptor",
        "Contributes to gentle nose touch and to the foraging movements of the "
        "head, along with IL1.",
        ("touch", "escape"),
    ),
    "OLL": Annotation(
        S, "nose touch receptor",
        "Mechanosensory cell of the head, involved in the response to touch "
        "delivered to the tip.",
        ("touch",),
    ),
    "IL1": Annotation(
        S, "inner labial mechanosensor and head motor",
        "Unusual in being both sensory and motor: it senses head touch and also "
        "innervates head muscle directly, contributing to foraging head "
        "movements and the head withdrawal reflex.",
        ("touch", "steering"),
    ),
    "IL2": Annotation(
        S, "inner labial chemosensor",
        "Chemosensory cell of the head, and central to the nictation behaviour "
        "of dauer larvae.",
        (),
    ),
    "CEP": Annotation(
        S, "dopaminergic texture sensor",
        "Senses the mechanical texture of a bacterial lawn. Its dopamine is what "
        "slows the animal down on encountering food -- the basal slowing "
        "response.",
        ("feeding", "touch"),
    ),
    "ADE": Annotation(
        S, "dopaminergic mechanosensor",
        "With CEP and PDE, part of the dopaminergic set that reports food "
        "texture and modulates locomotion speed.",
        ("feeding", "touch"),
    ),
    "PDE": Annotation(
        S, "dopaminergic mechanosensor",
        "Posterior member of the dopaminergic group; contributes to the slowing "
        "response and to area-restricted search.",
        ("feeding", "touch"),
    ),
    "ALN": Annotation(S, "putative mechanosensor",
        "Process runs alongside ALM; sensory function is presumed but not well "
        "established. Also linked to oxygen responses.", ()),
    "PLN": Annotation(S, "putative mechanosensor",
        "Posterior counterpart of ALN, with a similarly unsettled role.", ()),
    "SDQ": Annotation(S, "putative mechanosensor",
        "Body cavity neuron with a presumed mechanosensory or oxygen-sensing "
        "role.", ()),
    "BDU": Annotation(I, "poorly characterised interneuron",
        "Process runs along the excretory canal. Its function is not well "
        "established.", ()),
    # -- oxygen, CO2, temperature ----------------------------------------
    "URX": Annotation(
        S, "oxygen sensor",
        "Reports rising oxygen and drives the animal away from high O2 toward "
        "the ~7% it prefers. Central to aerotaxis and to aggregation behaviour.",
        ("aerotaxis",),
    ),
    "AQR": Annotation(S, "oxygen sensor",
        "Body cavity neuron exposed to the pseudocoelom, sensing internal "
        "oxygen alongside URX and PQR.", ("aerotaxis",)),
    "PQR": Annotation(S, "oxygen sensor",
        "Posterior body cavity oxygen sensor, completing the URX/AQR/PQR set.",
        ("aerotaxis",)),
    "BAG": Annotation(S, "carbon dioxide sensor",
        "Detects CO2 and also oxygen decreases; drives avoidance of high CO2.",
        ("aerotaxis",)),
    "AFD": Annotation(
        S, "primary thermosensor",
        "The main thermosensory neuron, remarkable for remembering the "
        "temperature at which the animal was cultivated and responding to "
        "deviations from it. Drives thermotaxis through AIY.",
        ("thermotaxis",),
    ),
    "AUA": Annotation(I, "interneuron, thermal and oxygen related",
        "Paired with AFD and URX; involved in thermosensory and oxygen "
        "responses.", ("thermotaxis", "aerotaxis")),
    # -- head motor neurons ----------------------------------------------
    "SMD": Annotation(
        M, "head motor neuron, steering",
        "Innervates dorsal or ventral head muscle according to subtype (SMDD "
        "versus SMDV) and sets the amplitude of head bending. Together with RMD "
        "it is the output stage of steering, and it is what RIA acts on.",
        ("steering",),
    ),
    "RMD": Annotation(
        M, "head motor neuron",
        "Head muscle motor neuron with dorsal and ventral subtypes, involved in "
        "head oscillation, foraging movements and the head withdrawal reflex.",
        ("steering", "escape"),
    ),
    "RME": Annotation(
        M, "head motor neuron, GABAergic",
        "Inhibitory head motor neuron that limits the amplitude of head "
        "swings; loss of RME gives exaggerated head bending.",
        ("steering",),
    ),
    "RIV": Annotation(
        M, "ventral head motor neuron",
        "Active during the ventral turn that follows a reversal, and so a key "
        "part of the omega turn.",
        ("steering", "escape"),
    ),
    "SMB": Annotation(M, "head and neck motor neuron",
        "Sets the amplitude of head and neck bending; ablation gives "
        "exaggerated, loopy foraging movements.", ("steering",)),
    "SAA": Annotation(I, "head motor interneuron, proprioceptive",
        "Carries head-bend information back into the head circuit, and is one "
        "of the routes by which posture feeds into steering.", ("steering",)),
    "SAB": Annotation(M, "anterior body motor neuron",
        "Innervates anterior body muscle; proprioceptive role has been "
        "proposed.", ("locomotion",)),
    "SIA": Annotation(I, "head interneuron, poorly characterised",
        "Projects into the head; function not well established.", ()),
    "SIB": Annotation(I, "head interneuron, poorly characterised",
        "Projects into the head; function not well established.", ()),
    "URA": Annotation(M, "head motor neuron",
        "Innervates head muscle, contributing to head movement and to feeding "
        "related motions.", ("steering",)),
    "URB": Annotation(I, "head neuron, poorly characterised", "Ends in the head; function unclear.", ()),
    "URY": Annotation(S, "putative head mechanosensor",
        "Ciliated ending in the head, presumed mechanosensory.", ()),
    "RMF": Annotation(I, "ring interneuron, poorly characterised", "Function not well established.", ()),
    "RMG": Annotation(
        I, "hub interneuron for pheromone and aggregation",
        "Forms a gap junction hub with several sensory neurons including ASH "
        "and URX, and gates aggregation and pheromone responses.",
        ("aerotaxis",),
    ),
    "RMH": Annotation(I, "ring interneuron, poorly characterised", "Function not well established.", ()),
    "RIH": Annotation(I, "ring interneuron, serotonergic",
        "Hub interneuron in the nerve ring receiving from many head sensory "
        "cells.", ()),
    "RIF": Annotation(I, "interneuron, poorly characterised",
        "Receives from AIB among others; contributes to locomotory state.", ()),
    "RIG": Annotation(I, "interneuron, poorly characterised", "Function not well established.", ()),
    "RIR": Annotation(I, "interneuron, poorly characterised", "Function not well established.", ()),
    "RIP": Annotation(
        I, "the only link to the pharyngeal nervous system",
        "RIP is the single connection between the somatic nervous system and "
        "the otherwise self-contained pharyngeal nervous system. Note that this "
        "dataset contains no synapse crossing that boundary at all, so the 20 "
        "pharyngeal cells here are completely isolated.",
        ("feeding",),
    ),
    "ALA": Annotation(I, "sleep and stress-response neuron",
        "Mediates the quiescence that follows cellular stress, and is involved "
        "in sleep behaviour.", ()),
    # -- ventral cord motor neurons ---------------------------------------
    "VA": Annotation(
        M, "ventral body muscle motor neuron, backward",
        "One of the A-type motor neurons that contract ventral body muscle "
        "during *backward* locomotion. Driven by AVA. Twelve members spaced "
        "along the ventral cord, each innervating a few adjacent muscles.",
        ("locomotion",),
    ),
    "DA": Annotation(
        M, "dorsal body muscle motor neuron, backward",
        "Dorsal counterpart of VA: A-type, active during backward locomotion, "
        "driven by AVA.",
        ("locomotion",),
    ),
    "VB": Annotation(
        M, "ventral body muscle motor neuron, forward",
        "B-type motor neuron contracting ventral muscle during *forward* "
        "locomotion, driven by AVB. B-type cells also carry the proprioceptive "
        "signal that propagates the body wave from head to tail.",
        ("locomotion",),
    ),
    "DB": Annotation(
        M, "dorsal body muscle motor neuron, forward",
        "Dorsal counterpart of VB: B-type, forward locomotion, and one of the "
        "cells thought to sense body curvature and pass the bend rearwards.",
        ("locomotion",),
    ),
    "DD": Annotation(
        M, "GABAergic cross-inhibitory motor neuron",
        "Inhibits dorsal muscle while the ventral side contracts, producing the "
        "alternation that makes an undulation rather than a simultaneous "
        "squeeze. Famously rewires its polarity during larval development.",
        ("locomotion",),
    ),
    "VD": Annotation(
        M, "GABAergic cross-inhibitory motor neuron",
        "Mirror of DD: inhibits ventral muscle while the dorsal side contracts. "
        "Loss of DD and VD together gives the shrinker phenotype, where the worm "
        "contracts both sides at once.",
        ("locomotion",),
    ),
    "AS": Annotation(M, "dorsal motor neuron, unpaired",
        "Innervates dorsal muscle and lacks a ventral partner; contributes to "
        "locomotion with a role less well defined than the A and B types.",
        ("locomotion",)),
    "VC": Annotation(M, "egg-laying motor neuron",
        "Innervates vulval muscle and ventral body muscle, and participates in "
        "the egg-laying circuit alongside HSN.", ("egg-laying",)),
    "HSN": Annotation(
        MOD, "egg-laying command neuron, serotonergic",
        "The main driver of egg laying, and a major serotonin source. Its "
        "activity gates the active egg-laying state.",
        ("egg-laying", "feeding"),
    ),
    "PDA": Annotation(M, "tail motor neuron", "Innervates posterior body muscle.", ("locomotion",)),
    "PDB": Annotation(M, "tail motor neuron", "Innervates posterior body muscle.", ("locomotion",)),
    "AVL": Annotation(M, "GABAergic motor neuron, defecation",
        "Drives the expulsion step of the defecation motor programme, together "
        "with DVB.", ()),
    "DVB": Annotation(M, "GABAergic motor neuron, defecation",
        "Contracts the enteric muscles during expulsion.", ()),
    # -- tail and process-bundle interneurons ------------------------------
    "DVA": Annotation(
        I, "stretch-sensitive interneuron",
        "Mechanically sensitive to body stretch, and feeds proprioceptive "
        "information back into the locomotor circuit, modulating bending "
        "amplitude.",
        ("locomotion",),
    ),
    "DVC": Annotation(I, "tail interneuron",
        "Interneuron of the tail ganglion, implicated in coordinating backward "
        "locomotion.", ("locomotion",)),
    "PVT": Annotation(I, "guidance and interneuron",
        "Important as a navigational landmark for axon guidance during "
        "development; adult signalling role less clear.", ()),
    "PVP": Annotation(I, "process bundle interneuron",
        "Long processes running the length of the animal; involved in "
        "locomotory coordination.", ("locomotion",)),
    "PVQ": Annotation(I, "process bundle interneuron",
        "Runs the length of the ventral cord; associated with chemosensory and "
        "pheromone processing.", ()),
    "PVR": Annotation(I, "tail interneuron", "Function not well established.", ()),
    "PVN": Annotation(I, "interneuron, poorly characterised", "Function not well established.", ()),
    "PVW": Annotation(I, "tail interneuron", "Function not well established.", ()),
    "LUA": Annotation(I, "tail interneuron",
        "Relays from the posterior sensory cells PLM and PHC toward the command "
        "interneurons.", ("touch",)),
    "PHA": Annotation(S, "phasmid chemosensor",
        "Tail chemosensory neuron mediating avoidance of repellents, working "
        "with PHB.", ("avoidance",)),
    "PHB": Annotation(S, "phasmid chemosensor",
        "Tail chemosensor; modulates the ASH-driven avoidance response.",
        ("avoidance",)),
    "PHC": Annotation(S, "tail mechanosensor and thermosensor",
        "Responds to harsh touch of the tail and to warm temperature.", ("touch",)),
    "AVF": Annotation(MOD, "interneuron, mating and egg laying",
        "Involved in the egg-laying circuit and in male mating behaviour.",
        ("egg-laying",)),
    "AVG": Annotation(I, "ventral cord pioneer interneuron",
        "Pioneers the ventral cord during development; adult role modest.", ()),
    "AVH": Annotation(I, "interneuron, poorly characterised", "Function not well established.", ()),
    "AVJ": Annotation(I, "interneuron, poorly characterised", "Function not well established.", ()),
    "AVK": Annotation(MOD, "peptidergic interneuron",
        "FLP-neuropeptide expressing interneuron modulating locomotion and "
        "body posture.", ("locomotion",)),
    "AIM": Annotation(I, "interneuron, glutamatergic to serotonergic switch",
        "Receives from many head sensory neurons; involved in pheromone and "
        "learning related behaviour.", ()),
    "AIN": Annotation(I, "interneuron, poorly characterised", "Function not well established.", ()),
    "ADA": Annotation(I, "ring interneuron",
        "Nerve ring interneuron connected to the command layer; role not "
        "sharply defined.", ()),
    # -- pharyngeal nervous system ----------------------------------------
    "I1": Annotation(PH, "pharyngeal interneuron",
        "Part of the pharynx's own nervous system, relaying to the pumping "
        "motor neurons and able to speed pumping in response to light.", ("feeding",)),
    "I2": Annotation(PH, "pharyngeal interneuron",
        "Inhibits pumping in response to light and to toxins.", ("feeding",)),
    "I3": Annotation(PH, "pharyngeal interneuron", "Pharyngeal interneuron.", ("feeding",)),
    "I4": Annotation(PH, "pharyngeal interneuron", "Pharyngeal interneuron.", ("feeding",)),
    "I5": Annotation(PH, "pharyngeal interneuron",
        "Connects the pharyngeal isthmus to the terminal bulb; involved in "
        "regulating isthmus peristalsis.", ("feeding",)),
    "I6": Annotation(PH, "pharyngeal interneuron", "Pharyngeal interneuron.", ("feeding",)),
    "M1": Annotation(PH, "pharyngeal motor neuron",
        "Drives the spitting reflex used to expel harmful material.", ("feeding",)),
    "M2": Annotation(PH, "pharyngeal motor neuron",
        "Cholinergic motor neuron of the isthmus, speeding peristalsis.", ("feeding",)),
    "M3": Annotation(PH, "pharyngeal motor neuron",
        "Inhibitory motor neuron controlling the timing of pump relaxation, and "
        "so how efficiently bacteria are trapped.", ("feeding",)),
    "M4": Annotation(PH, "pharyngeal motor neuron",
        "Essential for isthmus peristalsis; without it the animal starves.",
        ("feeding",)),
    "M5": Annotation(PH, "pharyngeal motor neuron", "Pharyngeal motor neuron.", ("feeding",)),
    "MC": Annotation(PH, "pharyngeal pacemaker",
        "Sets the rate of pharyngeal pumping -- effectively the pacemaker of "
        "feeding.", ("feeding",)),
    "MI": Annotation(PH, "pharyngeal motor neuron and interneuron", "Pharyngeal neuron.", ("feeding",)),
    "NSM": Annotation(
        PH, "pharyngeal serotonergic sensory neuron",
        "Detects bacteria as they enter the pharynx and releases serotonin, "
        "which slows locomotion and promotes dwelling -- the signal that the "
        "worm has found food.",
        ("feeding",),
    ),
}

#: Classes whose members carry a numeric suffix rather than a positional one.
_NUMBERED = {"VA", "VB", "VC", "VD", "DA", "DB", "DD", "AS"}


def class_of(name: str) -> str:
    """Map a cell name onto its class, e.g. ``SMDDL`` -> ``SMD``.

    Tries the name itself first and strips progressively, so single-cell classes
    that happen to end in a positional letter (AQR, PQR, PVR, AVL) are not
    mangled the way a blind rule would mangle them.
    """
    candidates = [name]
    stripped = re.sub(r"[LR]$", "", name)
    if stripped != name:
        candidates.append(stripped)
        no_dv = re.sub(r"[DV]$", "", stripped)
        if no_dv != stripped:
            candidates.append(no_dv)
    no_dv_direct = re.sub(r"[DV]$", "", name)
    if no_dv_direct != name:
        candidates.append(no_dv_direct)
    for candidate in list(candidates):
        numberless = re.sub(r"\d+$", "", candidate)
        if numberless != candidate and numberless in _NUMBERED:
            candidates.append(numberless)
    for candidate in candidates:
        if candidate in ANNOTATIONS:
            return candidate
    return name


def lookup(name: str) -> Annotation | None:
    """Annotation for a cell, or None if its class is not covered."""
    return ANNOTATIONS.get(class_of(name))

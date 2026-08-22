"""Checks that the simulation still behaves the way it was tuned to.

Each check is a measurement, not an assertion about the code's shape, so this
doubles as the evidence for the claims made in the README.

    python validate.py
"""

from __future__ import annotations

import math

import numpy as np

from celegans import connectome as connectome_module
from celegans.body import BodyParams, Locomotion
from celegans.neural import NeuralNetwork
from celegans.sensing import SensoryEncoder
from celegans.simulation import Simulation
from celegans.world import FoodPatch, Obstacle, World

PASS, FAIL = "  ok  ", " FAIL "
results: list[bool] = []


def check(name: str, ok: bool, detail: str) -> None:
    results.append(ok)
    print(f"[{PASS if ok else FAIL}] {name}\n         {detail}")


def clamped_run(conn, encoder, signal, seed, ms=12000, nose=False, ablate=None):
    """Hold one sensory channel at a fixed level and read the command balance."""
    net = NeuralNetwork(conn, rng=np.random.default_rng(seed))
    loco = Locomotion(conn)
    loco.calibrate(net)
    if ablate is not None:
        net.set_ablated(ablate, True)
    ext = np.zeros(conn.n_neurons, dtype=np.float32)
    g = encoder.p.chemo_gain
    if signal > 0:
        ext[encoder.on_cells] = g * signal
    elif signal < 0:
        ext[encoder.off_cells] = g * -signal
        ext[encoder.pirouette_cells] += encoder.p.pirouette_gain * -signal
    if nose:
        ext[encoder.nose_cells] += encoder.p.nose_gain
    balances = []
    for t in range(ms):
        net.step(1.0, ext)
        if t > 4000 and t % 50 == 0:
            loco.update(net, 0.05, BodyParams())
            balances.append(loco.command_balance)
    return np.array(balances)


def main() -> None:
    conn = connectome_module.load()
    encoder = SensoryEncoder(conn)

    check(
        "connectome loads",
        conn.n_neurons == 299 and conn.n_muscles == 94 and int((conn.W != 0).sum()) == 2279,
        f"{conn.n_neurons} neurons, {conn.n_muscles} muscles, "
        f"{int((conn.W != 0).sum())} synapses, {int((conn.W < 0).sum())} inhibitory",
    )

    bf = conn.body_fraction()
    head_first = bf[conn.index["IL1L"]] < bf[conn.index["AVM"]] < bf[conn.index["PLML"]]
    check(
        "body axis runs head to tail",
        head_first,
        f"IL1L {bf[conn.index['IL1L']]:.2f} < AVM {bf[conn.index['AVM']]:.2f} "
        f"< PLML {bf[conn.index['PLML']]:.2f}",
    )

    # Head motor neurons must stay segregated, the steering readout depends on it.
    d = conn.muscle_dorsal
    smdd = conn.M[conn.idx("SMDDL", "SMDDR")]
    smdv = conn.M[conn.idx("SMDVL", "SMDVR")]
    check(
        "SMD dorsal/ventral segregation",
        smdd[:, d].sum() > 3 * smdd[:, ~d].sum() and smdv[:, ~d].sum() > smdv[:, d].sum(),
        f"SMDD dorsal {smdd[:, d].sum():.0f} vs ventral {smdd[:, ~d].sum():.0f}; "
        f"SMDV dorsal {smdv[:, d].sum():.0f} vs ventral {smdv[:, ~d].sum():.0f}",
    )

    rates = []
    for seed in range(4):
        net = NeuralNetwork(conn, rng=np.random.default_rng(seed))
        for _ in range(5000):
            net.step(1.0)
        rates.append(float(net.rate.mean()))
    check(
        "network settles at a plausible spontaneous rate",
        3.0 < np.mean(rates) < 25.0 and np.std(rates) < 6.0,
        f"{np.mean(rates):.1f} +/- {np.std(rates):.1f} Hz across 4 seeds",
    )

    threshold = Locomotion(conn).p.reversal_threshold
    quiet = np.concatenate([clamped_run(conn, encoder, 0.0, s) for s in range(2)])
    rising = np.concatenate([clamped_run(conn, encoder, 1.0, s) for s in range(2)])
    falling = np.concatenate([clamped_run(conn, encoder, -1.0, s) for s in range(2)])
    check(
        "falling gradient triggers reversal, rising one does not",
        np.mean(falling > threshold) > 0.6 and np.mean(rising > threshold) < 0.05,
        f"over threshold {threshold}: falling {np.mean(falling > threshold):.0%}, "
        f"flat {np.mean(quiet > threshold):.0%}, rising {np.mean(rising > threshold):.0%}",
    )

    touched = np.concatenate([clamped_run(conn, encoder, 0.0, s, nose=True) for s in range(2)])
    check(
        "nose touch drives the backward command",
        touched.mean() > threshold > quiet.mean(),
        f"balance {touched.mean():.3f} touched vs {quiet.mean():.3f} at rest "
        f"(threshold {threshold})",
    )

    # Ablation is the whole point of being able to switch circuits off: if it
    # works, removing the nose-touch cells has to cost the worm the response
    # they carry.
    lesioned = np.concatenate([
        clamped_run(conn, encoder, 0.0, s, nose=True, ablate=encoder.nose_cells)
        for s in range(2)
    ])
    check(
        "ablating the nose-touch circuit removes the escape response",
        touched.mean() > threshold > lesioned.mean(),
        f"balance with ASH/FLP/OLQ intact {touched.mean():.3f}, "
        f"ablated {lesioned.mean():.3f} (threshold {threshold})",
    )

    # Closed loop: can it actually find food?
    reached, eaten = 0, []
    for seed in range(5):
        world = World()
        world.food = [FoodPatch(x=800.0, y=360.0, amount=2.0, sigma=200.0)]
        world.obstacles = []
        sim = Simulation(world=world, seed=seed)
        closest = 1e9
        for i in range(60 * 180):
            sim.step(1 / 60)
            if i % 10 == 0:
                closest = min(closest, math.hypot(sim.head[0] - 800, sim.head[1] - 360))
        reached += closest < 14.0
        eaten.append(sim.world.eaten)
    check(
        "worm climbs a gradient and reaches the food",
        reached >= 4,
        f"reached the patch in {reached}/5 runs from 350 units away, "
        f"ate {np.mean(eaten):.2f} on average",
    )

    # Closed loop: does hitting a wall make it back off?
    world = World(width=1000.0, height=720.0)
    world.food = []
    world.obstacles = [Obstacle(x=520.0, y=360.0, radius=90.0)]
    sim = Simulation(world=world, seed=0)
    sim.body.__init__(x=200.0, y=360.0, heading=0.0, params=sim.p.body)
    contacts = reversals = 0
    was = False
    for _ in range(60 * 120):
        sim.step(1 / 60)
        if sim.sensory_current[sim.encoder.nose_cells].max() > 0:
            contacts += 1
        if sim.state.reversing and not was:
            reversals += 1
        was = sim.state.reversing
    check(
        "collisions happen and are followed by reversals",
        contacts > 0 and reversals > 0,
        f"{contacts} frames of nose contact, {reversals} reversals over 120 s",
    )

    print()
    if all(results):
        print(f"all {len(results)} checks passed")
    else:
        print(f"{sum(results)}/{len(results)} checks passed")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Run the C. elegans simulation with graphics.

    python main.py                 # interactive window
    python main.py --seed 3        # a different noise realisation
    python main.py --headless 300  # 300 s with no window, prints a summary
"""

from __future__ import annotations

import argparse
import math

import pygame

from celegans import Simulation, SimulationParams
from celegans.circuits import CIRCUITS_BY_KEY
from celegans.world import FoodPatch, Obstacle
from celegans.render import Renderer
from celegans.inspector import Inspector


def run_headless(seconds: float, seed: int | None) -> None:
    sim = Simulation(seed=seed)
    dt = 1.0 / 60.0
    steps = int(seconds / dt)
    reversals = 0
    was_reversing = False
    for _ in range(steps):
        sim.step(dt)
        if sim.state.reversing and not was_reversing:
            reversals += 1
        was_reversing = sim.state.reversing

    s = sim.summary()
    print(f"simulated {s['t']:.0f} s")
    print(f"  mean firing rate   {s['rate']:.1f} Hz")
    print(f"  food eaten         {s['eaten']:.3f}")
    print(f"  reversals          {reversals}  ({reversals / seconds * 60:.1f} per minute)")
    print(f"  final odour at nose {s['concentration']:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--scale", type=float, default=1.0, help="window scale")
    parser.add_argument(
        "--headless", type=float, metavar="SECONDS",
        help="run without a window for this many simulated seconds",
    )
    args = parser.parse_args()

    if args.headless:
        run_headless(args.headless, args.seed)
        return

    pygame.init()
    params = SimulationParams()
    sim = Simulation(params=params, seed=args.seed)
    renderer = Renderer(sim, scale=args.scale)
    inspector = Inspector(sim, renderer.size)
    clock = pygame.time.Clock()

    paused = False
    probing = False
    showing_map = False
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_TAB:
                    showing_map = not showing_map
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    sim = Simulation(params=params, seed=args.seed)
                    renderer.sim = sim
                    inspector.attach(sim)
                    renderer._odour_age = 999
                elif event.key == pygame.K_a:
                    sim.net.restore_all()
                    inspector.clear_experiment()
                elif event.key == pygame.K_d and showing_map:
                    inspector.toggle_selected("ablate")
                elif event.key == pygame.K_s and showing_map:
                    inspector.toggle_selected("stimulate")
                elif event.key == pygame.K_e and showing_map:
                    inspector.run_experiment()
                elif event.key == pygame.K_p:
                    probing = not probing
                    if not probing:
                        sim.release()
                elif event.unicode in CIRCUITS_BY_KEY:
                    circuit = CIRCUITS_BY_KEY[event.unicode]
                    sim.net.toggle_ablated(circuit.indices(sim.conn))
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    params.time_scale = min(8.0, params.time_scale * 1.5)
                elif event.key == pygame.K_MINUS:
                    params.time_scale = max(0.1, params.time_scale / 1.5)
            elif event.type == pygame.MOUSEBUTTONDOWN and showing_map:
                # Left click only inspects, so looking at a neuron never
                # perturbs the run; right click is the destructive one.
                if event.button == 3:
                    inspector.ablate_at(event.pos)
                else:
                    inspector.select(event.pos)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                if mx < renderer.world_w:
                    wx, wy = mx / renderer.scale, my / renderer.scale
                    keys = pygame.key.get_pressed()
                    if keys[pygame.K_x]:
                        sim.world.obstacles.append(Obstacle(wx, wy, 45.0))
                    else:
                        sim.world.food.append(
                            FoodPatch(x=wx, y=wy, amount=1.0, sigma=140.0)
                        )
                    renderer._odour_age = 999

        dt = clock.tick(60) / 1000.0
        if not paused:
            # Probe mode freezes the body so a lesion can be measured against a
            # steady network instead of against a moving animal.
            if probing:
                sim.probe_step(dt)
            else:
                sim.step(dt)
            # Keep the display average warm whether or not the map is on screen,
            # so a baseline taken right after switching to it is already valid.
            inspector.update(dt)
        inspector.probing = probing
        if showing_map:
            inspector.draw(renderer.screen)
            pygame.display.flip()
        else:
            renderer.draw(paused, clock.get_fps())

    pygame.quit()


if __name__ == "__main__":
    main()

"""Ties the connectome network, the body and the world into one loop."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import connectome as connectome_module
from .body import Body, BodyParams, Locomotion, LocomotionParams, LocomotionState
from .neural import NeuralNetwork, NeuralParams
from .sensing import SensoryEncoder, SensoryParams
from .world import World, default_world


@dataclass
class SimulationParams:
    #: Neural integration step, in milliseconds.  The behavioural loop runs at
    #: the frame rate and takes as many neural steps as fit inside a frame.
    neural_dt: float = 1.0
    #: Simulated seconds per second of wall clock.
    time_scale: float = 1.0
    #: Food ingested per second while the head is on a patch.
    feeding_rate: float = 0.06
    body: BodyParams = field(default_factory=BodyParams)
    neural: NeuralParams = field(default_factory=NeuralParams)
    sensory: SensoryParams = field(default_factory=SensoryParams)
    locomotion: LocomotionParams = field(default_factory=LocomotionParams)


class Simulation:
    """One worm in one world."""

    def __init__(
        self,
        world: World | None = None,
        params: SimulationParams | None = None,
        seed: int | None = None,
    ) -> None:
        self.p = params or SimulationParams()
        self.rng = np.random.default_rng(seed)
        self.conn = connectome_module.load()
        self.world = world if world is not None else default_world(self.rng)

        self.net = NeuralNetwork(self.conn, self.p.neural, rng=self.rng)
        self.encoder = SensoryEncoder(self.conn, self.p.sensory)
        self.locomotion = Locomotion(self.conn, self.p.locomotion)
        self.body = Body(
            x=self.world.width * 0.5,
            y=self.world.height * 0.75,
            heading=-math.pi / 5.0,
            params=self.p.body,
        )

        # Measure the network's resting head asymmetry before anything moves,
        # so steering starts from a true zero.
        self.locomotion.calibrate(self.net)

        self.time = 0.0
        self.state: LocomotionState = self.locomotion.state
        self.sensory_current = np.zeros(self.conn.n_neurons, dtype=np.float32)
        #: Recent head positions, for drawing the track.
        self.trail: list[tuple[float, float]] = []
        self._neural_debt = 0.0
        #: Sensory input held constant while in probe mode, or None.
        self._frozen: np.ndarray | None = None

    # -- main loop --------------------------------------------------------

    #: Largest slice of simulated time a single behavioural update may cover.
    #: The behavioural loop is full of forward-Euler filters whose shortest time
    #: constant is 0.25 s, so handing one of them a step larger than that makes
    #: it overshoot, and a step larger than twice that makes it diverge.  With
    #: ``time_scale`` at 8 a single slow frame used to reach dt = 0.8 s and blow
    #: the steering signal up to 1e19.  Slicing keeps every filter inside its
    #: stable range and makes behaviour independent of frame rate.
    MAX_SUBSTEP = 0.02

    def step(self, dt: float) -> None:
        """Advance the simulation by ``dt`` seconds of wall clock."""
        remaining = min(dt, 0.1) * self.p.time_scale
        while remaining > 0.0:
            slice_dt = min(remaining, self.MAX_SUBSTEP)
            self._step_once(slice_dt)
            remaining -= slice_dt

    def _step_once(self, dt: float) -> None:
        """One behavioural update, guaranteed small enough for the filters."""
        if dt <= 0.0:
            return
        self.time += dt

        points = self.body.points()
        self.sensory_current = self.encoder.sense(
            self.world,
            points,
            self.p.body.radius,
            dt,
            head_bend=math.sin(self.body.phase),
        )

        # Run the network for the milliseconds this frame represents, carrying
        # any fractional step over to the next frame so timing stays exact.
        self._neural_debt += dt * 1000.0
        n_steps = int(self._neural_debt / self.p.neural_dt)
        self._neural_debt -= n_steps * self.p.neural_dt
        for _ in range(min(n_steps, 200)):
            self.net.step(self.p.neural_dt, self.sensory_current)

        self.state = self.locomotion.update(self.net, dt, self.p.body)
        self.body.advance(
            dt,
            self.state.direction,
            self.state.speed,
            self.state.amplitude,
            self.state.steering,
        )
        self._resolve_collisions()

        head_x, head_y = self.body.head
        self.world.eat(head_x, head_y, self.p.feeding_rate * dt)

        self.trail.append((head_x, head_y))
        if len(self.trail) > 1400:
            del self.trail[:200]

    def probe_step(self, dt: float) -> None:
        """Run only the network, holding the sensory input where it was.

        A lesion cannot be read off a freely behaving worm: its own state varies
        as much between two moments as the lesion does, so about a fifth of all
        neurons differ by more than 1.5 Hz with nothing done at all.  Freezing
        the body and the world removes that, and the same comparison drops to
        0.3% of neurons -- which is what makes a before/after readable.
        """
        if self._frozen is None:
            self._frozen = self.sensory_current.copy()
        self._neural_debt += min(dt, 0.1) * 1000.0
        n_steps = int(self._neural_debt / self.p.neural_dt)
        self._neural_debt -= n_steps * self.p.neural_dt
        for _ in range(min(n_steps, 200)):
            self.net.step(self.p.neural_dt, self._frozen)

    def release(self) -> None:
        """Leave probe mode, so the next ``step`` senses the world again."""
        self._frozen = None

    def _resolve_collisions(self) -> None:
        """Push the body back out of any wall it has run into."""
        radius = self.p.body.radius
        px = py = 0.0
        for x, y in self.body.points():
            dx, dy = self.world.wall_penetration(float(x), float(y), radius)
            if abs(dx) > abs(px):
                px = dx
            if abs(dy) > abs(py):
                py = dy
        if px or py:
            self.body.push_out(px, py)

    # -- readouts ---------------------------------------------------------

    @property
    def head(self) -> tuple[float, float]:
        return self.body.head

    def summary(self) -> dict[str, float | str]:
        return {
            "t": self.time,
            "rate": float(self.net.rate.mean()),
            "muscle": float(self.net.muscle.mean()),
            "balance": self.locomotion.command_balance,
            "steering": self.state.steering,
            "dc_dt": self.encoder.dc_dt,
            "concentration": self.encoder.concentration,
            "eaten": self.world.eaten,
            "behaviour": (
                "omega turn"
                if self.state.omega_turn
                else "reversal"
                if self.state.reversing
                else "forward"
            ),
        }

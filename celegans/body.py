"""Body and locomotion model.

The worm crawls by following its own head: a *leading end* lays down a path
whose curvature oscillates, and the rest of the body is sampled from that path
at fixed arc-length offsets.  A body wave therefore falls out of the head
trajectory instead of having to be propagated segment by segment, which is a
good description of how a real worm crawls inside the groove it cuts in agar.

Curvature is accumulated per unit *distance* rather than per unit time, so the
shape of the track is independent of how fast the worm is moving::

    dphi/ds = amplitude * sin(psi) + steering
    dpsi/ds = 2*pi / wavelength

What the connectome supplies is the drive, not the oscillation: the LIF network
has no gap junctions and no proprioceptive feedback, so it cannot generate an
undulation on its own.  :class:`Locomotion` reads three things out of it —

* direction, from the AVA/AVD/AVE (backward) versus AVB/PVC (forward) command
  competition,
* undulation amplitude and speed, from overall body muscle activation,
* steering, from the dorsal/ventral asymmetry of the *head* muscles, which in
  this dataset are innervated exclusively by SMD and RMD.

Reversals and omega turns are triggered by the command competition, so the
behavioural sequence is the network's, while the rhythm is this module's.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class BodyParams:
    """Geometry and gait constants.  Lengths are in world units."""

    length: float = 110.0
    n_segments: int = 24
    #: Undulation wavelength as a fraction of body length (~0.65 in vivo).
    wavelength_frac: float = 0.65
    #: Peak track curvature, in radians per unit distance.
    amplitude: float = 0.085
    #: Crawling speed at full forward drive, in world units per second.
    speed: float = 26.0
    #: Spacing of stored path points; smaller is smoother but costs more.
    path_ds: float = 1.2
    #: Radius used for body-wall collision tests.
    radius: float = 2.6
    #: Extra curvature applied while an omega turn is in progress.
    omega_curvature: float = 0.075
    omega_duration: float = 1.6
    #: How long a reversal lasts once triggered, in seconds.
    reversal_duration: float = 2.2


@dataclass
class LocomotionState:
    """Everything the renderer and the sensory layer need to know."""

    direction: int = 1  # +1 forward, -1 backward
    speed: float = 0.0
    amplitude: float = 0.0
    steering: float = 0.0
    reversing: bool = False
    omega_turn: bool = False


class Body:
    """Kinematic worm body: a path-following chain of points."""

    def __init__(
        self,
        x: float,
        y: float,
        heading: float = 0.0,
        params: BodyParams | None = None,
    ) -> None:
        self.p = params or BodyParams()
        self.heading = heading
        self.phase = 0.0
        # Path points run tail-end -> head-end with spacing ``path_ds``.  Seed it
        # with a straight body so the worm starts fully formed.
        n = int(self.p.length / self.p.path_ds) + 2
        self.path: deque[tuple[float, float]] = deque(
            (x - math.cos(heading) * i * self.p.path_ds,
             y - math.sin(heading) * i * self.p.path_ds)
            for i in range(n - 1, -1, -1)
        )
        self.tail_heading = heading + math.pi
        #: Distance travelled but not yet committed to a path point.
        self._pending = 0.0

    @property
    def head(self) -> tuple[float, float]:
        return self.path[-1]

    @property
    def tail(self) -> tuple[float, float]:
        return self.path[0]

    def points(self, n: int | None = None) -> np.ndarray:
        """Body outline sampled head-first as an ``(n, 2)`` array."""
        n = n or self.p.n_segments + 1
        pts = np.asarray(self.path, dtype=np.float32)
        # The body occupies the last ``length`` of arc measured from the head.
        span = min(len(pts) - 1, int(self.p.length / self.p.path_ds))
        idx = np.linspace(len(pts) - 1, len(pts) - 1 - span, n)
        return pts[np.round(idx).astype(int)]

    def advance(
        self,
        dt: float,
        direction: int,
        speed: float,
        amplitude: float,
        steering: float,
    ) -> None:
        """Extend the path by ``speed * dt`` at the current leading end."""
        p = self.p
        distance = max(0.0, speed) * dt
        if distance <= 0.0:
            return

        n_body = int(p.length / p.path_ds) + 2
        k_wave = 2.0 * math.pi / max(1e-6, p.wavelength_frac * p.length)

        # Path points must be laid down at exactly ``path_ds`` spacing, whatever
        # distance a frame happens to cover, or the stored path — and with it
        # the body length — silently rescales with the frame rate.
        self._pending += distance
        while self._pending >= p.path_ds:
            step = p.path_ds
            self._pending -= step

            self.phase += k_wave * step
            curvature = amplitude * math.sin(self.phase) + steering

            if direction >= 0:
                self.heading += curvature * step
                hx, hy = self.path[-1]
                self.path.append(
                    (hx + math.cos(self.heading) * step,
                     hy + math.sin(self.heading) * step)
                )
                while len(self.path) > n_body:
                    self.path.popleft()
            else:
                # Reversing: the tail leads and lays down new path behind it.
                self.tail_heading += curvature * step
                tx, ty = self.path[0]
                self.path.appendleft(
                    (tx + math.cos(self.tail_heading) * step,
                     ty + math.sin(self.tail_heading) * step)
                )
                while len(self.path) > n_body:
                    self.path.pop()

        # Keep the trailing end's heading consistent for the next reversal.
        if direction >= 0:
            (x0, y0), (x1, y1) = self.path[1], self.path[0]
            self.tail_heading = math.atan2(y1 - y0, x1 - x0)
        else:
            (x0, y0), (x1, y1) = self.path[-2], self.path[-1]
            self.heading = math.atan2(y1 - y0, x1 - x0)

    def push_out(self, dx: float, dy: float) -> None:
        """Translate the whole body, used to resolve wall penetration."""
        self.path = deque((x + dx, y + dy) for x, y in self.path)


@dataclass
class LocomotionParams:
    """How strongly each neural readout is allowed to affect the gait."""

    #: Command-competition value above which a reversal is triggered.
    reversal_threshold: float = 0.085
    #: A reversal this deep is followed by an omega turn.
    omega_threshold: float = 0.130
    #: Refractory period between reversals, in seconds.
    reversal_refractory: float = 1.1
    #: Maps head muscle dorsal/ventral imbalance to track curvature.
    steering_gain: float = 0.085
    #: Muscle activation producing the nominal undulation amplitude.
    muscle_reference: float = 0.35
    #: Head segments counted as "head" for the steering readout.  In this
    #: dataset SMD and RMD innervate exactly segments 1-8.
    head_segments: int = 8
    #: Smoothing applied to steering, in seconds.
    tau_steer: float = 0.25
    #: Smoothing applied to the command competition, in seconds.
    tau_balance: float = 0.45


class Locomotion:
    """Reads the spiking network and turns it into gait parameters.

    The forward/backward decision is a competition between the two command
    interneuron pools, normalised so that a network-wide change in excitability
    cancels out and only the *relative* balance steers behaviour.
    """

    def __init__(self, conn, params: LocomotionParams | None = None) -> None:
        self.p = params or LocomotionParams()
        self.conn = conn
        self.backward_cells = conn.idx("AVAL", "AVAR", "AVDL", "AVDR", "AVEL", "AVER")
        self.forward_cells = conn.idx("AVBL", "AVBR", "PVCL", "PVCR")

        self.state = LocomotionState()
        self._reversal_left = 0.0
        self._omega_left = 0.0
        self._refractory_left = 0.0
        self._steer = 0.0
        #: Normalised AVA-vs-AVB competition, exposed for the HUD.
        self.command_balance = 0.0
        #: Resting head dorsal/ventral imbalance, removed before steering.  The
        #: head muscles are not innervated symmetrically -- dorsal wins by about
        #: 0.1 with no sensory input at all -- and left in, that constant offset
        #: curls the track into a circle of roughly one body length and swamps
        #: the chemotactic signal.  :meth:`calibrate` measures it.
        self.steer_bias = 0.0

    def calibrate(self, net, seconds: float = 8.0, dt: float = 1.0) -> float:
        """Measure the resting head imbalance of an unstimulated network.

        Runs a throwaway copy of the network state forward, so the caller's
        network is left untouched.
        """
        import copy

        probe = copy.deepcopy(net)
        samples = []
        n_steps = int(seconds * 1000.0 / dt)
        for i in range(n_steps):
            probe.step(dt)
            if i * dt > 2000.0 and i % 25 == 0:
                dorsal, ventral = probe.segment_activation()
                n = self.p.head_segments
                d, v = float(dorsal[:n].sum()), float(ventral[:n].sum())
                samples.append((d - v) / (d + v + 1e-6))
        self.steer_bias = float(np.mean(samples)) if samples else 0.0
        return self.steer_bias

    def update(self, net, dt: float, body_params: BodyParams) -> LocomotionState:
        """Advance the behavioural state machine by ``dt`` seconds."""
        p = self.p
        backward = net.group_rate(self.backward_cells)
        forward = net.group_rate(self.forward_cells)
        raw_balance = (backward - forward) / (backward + forward + 1e-6)
        # Decide on a smoothed balance.  The instantaneous value fluctuates with
        # a standard deviation of ~0.065 around a near-zero baseline, so testing
        # it directly every frame turns membrane noise into constant reversals.
        # min(1.0, ...) so an oversized dt settles straight onto the target
        # instead of overshooting it; the caller is not guaranteed to be the
        # sub-stepping loop in Simulation.
        self.command_balance += min(1.0, dt / p.tau_balance) * (
            raw_balance - self.command_balance
        )

        dorsal, ventral = net.segment_activation()
        n_head = p.head_segments
        head_d = float(dorsal[:n_head].sum())
        head_v = float(ventral[:n_head].sum())
        imbalance = (head_d - head_v) / (head_d + head_v + 1e-6) - self.steer_bias
        self._steer += min(1.0, dt / p.tau_steer) * (imbalance - self._steer)

        self._refractory_left = max(0.0, self._refractory_left - dt)
        self._reversal_left = max(0.0, self._reversal_left - dt)
        self._omega_left = max(0.0, self._omega_left - dt)

        if (
            self._reversal_left <= 0.0
            and self._refractory_left <= 0.0
            and self.command_balance > p.reversal_threshold
        ):
            self._reversal_left = body_params.reversal_duration
            self._refractory_left = body_params.reversal_duration + p.reversal_refractory
            if self.command_balance > p.omega_threshold:
                self._omega_left = body_params.omega_duration + body_params.reversal_duration

        reversing = self._reversal_left > 0.0
        # The omega turn is the deep ventral sweep *after* the reversal ends.
        omega = self._omega_left > 0.0 and not reversing

        activation = min(1.0, float(net.muscle.mean()) / max(1e-6, p.muscle_reference))

        steering = self._steer * p.steering_gain
        if omega:
            steering += math.copysign(body_params.omega_curvature, self._steer or 1.0)

        state = self.state
        state.direction = -1 if reversing else 1
        state.speed = body_params.speed * (0.35 + 0.65 * activation)
        state.amplitude = body_params.amplitude * (0.5 + 0.5 * activation)
        state.steering = steering
        state.reversing = reversing
        state.omega_turn = omega
        return state

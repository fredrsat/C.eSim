"""The environment the worm lives in.

A rectangular arena holding circular obstacles and bacterial food patches.  Each
patch emits a Gaussian odour plume; the summed field is what the chemosensory
neurons read.  The field is evaluated analytically rather than on a grid so the
gradient stays smooth no matter how slowly the worm moves, with a cached
low-resolution raster kept only for drawing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class FoodPatch:
    x: float
    y: float
    #: Remaining bacteria; also the peak of this patch's odour plume.
    amount: float = 1.0
    #: Standard deviation of the plume, in world units.
    sigma: float = 130.0
    #: Worm has to get this close to actually eat.
    radius: float = 14.0

    @property
    def depleted(self) -> bool:
        return self.amount <= 1e-3


@dataclass
class Obstacle:
    x: float
    y: float
    radius: float


@dataclass
class World:
    width: float = 1000.0
    height: float = 720.0
    food: list[FoodPatch] = field(default_factory=list)
    obstacles: list[Obstacle] = field(default_factory=list)
    #: Total food consumed so far, in patch-amount units.
    eaten: float = 0.0

    # -- chemical field ---------------------------------------------------

    def concentration(self, x: float, y: float) -> float:
        """Odour concentration at a point, summed over all patches."""
        total = 0.0
        for patch in self.food:
            if patch.depleted:
                continue
            dx = x - patch.x
            dy = y - patch.y
            total += patch.amount * math.exp(
                -(dx * dx + dy * dy) / (2.0 * patch.sigma * patch.sigma)
            )
        return total

    def concentration_field(self, nx: int = 100, ny: int = 72) -> np.ndarray:
        """Raster of the odour field, ``(ny, nx)``, for rendering."""
        xs = np.linspace(0.0, self.width, nx, dtype=np.float32)
        ys = np.linspace(0.0, self.height, ny, dtype=np.float32)
        gx, gy = np.meshgrid(xs, ys)
        field_ = np.zeros_like(gx)
        for patch in self.food:
            if patch.depleted:
                continue
            d2 = (gx - patch.x) ** 2 + (gy - patch.y) ** 2
            field_ += patch.amount * np.exp(-d2 / (2.0 * patch.sigma**2))
        return field_

    # -- interaction ------------------------------------------------------

    def eat(self, x: float, y: float, rate: float) -> float:
        """Consume food near the head, returning how much was ingested."""
        ingested = 0.0
        for patch in self.food:
            if patch.depleted:
                continue
            if math.hypot(x - patch.x, y - patch.y) <= patch.radius:
                bite = min(patch.amount, rate)
                patch.amount -= bite
                ingested += bite
        self.eaten += ingested
        return ingested

    def wall_penetration(self, x: float, y: float, radius: float) -> tuple[float, float]:
        """Vector pushing a circle at ``(x, y)`` back out of any wall it overlaps.

        Returns ``(0.0, 0.0)`` when the circle is clear of every boundary and
        obstacle.
        """
        px = py = 0.0
        if x - radius < 0.0:
            px += radius - x
        elif x + radius > self.width:
            px += self.width - radius - x
        if y - radius < 0.0:
            py += radius - y
        elif y + radius > self.height:
            py += self.height - radius - y

        for obs in self.obstacles:
            dx = x - obs.x
            dy = y - obs.y
            dist = math.hypot(dx, dy)
            overlap = obs.radius + radius - dist
            if overlap > 0.0:
                if dist < 1e-6:
                    dx, dy, dist = 1.0, 0.0, 1.0
                px += dx / dist * overlap
                py += dy / dist * overlap
        return px, py

    def touching(self, x: float, y: float, radius: float) -> bool:
        px, py = self.wall_penetration(x, y, radius)
        return px != 0.0 or py != 0.0


def default_world(rng: np.random.Generator | None = None) -> World:
    """An arena with a few food patches and obstacles to bump into."""
    rng = rng or np.random.default_rng()
    world = World()
    world.food = [
        FoodPatch(x=790.0, y=180.0, amount=1.0, sigma=150.0),
        FoodPatch(x=250.0, y=560.0, amount=0.8, sigma=130.0),
        FoodPatch(x=560.0, y=350.0, amount=0.6, sigma=110.0),
    ]
    world.obstacles = [
        Obstacle(x=430.0, y=180.0, radius=52.0),
        Obstacle(x=700.0, y=520.0, radius=64.0),
        Obstacle(x=180.0, y=300.0, radius=40.0),
    ]
    return world

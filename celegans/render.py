"""Pygame rendering: the arena on the left, the nervous system on the right."""

from __future__ import annotations

import math

import numpy as np
import pygame

from .simulation import Simulation

PANEL_W = 400

#: Odour concentration drawn as full brightness.  Patch amounts start at <= 1.
ODOUR_REFERENCE = 1.0

#: Per-segment muscle activation drawn as a full-height bar.  Measured p95 is
#: 3.2 over a long run.  Fixed for the same reason as the odour reference: a
#: self-scaling chart hides whether the worm is contracting hard or barely at
#: all, and shows only the dorsal/ventral pattern.
MUSCLE_REFERENCE = 3.2

BG = (14, 16, 22)
PANEL_BG = (19, 22, 30)
GRID = (30, 35, 46)
TEXT = (196, 204, 220)
DIM = (110, 120, 140)
ACCENT = (255, 176, 74)
WORM = (236, 224, 200)
WORM_DARK = (150, 132, 104)
FOOD = (126, 200, 118)
OBSTACLE = (54, 60, 74)
DORSAL = (108, 176, 255)
VENTRAL = (255, 122, 140)
REVERSE = (255, 96, 96)
OMEGA = (255, 196, 64)


def _lerp(a, b, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(x + (y - x) * t) for x, y in zip(a, b))


class Renderer:
    """Draws a :class:`Simulation` into a pygame window."""

    def __init__(self, sim: Simulation, scale: float = 1.0) -> None:
        self.sim = sim
        self.scale = scale
        self.world_w = int(sim.world.width * scale)
        self.world_h = int(sim.world.height * scale)
        self.size = (self.world_w + PANEL_W, self.world_h)

        self.screen = pygame.display.set_mode(self.size)
        pygame.display.set_caption("C. elegans — connectome simulation")
        self.font = pygame.font.SysFont("menlo,dejavusansmono,monospace", 13)
        self.font_small = pygame.font.SysFont("menlo,dejavusansmono,monospace", 11)
        self.font_big = pygame.font.SysFont("menlo,dejavusansmono,monospace", 17, bold=True)

        self._odour_surface: pygame.Surface | None = None
        self._odour_age = 999
        self._neuron_xy = self._layout_neurons()

    # -- setup ------------------------------------------------------------

    def _layout_neurons(self) -> np.ndarray:
        """Project the real neuron positions into the side panel."""
        pos = self.sim.conn.positions
        ap, dv = pos[:, 1], pos[:, 2]
        x = (ap - ap.min()) / np.ptp(ap)
        y = (dv - dv.min()) / np.ptp(dv)
        return np.stack([x, y], axis=1)

    # -- world ------------------------------------------------------------

    def _to_screen(self, x: float, y: float) -> tuple[int, int]:
        return int(x * self.scale), int(y * self.scale)

    def _draw_odour(self) -> None:
        self._odour_age += 1
        if self._odour_surface is None or self._odour_age > 30:
            field = self.sim.world.concentration_field(100, 72)
            # Fixed reference, not the current maximum.  Normalising to the peak
            # would keep the plume at full brightness however much of the patch
            # the worm has eaten, so the display would never show food running
            # out -- the same trap as an auto-scaling activity map.
            norm = np.clip(field / ODOUR_REFERENCE, 0.0, 1.0) ** 0.65
            rgb = np.zeros((*norm.shape, 3), dtype=np.uint8)
            rgb[..., 0] = (BG[0] + norm * 26).astype(np.uint8)
            rgb[..., 1] = (BG[1] + norm * 70).astype(np.uint8)
            rgb[..., 2] = (BG[2] + norm * 34).astype(np.uint8)
            surf = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
            self._odour_surface = pygame.transform.smoothscale(
                surf, (self.world_w, self.world_h)
            )
            self._odour_age = 0
        self.screen.blit(self._odour_surface, (0, 0))

    def _draw_world(self) -> None:
        sim = self.sim
        self._draw_odour()

        for obs in sim.world.obstacles:
            cx, cy = self._to_screen(obs.x, obs.y)
            r = int(obs.radius * self.scale)
            pygame.draw.circle(self.screen, OBSTACLE, (cx, cy), r)
            pygame.draw.circle(self.screen, (74, 82, 100), (cx, cy), r, 2)

        for patch in sim.world.food:
            if patch.depleted:
                continue
            cx, cy = self._to_screen(patch.x, patch.y)
            r = max(3, int(patch.radius * self.scale * (0.4 + 0.6 * patch.amount)))
            pygame.draw.circle(self.screen, FOOD, (cx, cy), r)
            pygame.draw.circle(self.screen, (180, 240, 170), (cx, cy), r, 1)

        if len(sim.trail) > 2:
            pts = [self._to_screen(x, y) for x, y in sim.trail[-900:]]
            pygame.draw.lines(self.screen, (44, 52, 66), False, pts, 1)

        self._draw_worm()

    def _draw_worm(self) -> None:
        sim = self.sim
        pts = sim.body.points(41)
        screen_pts = [self._to_screen(float(x), float(y)) for x, y in pts]
        n = len(screen_pts)

        colour = WORM
        if sim.state.omega_turn:
            colour = OMEGA
        elif sim.state.reversing:
            colour = REVERSE

        # Taper the body from head to tail.
        for i in range(n - 1):
            t = i / (n - 1)
            width = max(1, int((5.5 - 3.6 * t) * self.scale))
            shade = _lerp(colour, WORM_DARK, t * 0.75)
            pygame.draw.line(self.screen, shade, screen_pts[i], screen_pts[i + 1], width)

        hx, hy = screen_pts[0]
        pygame.draw.circle(self.screen, colour, (hx, hy), max(2, int(3.4 * self.scale)))
        pygame.draw.circle(self.screen, (40, 30, 26), (hx, hy), max(1, int(1.4 * self.scale)))

    # -- side panel -------------------------------------------------------

    def _draw_panel(self) -> None:
        sim = self.sim
        x0 = self.world_w
        pygame.draw.rect(self.screen, PANEL_BG, (x0, 0, PANEL_W, self.world_h))
        pygame.draw.line(self.screen, GRID, (x0, 0), (x0, self.world_h))

        y = 14
        s = sim.summary()
        behaviour = str(s["behaviour"])
        bcol = OMEGA if behaviour == "omega turn" else REVERSE if behaviour == "reversal" else FOOD
        self.screen.blit(self.font_big.render(behaviour.upper(), True, bcol), (x0 + 16, y))
        y += 26
        self.screen.blit(
            self.font_small.render(
                f"t = {s['t']:6.1f} s     food eaten = {s['eaten']:.2f}", True, DIM
            ),
            (x0 + 16, y),
        )
        y += 24

        y = self._draw_neuron_map(x0, y)
        y = self._draw_muscles(x0, y)
        y = self._draw_groups(x0, y)
        self._draw_signals(x0, y)

    def _draw_neuron_map(self, x0: int, y: int) -> int:
        sim = self.sim
        w, h = PANEL_W - 32, 128
        rect = pygame.Rect(x0 + 16, y, w, h)
        pygame.draw.rect(self.screen, (24, 28, 38), rect)
        pygame.draw.rect(self.screen, GRID, rect, 1)
        self.screen.blit(
            self.font_small.render("299 neurons — head left, firing rate", True, DIM),
            (x0 + 18, y + 2),
        )

        rate = sim.net.rate
        norm = np.clip(rate / 45.0, 0.0, 1.0)
        for i in range(len(norm)):
            nx = rect.x + 8 + self._neuron_xy[i, 0] * (w - 16)
            ny = rect.y + 18 + self._neuron_xy[i, 1] * (h - 30)
            level = float(norm[i])
            colour = _lerp((52, 60, 78), ACCENT, level)
            radius = 1 + int(level * 2.4)
            pygame.draw.circle(self.screen, colour, (int(nx), int(ny)), radius)
        return y + h + 12

    def _draw_muscles(self, x0: int, y: int) -> int:
        sim = self.sim
        dorsal, ventral = sim.net.segment_activation()
        n = len(dorsal)
        w = PANEL_W - 32
        bar_w = w / n
        mid = y + 46
        peak = MUSCLE_REFERENCE

        self.screen.blit(
            self.font_small.render("body wall muscles — dorsal / ventral", True, DIM),
            (x0 + 16, y),
        )
        for i in range(n):
            bx = x0 + 16 + i * bar_w
            hd = int(dorsal[i] / peak * 28)
            hv = int(ventral[i] / peak * 28)
            pygame.draw.rect(
                self.screen, DORSAL, (bx, mid - hd, max(1, bar_w - 1.5), hd)
            )
            pygame.draw.rect(
                self.screen, VENTRAL, (bx, mid, max(1, bar_w - 1.5), hv)
            )
        pygame.draw.line(self.screen, GRID, (x0 + 16, mid), (x0 + 16 + w, mid))
        self.screen.blit(self.font_small.render("head", True, DIM), (x0 + 16, mid + 32))
        self.screen.blit(self.font_small.render("tail", True, DIM), (x0 + w - 8, mid + 32))
        return mid + 50

    def _draw_groups(self, x0: int, y: int) -> int:
        sim = self.sim
        conn = sim.conn
        groups = [
            ("AVA/AVD/AVE   backward cmd", sim.locomotion.backward_cells, REVERSE),
            ("AVB/PVC       forward cmd", sim.locomotion.forward_cells, FOOD),
            ("AIY           chemo interneuron", conn.idx("AIYL", "AIYR"), DORSAL),
            ("RIA           steering", conn.idx("RIAL", "RIAR"), DORSAL),
            ("SMD/RMD       head motor", conn.idx(
                "SMDDL", "SMDDR", "SMDVL", "SMDVR", "RMDDL", "RMDDR", "RMDVL", "RMDVR"
            ), ACCENT),
            ("ASH/FLP/OLQ   nose touch", sim.encoder.nose_cells, (200, 150, 255)),
        ]
        w = PANEL_W - 32
        for label, idx, colour in groups:
            rate = sim.net.group_rate(idx)
            frac = min(1.0, rate / 45.0)
            # Label above the bar, not on top of it -- pale text over a
            # saturated fill is unreadable exactly when the group is busiest.
            self.screen.blit(self.font_small.render(label, True, TEXT), (x0 + 16, y))
            value = self.font_small.render(f"{rate:5.1f} Hz", True, DIM)
            self.screen.blit(value, (x0 + 16 + w - value.get_width(), y))
            pygame.draw.rect(self.screen, (30, 35, 46), (x0 + 16, y + 14, w, 5))
            pygame.draw.rect(self.screen, colour, (x0 + 16, y + 14, int(w * frac), 5))
            y += 26
        return y + 6

    def _draw_signals(self, x0: int, y: int) -> None:
        sim = self.sim
        s = sim.summary()
        w = PANEL_W - 32

        def meter(label: str, low: str, high: str, value: float, span: float, colour) -> None:
            nonlocal y
            self.screen.blit(self.font_small.render(label, True, TEXT), (x0 + 16, y))
            shown = self.font_small.render(f"{value:+.4f}", True, DIM)
            self.screen.blit(shown, (x0 + 16 + w - shown.get_width(), y))
            bar = pygame.Rect(x0 + 16, y + 15, w, 7)
            pygame.draw.rect(self.screen, (30, 35, 46), bar)
            centre = bar.x + w // 2
            pygame.draw.line(self.screen, GRID, (centre, bar.y), (centre, bar.bottom))
            frac = max(-1.0, min(1.0, value / span))
            length = int(abs(frac) * (w // 2))
            if frac >= 0:
                pygame.draw.rect(self.screen, colour, (centre, bar.y, length, 7))
            else:
                pygame.draw.rect(self.screen, colour, (centre - length, bar.y, length, 7))
            self.screen.blit(self.font_small.render(low, True, (86, 94, 112)), (x0 + 16, y + 24))
            right = self.font_small.render(high, True, (86, 94, 112))
            self.screen.blit(right, (x0 + 16 + w - right.get_width(), y + 24))
            y += 44

        meter("dC/dt at nose", "worse", "better", float(s["dc_dt"]), 0.02, FOOD)
        meter("command balance", "forward AVB", "backward AVA",
              float(s["balance"]), 0.15, REVERSE)
        meter("steering", "ventral", "dorsal", float(s["steering"]), 0.06, DORSAL)

        for line in (
            "tab neuron map · space pause · r reset · +/- speed",
            "click add food · x+click add obstacle · 1-8 ablate circuit · a restore",
        ):
            self.screen.blit(self.font_small.render(line, True, (76, 84, 102)), (x0 + 16, y))
            y += 15

    # -- entry point ------------------------------------------------------

    def draw(self, paused: bool, fps: float) -> None:
        self.screen.fill(BG)
        self._draw_world()
        self._draw_panel()
        if paused:
            label = self.font_big.render("PAUSED", True, ACCENT)
            self.screen.blit(label, (16, 12))
        else:
            self.screen.blit(
                self.font_small.render(
                    f"{fps:4.0f} fps   x{self.sim.p.time_scale:.1f}", True, (60, 68, 86)
                ),
                (16, 14),
            )
        pygame.display.flip()

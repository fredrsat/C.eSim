"""Full-window neuron inspector.

The top half is a map of all 299 neurons, each labelled and shaded by firing
rate, grouped sensory / interneuron / motor and ordered head to tail.  The bottom
half explains whichever neuron is selected, in three columns that are kept
deliberately separate:

``LITERATURE``
    what the cell is believed to do, from :mod:`celegans.annotations`.  This is
    external knowledge, not something the dataset shows.
``MEASURED``
    structural facts computed from the connectome by :mod:`celegans.analysis`.
``WHY ACTIVE NOW``
    a live decomposition of the current arriving at the cell this millisecond,
    naming the presynaptic partners actually driving it.

Selecting is separate from perturbing, so inspecting a cell never changes the
simulation: left click selects, right click ablates, and ``s`` forces a cell on.
"""

from __future__ import annotations

import numpy as np
import pygame

from .analysis import Analysis
from .annotations import class_of, lookup
from .circuits import CIRCUITS, classify
from .experiment import Effect, measure
from .simulation import Simulation

BG = (12, 14, 19)
PANEL = (17, 20, 27)
CELL_BG = (28, 33, 44)
GRID = (44, 51, 66)
TEXT = (206, 214, 230)
DIM = (122, 132, 152)
FAINT = (86, 94, 112)
HEAT = (255, 176, 74)
HOT = (255, 238, 205)
ABLATED = (96, 40, 44)
STIM = (86, 140, 220)
EXC = (126, 200, 118)
INH = (255, 122, 140)
PRE = (108, 176, 255)
POST = (255, 176, 74)
SECTION = {
    "sensory": (126, 200, 118),
    "interneuron": (108, 176, 255),
    "motor": (255, 140, 120),
}

CELL_W = 62
CELL_H = 17
GAP = 2

#: Ceiling of the colour scale, in Hz.  This is deliberately *fixed* rather than
#: normalised to the current activity: a scale that tracks the network makes
#: every surviving neuron look hotter the moment you ablate the busiest ones,
#: which reads as the rest of the network compensating when nothing of the sort
#: has happened.  Each cell prints its rate anyway, so nothing is lost.
SCALE_HZ = 40.0
COOLER = (96, 150, 230)


def _lerp(a, b, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(x + (y - x) * t) for x, y in zip(a, b))


def _fmt(value: float) -> str:
    """Signed current, with enough precision to stay informative when small."""
    return f"{value:+.2f}" if abs(value) < 10.0 else f"{value:+.1f}"


def _wrap(font, text: str, width: int) -> list[str]:
    words, lines, line = text.split(), [], ""
    for word in words:
        trial = f"{line} {word}".strip()
        if font.size(trial)[0] <= width:
            line = trial
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


class Inspector:
    """Neuron map plus a detail panel for the selected cell."""

    def __init__(self, sim: Simulation, size: tuple[int, int]) -> None:
        self.sim = sim
        self.size = size
        self.analysis = Analysis(sim.conn)
        self.font = pygame.font.SysFont("menlo,dejavusansmono,monospace", 10)
        self.font_mid = pygame.font.SysFont("menlo,dejavusansmono,monospace", 12)
        self.font_big = pygame.font.SysFont(
            "menlo,dejavusansmono,monospace", 16, bold=True
        )
        self._rects: list[tuple[pygame.Rect, int]] = []
        self.selected: int | None = None
        self._order = self._build_order()
        self._panel_top = 0
        #: Slowly averaged firing rate, used for the map's absolute view.
        #: ``net.rate`` cannot be used directly: its time constant is 60 ms, so a
        #: single spike moves it by 16.7 Hz and 93% of cells differ from any
        #: instantaneous snapshot by more than 0.5 Hz with nothing changed at
        #: all.  Averaging over a few seconds brings that down to ~0.7 Hz, which
        #: is what makes a before/after comparison mean anything.
        self.smoothed = self.sim.net.rate.copy()
        #: Result of the last ablation experiment, or None.  This replaces the
        #: earlier hand-managed baseline: timing a snapshot yourself was fiddly
        #: and, in a freely behaving worm, measured mostly the animal moving on.
        self.effect: Effect | None = None
        #: Set by the host loop when the body is frozen for a clean measurement.
        self.probing = False

    #: Averaging window for the displayed rate, in seconds.
    TAU_DISPLAY = 3.0
    #: Changes smaller than this are treated as noise and not shown.
    DELTA_FLOOR = 1.5

    def attach(self, sim: Simulation) -> None:
        """Point the inspector at a new simulation, e.g. after a reset."""
        self.sim = sim
        self.analysis = Analysis(sim.conn)
        self.smoothed = sim.net.rate.copy()
        self.effect = None
        self.selected = None

    def update(self, dt: float) -> None:
        """Advance the display average.  Call every frame, map visible or not."""
        alpha = min(1.0, dt / self.TAU_DISPLAY)
        self.smoothed += alpha * (self.sim.net.rate - self.smoothed)

    def run_experiment(self) -> None:
        """Measure what the current ablation does, as a paired experiment."""
        self.effect = measure(self.sim)

    def clear_experiment(self) -> None:
        self.effect = None

    def _build_order(self) -> list[tuple[str, list[int]]]:
        conn = self.sim.conn
        labels = classify(conn)
        fraction = conn.body_fraction()
        groups: dict[str, list[int]] = {"sensory": [], "interneuron": [], "motor": []}
        for i, name in enumerate(conn.neurons):
            groups[labels[name]].append(i)
        for key in groups:
            groups[key].sort(key=lambda i: (fraction[i], conn.neurons[i]))
        return [(k, groups[k]) for k in ("sensory", "interneuron", "motor")]

    # -- interaction ------------------------------------------------------

    def neuron_at(self, pos: tuple[int, int]) -> int | None:
        for rect, index in self._rects:
            if rect.collidepoint(pos):
                return index
        return None

    def select(self, pos: tuple[int, int]) -> None:
        index = self.neuron_at(pos)
        if index is not None:
            self.selected = index

    def ablate_at(self, pos: tuple[int, int]) -> None:
        index = self.neuron_at(pos)
        if index is None:
            return
        self.selected = index
        net = self.sim.net
        net.set_ablated(np.array([index]), not bool(net.ablated[index]))

    def toggle_selected(self, what: str) -> None:
        if self.selected is None:
            return
        idx = np.array([self.selected])
        net = self.sim.net
        if what == "ablate":
            net.set_ablated(idx, not bool(net.ablated[self.selected]))
        else:
            net.set_stimulated(idx, not bool(net.stimulated[self.selected]))

    # -- drawing ----------------------------------------------------------

    def draw(self, screen: pygame.Surface) -> None:
        screen.fill(BG)
        self._rects.clear()
        self._draw_header(screen)
        y = self._draw_map(screen, self._panel_top)
        self._draw_detail(screen, max(y, self.size[1] - 300))

    def _draw_header(self, screen) -> None:
        sim = self.sim
        w = self.size[0]
        net = sim.net
        screen.blit(self.font_big.render("NEURON MAP", True, TEXT), (16, 8))
        mode = (
            "showing the EFFECT OF THE ABLATION (e to clear)"
            if self.effect is not None
            else f"colour 0-{SCALE_HZ:.0f} Hz fixed   ·   ablate something, then "
                 "press e to measure what it did"
        )
        if self.probing:
            mode = "PROBE — body frozen   ·   " + mode
        screen.blit(
            self.font_mid.render(
                f"{int(net.ablated.sum())} ablated   "
                f"{int(net.stimulated.sum())} forced on   "
                f"{sim.summary()['behaviour']}   ·   {mode}",
                True, HEAT if self.effect is not None else DIM,
            ),
            (160, 11),
        )

        x, y = 16, 32
        for circuit in CIRCUITS:
            indices = circuit.indices(sim.conn)
            on = indices.size and bool(net.ablated[indices].all())
            text = self.font.render(
                f"{circuit.key} {circuit.name}", True, (210, 120, 124) if on else DIM
            )
            box = pygame.Rect(x, y, text.get_width() + 12, 16)
            if box.right > w - 16:
                x, y = 16, y + 19
                box = pygame.Rect(x, y, text.get_width() + 12, 16)
            pygame.draw.rect(screen, ABLATED if on else CELL_BG, box)
            pygame.draw.rect(screen, GRID, box, 1)
            screen.blit(text, (box.x + 6, box.y + 3))
            x = box.right + 5
        self._panel_top = y + 24

    def _draw_map(self, screen, y: int) -> int:
        sim = self.sim
        w = self.size[0]
        rate = self.smoothed
        peak = SCALE_HZ
        pre_mask = post_mask = None
        if self.selected is not None:
            pre_mask, post_mask = self.analysis.partner_mask(self.selected)

        cols = max(1, (w - 24) // (CELL_W + GAP))
        for label, indices in self._order:
            screen.blit(
                self.font.render(f"{label.upper()} ({len(indices)})", True, SECTION[label]),
                (16, y),
            )
            y += 14
            for n, index in enumerate(indices):
                col, row = n % cols, n // cols
                rect = pygame.Rect(
                    16 + col * (CELL_W + GAP), y + row * (CELL_H + GAP), CELL_W, CELL_H
                )
                self._rects.append((rect, index))
                self._draw_cell(screen, rect, index, float(rate[index]), peak,
                                pre_mask, post_mask)
            y += ((len(indices) + cols - 1) // cols) * (CELL_H + GAP) + 8
        return y

    def _draw_cell(self, screen, rect, index, rate, peak, pre_mask, post_mask) -> None:
        net = self.sim.net
        name = self.sim.conn.neurons[index]

        if net.ablated[index]:
            pygame.draw.rect(screen, ABLATED, rect)
            colour = (182, 134, 136)
        elif net.stimulated[index]:
            pygame.draw.rect(screen, STIM, rect)
            colour = (232, 242, 255)
        elif self.effect is not None:
            # Diverging around the control arm: warm means the lesion made this
            # neuron busier, cool means quieter.  Anything inside the noise
            # floor stays neutral rather than being coloured in.
            delta = float(self.effect.delta[index])
            if abs(delta) < self.DELTA_FLOOR:
                pygame.draw.rect(screen, CELL_BG, rect)
                colour = TEXT
            else:
                level = min(1.0, (abs(delta) - self.DELTA_FLOOR) / 10.0)
                target = HEAT if delta > 0 else COOLER
                pygame.draw.rect(screen, _lerp(CELL_BG, target, level), rect)
                colour = HOT if level > 0.55 else TEXT
        else:
            level = min(1.0, rate / peak)
            pygame.draw.rect(screen, _lerp(CELL_BG, HEAT, level), rect)
            colour = HOT if level > 0.55 else TEXT

        # Partner highlighting: who talks to the selected cell, and who it talks to.
        if index == self.selected:
            pygame.draw.rect(screen, (255, 255, 255), rect, 2)
        elif pre_mask is not None and pre_mask[index]:
            pygame.draw.rect(screen, PRE, rect, 1)
        elif post_mask is not None and post_mask[index]:
            pygame.draw.rect(screen, POST, rect, 1)

        screen.blit(self.font.render(name[:7], True, colour), (rect.x + 3, rect.y + 3))
        if net.ablated[index]:
            return
        if self.effect is not None:
            delta = float(self.effect.delta[index])
            label = f"{delta:+.0f}" if abs(delta) >= self.DELTA_FLOOR else "·"
        elif rate >= 1.0:
            label = f"{rate:.0f}"
        else:
            return
        shown = self.font.render(label, True, colour)
        screen.blit(shown, (rect.right - 3 - shown.get_width(), rect.y + 3))

    # -- detail panel -----------------------------------------------------

    def _draw_detail(self, screen, y: int) -> None:
        w, h = self.size
        pygame.draw.rect(screen, PANEL, (0, y, w, h - y))
        pygame.draw.line(screen, GRID, (0, y), (w, y))
        if self.effect is not None:
            self._draw_effect(screen, y)
            return
        if self.selected is None:
            screen.blit(
                self.font_mid.render(
                    "click a neuron to inspect it   ·   right click ablates   ·   "
                    "s forces it on   ·   1-8 ablate a circuit   ·   "
                    "e measures what the ablation did   ·   a restore all",
                    True, FAINT,
                ),
                (16, y + 14),
            )
            return

        index = self.selected
        col_w = (w - 48) // 3
        self._draw_literature(screen, 16, y + 10, col_w, index)
        self._draw_measured(screen, 24 + col_w, y + 10, col_w, index)
        self._draw_why(screen, 32 + 2 * col_w, y + 10, col_w, index)

    def _heading(self, screen, x, y, text, colour) -> int:
        screen.blit(self.font.render(text, True, colour), (x, y))
        return y + 14

    def _draw_literature(self, screen, x, y, w, index) -> None:
        name = self.sim.conn.neurons[index]
        note = lookup(name)
        y = self._heading(screen, x, y, "LITERATURE — external knowledge, not from the dataset", FAINT)

        header = f"{name}   class {class_of(name)}"
        screen.blit(self.font_mid.render(header, True, TEXT), (x, y))
        y += 17
        if note is None:
            screen.blit(self.font.render("no annotation for this class", True, DIM), (x, y))
            return
        screen.blit(self.font.render(f"{note.kind} — {note.role}", True, HEAT), (x, y))
        y += 15
        for line in _wrap(self.font, note.detail, w)[:9]:
            screen.blit(self.font.render(line, True, DIM), (x, y))
            y += 12
        if note.circuits:
            y += 3
            screen.blit(
                self.font.render("circuits: " + ", ".join(note.circuits), True, FAINT),
                (x, y),
            )

    def _draw_measured(self, screen, x, y, w, index) -> None:
        facts = self.analysis.facts(index)
        net = self.sim.net
        y = self._heading(screen, x, y, "MEASURED — computed from the connectome", FAINT)

        rate_line = f"{self.smoothed[index]:.1f} Hz  (mean over {self.TAU_DISPLAY:.0f} s)"
        if self.effect is not None:
            rate_line += f"   {self.effect.delta[index]:+.1f} caused by the ablation"
        rows = [
            ("firing rate", rate_line),
            ("incoming", f"{facts.excitatory_in:.0f} exc / {facts.inhibitory_in:.0f} inh"
                         f"  over {facts.in_degree} partners"),
            ("outgoing", f"{facts.out_weight:.0f} over {facts.out_degree} partners"),
            ("drive rank", f"#{facts.drive_rank} of {self.sim.conn.n_neurons}"),
            ("position", f"{facts.body_position:.2f}  (0 head, 1 tail)"),
        ]
        if facts.muscle_weight:
            seg = facts.muscle_segments
            rows.append((
                "muscle",
                f"{facts.muscle_dorsal:.0f} dorsal / {facts.muscle_ventral:.0f} ventral"
                + (f", seg {seg[0]}-{seg[1]}" if seg else ""),
            ))
        rows.append((
            "→ command",
            f"{facts.to_backward:.3f} backward / {facts.to_forward:.3f} forward",
        ))
        if facts.isolated:
            rows.append(("note", "no synapses at all — cannot affect behaviour"))

        for label, value in rows:
            screen.blit(self.font.render(label, True, FAINT), (x, y))
            screen.blit(self.font.render(value, True, TEXT), (x + 78, y))
            y += 13

        pre, post = self.analysis.partners(index, top=5)
        y += 4
        if pre:
            screen.blit(self.font.render("strongest inputs", True, PRE), (x, y))
            y += 12
            screen.blit(
                self.font.render(
                    "  " + "  ".join(f"{n}{'+' if v > 0 else '-'}{abs(v):.0f}" for n, v in pre),
                    True, DIM),
                (x, y),
            )
            y += 14
        if post:
            screen.blit(self.font.render("strongest targets", True, POST), (x, y))
            y += 12
            screen.blit(
                self.font.render(
                    "  " + "  ".join(f"{n}{'+' if v > 0 else '-'}{abs(v):.0f}" for n, v in post),
                    True, DIM),
                (x, y),
            )

    def _draw_why(self, screen, x, y, w, index) -> None:
        sim = self.sim
        net = sim.net
        y = self._heading(screen, x, y, "WHY ACTIVE NOW — live current, this millisecond", FAINT)

        if net.ablated[index]:
            screen.blit(self.font_mid.render("ABLATED — silenced by hand", True, (210, 120, 124)), (x, y))
            return
        breakdown = net.input_breakdown(index, sim.sensory_current, top=7)

        total = float(breakdown["total"])
        threshold = float(breakdown["threshold"])
        screen.blit(
            self.font_mid.render(
                f"drive {_fmt(total)}   threshold {threshold:.0f}", True,
                EXC if total >= threshold else TEXT,
            ),
            (x, y),
        )
        y += 18

        parts = [
            ("tonic", float(breakdown["tonic"]), DIM),
            ("synaptic", float(breakdown["synaptic"]), EXC),
            ("adaptation", float(breakdown["adaptation"]), INH),
        ]
        if breakdown["external"]:
            parts.append(("sensory in", float(breakdown["external"]), HEAT))
        if net.stimulated[index]:
            parts.append(("forced on", net.p.stim_current, STIM))
        span = max(40.0, max(abs(v) for _, v, _ in parts))
        for label, value, colour in parts:
            screen.blit(self.font.render(label, True, FAINT), (x, y))
            bar_x = x + 66
            bar_w = w - 110
            centre = bar_x + bar_w // 2
            pygame.draw.line(screen, GRID, (centre, y), (centre, y + 9))
            length = int(abs(value) / span * (bar_w // 2))
            rect = (centre, y + 1, length, 8) if value >= 0 else (centre - length, y + 1, length, 8)
            pygame.draw.rect(screen, colour, rect)
            shown = self.font.render(_fmt(value), True, DIM)
            screen.blit(shown, (x + w - shown.get_width() - 4, y))
            y += 13

        contributors = breakdown["contributors"]
        y += 5
        screen.blit(
            self.font.render(
                "presynaptic partners driving it right now" if contributors
                else "no presynaptic partner is active right now",
                True, FAINT),
            (x, y),
        )
        y += 13
        if contributors:
            biggest = max(abs(v) for _, v in contributors)
            for pre_name, value in contributors:
                screen.blit(self.font.render(pre_name, True, TEXT), (x, y))
                bar_x = x + 60
                bar_w = w - 104
                centre = bar_x + bar_w // 2
                length = int(abs(value) / biggest * (bar_w // 2))
                colour = EXC if value > 0 else INH
                rect = (centre, y + 1, length, 7) if value >= 0 else (centre - length, y + 1, length, 7)
                pygame.draw.rect(screen, colour, rect)
                shown = self.font.render(_fmt(value), True, DIM)
                screen.blit(shown, (x + w - shown.get_width() - 4, y))
                y += 12

    def _draw_effect(self, screen, y: int) -> None:
        """Report of the last paired ablation experiment."""
        effect = self.effect
        assert effect is not None
        w = self.size[0]
        col = (w - 48) // 3
        x = 16

        screen.blit(
            self.font.render(
                "EFFECT OF THE ABLATION — two identical runs, same noise, "
                "differing only in the lesion", True, FAINT),
            (x, y + 8),
        )
        ablated = ", ".join(effect.ablated[:12]) + ("…" if len(effect.ablated) > 12 else "")
        screen.blit(self.font_mid.render(f"ablated  {ablated}", True, TEXT), (x, y + 24))
        screen.blit(self.font_mid.render(effect.headline(), True, HEAT), (x, y + 44))

        rows_y = y + 70
        screen.blit(self.font.render("what the worm can still do", True, FAINT), (x, rows_y))
        line = rows_y + 14
        for label, before, after, fmt in (
            ("backward cmd", effect.backward[0], effect.backward[1], "{:.1f} Hz"),
            ("forward cmd", effect.forward[0], effect.forward[1], "{:.1f} Hz"),
            ("command balance", effect.balance[0], effect.balance[1], "{:+.3f}"),
            ("muscle output", effect.muscle[0], effect.muscle[1], "{:.2f}"),
        ):
            screen.blit(self.font.render(label, True, FAINT), (x, line))
            text = f"{fmt.format(before)}  →  {fmt.format(after)}"
            colour = COOLER if after < before else HEAT if after > before else TEXT
            screen.blit(self.font.render(text, True, colour), (x + 104, line))
            line += 13
        screen.blit(
            self.font.render(effect.reversal_verdict(), True, HOT), (x, line + 4)
        )

        for offset, title, items, colour in (
            (col + 8, "quietened most", effect.fallers, COOLER),
            (2 * col + 16, "made busier", effect.risers, HEAT),
        ):
            cx = x + offset
            screen.blit(self.font.render(title, True, FAINT), (cx, rows_y))
            line = rows_y + 14
            if not items:
                screen.blit(self.font.render("none", True, DIM), (cx, line))
                continue
            biggest = max(abs(v) for _, v in items)
            for name, value in items:
                screen.blit(self.font.render(name, True, TEXT), (cx, line))
                length = int(abs(value) / biggest * (col - 150))
                pygame.draw.rect(screen, colour, (cx + 62, line + 2, length, 7))
                shown = self.font.render(f"{value:+.1f}", True, DIM)
                screen.blit(shown, (cx + col - 60, line))
                line += 13

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import random

QWERTY_LAYOUT = "qwertyuiopasdfghjklzxcvbnm"


@dataclass(frozen=True)
class Slot:
    name: str
    row: int
    col: float
    hand: str
    finger: str
    finger_rank: int
    effort: float


@dataclass(frozen=True)
class Geometry:
    name: str
    slots: tuple[Slot, ...]

    @property
    def slot_count(self) -> int:
        return len(self.slots)


STAGGERED_SLOTS = (
    Slot("T0", 0, 0.00, "L", "pinky", 0, 2.20),
    Slot("T1", 0, 1.00, "L", "ring", 1, 1.85),
    Slot("T2", 0, 2.00, "L", "middle", 2, 1.55),
    Slot("T3", 0, 3.00, "L", "index", 3, 1.30),
    Slot("T4", 0, 4.00, "L", "index", 3, 1.35),
    Slot("T5", 0, 5.00, "R", "index", 3, 1.35),
    Slot("T6", 0, 6.00, "R", "index", 3, 1.30),
    Slot("T7", 0, 7.00, "R", "middle", 2, 1.55),
    Slot("T8", 0, 8.00, "R", "ring", 1, 1.85),
    Slot("T9", 0, 9.00, "R", "pinky", 0, 2.20),
    Slot("H0", 1, 0.25, "L", "pinky", 0, 1.75),
    Slot("H1", 1, 1.25, "L", "ring", 1, 1.35),
    Slot("H2", 1, 2.25, "L", "middle", 2, 1.10),
    Slot("H3", 1, 3.25, "L", "index", 3, 1.00),
    Slot("H4", 1, 4.25, "L", "index", 3, 1.05),
    Slot("H5", 1, 5.25, "R", "index", 3, 1.05),
    Slot("H6", 1, 6.25, "R", "index", 3, 1.00),
    Slot("H7", 1, 7.25, "R", "middle", 2, 1.10),
    Slot("H8", 1, 8.25, "R", "ring", 1, 1.35),
    Slot("B0", 2, 0.75, "L", "pinky", 0, 2.10),
    Slot("B1", 2, 1.75, "L", "ring", 1, 1.75),
    Slot("B2", 2, 2.75, "L", "middle", 2, 1.35),
    Slot("B3", 2, 3.75, "L", "index", 3, 1.20),
    Slot("B4", 2, 4.75, "L", "index", 3, 1.25),
    Slot("B5", 2, 5.75, "R", "index", 3, 1.25),
    Slot("B6", 2, 6.75, "R", "index", 3, 1.35),
)

ORTHO_SLOTS = tuple(
    Slot(
        f"{'T' if r==0 else 'H' if r==1 else 'B'}{c}",
        r,
        float(c),
        "L" if c < 5 else "R",
        ["pinky", "ring", "middle", "index", "index", "index", "index", "middle", "ring", "pinky"][c],
        [0, 1, 2, 3, 3, 3, 3, 2, 1, 0][c],
        1.0 + 0.1 * abs(r - 1) + 0.1 * abs(c - (4.5 if c < 5 else 5.5)),
    )
    for r in range(3)
    for c in range(10)
)

GEOMETRIES = {
    "staggered": Geometry("staggered", STAGGERED_SLOTS),
    "ortho": Geometry("ortho", ORTHO_SLOTS),
}


def random_layout(rng: random.Random, charset: str) -> str:
    chars = list(charset)
    rng.shuffle(chars)
    return "".join(chars)


def mutate_layout(layout: str, rng: random.Random, swaps: int) -> str:
    chars = list(layout)
    for _ in range(max(1, swaps)):
        left, right = rng.sample(range(len(chars)), 2)
        chars[left], chars[right] = chars[right], chars[left]
    return "".join(chars)


def crossover_layout(parent_a: str, parent_b: str, rng: random.Random) -> str:
    size = len(parent_a)
    keep_positions = set(rng.sample(range(size), rng.randint(size // 4, size // 2)))

    # Count instead of using a set: filler spaces repeat, and each one must survive.
    needed = Counter(parent_a) - Counter(parent_a[index] for index in keep_positions)
    fill = []
    for char in parent_b:
        if needed[char] > 0:
            fill.append(char)
            needed[char] -= 1

    fill_chars = iter(fill)
    return "".join(parent_a[index] if index in keep_positions else next(fill_chars) for index in range(size))


def pretty_layout(layout: str) -> str:
    # Simplified version for staggered 10-9-7
    if len(layout) == 26:
        top, home, bottom = layout[:10], layout[10:19], layout[19:]
        return "\n".join(
            [
                " ".join(top),
                " " + " ".join(home),
                "  " + " ".join(bottom),
            ]
        )
    return layout

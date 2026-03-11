from __future__ import annotations

from dataclasses import dataclass
import random

ALPHABET = "abcdefghijklmnopqrstuvwxyz"
QWERTY_LAYOUT = "qwertyuiopasdfghjklzxcvbnm"
SLOT_COUNT = 26


@dataclass(frozen=True)
class Slot:
    name: str
    row: int
    col: float
    hand: str
    finger: str
    finger_rank: int
    effort: float


SLOTS = (
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


def normalize_layout(layout: str) -> str:
    cleaned = "".join(ch for ch in layout.lower() if ch.isalpha())
    if len(cleaned) != SLOT_COUNT:
        raise ValueError(
            "layout must contain exactly 26 letters ordered as top row, home row, bottom row"
        )
    if set(cleaned) != set(ALPHABET):
        raise ValueError("layout must contain each letter a-z exactly once")
    return cleaned


def layout_rows(layout: str) -> tuple[str, str, str]:
    normalized = normalize_layout(layout)
    return normalized[:10], normalized[10:19], normalized[19:]


def pretty_layout(layout: str) -> str:
    top, home, bottom = layout_rows(layout)
    return "\n".join(
        [
            " ".join(top),
            " " + " ".join(home),
            "  " + " ".join(bottom),
        ]
    )


def slot_map_for_layout(layout: str) -> dict[str, Slot]:
    normalized = normalize_layout(layout)
    return {letter: slot for letter, slot in zip(normalized, SLOTS, strict=True)}


def random_layout(rng: random.Random) -> str:
    letters = list(ALPHABET)
    rng.shuffle(letters)
    return "".join(letters)


def mutate_layout(layout: str, rng: random.Random, swaps: int) -> str:
    letters = list(normalize_layout(layout))
    for _ in range(max(1, swaps)):
        left, right = rng.sample(range(SLOT_COUNT), 2)
        letters[left], letters[right] = letters[right], letters[left]
    return "".join(letters)


def crossover_layout(parent_a: str, parent_b: str, rng: random.Random) -> str:
    first = normalize_layout(parent_a)
    second = normalize_layout(parent_b)
    keep_count = rng.randint(8, 18)
    keep_positions = set(rng.sample(range(SLOT_COUNT), keep_count))

    child: list[str | None] = [None] * SLOT_COUNT
    used: set[str] = set()

    for index in keep_positions:
        child[index] = first[index]
        used.add(first[index])

    fill = [letter for letter in second if letter not in used]
    fill_index = 0
    for index in range(SLOT_COUNT):
        if child[index] is None:
            child[index] = fill[fill_index]
            fill_index += 1

    return "".join(child)  # type: ignore[arg-type]


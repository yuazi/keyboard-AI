from __future__ import annotations

from dataclasses import dataclass
import math

from .corpus import CorpusStats
from .layout import Slot, Geometry


@dataclass(frozen=True)
class LayoutAnalysis:
    score: float
    effort_cost: float
    same_finger_cost: float
    same_hand_cost: float
    row_jump_cost: float
    repetition_cost: float
    alternation_bonus: float
    roll_bonus: float
    redirect_cost: float

    def to_dict(self) -> dict[str, float]:
        return {
            "score": self.score,
            "effort_cost": self.effort_cost,
            "same_finger_cost": self.same_finger_cost,
            "same_hand_cost": self.same_hand_cost,
            "row_jump_cost": self.row_jump_cost,
            "repetition_cost": self.repetition_cost,
            "alternation_bonus": self.alternation_bonus,
            "roll_bonus": self.roll_bonus,
            "redirect_cost": self.redirect_cost,
        }


def _distance(left: Slot, right: Slot) -> float:
    return math.hypot(left.row - right.row, left.col - right.col)


def _roll_direction(first: Slot, second: Slot, third: Slot) -> int:
    if len({first.hand, second.hand, third.hand}) != 1:
        return 0

    ranks = (first.finger_rank, second.finger_rank, third.finger_rank)
    if len(set(ranks)) < 3:
        return 0
    if ranks[0] < ranks[1] < ranks[2]:
        return 1
    if ranks[0] > ranks[1] > ranks[2]:
        return -1
    return 0


def analyze_layout(layout: str, geometry: Geometry, corpus: CorpusStats) -> LayoutAnalysis:
    # Build mapping from char -> Slot
    mapping = {char: slot for char, slot in zip(layout, geometry.slots, strict=True)}
    
    scale = max(corpus.letter_count, 1)

    effort_cost = 0.0
    same_finger_cost = 0.0
    same_hand_cost = 0.0
    row_jump_cost = 0.0
    repetition_cost = 0.0
    alternation_bonus = 0.0
    roll_bonus = 0.0
    redirect_cost = 0.0

    for letter, count in corpus.unigrams.items():
        if letter in mapping:
            effort_cost += mapping[letter].effort * count

    for pair, count in corpus.bigrams.items():
        if pair[0] not in mapping or pair[1] not in mapping:
            continue
            
        first = mapping[pair[0]]
        second = mapping[pair[1]]

        if pair[0] == pair[1]:
            repetition_cost += 0.75 * count
            continue

        if first.hand == second.hand and first.finger == second.finger:
            same_finger_cost += (1.8 + 0.9 * _distance(first, second)) * count
            continue

        if first.hand == second.hand:
            same_hand_cost += (0.45 + 0.05 * abs(first.row - second.row)) * count
            row_jump_cost += 0.12 * abs(first.row - second.row) * count
            continue

        alternation_bonus += 0.30 * count

    for trigram, count in corpus.trigrams.items():
        if trigram[0] not in mapping or trigram[1] not in mapping or trigram[2] not in mapping:
            continue
            
        first = mapping[trigram[0]]
        second = mapping[trigram[1]]
        third = mapping[trigram[2]]
        direction = _roll_direction(first, second, third)

        if direction == 1:
            roll_bonus += 0.35 * count
        elif direction == -1:
            roll_bonus += 0.18 * count
        elif len({first.hand, second.hand, third.hand}) == 1:
            redirect_cost += 0.28 * count

    total_cost = (
        effort_cost
        + same_finger_cost
        + same_hand_cost
        + row_jump_cost
        + repetition_cost
        + redirect_cost
        - alternation_bonus
        - roll_bonus
    )

    return LayoutAnalysis(
        score=-(total_cost / scale),
        effort_cost=effort_cost / scale,
        same_finger_cost=same_finger_cost / scale,
        same_hand_cost=same_hand_cost / scale,
        row_jump_cost=row_jump_cost / scale,
        repetition_cost=repetition_cost / scale,
        alternation_bonus=alternation_bonus / scale,
        roll_bonus=roll_bonus / scale,
        redirect_cost=redirect_cost / scale,
    )


def score_layout(layout: str, geometry: Geometry, corpus: CorpusStats) -> float:
    return analyze_layout(layout, geometry, corpus).score

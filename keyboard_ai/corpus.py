from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Iterable

WORD_RE = re.compile(r"[a-z]+")


@dataclass
class CorpusStats:
    unigrams: Counter[str] = field(default_factory=Counter)
    bigrams: Counter[str] = field(default_factory=Counter)
    trigrams: Counter[str] = field(default_factory=Counter)
    token_count: int = 0
    letter_count: int = 0

    @classmethod
    def from_text(cls, text: str) -> "CorpusStats":
        stats = cls()
        for token in WORD_RE.findall(text.lower()):
            stats.token_count += 1
            stats.letter_count += len(token)
            stats.unigrams.update(token)
            stats.bigrams.update(token[index : index + 2] for index in range(len(token) - 1))
            stats.trigrams.update(token[index : index + 3] for index in range(len(token) - 2))
        return stats

    @classmethod
    def from_files(cls, paths: Iterable[Path]) -> "CorpusStats":
        stats = cls()
        for path in paths:
            stats.merge(cls.from_text(path.read_text(encoding="utf-8")))
        return stats

    def merge(self, other: "CorpusStats") -> None:
        self.unigrams.update(other.unigrams)
        self.bigrams.update(other.bigrams)
        self.trigrams.update(other.trigrams)
        self.token_count += other.token_count
        self.letter_count += other.letter_count

    def top_unigrams(self, limit: int = 8) -> list[tuple[str, int]]:
        return self.unigrams.most_common(limit)

    def top_bigrams(self, limit: int = 8) -> list[tuple[str, int]]:
        return self.bigrams.most_common(limit)

    def to_dict(self) -> dict[str, object]:
        return {
            "unigrams": dict(self.unigrams),
            "bigrams": dict(self.bigrams),
            "trigrams": dict(self.trigrams),
            "token_count": self.token_count,
            "letter_count": self.letter_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "CorpusStats":
        return cls(
            unigrams=Counter(data.get("unigrams", {})),
            bigrams=Counter(data.get("bigrams", {})),
            trigrams=Counter(data.get("trigrams", {})),
            token_count=int(data.get("token_count", 0)),
            letter_count=int(data.get("letter_count", 0)),
        )


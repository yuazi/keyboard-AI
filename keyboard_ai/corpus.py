from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Iterable

WORD_RE = re.compile(r"[a-z.,;:'\-?!]+")
DEFAULT_CHARSET = "abcdefghijklmnopqrstuvwxyz"


@dataclass
class CorpusStats:
    unigrams: Counter[str] = field(default_factory=Counter)
    bigrams: Counter[str] = field(default_factory=Counter)
    trigrams: Counter[str] = field(default_factory=Counter)
    token_count: int = 0
    letter_count: int = 0

    @classmethod
    def from_text(cls, text: str, charset: str = DEFAULT_CHARSET) -> "CorpusStats":
        stats = cls()
        charset_set = set(charset.lower())
        
        # We split by whitespace but keep the punctuation attached to words
        # and then filter the characters based on the charset.
        for raw_token in text.lower().split():
            # Filter the token to only include characters in our charset
            token = "".join(ch for ch in raw_token if ch in charset_set)
            if not token:
                continue
                
            stats.token_count += 1
            stats.letter_count += len(token)
            stats.unigrams.update(token)
            stats.bigrams.update(token[index : index + 2] for index in range(len(token) - 1))
            stats.trigrams.update(token[index : index + 3] for index in range(len(token) - 2))
        return stats

    @classmethod
    def from_files(cls, paths: Iterable[Path], charset: str = DEFAULT_CHARSET) -> "CorpusStats":
        stats = cls()
        for path in paths:
            stats.merge(cls.from_text(path.read_text(encoding="utf-8"), charset=charset))
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
            unigrams=Counter(data.get("unigrams", {})),  # type: ignore
            bigrams=Counter(data.get("bigrams", {})),  # type: ignore
            trigrams=Counter(data.get("trigrams", {})),  # type: ignore
            token_count=int(data.get("token_count", 0)),  # type: ignore
            letter_count=int(data.get("letter_count", 0)),  # type: ignore
        )


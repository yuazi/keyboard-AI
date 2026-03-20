from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Optional

from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

from .corpus import CorpusStats
from .layout import (
    QWERTY_LAYOUT,
    Geometry,
    GEOMETRIES,
    crossover_layout,
    mutate_layout,
    random_layout,
)
from .scoring import score_layout


@dataclass(frozen=True)
class EvolutionConfig:
    generations: int = 300
    population_size: int = 64
    elite_size: int = 8
    initial_sigma: float = 3.0
    min_sigma: float = 1.0
    max_sigma: float = 6.0
    crossover_rate: float = 0.40
    sigma_learning_rate: float = 0.24


@dataclass
class Candidate:
    layout: str
    sigma: float
    score: float | None = None


@dataclass(frozen=True)
class OptimizationResult:
    best_layout: str
    best_score: float
    baseline_layout: str
    baseline_score: float
    starting_layout: str
    starting_score: float
    generations: int
    seed: int | None
    history: list[float]
    corpus_tokens: int
    corpus_letters: int
    config: EvolutionConfig
    geometry_name: str

    def to_dict(self) -> dict[str, object]:
        return {
            "best_layout": self.best_layout,
            "best_score": self.best_score,
            "baseline_layout": self.baseline_layout,
            "baseline_score": self.baseline_score,
            "starting_layout": self.starting_layout,
            "starting_score": self.starting_score,
            "generations": self.generations,
            "seed": self.seed,
            "history": self.history,
            "corpus_tokens": self.corpus_tokens,
            "corpus_letters": self.corpus_letters,
            "geometry_name": self.geometry_name,
            "config": {
                "generations": self.config.generations,
                "population_size": self.config.population_size,
                "elite_size": self.config.elite_size,
                "initial_sigma": self.config.initial_sigma,
                "min_sigma": self.config.min_sigma,
                "max_sigma": self.config.max_sigma,
                "crossover_rate": self.config.crossover_rate,
                "sigma_learning_rate": self.config.sigma_learning_rate,
            },
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_file(cls, path: Path) -> "OptimizationResult":
        data = json.loads(path.read_text(encoding="utf-8"))
        config_data = data["config"]
        config = EvolutionConfig(
            generations=int(config_data["generations"]),
            population_size=int(config_data["population_size"]),
            elite_size=int(config_data["elite_size"]),
            initial_sigma=float(config_data["initial_sigma"]),
            min_sigma=float(config_data["min_sigma"]),
            max_sigma=float(config_data["max_sigma"]),
            crossover_rate=float(config_data["crossover_rate"]),
            sigma_learning_rate=float(config_data["sigma_learning_rate"]),
        )
        return cls(
            best_layout=str(data["best_layout"]),
            best_score=float(data["best_score"]),
            baseline_layout=str(data["baseline_layout"]),
            baseline_score=float(data["baseline_score"]),
            starting_layout=str(data["starting_layout"]),
            starting_score=float(data["starting_score"]),
            generations=int(data["generations"]),
            seed=None if data["seed"] is None else int(data["seed"]),
            history=[float(value) for value in data["history"]],
            corpus_tokens=int(data["corpus_tokens"]),
            corpus_letters=int(data["corpus_letters"]),
            config=config,
            geometry_name=str(data.get("geometry_name", "staggered")),
        )


class SelfLearningLayoutAI:
    def __init__(
        self,
        corpus: CorpusStats,
        geometry: Geometry,
        charset: str,
        config: EvolutionConfig | None = None,
        seed: int | None = None,
    ) -> None:
        self.corpus = corpus
        self.geometry = geometry
        self.charset = charset
        self.config = config or EvolutionConfig()
        self.seed = seed
        self.rng = random.Random(seed)

        if self.config.population_size < 4:
            raise ValueError("population_size must be at least 4")
        if not 1 <= self.config.elite_size < self.config.population_size:
            raise ValueError("elite_size must be between 1 and population_size - 1")

    def train(self, initial_layout: Optional[str] = None) -> OptimizationResult:
        if initial_layout is None:
            initial_layout = QWERTY_LAYOUT if self.geometry.name == "staggered" else random_layout(self.rng, self.charset, self.geometry)
            
        baseline_score = score_layout(QWERTY_LAYOUT, GEOMETRIES["staggered"], self.corpus) if self.geometry.name == "staggered" else 0.0
        starting_score = score_layout(initial_layout, self.geometry, self.corpus)

        population = self._initial_population(initial_layout)
        self._evaluate(population)
        population.sort(key=lambda candidate: candidate.score or float("-inf"), reverse=True)

        best = Candidate(population[0].layout, population[0].sigma, population[0].score)
        history = [best.score or float("-inf")]

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("[bold blue]Score: {task.fields[score]:.4f}"),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("Evolving...", total=self.config.generations, score=best.score)

            for _ in range(self.config.generations):
                elites = [
                    Candidate(candidate.layout, candidate.sigma, candidate.score)
                    for candidate in population[: self.config.elite_size]
                ]
                offspring: list[Candidate] = []

                while len(offspring) < self.config.population_size - self.config.elite_size:
                    parent = self._pick_parent(elites)
                    sigma = self._mutate_sigma(parent.sigma)

                    if self.rng.random() < self.config.crossover_rate:
                        partner = self._pick_parent(elites)
                        child_layout = crossover_layout(parent.layout, partner.layout, self.rng)
                    else:
                        child_layout = parent.layout

                    child_layout = mutate_layout(child_layout, self.rng, max(1, round(sigma)))
                    offspring.append(Candidate(child_layout, sigma))

                population = elites + offspring
                self._evaluate(population)
                population.sort(key=lambda candidate: candidate.score or float("-inf"), reverse=True)

                if (population[0].score or float("-inf")) > (best.score or float("-inf")):
                    best = Candidate(population[0].layout, population[0].sigma, population[0].score)

                history.append(best.score or float("-inf"))
                progress.update(task, advance=1, score=best.score)

        return OptimizationResult(
            best_layout=best.layout,
            best_score=best.score or float("-inf"),
            baseline_layout=QWERTY_LAYOUT if self.geometry.name == "staggered" else "N/A",
            baseline_score=baseline_score,
            starting_layout=initial_layout,
            starting_score=starting_score,
            generations=self.config.generations,
            seed=self.seed,
            history=history,
            corpus_tokens=self.corpus.token_count,
            corpus_letters=self.corpus.letter_count,
            config=self.config,
            geometry_name=self.geometry.name,
        )

    def _initial_population(self, initial_layout: str) -> list[Candidate]:
        population = [Candidate(initial_layout, self.config.initial_sigma)]
        seen = {initial_layout}

        while len(population) < self.config.population_size:
            if len(population) < self.config.population_size // 3:
                layout = mutate_layout(
                    initial_layout,
                    self.rng,
                    self.rng.randint(1, max(2, round(self.config.initial_sigma))),
                )
                sigma = self.config.initial_sigma * self.rng.uniform(0.8, 1.2)
            else:
                layout = random_layout(self.rng, self.charset, self.geometry)
                sigma = self.config.initial_sigma * self.rng.uniform(0.6, 1.4)

            if layout in seen:
                continue

            seen.add(layout)
            population.append(Candidate(layout, sigma))

        return population

    def _evaluate(self, population: list[Candidate]) -> None:
        for candidate in population:
            if candidate.score is None:
                candidate.score = score_layout(candidate.layout, self.geometry, self.corpus)

    def _pick_parent(self, candidates: list[Candidate]) -> Candidate:
        total_weight = sum(range(1, len(candidates) + 1))
        ticket = self.rng.uniform(0, total_weight)
        cumulative = 0.0

        for index, candidate in enumerate(candidates):
            weight = len(candidates) - index
            cumulative += weight
            if ticket <= cumulative:
                return candidate

        return candidates[0]

    def _mutate_sigma(self, sigma: float) -> float:
        updated = sigma * math.exp(self.config.sigma_learning_rate * self.rng.gauss(0.0, 1.0))
        return min(self.config.max_sigma, max(self.config.min_sigma, updated))

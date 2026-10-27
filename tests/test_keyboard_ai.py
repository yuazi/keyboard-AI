from __future__ import annotations

import os
from pathlib import Path
import random
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from keyboard_ai.corpus import CorpusStats
from keyboard_ai.layout import QWERTY_LAYOUT, GEOMETRIES, crossover_layout
from keyboard_ai.optimizer import EvolutionConfig, OptimizationResult, SelfLearningLayoutAI
from keyboard_ai.scoring import score_layout


REPO_ROOT = Path(__file__).resolve().parents[1]


class CorpusStatsTests(unittest.TestCase):
    def test_ngram_counts_are_collected(self) -> None:
        corpus = CorpusStats.from_text("Hello, hello.")

        self.assertEqual(corpus.token_count, 2)
        self.assertEqual(corpus.letter_count, 10)
        self.assertEqual(corpus.unigrams["h"], 2)
        self.assertEqual(corpus.bigrams["he"], 2)
        self.assertEqual(corpus.trigrams["hel"], 2)

    def test_punctuation_counts_are_collected(self) -> None:
        corpus = CorpusStats.from_text("Hello, world!", charset="abcdefghijklmnopqrstuvwxyz!,")
        self.assertEqual(corpus.unigrams[","], 1)
        self.assertEqual(corpus.unigrams["!"], 1)


class OptimizerTests(unittest.TestCase):
    def test_training_improves_on_a_biased_corpus(self) -> None:
        text = ("qaqaqa opopop zxzxzx qaqaqa opopop zxzxzx " * 300).strip()
        corpus = CorpusStats.from_text(text)
        geometry = GEOMETRIES["staggered"]
        baseline = score_layout(QWERTY_LAYOUT, geometry, corpus)

        trainer = SelfLearningLayoutAI(
            corpus,
            geometry,
            "abcdefghijklmnopqrstuvwxyz",
            config=EvolutionConfig(generations=50, population_size=32, elite_size=4),
            seed=7,
        )
        result = trainer.train(initial_layout=QWERTY_LAYOUT)

        self.assertGreater(result.best_score, baseline)
        self.assertEqual(len(result.history), 51)
        self.assertNotEqual(result.best_layout, QWERTY_LAYOUT)

    def test_saved_models_round_trip(self) -> None:
        corpus = CorpusStats.from_text("alpha beta gamma delta " * 50)
        geometry = GEOMETRIES["staggered"]
        trainer = SelfLearningLayoutAI(
            corpus,
            geometry,
            "abcdefghijklmnopqrstuvwxyz",
            config=EvolutionConfig(generations=10, population_size=16, elite_size=2),
            seed=3,
        )
        result = trainer.train(initial_layout=QWERTY_LAYOUT)

        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            result.save(path)
            restored = OptimizationResult.from_file(path)

        self.assertEqual(restored.best_layout, result.best_layout)
        self.assertEqual(restored.generations, result.generations)
        self.assertAlmostEqual(restored.best_score, result.best_score)

    def test_crossover_keeps_every_character_including_repeated_filler(self) -> None:
        parent = "abcdefghijklmnopqrstuvwxyz    "
        rng = random.Random(1)
        for _ in range(200):
            other = list(parent)
            rng.shuffle(other)
            child = crossover_layout(parent, "".join(other), rng)
            self.assertEqual(sorted(child), sorted(parent))

    def test_starting_layout_must_match_charset(self) -> None:
        trainer = SelfLearningLayoutAI(
            CorpusStats.from_text("alpha beta"),
            GEOMETRIES["staggered"],
            "abcdefghijklmnopqrstuvwxy;",
        )
        with self.assertRaises(ValueError):
            trainer.train(initial_layout=QWERTY_LAYOUT)


class CLITests(unittest.TestCase):
    def run_cli(
        self,
        *args: str,
        input_text: str | None = None,
        cwd: Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO_ROOT)
        return subprocess.run(
            [sys.executable, "-m", "keyboard_ai.cli", *args],
            cwd=cwd or REPO_ROOT,
            env=env,
            input=input_text,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_train_accepts_stdin_text(self) -> None:
        result = self.run_cli(
            "train",
            "--stdin",
            "--generations",
            "5",
            "--population",
            "8",
            "--elite",
            "2",
            "--seed",
            "7",
            input_text="This is my typing sample.\nIt should teach the layout tool.\n",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Sources: stdin text", result.stdout)
        self.assertIn("Best learned score:", result.stdout)
        self.assertIn("Layout string:", result.stdout)

    def test_train_uses_bundled_corpus_when_no_input_source_is_given(self) -> None:
        result = self.run_cli(
            "train",
            "--generations",
            "5",
            "--population",
            "8",
            "--elite",
            "2",
            "--seed",
            "7",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Sources: bundled sample corpus", result.stdout)

    def test_score_accepts_stdin_text(self) -> None:
        result = self.run_cli(
            "score",
            "--stdin",
            "--layout",
            QWERTY_LAYOUT,
            input_text="Typing practice text for scoring.\n",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Sources: stdin text", result.stdout)
        self.assertIn("Score", result.stdout)
        self.assertIn("Effort cost", result.stdout)

    def test_train_rejects_corpus_file_without_charset_characters(self) -> None:
        with TemporaryDirectory() as directory:
            corpus = Path(directory) / "digits.txt"
            corpus.write_text("12345 67890\n", encoding="utf-8")
            result = self.run_cli("train", "--corpus", str(corpus), "--generations", "2")

        self.assertEqual(result.returncode, 1)
        self.assertIn("did not contain any characters", result.stdout)

    def test_ortho_training_keeps_all_letters_and_resumes(self) -> None:
        fast = ("--generations", "3", "--population", "8", "--elite", "2")
        with TemporaryDirectory() as directory:
            model = Path(directory) / "ortho.json"
            trained = self.run_cli("train", "--geometry", "ortho", *fast, "--output", str(model))
            resumed = self.run_cli("train", "--resume", str(model), *fast)
            layout = OptimizationResult.from_file(model).best_layout

        self.assertEqual(trained.returncode, 0, trained.stdout)
        self.assertTrue(set("abcdefghijklmnopqrstuvwxyz") <= set(layout), layout)
        self.assertEqual(resumed.returncode, 0, resumed.stdout)
        self.assertIn("Geometry: ortho", resumed.stdout)


if __name__ == "__main__":
    unittest.main()

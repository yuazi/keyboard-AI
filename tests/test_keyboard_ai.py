from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from keyboard_ai.corpus import CorpusStats
from keyboard_ai.layout import QWERTY_LAYOUT, GEOMETRIES
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


if __name__ == "__main__":
    unittest.main()

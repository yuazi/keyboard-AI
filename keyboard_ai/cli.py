from __future__ import annotations

import argparse
from importlib import resources
from pathlib import Path
import sys
from typing import Iterable

from .corpus import CorpusStats
from .layout import QWERTY_LAYOUT, normalize_layout, pretty_layout
from .optimizer import EvolutionConfig, OptimizationResult, SelfLearningLayoutAI
from .scoring import analyze_layout


class CorpusInputError(ValueError):
    pass


def _resource_corpus() -> CorpusStats:
    text = resources.files("keyboard_ai").joinpath("sample_corpus.txt").read_text(encoding="utf-8")
    return CorpusStats.from_text(text)


def _read_stdin_text() -> str:
    if sys.stdin.isatty():
        print("Paste or type text, then press Ctrl-D when finished.", file=sys.stderr)
    return sys.stdin.read()


def _load_corpus(paths: list[str] | None, use_stdin: bool = False) -> tuple[CorpusStats, list[str]]:
    if use_stdin:
        corpus = CorpusStats.from_text(_read_stdin_text())
        if corpus.letter_count == 0:
            raise CorpusInputError("stdin text did not contain any alphabetic characters")
        return corpus, ["stdin text"]

    if not paths:
        return _resource_corpus(), ["bundled sample corpus"]

    resolved = [Path(path).expanduser().resolve() for path in paths]
    return CorpusStats.from_files(resolved), [str(path) for path in resolved]


def _format_top_items(items: Iterable[tuple[str, int]]) -> str:
    return ", ".join(f"{item}:{count}" for item, count in items)


def cmd_train(args: argparse.Namespace) -> int:
    corpus, sources = _load_corpus(args.corpus, use_stdin=args.stdin)

    initial_layout = args.layout
    if args.resume is not None:
        saved = OptimizationResult.from_file(Path(args.resume))
        initial_layout = saved.best_layout

    config = EvolutionConfig(
        generations=args.generations,
        population_size=args.population,
        elite_size=args.elite,
        initial_sigma=args.sigma,
    )
    trainer = SelfLearningLayoutAI(corpus, config=config, seed=args.seed)
    result = trainer.train(initial_layout=initial_layout)

    print(f"Sources: {', '.join(sources)}")
    print(f"Corpus: {corpus.token_count} tokens, {corpus.letter_count} letters")
    print(f"Top letters: {_format_top_items(corpus.top_unigrams(6))}")
    print(f"Top bigrams: {_format_top_items(corpus.top_bigrams(6))}")
    print()
    print(f"Baseline QWERTY score: {result.baseline_score:.4f}")
    print(f"Starting layout score: {result.starting_score:.4f}")
    print(f"Best learned score:    {result.best_score:.4f}")
    print(f"Improvement vs QWERTY: {result.best_score - result.baseline_score:.4f}")
    print()
    print("Best layout:")
    print(pretty_layout(result.best_layout))
    print()
    print(f"Layout string: {result.best_layout}")

    if args.output is not None:
        output = Path(args.output).expanduser()
        result.save(output)
        print(f"Saved model: {output}")

    return 0


def cmd_score(args: argparse.Namespace) -> int:
    corpus, sources = _load_corpus(args.corpus, use_stdin=args.stdin)
    layout = normalize_layout(args.layout)
    analysis = analyze_layout(layout, corpus)

    print(f"Sources: {', '.join(sources)}")
    print()
    print(pretty_layout(layout))
    print()
    print(f"Score:             {analysis.score:.4f}")
    print(f"Effort cost:       {analysis.effort_cost:.4f}")
    print(f"Same-finger cost:  {analysis.same_finger_cost:.4f}")
    print(f"Same-hand cost:    {analysis.same_hand_cost:.4f}")
    print(f"Row-jump cost:     {analysis.row_jump_cost:.4f}")
    print(f"Repetition cost:   {analysis.repetition_cost:.4f}")
    print(f"Alternation bonus: {analysis.alternation_bonus:.4f}")
    print(f"Roll bonus:        {analysis.roll_bonus:.4f}")
    print(f"Redirect cost:     {analysis.redirect_cost:.4f}")

    return 0


def cmd_show(args: argparse.Namespace) -> int:
    if args.model is not None:
        layout = OptimizationResult.from_file(Path(args.model)).best_layout
    else:
        layout = args.layout

    print(pretty_layout(layout))
    print()
    print(normalize_layout(layout))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="keyboard-ai",
        description="Self-learning keyboard layout optimizer",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="learn a layout from text")
    train_inputs = train_parser.add_mutually_exclusive_group()
    train_inputs.add_argument("--corpus", nargs="+", help="text files to learn from")
    train_inputs.add_argument("--stdin", action="store_true", help="read text from standard input")
    train_parser.add_argument("--layout", default=QWERTY_LAYOUT, help="starting layout string")
    train_parser.add_argument("--resume", help="resume from a saved model JSON file")
    train_parser.add_argument("--generations", type=int, default=300)
    train_parser.add_argument("--population", type=int, default=64)
    train_parser.add_argument("--elite", type=int, default=8)
    train_parser.add_argument("--sigma", type=float, default=3.0, help="initial mutation strength")
    train_parser.add_argument("--seed", type=int, default=7)
    train_parser.add_argument("--output", help="save the learned model JSON to this path")
    train_parser.set_defaults(func=cmd_train)

    score_parser = subparsers.add_parser("score", help="score a layout")
    score_inputs = score_parser.add_mutually_exclusive_group()
    score_inputs.add_argument("--corpus", nargs="+", help="text files to learn from")
    score_inputs.add_argument("--stdin", action="store_true", help="read text from standard input")
    score_parser.add_argument("--layout", required=True, help="layout string")
    score_parser.set_defaults(func=cmd_score)

    show_parser = subparsers.add_parser("show", help="print a layout")
    show_group = show_parser.add_mutually_exclusive_group(required=True)
    show_group.add_argument("--layout", help="layout string")
    show_group.add_argument("--model", help="saved model JSON")
    show_parser.set_defaults(func=cmd_show)

    return parser


def main() -> int:
    parser = build_parser()
    try:
        args = parser.parse_args()
        return args.func(args)
    except CorpusInputError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

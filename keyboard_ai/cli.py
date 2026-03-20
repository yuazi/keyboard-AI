from __future__ import annotations

import argparse
from importlib import resources
from pathlib import Path
import sys
from typing import Iterable, Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from .corpus import CorpusStats, DEFAULT_CHARSET
from .layout import QWERTY_LAYOUT, GEOMETRIES, Geometry
from .optimizer import EvolutionConfig, OptimizationResult, SelfLearningLayoutAI
from .scoring import analyze_layout
from .export import export_karabiner, export_qmk

console = Console()


class CorpusInputError(ValueError):
    pass


def _resource_corpus(charset: str) -> CorpusStats:
    text = resources.files("keyboard_ai").joinpath("sample_corpus.txt").read_text(encoding="utf-8")
    return CorpusStats.from_text(text, charset=charset)


def _read_stdin_text() -> str:
    if sys.stdin.isatty():
        console.print("[yellow]Paste or type text, then press Ctrl-D when finished.[/yellow]", style="italic")
    return sys.stdin.read()


def _load_corpus(paths: list[str] | None, charset: str, use_stdin: bool = False) -> tuple[CorpusStats, list[str]]:
    if use_stdin:
        corpus = CorpusStats.from_text(_read_stdin_text(), charset=charset)
        if corpus.letter_count == 0:
            raise CorpusInputError(f"stdin text did not contain any characters from charset: {charset}")
        return corpus, ["stdin text"]

    if not paths:
        return _resource_corpus(charset), ["bundled sample corpus"]

    resolved = [Path(path).expanduser().resolve() for path in paths]
    return CorpusStats.from_files(resolved, charset=charset), [str(path) for path in resolved]


def _format_top_items(items: Iterable[tuple[str, int]]) -> str:
    return ", ".join(f"[bold cyan]{item}[/bold cyan]:{count}" for item, count in items)


def render_layout(layout: str, geometry: Geometry, corpus: Optional[CorpusStats] = None) -> Panel:
    table = Table.grid(padding=(0, 1))
    
    rows: dict[int, list[tuple[float, str]]] = {}
    for char, slot in zip(layout, geometry.slots, strict=True):
        if slot.row not in rows:
            rows[slot.row] = []
        rows[slot.row].append((slot.col, char))
        
    for r in rows:
        rows[r].sort()
        
    max_count = max(corpus.unigrams.values()) if corpus and corpus.unigrams else 1

    for r in sorted(rows.keys()):
        row_text = Text()
        indent = int(rows[r][0][0] * 2)
        row_text.append(" " * indent)
        
        for _, char in rows[r]:
            style = ""
            if corpus and char in corpus.unigrams:
                freq = corpus.unigrams[char] / max_count
                if freq > 0.8:
                    style = "bold red"
                elif freq > 0.5:
                    style = "bold yellow"
                elif freq > 0.2:
                    style = "bold green"
                else:
                    style = "bold blue"
            
            row_text.append(char + " ", style=style)
        table.add_row(row_text)
        
    return Panel(table, title=f"[bold]{geometry.name.upper()} Layout[/bold]", border_style="bright_black")


def cmd_train(args: argparse.Namespace) -> int:
    geometry = GEOMETRIES.get(args.geometry)
    if not geometry:
        console.print(f"[red]Error: Unknown geometry '{args.geometry}'[/red]")
        return 1
        
    charset = args.charset or (DEFAULT_CHARSET if args.geometry == "staggered" else "".join(sorted(set(DEFAULT_CHARSET + ".,;:'-?!"))))
    
    if len(charset) < geometry.slot_count:
        charset = charset.ljust(geometry.slot_count)
    elif len(charset) > geometry.slot_count:
        charset = charset[:geometry.slot_count]

    try:
        corpus, sources = _load_corpus(args.corpus, charset, use_stdin=args.stdin)
    except CorpusInputError as e:
        console.print(f"[red]Error: {e}[/red]")
        return 1

    initial_layout = args.layout
    if args.resume is not None:
        saved = OptimizationResult.from_file(Path(args.resume))
        initial_layout = saved.best_layout
    elif initial_layout == QWERTY_LAYOUT and args.geometry != "staggered":
        initial_layout = None

    config = EvolutionConfig(
        generations=args.generations,
        population_size=args.population,
        elite_size=args.elite,
        initial_sigma=args.sigma,
    )
    trainer = SelfLearningLayoutAI(corpus, geometry, charset, config=config, seed=args.seed)
    
    console.print("[bold]Training starting...[/bold]")
    console.print(f"Sources: [dim]{', '.join(sources)}[/dim]")
    console.print(f"Corpus: [bold]{corpus.token_count}[/bold] tokens, [bold]{corpus.letter_count}[/bold] letters")
    console.print(f"Geometry: [bold cyan]{geometry.name}[/bold cyan]")
    console.print()

    result = trainer.train(initial_layout=initial_layout)

    console.print()
    if result.baseline_score != 0:
        console.print(f"Baseline QWERTY score: [bold]{result.baseline_score:.4f}[/bold]")
    console.print(f"Starting layout score: [bold]{result.starting_score:.4f}[/bold]")
    console.print(f"Best learned score:    [bold green]{result.best_score:.4f}[/bold green]")
    if result.baseline_score != 0:
        console.print(f"Improvement vs QWERTY: [bold green]{result.best_score - result.baseline_score:.4f}[/bold green]")
    console.print()
    
    console.print(render_layout(result.best_layout, geometry, corpus))
    console.print()
    console.print(f"Layout string: [bold cyan]{result.best_layout}[/bold cyan]")

    if args.output is not None:
        output = Path(args.output).expanduser()
        result.save(output)
        console.print(f"Saved model: [dim]{output}[/dim]")

    return 0


def cmd_score(args: argparse.Namespace) -> int:
    geometry = None
    for g in GEOMETRIES.values():
        if g.slot_count == len(args.layout):
            geometry = g
            break
    
    if not geometry:
        console.print(f"[red]Error: No geometry found for layout of length {len(args.layout)}[/red]")
        return 1

    charset = "".join(sorted(set(args.layout)))
    corpus, sources = _load_corpus(args.corpus, charset, use_stdin=args.stdin)
    analysis = analyze_layout(args.layout, geometry, corpus)

    console.print(f"Sources: [dim]{', '.join(sources)}[/dim]")
    console.print()
    console.print(render_layout(args.layout, geometry, corpus))
    console.print()
    
    table = Table(title="Analysis", show_header=False, box=None)
    table.add_row("Score", f"[bold]{analysis.score:.4f}[/bold]")
    table.add_row("Effort cost", f"{analysis.effort_cost:.4f}")
    table.add_row("Same-finger cost", f"{analysis.same_finger_cost:.4f}")
    table.add_row("Same-hand cost", f"{analysis.same_hand_cost:.4f}")
    table.add_row("Row-jump cost", f"{analysis.row_jump_cost:.4f}")
    table.add_row("Repetition cost", f"{analysis.repetition_cost:.4f}")
    table.add_row("Alternation bonus", f"[green]{analysis.alternation_bonus:.4f}[/green]")
    table.add_row("Roll bonus", f"[green]{analysis.roll_bonus:.4f}[/green]")
    table.add_row("Redirect cost", f"{analysis.redirect_cost:.4f}")
    
    console.print(table)

    return 0


def cmd_show(args: argparse.Namespace) -> int:
    if args.model is not None:
        res = OptimizationResult.from_file(Path(args.model))
        layout = res.best_layout
        geometry = GEOMETRIES.get(res.geometry_name, GEOMETRIES["staggered"])
    else:
        layout = args.layout
        geometry = None
        for g in GEOMETRIES.values():
            if g.slot_count == len(layout):
                geometry = g
                break
        if not geometry:
            console.print(f"[red]Error: Unknown layout length {len(layout)}[/red]")
            return 1

    if geometry:
        console.print(render_layout(layout, geometry))
    console.print()
    console.print(f"[bold cyan]{layout}[/bold cyan]")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    if args.model is not None:
        res = OptimizationResult.from_file(Path(args.model))
        layout = res.best_layout
        geometry_name = res.geometry_name
    else:
        layout = args.layout
        geometry_name = args.geometry
        
    output = Path(args.output).expanduser()
    
    if args.format == "karabiner":
        export_karabiner(layout, geometry_name, output)
    elif args.format == "qmk":
        export_qmk(layout, geometry_name, output)
    
    console.print(f"Exported [bold cyan]{args.format}[/bold cyan] config to [dim]{output}[/dim]")
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
    train_parser.add_argument("--geometry", default="staggered", choices=list(GEOMETRIES.keys()))
    train_parser.add_argument("--charset", help="characters to include in optimization")
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

    export_parser = subparsers.add_parser("export", help="export layout to external formats")
    export_source = export_parser.add_mutually_exclusive_group(required=True)
    export_source.add_argument("--layout", help="layout string")
    export_source.add_argument("--model", help="saved model JSON")
    export_parser.add_argument("--format", choices=["karabiner", "qmk"], required=True)
    export_parser.add_argument("--geometry", default="staggered", choices=list(GEOMETRIES.keys()))
    export_parser.add_argument("--output", required=True, help="output file path")
    export_parser.set_defaults(func=cmd_export)

    parser.add_argument("--debug", action="store_true", help="show full stack trace on error")

    return parser


def main() -> int:
    parser = build_parser()
    try:
        args = parser.parse_args()
        return args.func(args)
    except Exception as e:
        if "--debug" in sys.argv:
            console.print_exception()
        else:
            console.print(f"[red]Error: {e}[/red]")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

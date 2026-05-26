# Keyboard AI

[![Python](https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![CI](https://github.com/yuazi/keyboard-AI/actions/workflows/ci.yml/badge.svg)](https://github.com/yuazi/keyboard-AI/actions)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

`keyboard-ai` is a Python CLI that learns character patterns from text and searches for keyboard layouts with lower ergonomic cost. It combines corpus analysis, evolutionary optimization, layout scoring, and export helpers in a small package-style project.

This is not a claim to find a universal "best" keyboard layout. The tool finds the best layout it can for the corpus, geometry, character set, and scoring model you give it.

## What It Demonstrates

- Corpus ingestion from files, stdin, or a bundled sample text.
- Unigram, bigram, and trigram statistics for layout scoring.
- Evolutionary search with configurable generations, population size, elite count, mutation strength, and seed.
- Ergonomic scoring for effort, same-finger movement, same-hand usage, row jumps, repetition, alternation, rolls, and redirects.
- Support for staggered and ortholinear keyboard geometries.
- Saved model JSON output plus export helpers for Karabiner and QMK-style configs.
- Regression tests for corpus statistics, training behavior, saved-model round trips, and CLI flows.

## Installation

```bash
git clone https://github.com/yuazi/keyboard-AI.git
cd keyboard-AI
python3 -m pip install -e .
```

Check that the CLI is available:

```bash
keyboard-ai --help
```

## Quick Start

Train against the bundled sample corpus:

```bash
keyboard-ai train --generations 250 --population 64
```

Train from pasted text:

```bash
keyboard-ai train --stdin
```

Train from one or more text files and save the learned model:

```bash
keyboard-ai train --corpus notes.txt emails.txt --output model.json
```

Score an existing layout:

```bash
keyboard-ai score --layout qwertyuiopasdfghjklzxcvbnm --corpus notes.txt
```

Render a saved model or layout:

```bash
keyboard-ai show --model model.json
keyboard-ai show --layout qwertyuiopasdfghjklzxcvbnm
```

Export a layout:

```bash
keyboard-ai export --model model.json --format karabiner --output karabiner.json
keyboard-ai export --model model.json --format qmk --output qmk_keymap.c
```

## Layout Format

For the default staggered geometry, layouts are encoded as 26 lowercase letters in physical key order:

```text
top row (10) + home row (9) + bottom row (7)
```

QWERTY example:

```text
qwertyuiopasdfghjklzxcvbnm
```

The CLI also supports an `ortho` geometry for 30 slots. If the provided character set is shorter than the geometry, the optimizer pads remaining slots with spaces.

## CLI Reference

| Command | Purpose |
| --- | --- |
| `keyboard-ai train` | Learn a layout from a corpus, stdin, or the bundled sample text. |
| `keyboard-ai score` | Score an existing layout against a corpus. |
| `keyboard-ai show` | Print a layout or saved model in a readable form. |
| `keyboard-ai export` | Export a layout or saved model to Karabiner or QMK output. |

Common training flags:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--corpus` | bundled sample | One or more text files to learn from. |
| `--stdin` | false | Read training text from standard input. |
| `--layout` | QWERTY | Starting layout string. |
| `--resume` | none | Continue from a saved model JSON file. |
| `--geometry` | `staggered` | Keyboard geometry: `staggered` or `ortho`. |
| `--charset` | alphabet / geometry-aware default | Characters included in optimization. |
| `--generations` | `300` | Evolutionary search generations. |
| `--population` | `64` | Candidate layouts per generation. |
| `--elite` | `8` | Best candidates preserved per generation. |
| `--sigma` | `3.0` | Initial mutation strength. |
| `--seed` | `7` | Deterministic random seed. |
| `--output` | none | Save learned model JSON. |

## Project Structure

```text
.
├── keyboard_ai/
│   ├── cli.py          # argparse CLI and Rich output
│   ├── corpus.py       # text ingestion and n-gram statistics
│   ├── export.py       # Karabiner and QMK export helpers
│   ├── layout.py       # keyboard geometries and layout operations
│   ├── optimizer.py    # evolutionary search and saved model format
│   ├── scoring.py      # ergonomic cost model
│   └── sample_corpus.txt
├── tests/
│   └── test_keyboard_ai.py
├── pyproject.toml
└── README.md
```

## Development

Run the test suite:

```bash
python3 -m unittest discover -s tests -v
```

Install locally in editable mode before testing the command-line entry point:

```bash
python3 -m pip install -e .
keyboard-ai train --generations 5 --population 8
```

## Roadmap

- Tune the ergonomic weights with better empirical defaults.
- Add visual score breakdowns for learned layouts.
- Improve duplicate/filler handling for larger custom character sets.
- Add more export targets and example configs.

## License

MIT License. See [LICENSE](LICENSE) for details.

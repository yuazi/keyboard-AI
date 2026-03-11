# Keyboard AI

`Keyboard AI` is a self-learning keyboard layout optimizer.

It does two things:

1. It learns the character patterns in a text corpus by building unigram, bigram, and trigram statistics.
2. It evolves keyboard layouts to minimize an ergonomic cost model that rewards easy keys, hand alternation, and smooth finger rolls while penalizing same-finger stretches and awkward redirects.

The result is not a universal "best" layout for every person. It is the best layout the optimizer can find for the corpus and scoring model you give it.

## Quick Start

Run the bundled demo corpus:

```bash
python3 -m keyboard_ai.cli train --generations 250 --population 64 --seed 7
```

Paste your own text directly into the terminal:

```bash
python3 -m keyboard_ai.cli train --stdin
```

Type or paste your sample text, then press `Ctrl-D`.

Train on your own writing:

```bash
python3 -m keyboard_ai.cli train --corpus notes.txt chatlog.txt code_comments.txt --generations 400 --population 80 --output artifacts/my-layout.json
```

Resume training from a saved model:

```bash
python3 -m keyboard_ai.cli train --corpus notes.txt --resume artifacts/my-layout.json --generations 300 --output artifacts/my-layout-v2.json
```

Score any layout string:

```bash
python3 -m keyboard_ai.cli score --corpus notes.txt --layout qwertyuiopasdfghjklzxcvbnm
```

Score a layout against pasted terminal text:

```bash
python3 -m keyboard_ai.cli score --stdin --layout qwertyuiopasdfghjklzxcvbnm
```

Show a layout as rows:

```bash
python3 -m keyboard_ai.cli show --layout qwertyuiopasdfghjklzxcvbnm
```

## Layout Format

Layouts are encoded as 26 letters in physical key order:

`top row (10)` + `home row (9)` + `bottom row (7)`

Example:

`qwertyuiopasdfghjklzxcvbnm`

## Commands

- `train`: learn from a corpus and search for a better layout
- `score`: inspect the score and ergonomic breakdown of a layout
- `show`: print a layout from a raw string or a saved model

`train` and `score` can learn from:

- `--corpus file1.txt file2.txt`
- `--stdin` for pasted terminal text
- no custom source, which falls back to the bundled sample corpus

Models are only saved when `--output <path>` is provided.

## Verification

Run the test suite with:

```bash
python3 -m unittest discover -s tests -v
```

# Keyboard AI

`keyboard-ai` is a self-learning keyboard layout optimizer for English text. It analyzes your writing, learns character patterns, and evolves alternative layouts that aim to reduce strain and improve typing comfort.

It does two main things:

- Learns character patterns from a text corpus by building unigram, bigram, and trigram statistics.
- Evolves keyboard layouts to minimize an ergonomic cost model that rewards easy keys, hand alternation, and smooth finger rolls, while penalizing same-finger stretches and awkward redirects.

The result is not a universal "best" layout for everyone, but the best layout the optimizer can find for the corpus and scoring model you give it.

## Installation

From source (editable install):

```bash
git clone https://github.com/yuazi/keyboard-AI.git
cd keyboard-AI
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

You can then run the CLI via:

```bash
keyboard-ai --help
```

## Quick start

Run the bundled demo corpus:

```bash
keyboard-ai train --generations 250 --population 64 --seed 7
```

Paste your own text directly into the terminal:

```bash
keyboard-ai train --stdin
```

Type or paste your sample text, then press `Ctrl-D` (Unix/macOS) to start training.

Train on your own writing:

```bash
keyboard-ai train --corpus notes.txt chatlog.txt code_comments.txt --generations 400 --population 80 --output artifacts/my-layout.json
```

Resume training from a saved model:

```bash
keyboard-ai train --corpus notes.txt --resume artifacts/my-layout.json --generations 300 --output artifacts/my-layout-v2.json
```

Score any layout string:

```bash
keyboard-ai score --corpus notes.txt --layout qwertyuiopasdfghjklzxcvbnm
```

Score a layout against pasted terminal text:

```bash
keyboard-ai score --stdin --layout qwertyuiopasdfghjklzxcvbnm
```

Show a layout as rows:

```bash
keyboard-ai show --layout qwertyuiopasdfghjklzxcvbnm
```

---

## Layout format

Layouts are encoded as 26 lowercase letters in physical key order:

`top row (10)` + `home row (9)` + `bottom row (7)`

Example (QWERTY):

```text
qwertyuiopasdfghjklzxcvbnm
```

This format is accepted by the `train`, `score`, and `show` commands wherever a `--layout` argument is used.

---

## Commands

`keyboard-ai` currently exposes three CLI commands:

- `train`: Learn from a corpus and search for a better layout.
- `score`: Inspect the score and ergonomic breakdown of a layout.
- `show`: Print a layout from a raw string or a saved model.

`train` and `score` can learn from:

- `--corpus file1.txt file2.txt`
- `--stdin` for pasted terminal text
- No custom source, which falls back to the bundled sample corpus.

Models are only saved when `--output` is provided.

---

## Verification

Run the test suite with:

```bash
python3 -m unittest discover -s tests -v
```

---

## Roadmap

Planned improvements:

- Better default ergonomic model and tunable weights.
- Support for additional characters and symbols beyond 26 letters.
- Visualization of layouts and score breakdowns.
- Configurable optimization strategies and search parameters.

---

## License

MIT License. See `LICENSE` for details.

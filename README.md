# Keyboard AI

`keyboard-ai` is a self-learning keyboard layout optimizer for English text. It analyzes your writing, learns character patterns, and evolves alternative layouts that aim to reduce strain and improve typing comfort.

It does two main things:

- Learns character patterns from a text corpus by building unigram, bigram, and trigram statistics.
- Evolves keyboard layouts to minimize an ergonomic cost model that rewards easy keys, hand alternation, and smooth finger rolls, while penalizing same-finger stretches and awkward redirects.

The result is not a universal "best" layout for everyone, but the best layout the optimizer can find for the corpus and scoring model you give it.

## Installation

The easiest way to set up the project locally for development is to install it in "editable" mode. This makes the `keyboard-ai` command available everywhere in your terminal:

```bash
git clone https://github.com/yuazi/keyboard-AI.git
cd keyboard-AI
pip install -e .
```

After installing, check that it's working:
```bash
keyboard-ai --help
```

---

## Quick start

### Training a layout
You can train a layout on the **bundled sample corpus** with:
```bash
keyboard-ai train --generations 250 --population 64
```

Or paste your own writing directly (Ctrl-D when finished):
```bash
keyboard-ai train --stdin
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

`keyboard-ai` currently exposes two main commands:

- `train`: Search for a better layout by learning from text.
- `export`: Generate configuration files for external tools.

### Input Sources

`train` and `score` can learn from:

- `--corpus file1.txt file2.txt` (local files)
- `--stdin` (pasted text in terminal)
- *Default*: The bundled sample corpus if no input is given.

Models are only saved when you provide the `--output` flag.

---

## Development & Verification

I've set up some modern tools to keep the code clean and correct:

### Run Tests
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

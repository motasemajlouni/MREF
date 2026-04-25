# MREF

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Automatic evaluation metric accompanying the paper **MREF: A Multi-Aspect Reference-Free Evaluation Framework for Text Simplification**.

## Authors

* AlMotasem Bellah Al Ajlouni
* Jinlong Li

## Overview

MREF is a modular, reference-free evaluation framework for text simplification (TS). It evaluates output quality along three dimensions:

* **S-axis**: Simplicity
* **M-axis**: Meaning preservation
* **G-axis**: Grammaticality

The package returns four scores:

* `S_axis`
* `M_axis`
* `G_axis`
* `MREFscore`

## Features

* Reference-free scoring
* Interpretable multi-axis evaluation
* Single-pair scoring from strings
* Corpus scoring from line-aligned text files
* Optional axis-only CLI modes
* Optional task-specific weighting through both the Python API and the CLI
* Automatic generation of `pos_prior_<language>.json` from Tatoeba if the file is missing

## Installation

### From source

```bash
git clone https://github.com/motasemajlouni/MREF
cd MREF
python -m pip install .
```

### Recommended environment

A clean Python 3.10 or 3.11 environment is recommended.

For example with conda:

```bash
conda create -n mref python=3.10 -y
conda activate mref
set PYTHONNOUSERSITE=1
python -m pip install --upgrade pip setuptools wheel
python -m pip install .
```

## Resources

The final equation-based version expects a grammar resource file named:

```text
MREF/pos_prior_en.json
```

for English.

If this file is not found at the default path, MREF will automatically build it from the Tatoeba corpus on first use. This means:

* the **first run may take longer**
* **internet access is required** for the initial build
* later runs will reuse the saved resource file

The first run may also download the spaCy English model if it is not already installed.

## Python usage

### Basic example

```python
from MREF import MREF

metric = MREF(language="en")
result = metric.score_axes(
    comp="The man, who was very tall, entered the house.",
    simp="The tall man entered the house."
)

print(result)
```

Example output:

```python
{
    "G_axis": 0.827862,
    "M_axis": 0.830623,
    "S_axis": 0.580331,
    "MREFscore": 0.755441
}
```

### Python usage with custom weights

You can override the default equal-weight aggregation by passing custom `AxisWeights`
either when creating the `MREF` object or when calling `score()` / `score_axes()`.

#### Set custom default weights for the metric instance

```python
from MREF import MREF, AxisWeights

metric = MREF(
    language="en",
    axis_weights=AxisWeights(w_G=0.2, w_M=0.5, w_S=0.3),
)

result = metric.score_axes(
    comp="The man, who was very tall, entered the house.",
    simp="The tall man entered the house."
)

print(result)
```

#### Override the weights for a single call

```python
from MREF import MREF, AxisWeights

metric = MREF(language="en")

result = metric.score_axes(
    comp="The man, who was very tall, entered the house.",
    simp="The tall man entered the house.",
    axis_weights=AxisWeights(w_G=0.2, w_M=0.5, w_S=0.3),
)

print(result)
```

#### Get only the scalar score with custom weights

```python
from MREF import MREF, AxisWeights

metric = MREF(language="en")

score = metric.score(
    comp="The man, who was very tall, entered the house.",
    simp="The tall man entered the house.",
    axis_weights=AxisWeights(w_G=0.2, w_M=0.5, w_S=0.3),
)

print(score)
```

### Notes on weights

* `w_G` controls the contribution of grammaticality
* `w_M` controls the contribution of meaning preservation
* `w_S` controls the contribution of simplicity

The weights are normalized internally, so they do not need to sum to 1 exactly.

## Command Line Interface (CLI)

### Single sentence pair

```bash
mref -r "The man, who was very tall, entered the house." -c "The tall man entered the house."
```

Example output:

```text
Saxis = 0.580331
Maxis = 0.830623
Gaxis = 0.827862
MREFscore = 0.755441
```

### Return only one score

```bash
mref -r "The man, who was very tall, entered the house." -c "The tall man entered the house." --saxis-only
mref -r "The man, who was very tall, entered the house." -c "The tall man entered the house." --maxis-only
mref -r "The man, who was very tall, entered the house." -c "The tall man entered the house." --gaxis-only
mref -r "The man, who was very tall, entered the house." -c "The tall man entered the house." --mref-only
```

### Custom task-specific weights from the CLI

You can also pass custom weights directly from the command line:

```bash
mref -r "The man, who was very tall, entered the house." \
     -c "The tall man entered the house." \
     --w-g 0.2 --w-m 0.5 --w-s 0.3
```

To combine custom weights with one-score output:

```bash
mref -r "The man, who was very tall, entered the house." \
     -c "The tall man entered the house." \
     --w-g 0.2 --w-m 0.5 --w-s 0.3 \
     --mref-only
```

### Corpus mode

If both `-r` and `-c` are file paths, MREF treats them as **line-aligned corpora**:

* line *i* in the reference file is paired with line *i* in the candidate file
* corpus-level averages are printed
* sentence-level scores are written to the output file

```bash
mref -r example/comp.txt -c example/simp.txt -o results.txt
```

Example output:

```text
corpus Saxis = 0.533102
corpus Maxis = 0.935656
corpus Gaxis = 0.714641
corpus MREFscore = 0.752744
```

### Corpus mode with one score only

```bash
mref -r example/comp.txt -c example/simp.txt --mref-only -o mref_only.txt
```

### Corpus mode with custom weights

```bash
mref -r example/comp.txt -c example/simp.txt \
     --w-g 0.2 --w-m 0.5 --w-s 0.3 \
     --mref-only -o results.txt
```

### Language selection

```bash
mref -r "complex sentence" -c "simple sentence" -l en
```

### Limit the automatic Tatoeba build

This is useful for testing:

```bash
mref -r "complex sentence" -c "simple sentence" --max-tatoeba-sentences 50000
```

### Notes on CLI weights

If you pass custom weights from the CLI, you must provide all three together:

* `--w-g`
* `--w-m`
* `--w-s`

The weights must be non-negative, and at least one must be positive. They are normalized internally before scoring.

## Output files

In corpus mode, the output file contains sentence-level scores.

Examples:

### All scores

```text
S_axis ; M_axis ; G_axis ; MREFscore
0.587394 ; 0.922936 ; 0.364495 ; 0.665757
...
```

### One score only

```text
MREFscore
0.755441
...
```

## Repository structure

```text
MREF/
├── .gitignore
├── LICENSE
├── MANIFEST.in
├── README.md
├── setup.py
├── upload_pypi.sh
├── MREF/
│   ├── __init__.py
│   ├── cli.py
│   ├── mref_metric.py
│   └── pos_prior_en.json
└── example/
    ├── comp.txt
    └── simp.txt
```

## Notes

* The public class is `MREF`.
* The current implementation is based on the equation-matching final version.
* If you distribute prebuilt resources, place them inside the `MREF/` package directory.
* For reproducibility, the first run after deleting `pos_prior_en.json` may take longer because the resource will be rebuilt automatically.

## License

MIT

# MREF

[![made-with-python]((https://img.shields.io/badge/Made%20with-Python-red.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Automatic evaluation metric accompanying the paper **MREF: A Multi-Aspect Reference-Free Evaluation Framework for Text Simplification**.

## Authors

* AlMotasem Bellah Al Ajlouni
* Jinlong Li
* Huanhuan Chen

## Overview

MREF is a modular, reference-free evaluation framework for text simplification (TS). It evaluates output quality along three dimensions:

* **S-axis**: Simplicity
* **M-axis**: Meaning preservation
* **G-axis**: Grammaticality

The package returns four scores:

* `S_axis`
* `M_axis`
* `G_axis`
* `MREF`

## Features

* Reference-free scoring
* Interpretable multi-axis evaluation
* Single-pair scoring from strings
* Corpus scoring from line-aligned text files
* Optional axis-only CLI modes
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

## Python usage

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
    "MREF": 0.755441
}
```

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
corpus Maxis = 0.935645
corpus Gaxis = 0.714641
corpus MREFscore = 0.752739
```

### Corpus mode with one score only

```bash
mref -r example/comp.txt -c example/simp.txt --mref-only -o mref_only.txt
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

## Citation

```bibtex
@article{alajlouni2026mref,
  title   = {MREF: A Multi-Aspect Reference-Free Evaluation Framework for Text Simplification},
  author  = {AlMotasem Bellah Al Ajlouni and Jinlong Li and Huanhuan Chen},
  journal = {Language Resources and Evaluation},
  year    = {under review}
}
```

## License

MIT


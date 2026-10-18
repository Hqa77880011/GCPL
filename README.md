# GCPL

Implementation of **Relation-preserving graph continual prototype learning for replay-free class-incremental vibration fault diagnosis**.

## Installation

Requires Python >=3.11.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5, NumPy 2.1.3, SciPy 1.15.3 and scikit-learn 1.6.1; neural entries use the installed PyTorch 2.7.1+cu118. Checks cover Python syntax, TOML/JSON and CLI options against parser definitions. Training and evaluation were not run.

## Data and preprocessing

Download original [CWRU recordings](https://engineering.case.edu/bearingdatacenter/download-data-file) or equivalent bearing signals and convert single-channel vectors to NPY. A JSON recording manifest has `file` (relative NPY path), `label`, `group` and `split` (`train,dev,test`). `configs/sessions.json` fixes class-introduction sessions. Keep complete recordings together; base-session development data need at least two windows in a class for dispersion estimation.

```bash
gcpl prepare --manifest data/recordings.json --sessions configs/sessions.json --output data/prepared
```

## Training and inference

```bash
gcpl train --data data/prepared --output runs/full --epochs 40 --seed 42
```

## Evaluation

```bash
gcpl evaluate --data data/prepared --checkpoints runs/full --output runs/full/evaluation
```

## Ablations

Use `--anchor-weight 0`, `--edge-weight 0`, `--rank-weight 0`, `--boundary-weight 0`, `--no-reliability` or `--global-rebuild` for component controls. For few-shot increments set `--shots 1` or `--shots 5` consistently across methods.

```bash
gcpl train --data data/prepared --output runs/no_graph --alpha 0 --edge-weight 0 --rank-weight 0 --boundary-weight 0 --epochs 40 --seed 42
gcpl evaluate --data data/prepared --checkpoints runs/no_graph --output runs/no_graph/evaluation
```

## Outputs

`data/prepared/` contains session-specific window/STFT archives. `runs/full/session_*.pt` stores encoders, prototypes, dispersions and sparse relations. Evaluation writes `metrics.json` and `incremental.png`.

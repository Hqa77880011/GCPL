# GCPL

Implementation of **Relation-preserving graph continual prototype learning for replay-free class-incremental vibration fault diagnosis**.

GCPL stores class prototypes, diagonal dispersions, and sparse relation weights between incremental sessions. New classes insert local edges; classification uses graph-smoothed prototypes. Prototype anchoring, old-edge consistency, neighbor ranking, and graph-implied old/new boundary losses constrain updates without retaining old training windows.

## Install and prepare

```bash
python -m venv .venv
python -m pip install -e .
```

Use Python 3.11+, activate the environment, and select a suitable PyTorch build for CPU or CUDA. Obtain the original CWRU/Paderborn/MFPT/SEU bearing recordings from their maintainers and map the chosen fault taxonomy explicitly. Convert the selected single-channel vectors to NPY. Assign source recording or bearing groups to partitions before windowing.

Create a JSON manifest whose entries have `file` (NPY path relative to the manifest), `label` (class name), `group` (recording/bearing ID), and `split` (`train`, `dev`, or `test`). Each class must occur in exactly one introduction session, with training/test data for that class. Base-session `dev` data must contain at least two windows in a class to estimate dispersion. The sample `configs/sessions.json` introduces two base classes and then outer-race/ball faults; adapt it to the dataset without inventing unavailable labels.

```bash
gcpl prepare --manifest recordings.json --sessions configs/sessions.json --output data/prepared
gcpl train --data data/prepared --output runs/full --epochs 40 --seed 42
gcpl evaluate --data data/prepared --checkpoints runs/full --output outputs/full
```

Preparation standardizes each 2048-sample window and computes log-magnitude STFT using 256-sample frames and 128 overlap. The default window stride is 1024 and never crosses the supplied recording. Training builds a compact 2D convolutional encoder with 64-dimensional normalized embeddings. AdamW uses 0.001. Set `--device cuda` to use a GPU or keep the CPU default.

For few-shot increments, `--shots 1` or `--shots 5` limits examples per new class after the base session. Old training NPZ files remain available for independent runs, but the training loop reads only the current session's training file. Previously learned encoder parameters and prototype statistics are restored; only base `dev` data supply a dispersion prior. That prior is recomputed after base training and frozen for all later one-shot classes.

## Comparisons and ablations

```bash
gcpl train --data data/prepared --output runs/no_graph --alpha 0 --edge-weight 0 --rank-weight 0 --boundary-weight 0
gcpl train --data data/prepared --output runs/no_anchor --anchor-weight 0
gcpl train --data data/prepared --output runs/no_edges --edge-weight 0
gcpl train --data data/prepared --output runs/no_rank --rank-weight 0
gcpl train --data data/prepared --output runs/no_boundary --boundary-weight 0
gcpl train --data data/prepared --output runs/no_reliability --no-reliability
gcpl train --data data/prepared --output runs/global --global-rebuild
```

The no-graph control is nearest-prototype continual learning with the same encoder and anchor. Other runs remove one named mechanism or globally rebuild nearest-neighbor support at each update. `--neighbors` is five, `--temperature` is 0.2, reliability scale `--eta` is 0.1, smoothing `--alpha` is 0.3, and classification distance temperature is 0.1. Objective weights and margins are exposed in `gcpl train --help`; select them on a separate registered base-validation protocol before test evaluation.

Local insertion retains the previous old/old edge support and adds new-node connections if the new node enters an old node's nearest-neighbor list. This preserves the previous support at insertion, but old-node degree can grow beyond five across sessions. Actual serialized sparse-state bytes are reported instead of assuming a strict fixed-degree storage bound. The global-rebuild control saves its final recomputed support, matching the graph used during its last optimization update.

## Outputs

Each `session_*.pt` stores the encoder, prototypes, dispersions, frozen prior, sparse edge indices/weights, and session training time. No raw waveform or historical embedding is stored in the model. `metrics.json` reports average incremental accuracy (AIA), cumulative/old/new accuracy, balanced accuracy, old-class forgetting against the best prior accuracy, 15-bin confidence ECE, base-edge drift, neighbor-rank drift, serialized relation-statistic bytes, training seconds, and evaluation time per window. `incremental.png` plots cumulative accuracy by session.

Rank drift is the fraction of retained base-neighbor ranks that move by more than one position; edge drift is the Frobenius norm of the current base block minus its original adjacency. Both compare the same base classes throughout learning. Evaluation uses test windows only to calculate these outputs, never to change prototypes. Repeat class orders, seeds, working-condition manifests, and few-shot settings as separate runs. No experiment scores are included.

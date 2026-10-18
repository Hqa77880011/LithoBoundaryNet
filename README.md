# LithoBoundaryNet

Implementation of **Topology-preserving SEM defect segmentation**.

## Installation

Requires Python `>=3.11`. Run from the repository root in an activated environment.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5; numpy 2.1.3, scipy 1.15.3, torch 2.7.1+cu118, torchvision 0.22.1, scikit-learn 1.6.1, matplotlib 3.10.0, pillow 12.3.0. Checks: Python syntax, configuration parsing and CLI/source argument matching. Training and full evaluation were not run.

## Data and preprocessing

Supply SEM image/mask records from your licensed inspection dataset; this repository has no bundled fab data. Preserve source_defect_id across every augmented image.

Store `x` as float32 [N,C,H,W], `y` as binary [N,1,H,W], and `group` as source defect IDs. Optional `instances` is the integer source-instance mask [N,H,W]. Images use a fixed 512x512 grid. Foreground labels use nearest-neighbor resizing. Intensities are scaled to [0,1].

Create `data/manifest.jsonl` with one row per source sample, for example `{"x":"images/1.png","y":"masks/1.png","group":"defect-1"}`; paths resolve relative to the manifest.

```bash
python -m lithoboundarynet.pack --manifest data/manifest.jsonl --output data/raw.npz --channels 1 --target mask
python -m lithoboundarynet prepare --source data/raw.npz --output data/prepared --seed 17
```

Use `group` to keep every derived view of one source object in the same partition. An explicit integer `split` array (0=train, 1=validation, 2=test) preserves a registered split. Otherwise groups are assigned 60/20/20. Every stored field must share the sample axis. Files are loaded without pickle.

## Training and inference

Set input dimensions, device, training budget and `seed` in `configs/default.json` (default seed 17).

```bash
python -m lithoboundarynet train --config configs/default.json --data data/prepared --output runs/main
```

## Evaluation

```bash
python -m lithoboundarynet evaluate --checkpoint runs/main/lithoboundarynet/model.pt --data data/prepared/test.npz --output results/lithoboundarynet
```

## Ablations

```bash
python -m lithoboundarynet suite --config configs/default.json --data data/prepared --output runs/comparison
```

Select a subset with `--methods`: `lithoboundarynet`, `unet`, `no_refinement`, `no_boundary`, `no_distance`, `no_topology`, `uniform_topology`.

## Outputs

`runs/main/lithoboundarynet/model.pt` stores the validation-selected weights, method and configuration; `history.jsonl` records measured epoch losses and time. Evaluation writes `predictions.npz` and `metrics.json` under `results/lithoboundarynet`.

```bash
python -m lithoboundarynet plot --inputs runs/main/lithoboundarynet results/lithoboundarynet --output figures
```

`figures/` contains loss curves and metric plots from the supplied run directories.

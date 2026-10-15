# LithoBoundaryNet

Implementation of **Topology-preserving SEM defect segmentation**.

Implements ResNet-34 U-Net features, boundary/distance heads, spatial residual refinement and instance-wise elongation-adaptive clDice. The region, boundary, distance and topology loss weights are 1, 0.3, 0.2 and 0.2.

## Install

Use Python 3.11 or newer. Install a matching PyTorch/TorchVision build for the target CPU or CUDA device, then install the package.

```bash
git clone https://github.com/Hqa77880011/LithoBoundaryNet.git
cd LithoBoundaryNet
python -m venv .venv
```

Activate with `source .venv/bin/activate` or `.venv\Scripts\Activate.ps1` in PowerShell.

```bash
python -m pip install -e .
```

## Prepare data

Use SEM image/mask records from the inspection dataset. The manuscript does not give a public download for the synthetic and real fab sets. Preserve source_defect_id across every augmented image.

Store `x` as float32 [N,C,H,W], `y` as binary [N,1,H,W], and `group` as source defect IDs. Optional `instances` is the integer source-instance mask [N,H,W]. Images use a fixed 512x512 grid. Foreground labels use nearest-neighbor resizing. Intensities are scaled to [0,1].

```bash
python -m lithoboundarynet prepare --source data/raw.npz --output data/prepared --seed 17
```

Use `group` to keep every derived view of one source object in the same partition. An explicit integer `split` array (0=train, 1=validation, 2=test) preserves a registered split. Otherwise groups are assigned 60/20/20. Every stored field must share the sample axis. Files are loaded without pickle.

## Train

Edit `configs/default.json` for the input dimensions, device and training budget.

```bash
python -m lithoboundarynet train --config configs/default.json --data data/prepared --output runs/main
python -m lithoboundarynet suite --config configs/default.json --data data/prepared --output runs/comparison
```

`--methods` selects a subset. lithoboundarynet, unet, no_refinement, no_boundary, no_distance, no_topology and uniform_topology.

The common driver uses AdamW with cosine learning-rate decay, validation-objective early stopping and a fixed seed. Custom federated/continual drivers use their registered round/task budgets. It saves only a validation-selected model and measured epoch history. `learning_rate`, `weight_decay`, `batch_size`, `epochs`, `patience` and `device` control optimization. Use separate output roots for independent runs.

## Evaluate and visualize

```bash
python -m lithoboundarynet evaluate --checkpoint runs/main/lithoboundarynet/model.pt --data data/prepared/test.npz --output results/lithoboundarynet
python -m lithoboundarynet plot --inputs runs/main/lithoboundarynet results/lithoboundarynet --output figures
```

`model.pt` contains model parameters, configuration and method identity. `history.jsonl` contains measured training/validation loss and epoch time. Evaluation writes `predictions.npz` and `metrics.json`. Plotting creates loss curves and metric comparisons from supplied run directories.

Dice, foreground IoU, two-class mIoU, three-pixel inside-boundary IoU, HD95 in pixels, clDice, small-instance recall and elongated-instance fragmentation, plus parameters and measured inference time. Empty one-sided masks receive the image diagonal as HD95. Instance recall uses one-to-one IoU>=0.5 matching; components below four pixels do not count as fragmentation.

## Implementation conventions

Without supplied instance IDs, connected components define instances. Fewer than 16 foreground pixels receive unit topology weight. Default early stopping uses validation objective; the backbone and loss are shared across component controls. Augmentation and domain splits should be prepared before NPZ conversion.


## Pack image records

A JSONL manifest can point to images, masks and NPY feature arrays. Paths are relative to the manifest. All rows use identical keys; group stores the source object ID. For example, an image-classification row is `{"x":"images/1.png","y":0,"group":"object-1"}`.

```bash
python -m lithoboundarynet.pack --manifest data/manifest.jsonl --output data/raw.npz --channels 1 --target mask
```

Inference timing runs the deployment forward pass separately from calibration, commutation and other evaluator diagnostics, after one warmup. Batch P50/P95 and examples/s retain the configured batch size. MAC/FLOP counts cover Conv/Linear/GRU multiply-adds; process RSS includes the dataset and Python runtime.

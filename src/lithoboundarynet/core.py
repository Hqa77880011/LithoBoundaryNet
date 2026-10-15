import numpy as np
import torch
from scipy import ndimage
from torch import nn
from torch.nn import functional as F

from .vision import ResNetUNet, cldice_loss, dice_loss, segmentation_metrics


METHODS = [
    "lithoboundarynet",
    "unet",
    "no_refinement",
    "no_boundary",
    "no_distance",
    "no_topology",
    "uniform_topology",
]


class Model(nn.Module):
    def __init__(self, cfg, method):
        super().__init__()
        self.method = method
        width = cfg["width"]
        self.backbone = ResNetUNet(cfg["channels"], width)
        self.boundary = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, 1, 1),
        )
        self.distance = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, 1, 1),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(width + 2, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.Sigmoid(),
        )
        self.head = nn.Conv2d(width, 1, 1)

    def forward(self, x):
        feature = self.backbone.features(x.float())
        boundary = self.boundary(feature)
        distance = self.distance(feature).tanh()
        if self.method in ("unet", "no_refinement"):
            refined = feature
        else:
            b = torch.zeros_like(boundary) if self.method == "no_boundary" else boundary.sigmoid()
            d = torch.zeros_like(distance) if self.method == "no_distance" else distance
            refined = feature + feature * self.gate(torch.cat([feature, b, d], 1))
        return self.head(refined), boundary, distance


def targets(y):
    masks = y.detach().cpu().numpy()
    boundaries, distances = [], []
    for mask in masks:
        binary = mask.squeeze() > 0.5
        boundary = binary & ~ndimage.binary_erosion(binary)
        distance = ndimage.distance_transform_edt(binary) - ndimage.distance_transform_edt(~binary)
        distance = distance / max(np.abs(distance).max(), 1)
        boundaries.append(boundary[None])
        distances.append(distance[None])
    return (
        torch.as_tensor(np.stack(boundaries), device=y.device, dtype=torch.float32),
        torch.as_tensor(np.stack(distances), device=y.device, dtype=torch.float32),
    )


def topology(probability, y, instances, uniform=False):
    losses, weights = [], []
    for index in range(len(y)):
        labels = (
            instances[index].detach().cpu().numpy().squeeze()
            if instances is not None
            else ndimage.label(y[index].detach().cpu().numpy().squeeze() > 0.5)[0]
        )
        for label in np.unique(labels):
            if label == 0:
                continue
            binary = labels == label
            coordinates = np.argwhere(binary)
            if len(coordinates) < 16 or uniform:
                weight = 1.0
            else:
                eigen = np.linalg.eigvalsh(np.cov(coordinates.T))
                elongation = 1 - (eigen[0] + 1e-7) / (eigen[-1] + 1e-7)
                weight = 1 + min(elongation / 0.75, 1)
            low = np.maximum(coordinates.min(0) - 3, 0)
            high = np.minimum(coordinates.max(0) + 4, binary.shape)
            p = probability[index : index + 1, :, low[0] : high[0], low[1] : high[1]]
            target = torch.as_tensor(
                binary[low[0] : high[0], low[1] : high[1]], device=y.device, dtype=y.dtype
            )[None, None]
            losses.append(cldice_loss(p, target) * weight)
            weights.append(weight)
    return torch.stack(losses).sum() / sum(weights) if losses else probability.sum() * 0


def loss(model, batch, cfg):
    y = batch["y"].float()
    if y.ndim == 3:
        y = y[:, None]
    logits, boundary, distance = model(batch["x"])
    probability = logits.sigmoid()
    result = F.binary_cross_entropy_with_logits(logits, y) + dice_loss(probability, y)
    if model.method == "unet":
        return result
    b, d = targets(y)
    if model.method != "no_boundary":
        result = result + 0.3 * (
            F.binary_cross_entropy_with_logits(boundary, b) + dice_loss(boundary.sigmoid(), b)
        )
    if model.method != "no_distance":
        result = result + 0.2 * F.smooth_l1_loss(distance, d)
    if model.method != "no_topology":
        result = result + 0.2 * topology(
            probability, y, batch.get("instances"), model.method == "uniform_topology"
        )
    return result


def predict(model, batch, cfg, calibration=None):
    p, b, d = model(batch["x"])
    return {"probability": p.sigmoid(), "boundary": b.sigmoid(), "distance": d}


def metrics(prediction, data, cfg):
    return segmentation_metrics(prediction["probability"], data["y"], data.get("instances"))


def inference(model, batch, cfg):
    return model(batch["x"])

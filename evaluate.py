from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import RadarPoseDataset
from models import AgileMmWaveHPE
from train import load_config, select_device


def pa_majpe(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample similarity Procrustes alignment, followed by MAJPE in metres."""
    prediction = prediction - prediction.mean(dim=1, keepdim=True)
    target_centered = target - target.mean(dim=1, keepdim=True)
    covariance = prediction.transpose(1, 2) @ target_centered
    left, _, right_t = torch.linalg.svd(covariance)
    rotation = right_t.transpose(1, 2) @ left.transpose(1, 2)
    determinant = torch.det(rotation)
    correction = torch.eye(3, device=prediction.device).expand(len(prediction), -1, -1).clone()
    correction[:, 2, 2] = determinant.sign()
    rotation = right_t.transpose(1, 2) @ correction @ left.transpose(1, 2)
    scale = (target_centered * (prediction @ rotation)).sum(dim=(1, 2)) / prediction.square().sum(dim=(1, 2)).clamp_min(1e-8)
    return ((prediction @ rotation) * scale[:, None, None] - target_centered).norm(dim=-1).mean(dim=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    arguments = parser.parse_args()
    config, device = load_config(arguments.config), select_device(load_config(arguments.config)["device"])
    checkpoint = torch.load(arguments.checkpoint, map_location=device)
    model = AgileMmWaveHPE(checkpoint.get("model", config["model"])).to(device)
    model.load_state_dict(checkpoint["state_dict"]); model.eval()
    data_cfg = config["data"]
    loader = DataLoader(RadarPoseDataset(Path(data_cfg["root"]) / data_cfg["test_file"]), batch_size=config["train"]["batch_size"], num_workers=data_cfg["num_workers"])
    absolute_errors, aligned_errors = [], []
    with torch.no_grad():
        for radar, joints in loader:
            prediction, joints = model(radar.to(device)), joints.to(device)
            absolute_errors.append((prediction - joints).norm(dim=-1).reshape(-1))
            aligned_errors.append(pa_majpe(prediction, joints))
    print(f"MAJPE: {torch.cat(absolute_errors).mean().item() * 1000:.3f} mm")
    print(f"PA-MAJPE: {torch.cat(aligned_errors).mean().item() * 1000:.3f} mm")


if __name__ == "__main__":
    main()
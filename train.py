from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.nn import functional as functional
from torch.utils.data import DataLoader

from data import RadarPoseDataset
from models import AgileMmWaveHPE


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def select_device(requested: str) -> torch.device:
    return torch.device("cuda" if requested == "auto" and torch.cuda.is_available() else "cpu" if requested == "auto" else requested)


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    errors = []
    with torch.no_grad():
        for radar, joints in loader:
            prediction = model(radar.to(device))
            errors.append((prediction.cpu() - joints).norm(dim=-1).reshape(-1))
    return torch.cat(errors).mean().item() * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Agile mmWave HPE on canonical HuPR NPZ splits.")
    parser.add_argument("--config", required=True)
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    seed = config["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = select_device(config["device"])
    root, data_cfg, train_cfg = Path(config["data"]["root"]), config["data"], config["train"]
    train_loader = DataLoader(RadarPoseDataset(root / data_cfg["train_file"]), batch_size=train_cfg["batch_size"], shuffle=True, num_workers=data_cfg["num_workers"])
    val_loader = DataLoader(RadarPoseDataset(root / data_cfg["val_file"]), batch_size=train_cfg["batch_size"], num_workers=data_cfg["num_workers"])
    model = AgileMmWaveHPE(config["model"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg["epochs"])
    checkpoint_dir = Path(train_cfg["checkpoint_dir"]); checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{Path(arguments.config).stem}.pth"
    best_error = float("inf")
    for epoch in range(1, train_cfg["epochs"] + 1):
        model.train()
        for radar, joints in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = functional.mse_loss(model(radar.to(device)), joints.to(device))
            loss.backward(); optimizer.step()
        scheduler.step()
        val_error = evaluate(model, val_loader, device)
        print(f"epoch={epoch:03d} val_majpe_mm={val_error:.3f}")
        if val_error < best_error:
            best_error = val_error
            torch.save({"state_dict": model.state_dict(), "model": config["model"], "config": config, "val_majpe_mm": val_error}, checkpoint_path)


if __name__ == "__main__":
    main()
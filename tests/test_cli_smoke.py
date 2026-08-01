import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml


def test_train_evaluate_and_export(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    data_root = tmp_path / "HuPR"
    data_root.mkdir()
    for split in ("train", "val", "test"):
        np.savez(data_root / f"{split}.npz", rad=np.random.rand(2, 8, 8, 4).astype(np.float32), joints=np.random.rand(2, 2, 3).astype(np.float32))
    config = {
        "seed": 1, "device": "cpu",
        "data": {"root": str(data_root), "train_file": "train.npz", "val_file": "val.npz", "test_file": "test.npz", "num_workers": 0},
        "model": {"range_bins": 8, "angle_bins": 8, "doppler_bins": 4, "num_joints": 2, "range_resolution_m": 0.1, "angle_min_deg": -60, "angle_max_deg": 60, "max_velocity_mps": 2.0, "spatial_bounds_m": [0.0, 1.0], "spatial_bounds_deg": [-60, 60], "velocity_bounds_mps": [0.0, 2.0], "velocity_std_bounds_mps": [0.0, 2.0], "local_window_radius": 1, "coarse_kernel": [2, 2, 2], "medium_kernel": [1, 1, 1], "hidden_dims": [8, 8]},
        "train": {"batch_size": 2, "epochs": 1, "learning_rate": 0.001, "weight_decay": 0.0, "checkpoint_dir": str(tmp_path / "checkpoints")},
    }
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    subprocess.run([sys.executable, "train.py", "--config", str(config_path)], cwd=root, check=True)
    checkpoint = tmp_path / "checkpoints" / "smoke.pth"
    subprocess.run([sys.executable, "evaluate.py", "--config", str(config_path), "--checkpoint", str(checkpoint)], cwd=root, check=True)
    subprocess.run([sys.executable, "export_onnx.py", "--checkpoint", str(checkpoint), "--output", str(tmp_path / "model.onnx")], cwd=root, check=True)
    assert (tmp_path / "model.onnx").exists()
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class RadarPoseDataset(Dataset):
    """Canonical dataset: rad [N,R,A,D], joints [N,J,3], stored in one NPZ file."""

    def __init__(self, path: str | Path) -> None:
        archive = np.load(path)
        if "rad" not in archive or "joints" not in archive:
            raise KeyError(f"{path} must contain 'rad' and 'joints' arrays.")
        self.rad = archive["rad"]
        self.joints = archive["joints"].astype(np.float32)
        if self.rad.ndim != 4 or self.joints.ndim != 3 or len(self.rad) != len(self.joints):
            raise ValueError("Expected rad [N,R,A,D] and joints [N,J,3] with matching N.")

    def __len__(self) -> int:
        return len(self.rad)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Original RAD tensors can be complex; SSP/MCP operate on their reflection magnitude.
        radar = np.abs(self.rad[index]).astype(np.float32, copy=False)
        return torch.from_numpy(radar), torch.from_numpy(self.joints[index])
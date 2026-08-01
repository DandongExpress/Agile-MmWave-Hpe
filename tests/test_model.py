import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models import AgileMmWaveHPE


def test_forward_shape_and_deterministic_front_end() -> None:
    root = Path(__file__).resolve().parents[1]
    with open(root / "configs" / "balanced.yaml", encoding="utf-8") as file:
        model = AgileMmWaveHPE(yaml.safe_load(file)["model"])
    radar = torch.rand(2, 64, 64, 16)
    output = model(radar)
    assert output.shape == (2, 17, 3)
    assert sum(parameter.numel() for parameter in model.ssp.parameters()) == 0
    assert sum(parameter.numel() for parameter in model.mcp.parameters()) == 0
    assert sum(parameter.numel() for parameter in model.hmsf.parameters()) == 0
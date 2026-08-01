from __future__ import annotations

import argparse

import torch

from models import AgileMmWaveHPE


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="balanced.onnx")
    arguments = parser.parse_args()
    checkpoint = torch.load(arguments.checkpoint, map_location="cpu")
    config = checkpoint["model"]
    model = AgileMmWaveHPE(config).eval()
    model.load_state_dict(checkpoint["state_dict"])
    sample = torch.zeros(1, config["range_bins"], config["angle_bins"], config["doppler_bins"])
    torch.onnx.export(model, sample, arguments.output, opset_version=15, input_names=["radar_cube"], output_names=["joints"], dynamic_axes={"radar_cube": {0: "batch"}, "joints": {0: "batch"}})
    print(f"Exported {arguments.output}")


if __name__ == "__main__":
    main()
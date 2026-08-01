"""Convert exported HuPR RAD/pose arrays to the repository's canonical NPZ format."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rad", required=True, help="Numpy file with [N,64,64,16] RAD cubes.")
    parser.add_argument("--joints", required=True, help="Numpy file with [N,17,3] joints in metres.")
    parser.add_argument("--output", required=True, help="Output .npz path, e.g. data/HuPR/train.npz.")
    arguments = parser.parse_args()
    rad, joints = np.load(arguments.rad), np.load(arguments.joints)
    if rad.ndim != 4 or joints.ndim != 3 or rad.shape[0] != joints.shape[0]:
        raise ValueError("Expected RAD [N,R,A,D] and joints [N,J,3] with matching sample count.")
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, rad=rad, joints=joints.astype(np.float32))


if __name__ == "__main__":
    main()
from __future__ import annotations

import argparse
import time

import numpy as np
import onnxruntime as ort


def main() -> None:
    parser = argparse.ArgumentParser(description="Run exported Agile mmWave HPE ONNX inference on a RAD cube.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--rad", required=True, help="One .npy RAD cube [64,64,16]; complex input is converted to magnitude.")
    arguments = parser.parse_args()
    session = ort.InferenceSession(arguments.model, providers=["CPUExecutionProvider"])
    radar = np.abs(np.load(arguments.rad)).astype(np.float32)
    if radar.ndim != 3:
        raise ValueError("Expected one radar cube with shape [range, angle, doppler].")
    start = time.perf_counter()
    joints = session.run(None, {session.get_inputs()[0].name: radar[None]})[0]
    print(f"latency_ms={(time.perf_counter() - start) * 1000:.2f}")
    print(joints[0])


if __name__ == "__main__":
    main()
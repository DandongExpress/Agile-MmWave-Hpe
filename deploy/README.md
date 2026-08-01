# Raspberry Pi 5 Deployment

The paper uses Raspberry Pi 5 CPU-only inference with a TI AWR1843BOOST producing $64\times64\times16$ RAD cubes. Install the dependencies from `requirements.txt`, export a trained checkpoint, then run:

```bash
python deploy/run_rpi.py --model balanced.onnx --rad frame.npy
```

`frame.npy` must contain one range-angle-Doppler cube in that axis order. Radar packet acquisition, ADC reconstruction, and FFT generation are board/firmware-specific and must produce this tensor before this runner is invoked.
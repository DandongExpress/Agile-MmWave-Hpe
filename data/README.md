# HuPR Data Contract

The published HuPR release may require its own extraction step. Export each split into two aligned NumPy arrays:

- `rad`: `[N, 64, 64, 16]` RAD cubes. Complex arrays are accepted and converted to magnitude at load time.
- `joints`: `[N, 17, 3]` 3D joint coordinates in metres.

Create the files expected by the default configuration:

```powershell
python data/prepare_hupr.py --rad <train-rad.npy> --joints <train-joints.npy> --output data/HuPR/train.npz
python data/prepare_hupr.py --rad <val-rad.npy> --joints <val-joints.npy> --output data/HuPR/val.npz
python data/prepare_hupr.py --rad <test-rad.npy> --joints <test-joints.npy> --output data/HuPR/test.npz
```

The split must follow HuPR's official partition. RAD axes must remain range, azimuth angle, Doppler; do not transpose them.
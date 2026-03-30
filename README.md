**Agile-Mmwave-Hpe**
> Physics-guided mmWave preprocessing for agile human pose estimation. Replaces learned front-ends with deterministic SSP/MCP/HMSF modules, reducing parameters by 55.7–88.9% while enabling real-time deployment on Raspberry Pi 5.
[Paper & Supplementary]](https://arxiv.org/abs/2603.08236)**

---

**Overview**
Existing mmWave-based HPE systems are over-parameterized yet underperform vision baselines — we show this mismatch originates primarily in learned front-end preprocessing modules. This repository provides a physics-guided alternative that replaces those modules with three deterministic components:
SSP (Spatial Structure Preservation) — range–angle masking based on anthropometric priors
MCP (Motion Continuity Preservation) — Doppler-based dominant velocity extraction and local consistency filtering
HMSF (Hierarchical Multi-Scale Fusion) — 3D pooling aligned with torso, limb, and joint scales
A lightweight MLP regressor then maps the resulting compact descriptor to 3D joint coordinates. The full pipeline runs at 18.2 FPS on a Raspberry Pi 5 with only 7.3 MB peak RAM.

---

**Results**
Method	Params (M)	MAJPE (mm) ↓	PA-MAJPE (mm) ↓	FLOPs (M)
HuprModel	324.9	65.37	58.11	454
MVDoppler-Pose	36.7	69.71	66.56	84
Ours (Full)	5.1	64.16	60.29	14
Evaluated on the HuPR benchmark. See the paper for full comparisons and bidirectional replacement experiments.

---
**Getting Started**
Requirements
```bash
pip install -r requirements.txt
```
Key dependencies: `torch`, `numpy`, `onnxruntime`
Data Preparation
Download the HuPR dataset and place it under `data/HuPR/`. Follow the preprocessing instructions in `data/README.md`.
Training
```bash
python train.py --config configs/balanced.yaml
```
Evaluation
```bash
python evaluate.py --config configs/balanced.yaml --checkpoint checkpoints/balanced.pth
```
Raspberry Pi Deployment
Export to ONNX and run on-device:
```bash
python export_onnx.py --checkpoint checkpoints/balanced.pth
# Copy the exported .onnx file to the Raspberry Pi, then:
python deploy/run_rpi.py --model balanced.onnx
```
See `deploy/README.md` for full hardware setup and radar configuration details.

---

**Runtime Profiles**
All profiles share the same 5.1M-parameter back end; only front-end hyperparameters differ.
Profile	FLOPs (M)	Peak RAM (MB)	MAJPE (mm)
Ultra-Light	2.1	2.8	72.43
Light	3.5	4.6	67.28
Balanced	5.8	7.3	64.16
High-Precision	8.2	10.5	62.87
Ultra-Precision	11.4	14.2	62.15


---
**Repository Structure**
```
agile-mmwave-hpe/
├── configs/          # Hyperparameter profiles (yaml)
├── data/             # Dataset preparation scripts
├── deploy/           # Raspberry Pi deployment scripts and instructions
├── models/           # SSP, MCP, HMSF modules and MLP regressor
├── checkpoints/      # Pretrained weights (see Releases)
├── train.py
├── evaluate.py
└── export_onnx.py
```
---
**Citation**
If you find this work useful, please cite:
```bibtex
@article{zheng2026learn,
  title={Why Learn What Physics Already Knows? Realizing Agile mmWave-based Human Pose Estimation via Physics-Guided Preprocessing},
  author={Zheng, Shuntian and Li, Jiaqi and Ni, Minzhe and Lu, Xiaoman and Guan, Yu},
  journal={arXiv preprint arXiv:2603.08236},
  year={2026}
}
```
---
**Acknowledgement**
This research is funded by the Department of Computer Science, University of Warwick.
---
License
This project is released under the MIT License.

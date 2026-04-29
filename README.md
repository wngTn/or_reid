# Beyond role-based surgical domain modeling: Generalizable re-identification in the operating room

[![Paper](https://img.shields.io/badge/MedIA'25-Paper-blue)](https://www.sciencedirect.com/science/article/pii/S1361841525002348)
[![arXiv](https://img.shields.io/badge/arXiv-2503.13028-B31B1B.svg)](https://arxiv.org/abs/2503.13028)
[![Website](https://img.shields.io/badge/Project-Website-green)](https://wngtn.github.io/or_reid/)

Official implementation of the paper:  
**"Beyond role-based surgical domain modeling: Generalizable re-identification in the operating room"**  
*Medical Image Analysis (MedIA), 2025.*

[Tony Danjun Wang](https://wngtn.github.io)<sup>\*,1</sup>, [Lennart Bastian](https://www.cs.cit.tum.de/camp/members/lennart-bastian/)<sup>\*,1,2</sup>, [Tobias Czempiel](https://profiles.ucl.ac.uk/98117-tobias-czempiel)<sup>3</sup>, [Christian Heiliger](https://de.linkedin.com/in/christian-heiliger)<sup>4</sup>, [Nassir Navab](https://www.professoren.tum.de/en/navab-nassir)<sup>1,2</sup>

<sup>1</sup>Technical University of Munich, <sup>2</sup>Munich Center for Machine Learning, <sup>3</sup>University College London, <sup>4</sup>University Hospital of Munich (LMU).  
<sup>\*</sup>Equal Contribution.

---

## Overview

Surgical data science increasingly relies on role-based domain models. However, these fail to capture the significance of individual team members and their collaborative dynamics on surgical outcomes. We propose a paradigm shift towards **staff-centric** surgical domain models, which model individual traits as opposed to considering staff as interchangeable surgical roles. 

To achieve this, we address the necessary problem of **person re-identification** in the operating room (OR), which has been hindered due to the challenging visual environment where traditional biometric cues are obscured. To overcome monotonous texture appearances due to standardized attire, we introduce a novel approach that **leverages 3D shape and articulated motion cues** to achieve robust, invariant biometric signatures for personnel re-identification.

### Key highlights
- **Geometry over Texture:** Utilizes 3D point clouds and articulated motion to bypass appearance-based biases (scrubs, gowns).
- **Clinical Performance:** Outperforms RGB counterparts by a **12% margin** in accuracy on authentic clinical data.
- **Robust Tracking (TrackOR):** Integrates geometric signatures into an online tracking pipeline for persistent identity maintenance.
- **3D Activity Imprints:** Novel visualization for surgical team coordination and space utilization.

---

## Installation

This project uses [uv](https://docs.astral.sh/uv/getting-started/installation/) as the package manager.

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Synchronize dependencies
uv sync
```

**Tested With:**
- Python 3.11
- PyTorch 2.7.0 with CUDA 12.8 wheels

---

## Data preparation

We provide scripts to download the datasets used in our study. 
Please fill in the Google Forms [here](https://forms.gle/EGdZBCwjaRHFsfyF7) to get access to the `OR_ReID_13` as well as the `4D-OR_ReID` datasets.
After downloading the datasets, please put them under `./data`

## Checkpoints

Download checkpoints:

```bash
gdown --no-cookies --folder 1BQk-OKbhUBlcqlePo62j-7Y7q-e5amln
```

Expected local layout:

```text
ckpts/
  manifest.yaml
  or_reid_13/
    depth/split_0.pt ... split_3.pt
    rgb/split_0.pt ... split_3.pt
  sustech1k/
    model.pt
```

The OR_ReID_13 checkpoints have one weight file per split (`0,1,2,3`).
SUSTech1K has one checkpoint only. `--ckpt` accepts a direct file, a
directory, or one of the manifest aliases: `or_reid_13/depth`,
`or_reid_13/rgb`, or `sustech1k`.

Available checkpoint presets:

| Alias | Training data | Modality | Splits | Files |
| --- | --- | --- | --- | --- |
| `or_reid_13/depth` | OR_ReID_13 | Depth / point cloud | 4 | `ckpts/or_reid_13/depth/split_0.pt` ... `split_3.pt` |
| `or_reid_13/rgb` | OR_ReID_13 | RGB | 4 | `ckpts/or_reid_13/rgb/split_0.pt` ... `split_3.pt` |
| `sustech1k` | SUSTech1K | Shared pretrained model | 1 | `ckpts/sustech1k/model.pt` |

For OR_ReID_13 presets, `src/eval.py` automatically selects the checkpoint
matching the requested split. For SUSTech1K, the same checkpoint is reused
because SUSTech1K has no four-split setup in this repository.

---

## Evaluation of the checkpoints

Evaluates OR_ReID_13 trained **Depth** model on OR_ReID_13 **Depth**
```bash
python src/eval.py --cfgs configs/PointCloud/intra/OR_ReID_13.yaml --splits all --ckpt or_reid_13/depth
```

Evaluates OR_ReID_13 trained **RGB** model on OR_ReID_13 **RGB**
```bash
python src/eval.py --cfgs configs/RGB/intra/OR_ReID_13.yaml --splits all --ckpt or_reid_13/rgb
```

Evaluates OR_ReID_13 trained **Depth** model on 4D-OR_ReID **Depth**
```bash
python src/eval.py --cfgs configs/PointCloud/intra/OR_ReID_13.yaml \
  --splits all --ckpt or_reid_13/depth --test_dataset 4D-OR_ReID
```

Evaluates OR_ReID_13 trained **RGB** model on 4D-OR_ReID **RGB**
```bash
python src/eval.py --cfgs configs/RGB/intra/OR_ReID_13.yaml \
  --splits all --ckpt or_reid_13/rgb --test_dataset 4D-OR_ReID
```

Evaluates the SUSTech1K pretrained model on OR_ReID_13 **Depth**
```bash
python src/eval.py --cfgs configs/PointCloud/intra/OR_ReID_13.yaml --ckpt sustech1k --splits 0
```

## Usage

### Training

The project supports 4-fold cross-validation by default.

```bash
# Point cloud on OR_ReID_13 — full 4-fold CV
python src/train.py --cfgs configs/PointCloud/intra/OR_ReID_13.yaml

# RGB on 4D-OR_ReID — single split (split 0)
python src/train.py --cfgs configs/RGB/intra/4D-OR_ReID.yaml --splits 0

# Fine-tune from SUSTech1K weights
python src/train.py --cfgs configs/PointCloud/intra/OR_ReID_13.yaml --ckpt sustech1k --splits 0

# Multi-GPU training
torchrun --nproc_per_node=4 src/train.py --cfgs configs/PointCloud/intra/OR_ReID_13.yaml
```

### Evaluation

```bash
# Evaluate all splits from a trained model
python src/eval.py --cfgs configs/PointCloud/intra/OR_ReID_13.yaml --exp_name my_exp

# Cross-Dataset Evaluation (Train on OR_ReID_13, test on 4D-OR_ReID)
python src/eval.py --cfgs configs/PointCloud/intra/OR_ReID_13.yaml --exp_name my_exp --test_dataset 4D-OR_ReID

# Evaluate a specific pre-trained checkpoint
python src/eval.py --cfgs configs/PointCloud/intra/OR_ReID_13.yaml --ckpt or_reid_13/depth
```

---

## Configuration structure

```text
configs/
  PointCloud/
    intra/            # Train & eval on the same dataset (point cloud)
      OR_ReID_13.yaml
      4D-OR_ReID.yaml
    pretrained/       # Eval pre-trained SUSTech1K checkpoint
      SUSTech1K_OR_ReID_13.yaml
      SUSTech1K_4D-OR_ReID.yaml
  RGB/
    intra/            # Train & eval on the same dataset (RGB)
      OR_ReID_13.yaml
      4D-OR_ReID.yaml
    pretrained/       # Eval pre-trained SUSTech1K checkpoint
      SUSTech1K_OR_ReID_13.yaml
      SUSTech1K_4D-OR_ReID.yaml
```

---

## Output structure

```text
output/<exp_name>/<save_name>/
├── checkpoints/    # Model checkpoints per split
├── logs/           # Training/eval logs
└── results.pkl     # Per-split metrics DataFrame
```

---

## Citation

If you find this work useful in your research, please cite:

```bibtex
@article{wang2025_beyond_role,
    title   = {Beyond role-based surgical domain modeling: Generalizable re-identification in the operating room},
    journal = {Medical Image Analysis},
    volume  = {105},
    pages   = {103687},
    year    = {2025},
    author  = {Tony Danjun Wang and Lennart Bastian and Tobias Czempiel and Christian Heiliger and Nassir Navab}
}

@inproceedings{wang2025_trackor,
    title     = {TrackOR: Towards Personalized Intelligent Operating Rooms Through Robust Tracking},
    author    = {Tony Danjun Wang and Christian Heiliger and Nassir Navab and Lennart Bastian},
    booktitle = {Workshop Collaborative Intelligence and Autonomy in Image-guided Surgery (COLAS) at MICCAI},
    year      = {2025},
    publisher = {Springer Nature}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).

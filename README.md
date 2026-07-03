# Brep2Shape

Official implementation of **Brep2Shape: Boundary and Shape Representation Alignment via
Self-Supervised Transformers** (ICML 2026).

[Paper](https://arxiv.org/abs/2602.07429)

> **Release status:** The Brep2Shape pretraining dataset and pretrained checkpoint will be
> released soon. Download links and usage instructions will be added here when they are available.

Brep2Shape learns representations of boundary-representation (B-rep) CAD models by predicting
dense spatial points from parametric Bezier control points. Its Dual Transformer processes face
and edge primitives in parallel, while topology-aware attention exchanges information through
their B-rep connectivity. The pretrained representation can be transferred to solid
classification and face segmentation.

## Highlights

- **Boundary-to-shape self-supervision:** learn from geometric reconstruction targets without
  manual class annotations.
- **Dual Transformer:** encode surface and curve primitives in parallel.
- **Topology-aware attention:** model face-edge dependencies and shared-boundary consistency.
- **Generalizable representations:** transfer the pretrained encoders to classification and
  segmentation tasks.

## Installation

The code targets Python 3.10, PyTorch 2.2, CUDA 12.1, and DGL 2.1. Create the maintained Conda
environment with:

```bash
conda env create -f environment.yml
conda activate brep2shape
```

## Data Interface

Pass a dataset root through `--dataset_dir`. Paths inside a split file may be absolute or relative
to that root. 

### Self-supervised pretraining

Pretraining reads `datasplit.json`. No class label is required:

```json
{
  "train": [
    {
      "face": "features/triangles/sample.pt",
      "topo": "features/topology/sample.pt",
      "graph": "graphs/sample.bin",
      "line_graph": "line_graphs/sample.bin"
    }
  ],
}
```

### Fine-tuning

Classification and segmentation read `datasplit.json` and additionally require `label`:

```json
{
  "train": [
    {
      "face": "features/triangles/sample.pt",
      "topo": "features/topology/sample.pt",
      "graph": "graphs/sample.bin",
      "line_graph": "line_graphs/sample.bin",
      "label": "labels/sample.txt"
    }
  ],
}
```

For classification, `label` may be an integer class ID or a path to one. For segmentation, it
must provide one class ID per B-rep face.

The geometry fields have the following roles:

| Field | Description |
| --- | --- |
| `face` | Bezier-triangle face primitives and sampled face targets |
| `topo` | Bezier-curve edge primitives and face/edge incidence data |
| `graph` | DGL face adjacency graph |
| `line_graph` | DGL edge adjacency graph |
| `label` | Downstream solid-level or face-level annotation |

## Model API

The model-only API is independent of command-line parsing:

```python
from models import Brep2ShapeConfig, DualClassification

config = Brep2ShapeConfig(num_classes=10)
model = DualClassification(config)
```

`models` also exports `DualSegmentation`, `UVPointPrediction`, and the face, edge, and graph
encoders.

## Training

Self-supervised pretraining:

```bash
python pretrain.py train \
  --dataset_dir /path/to/brep2shape_data \
  --batch_size 32 \
  --num_workers 8
```

Classification:

```bash
python classification.py train \
  --dataset_dir /path/to/brep2shape_data \
  --num_classes 10 \
  --batch_size 32 \
  --num_workers 4
```

Face segmentation:

```bash
python segmentation.py train \
  --dataset_dir /path/to/brep2shape_data \
  --num_classes 25 \
  --batch_size 32 \
  --num_workers 4
```

To initialize a downstream model from the self-supervised encoders, provide a Lightning
checkpoint:

```bash
python segmentation.py train \
  --dataset_dir /path/to/brep2shape_data \
  --num_classes 25 \
  --pretrain_checkpoint /path/to/pretraining.ckpt
```

Run artifacts are written to `results/{experiment_name}/{month_day}/{run_id}/`. They include the
resolved arguments, model description, TensorBoard logs, and checkpoints.

## Evaluation

Evaluate a classification checkpoint:

```bash
python classification.py test \
  --dataset_dir /path/to/brep2shape_data \
  --num_classes 10 \
  --checkpoint /path/to/best.ckpt
```

For segmentation, `--checkpoint` accepts either one checkpoint or a directory of `.ckpt` files:

```bash
python segmentation.py test \
  --dataset_dir /path/to/brep2shape_data \
  --num_classes 25 \
  --checkpoint /path/to/best.ckpt
```

Reusable shell wrappers are available in `scripts/` and require paths through environment
variables:

```bash
DATASET_DIR=/path/to/brep2shape_data bash scripts/pretrain.sh
DATASET_DIR=/path/to/brep2shape_data \
  PRETRAIN_CHECKPOINT=/path/to/pretraining.ckpt \
  bash scripts/finetune_with_pretrain.sh
DATASET_DIR=/path/to/brep2shape_data \
  CHECKPOINT=/path/to/best.ckpt \
  TASK=classification \
  bash scripts/test.sh
```

## Results in the Paper

The paper evaluates classification on **FabWave** and **TMCAD**, and face segmentation on
**MFCAD++** and **Fusion360Seg**. Across these tasks, Brep2Shape reports competitive or
state-of-the-art accuracy and faster downstream convergence. It also studies pretraining scale,
sampling density, topology attention, edge-level supervision, and cross-dataset transfer. See the
paper for the complete experimental setup, numerical results, and ablations.

The pretraining dataset and pretrained checkpoint are not included in the current repository
snapshot and will be released soon. The original data preprocessing pipeline is not part of the
planned release. Until the data and checkpoint are available, the published results should be
treated as paper results rather than as metrics reproduced automatically by this repository.

## Citation

If you use this code or method, please cite:

```bibtex
@inproceedings{sunbrep2shape,
  title={Brep2Shape: Boundary and Shape Representation Alignment via Self-supervised Transformers},
  author={Sun, Yuanxu and Ma, Yuezhou and Wu, Haixu and Zeng, Guanyang and Chen, Muye and Wang, Jianmin and Long, Mingsheng},
  booktitle={Forty-third International Conference on Machine Learning}
}
```

## Acknowledgements

This implementation builds on ideas and tooling from prior B-rep representation learning work,
including:

- Qiang Zou and Lizhen Zhu. "Bringing Attention to CAD: Boundary Representation Learning via
  Transformer." *Computer-Aided Design*, 2025.
- Pradeep Kumar Jayaraman, Aditya Sanghi, Joseph G. Lambourne, Karl D. D. Willis, Thomas Davies,
  Hooman Shayani, and Nigel Morris. "UV-Net: Learning from Boundary Representations." *CVPR*, 2021.

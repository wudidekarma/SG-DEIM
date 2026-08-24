# SG-DEIM: Context-Guided Spectral Redistribution and Dual-Stage Gating for End-to-End Small Object Detection

SG-DEIM is an end-to-end detector designed for small object detection. It focuses on preserving weak small-object cues during feature extraction, reducing cross-scale interference during feature fusion, and suppressing irrelevant attention-head responses during decoder query updating.

The framework contains three main components:

- **Context-Guided Spectral Redistribution (CGSR):** enhances useful spectral information with adjacent-scale semantic guidance before encoder projection.
- **Cross-Scale Adaptive Interaction (CSAI):** performs bidirectional interaction between neighboring feature levels before multi-scale aggregation.
- **Query-Conditioned Attention Gating (QCAG):** adaptively filters attention-head responses according to the current object query before query updating.

SG-DEIM is evaluated on **COCO 2017**, **VisDrone2019-DET**, and **AI-TODv1**.

<p align="center">
  <img src="tools/third_party/Figure_1a.png" width="48%" alt="SG-DEIM accuracy result 1">
  <img src="tools/third_party/Figure_1b.png" width="48%" alt="SG-DEIM accuracy result 2">
</p>

---

## Environment

The experiments were conducted with:

- Ubuntu 22.04.5 LTS
- Python 3.11
- PyTorch 2.5.1
- CUDA 13.0
- NVIDIA GeForce RTX 4090

Create the environment:

```bash
conda create -n sgdeim python=3.11.14 -y
conda activate sgdeim
```

Clone the repository:

```bash
git clone https://github.com/wudidekarma/SG-DEIM.git
cd SG-DEIM
```

Install the required packages:

```bash
pip install -r requirements-SGDEIM.txt
```

---

## Pretrained Weights

Download the pretrained backbone checkpoint corresponding to the selected model.

| Model | Backbone | Pretrained Weight |
| --- | --- | --- |
| SG-DEIM-S | DEIMv2 ViT-Tiny | [`vitt_distill.pt`](https://drive.google.com/file/d/1YMTq_woOLjAcZnHSYNTsNg7f0ahj5LPs/view?usp=sharing) |
| SG-DEIM-M | DEIMv2 ViT-Tiny+ | [`vittplus_distill.pt`](https://drive.google.com/file/d/1COHfjzq5KfnEaXTluVGEOMdhpuVcG6Jt/view?usp=sharing) |
| SG-DEIM-L | DINOv3 ViT-S/16 | `dinov3_vits16_pretrain_lvd1689m-08c60483.pth` |
| SG-DEIM-X | DINOv3 ViT-S+/16 | `dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth` |

For SG-DEIM-L and SG-DEIM-X, please download the corresponding pretrained models from the official [DINOv3 repository](https://github.com/facebookresearch/dinov3#pretrained-models).

Place all pretrained weights in:

```text
SG-DEIM/
└── ckpts/
    ├── vitt_distill.pt
    ├── vittplus_distill.pt
    ├── dinov3_vits16_pretrain_lvd1689m-08c60483.pth
    └── dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth
```

Make sure that `weights_path` in the corresponding configuration file matches the checkpoint path.

---

## Datasets

The following datasets are used in this project.

| Dataset | Download |
| --- | --- |
| COCO 2017 | [Official COCO](https://cocodataset.org/#download) |
| VisDrone2019-DET | [Official VisDrone Repository](https://github.com/VisDrone/VisDrone-Dataset) |
| AI-TODv1 | [Official AI-TOD Repository](https://github.com/jwwangchn/AI-TOD) |

A recommended directory structure is:

```text
data/
├── coco/
│   ├── train2017/
│   ├── val2017/
│   └── annotations/
├── visdrone2019/
└── ai-tod-v1/
```

Dataset paths can be modified in:

```text
configs/dataset/coco_detection.yml
configs/dataset/visdrone_detection.yml
configs/dataset/aitod_detection.yml
```

---

## Training

Two-GPU distributed training is recommended.

### COCO

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --standalone \
  --nproc_per_node=2 \
  train.py \
  --use-amp \
  --seed 0 \
  -c configs/sgdeim/sgdeim_dinov3_s_coco.yml
```

### VisDrone2019

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --standalone \
  --nproc_per_node=2 \
  train.py \
  --use-amp \
  --seed 0 \
  -c configs/sgdeim/sgdeim_dinov3_s_visdrone800.yml
```

### AI-TODv1

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --standalone \
  --nproc_per_node=2 \
  train.py \
  --use-amp \
  --seed 0 \
  -c configs/sgdeim/sgdeim_dinov3_s_aitod800.yml
```

For other model scales, replace the configuration file with the corresponding file under:

```text
configs/sgdeim/
```

---

## Acknowledgments

This project is built upon the excellent open-source work of **DEIMv2**, **DEIM**, **D-FINE**, **RT-DETR**, and **DINOv3**.

We also thank the authors of **COCO**, **VisDrone2019**, and **AI-TOD** for providing the datasets and evaluation resources.

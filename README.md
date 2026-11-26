# Context-Guided Spectral Redistribution and Dual-Stage Gating for End-to-End Small Object Detection

## Overview

Small object detection in aerial and remote sensing images remains challenging because weak object cues may be attenuated during feature extraction and diluted by subsequent feature aggregation. Fixed spectral gains cannot accommodate scene-dependent spectral distributions, while conventional feature aggregation lacks explicit content-dependent regulation of individual feature contributions.

To address these limitations, we propose **SG-DEIM**, an end-to-end small object detector based on **context-guided spectral redistribution and dual-stage gating**. SG-DEIM regulates feature information at three successive stages: encoder input, multi-scale pyramid fusion, and decoder query updating.

Extensive experiments on **VisDrone2019**, **AI-TODv1**, **TinyPerson**, **DIOR**, and **COCO** demonstrate competitive detection performance across diverse scenes and object scales.

---

## Core Contributions

- **Context-Guided Spectral Redistribution (CGSR):** Adaptively redistributes spectral magnitudes across soft radial frequency bands using current-scale spectral statistics and adjacent coarser-scale semantics. Phase preservation, gain centering, and per-channel energy normalization constrain the spectral modulation.

- **Cross-Scale Adaptive Interaction (CSAI):** Performs bidirectional content-dependent modulation between aligned adjacent-level features, allowing fine spatial details and coarse semantic features to condition one another before pyramid aggregation.

- **Query-Conditioned Attention Gating (QCAG):** Generates query-dependent head-wise gates to regulate individual self-attention and cross-attention outputs before aggregation, enabling selective information regulation during decoder query updates.

---

## Environment

- Ubuntu 22.04.5 LTS
- Python 3.11
- PyTorch 2.5.1
- CUDA 13.0
- NVIDIA GeForce RTX 4090

### Installation

```bash
conda create -n sgdeim python=3.11.14 -y
conda activate sgdeim

git clone https://github.com/wudidekarma/SG-DEIM.git
cd SG-DEIM

pip install -r requirements-SGDEIM.txt
```

---

## Pretrained Weights

SG-DEIM models are initialized using DINOv3 or DINOv3-distilled backbone weights.

| Model | Backbone | Pretrained Weights |
| :--- | :--- | :--- |
| SG-DEIM-S | ViT-Tiny | [vitt_distill.pt](https://drive.google.com/file/d/1YMTq_woOLjAcZnHSYNTsNg7f0ahj5LPs/view?usp=sharing) |
| SG-DEIM-M | ViT-Tiny+ | [vittplus_distill.pt](https://drive.google.com/file/d/1COHfjzq5KfnEaXTluVGEOMdhpuVcG6Jt/view?usp=sharing) |
| SG-DEIM-L | DINOv3 ViT-S/16 | [DINOv3](https://github.com/facebookresearch/dinov3#pretrained-models) |
| SG-DEIM-X | DINOv3 ViT-S+/16 | [DINOv3](https://github.com/facebookresearch/dinov3#pretrained-models) |

Place the downloaded checkpoints in the `ckpts/` directory and ensure that `weights_path` in the corresponding configuration file points to the correct checkpoint.

---

## Datasets

SG-DEIM is evaluated on five public benchmark datasets.

| Dataset | Application | Data Source |
| :--- | :--- | :--- |
| COCO 2017 | General object detection | [COCO](https://cocodataset.org/#download) |
| VisDrone2019-DET | UAV object detection | [VisDrone](https://github.com/VisDrone/VisDrone-Dataset) |
| AI-TODv1 | Aerial tiny-object detection | [AI-TOD](https://github.com/jwwangchn/AI-TOD) |
| TinyPerson | Long-range pedestrian detection | [TinyBenchmark](https://github.com/w-sugar/TinyBenchmark) |
| DIOR | Optical remote sensing object detection | [DIOR](https://gcheng-nwpu.github.io/#Datasets) |

Please download the datasets from their corresponding sources and configure the dataset paths under:

```text
configs/dataset/
```

---

## Training

The training configurations for different datasets and model scales are available under:

```text
configs/sgdeim/
```

To train SG-DEIM, use:

```bash
python train.py -c <config_path> --use-amp
```

Please select the corresponding configuration file and ensure that the dataset paths and pretrained weights are correctly configured before training.

---

## Acknowledgments

SG-DEIM is developed upon **DEIMv2** and benefits from several excellent open-source projects, including **DEIM**, **D-FINE**, **RT-DETR**, and **DINOv3**.

We sincerely thank the authors and maintainers of **COCO**, **VisDrone2019**, **AI-TOD**, **TinyPerson**, and **DIOR** for providing the datasets, annotations, and evaluation resources used in this work.

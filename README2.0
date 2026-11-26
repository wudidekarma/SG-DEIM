# Context-Guided Spectral Redistribution and Dual-Stage Gating for End-to-End Small Object Detection

## Overview

Small object detection in aerial and remote sensing images remains challenging because weak object cues can be attenuated during feature extraction and diluted by subsequent feature aggregation. Fixed spectral gains cannot accommodate scene-dependent spectral distributions, while direct cross-scale addition and conventional attention aggregation lack explicit mechanisms for regulating individual feature contributions according to contextual information.

These limitations motivate two design requirements: context-dependent spectral redistribution under an energy constraint and content-dependent regulation of information aggregation throughout the detection pathway.

To this end, we propose **SG-DEIM**, an end-to-end small object detector based on **context-guided spectral redistribution and dual-stage gating**. SG-DEIM regulates feature information at three successive stages: **encoder input**, **multi-scale pyramid fusion**, and **decoder query updating**.

Extensive experiments on **VisDrone2019**, **AI-TODv1**, **TinyPerson**, **DIOR**, and **COCO** demonstrate competitive detection performance across diverse scenes and object scales, including dense UAV imagery, aerial tiny objects, long-range pedestrians, optical remote sensing images, and general object detection.

---

## Core Contributions

### 1. Context-Guided Spectral Redistribution

- **Context-Guided Spectral Redistribution (CGSR):** Adaptively redistributes spectral magnitudes across soft radial frequency bands using current-scale spectral statistics and adjacent coarser-scale semantics. Phase preservation, gain centering, and per-channel energy normalization constrain the spectral modulation before residual feature correction. This enables context-dependent spectral adjustment rather than fixed frequency enhancement.

### 2. Dual-Stage Information Gating

- **Cross-Scale Adaptive Interaction (CSAI):** Replaces direct adjacent-level addition with bidirectional content-dependent modulation. Fine spatial details and coarse semantic features condition one another before pyramid aggregation, allowing their contributions to vary with feature content.

- **Query-Conditioned Attention Gating (QCAG):** Generates query-dependent, head-wise gates and applies them to individual self-attention and cross-attention outputs before multi-head aggregation. This mechanism regulates the information used for decoder query updates.

Together, these mechanisms address spectral adjustment, cross-scale feature interaction, and query-dependent response selection at successive stages of the end-to-end detection pipeline.

---

## Performance Comparison

<p align="center">
  <img src="tools/third_party/Figure_1a.png" width="48%" alt="COCO parameter-accuracy comparison">
  <img src="tools/third_party/Figure_1b.png" width="48%" alt="VisDrone2019 parameter-accuracy comparison">
</p>

<p align="center">
  <em>Parameter–accuracy comparison of SG-DEIM and representative object detectors on COCO and VisDrone2019.</em>
</p>

---

## Benchmark Results

Representative results of SG-DEIM-S on five public benchmarks are summarized below.

| Dataset | Input Size | Metric | DEIMv2-S | SG-DEIM-S |
| :--- | :---: | :---: | ---: | ---: |
| COCO 2017 | 640 × 640 | AP | 50.7 | **51.8** |
| COCO 2017 | 640 × 640 | AP_S | 31.1 | **33.8** |
| VisDrone2019-DET | 640 × 640 | AP | 26.9 | **30.0** |
| VisDrone2019-DET | 640 × 640 | AP50 | 45.7 | **49.8** |
| VisDrone2019-DET | 800 × 800 | AP | — | **33.6** |
| AI-TODv1 | 800 × 800 | AP | 24.2 | **27.4** |
| AI-TODv1 | 800 × 800 | AP50 | 52.3 | **57.0** |
| DIOR | 800 × 800 | AP50 | 82.4 | **84.8** |
| TinyPerson | 640 × 640 | AP50_tiny | — | **52.94** |

All accuracy results are reported in percentages. The evaluation metrics follow the corresponding dataset protocols. The DEIMv2-S results are obtained from the baseline evaluations reported in the paper. A dash indicates that the corresponding baseline result is not reported.

### Results on COCO 2017

| Model | Parameters (M) | GFLOPs | AP | AP50 | AP75 | AP_S | AP_M | AP_L |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SG-DEIM-S | 11.3 | 32.6 | 51.8 | 69.2 | 56.4 | 33.8 | 56.1 | 70.4 |
| SG-DEIM-M | 21 | 65 | 53.3 | 70.6 | 58.0 | 35.7 | 57.8 | 71.5 |
| SG-DEIM-L | 34 | 106 | 56.4 | 74.1 | 61.5 | 38.1 | 61.3 | 74.9 |
| SG-DEIM-X | 53 | 164 | 57.9 | 75.6 | 63.2 | 39.1 | 62.5 | 75.2 |

### Results on VisDrone2019

| Model | Input Size | Parameters (M) | GFLOPs | AP | AP50 | FPS |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: |
| SG-DEIM-S (C2 bypass) | 640 × 640 | 11.3 | 32.6 | 28.4 | 47.4 | 419 |
| SG-DEIM-S (Full C2) | 640 × 640 | 12.6 | 72.6 | 30.0 | 49.8 | 306 |
| SG-DEIM-S (Full C2) | 800 × 800 | 12.6 | 113.1 | 33.6 | 54.8 | 231 |
| SG-DEIM-M (Full C2) | 800 × 800 | 24.9 | 254.7 | 35.4 | 57.3 | 156 |

Inference speed is measured on a single NVIDIA GeForce RTX 4090 using TensorRT FP16 with a batch size of 1. The reported FPS covers the engine forward pass and excludes preprocessing, host-to-device data transfer, and postprocessing.

---

## Environment

The experiments were conducted using the following hardware and software environment:

- Ubuntu 22.04.5 LTS
- Python 3.11
- PyTorch 2.5.1
- CUDA 13.0
- Intel Core i9-14900KF
- 64 GB RAM
- 2 × NVIDIA GeForce RTX 4090

### Installation

Create a Conda environment and install the required dependencies:

```bash
conda create -n sgdeim python=3.11.14 -y
conda activate sgdeim

git clone https://github.com/wudidekarma/SG-DEIM.git
cd SG-DEIM

pip install -r requirements-SGDEIM.txt
```

---

## Pretrained Weights

SG-DEIM models are initialized using DINOv3 or DINOv3-distilled backbone weights, depending on the model scale.

| Model | Backbone | Pretrained Weights |
| :--- | :--- | :--- |
| SG-DEIM-S | ViT-Tiny | [vitt_distill.pt](https://drive.google.com/file/d/1YMTq_woOLjAcZnHSYNTsNg7f0ahj5LPs/view?usp=sharing) |
| SG-DEIM-M | ViT-Tiny+ | [vittplus_distill.pt](https://drive.google.com/file/d/1COHfjzq5KfnEaXTluVGEOMdhpuVcG6Jt/view?usp=sharing) |
| SG-DEIM-L | DINOv3 ViT-S/16 | [DINOv3 Pretrained Models](https://github.com/facebookresearch/dinov3#pretrained-models) |
| SG-DEIM-X | DINOv3 ViT-S+/16 | [DINOv3 Pretrained Models](https://github.com/facebookresearch/dinov3#pretrained-models) |

Place the downloaded checkpoints in the `ckpts/` directory:

```text
SG-DEIM/
└── ckpts/
    ├── vitt_distill.pt
    ├── vittplus_distill.pt
    ├── dinov3_vits16_pretrain_lvd1689m-08c60483.pth
    └── dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth
```

Ensure that `weights_path` in the corresponding configuration file points to the correct pretrained checkpoint.

---

## Datasets

SG-DEIM is evaluated on five public benchmark datasets covering complementary scene characteristics and object-scale distributions.

| Dataset | Application | Input Size | Data Source |
| :--- | :--- | :---: | :--- |
| COCO 2017 | General object detection | 640 × 640 | [COCO Official Website](https://cocodataset.org/#download) |
| VisDrone2019-DET | Dense UAV object detection | 640 × 640 / 800 × 800 | [VisDrone Official Repository](https://github.com/VisDrone/VisDrone-Dataset) |
| AI-TODv1 | Aerial tiny-object detection | 800 × 800 | [AI-TOD Official Repository](https://github.com/jwwangchn/AI-TOD) |
| TinyPerson | Long-range tiny pedestrian detection | 640 × 640 | [TinyPerson / TinyBenchmark](https://github.com/w-sugar/TinyBenchmark) |
| DIOR | Optical remote sensing object detection | 800 × 800 | [DIOR Dataset](https://gcheng-nwpu.github.io/#Datasets) |

### Dataset Preparation

A recommended dataset directory structure is:

```text
data/
├── coco/
├── visdrone2019/
├── ai-tod-v1/
├── tinyperson/
└── dior/
```

The corresponding dataset paths can be configured under:

```text
configs/dataset/
```

Please download each dataset from its official source and follow the corresponding dataset protocol when preparing images, annotations, and training and evaluation splits.

For DIOR, the horizontal bounding box annotations are used for object detection.

---

## Model Configurations

SG-DEIM supports two configurations for utilizing high-resolution shallow features.

### C2-to-C3 Bypass

The shallow C2 feature is spectrally adjusted and transferred to the C3 pathway without retaining C2 as a complete encoder feature level.

This configuration preserves a three-level feature pyramid and reduces the computational overhead associated with high-resolution feature processing.

The C2-to-C3 bypass is used in the experiments on:

- COCO 2017
- DIOR

### Full C2 Feature Level

The high-resolution C2 feature is retained as an additional pyramid level to provide finer spatial representations for small object detection.

This configuration is used in the experiments on:

- VisDrone2019-DET
- AI-TODv1
- TinyPerson

The full C2 configuration provides higher detection accuracy on VisDrone2019, while the bypass configuration offers a lower computational cost. The choice between these configurations depends on the target-scale distribution and available computational resources.

---

## Training

### Training SG-DEIM on COCO

For example, SG-DEIM-S can be trained on COCO using two GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --standalone \
  --nproc_per_node=2 \
  train.py \
  --use-amp \
  --seed 0 \
  -c configs/sgdeim/sgdeim_dinov3_s_coco.yml
```

For other datasets or model scales, replace the configuration file with the corresponding configuration under:

```text
configs/sgdeim/
```

Before training, ensure that:

- The dataset paths are correctly configured.
- The corresponding pretrained backbone weights are available.
- The input resolution and model scale match the desired experimental setting.
- The C2 feature configuration is consistent with the corresponding experiment.

### Training Schedules

The training schedules reported in the paper are summarized below.

| Dataset | Model | Training Epochs |
| :--- | :--- | ---: |
| COCO 2017 | SG-DEIM-S | 140 |
| COCO 2017 | SG-DEIM-M | 100 |
| COCO 2017 | SG-DEIM-L | 70 |
| COCO 2017 | SG-DEIM-X | 50 |
| VisDrone2019-DET | SG-DEIM | 200 |
| AI-TODv1 | SG-DEIM | 140 |
| TinyPerson | SG-DEIM | 120 |
| DIOR | SG-DEIM-S | Refer to the corresponding configuration |

All models are trained using the AdamW optimizer with a flat-cosine learning rate schedule and linear warm-up. A smaller learning rate is assigned to the pretrained backbone, while larger learning rates are used for the encoder, decoder, and detection heads.

Training augmentations include Mosaic, random photometric distortion, random scale expansion, random IoU cropping, horizontal flipping, MixUp, and CopyBlend. Strong augmentations are disabled during the final stage of training.

Please refer to the corresponding configuration files for detailed hyperparameter settings.

---

## Acknowledgments

SG-DEIM is developed upon the open-source implementation of **DEIMv2** and benefits from several object detection and visual representation learning projects:

- **DEIMv2**
- **DEIM**
- **D-FINE**
- **RT-DETR**
- **DINOv3**

We sincerely thank the authors and maintainers of **COCO**, **VisDrone2019**, **AI-TOD**, **TinyPerson / TinyBenchmark**, and **DIOR** for providing the public datasets, annotations, and evaluation resources used in this work.

We also acknowledge the broader open-source computer vision community for its contributions to end-to-end object detection and remote sensing image analysis.

# SG-DEIM: Context-Guided Spectral Redistribution and Dual-Stage Gating for End-to-End Small Object Detection

SG-DEIM is an end-to-end detector designed for small object detection. Small objects usually occupy only a few pixels, so their classification and localization depend heavily on boundaries, corners, short textures, and local intensity variations. Such weak evidence can be attenuated during feature extraction, disturbed by cross-scale interference during pyramid aggregation, and mixed with irrelevant attention-head responses during decoder query updates. SG-DEIM therefore regulates information flow at three consecutive stages?encoder input, pyramid fusion, and decoder query updating?rather than treating its three modules as isolated enhancements.

## Main Contributions

- **Context-Guided Spectral Redistribution (CGSR):** Before encoder projection, CGSR applies an orthonormal RFFT to backbone features, partitions the one-sided spectrum into overlapping soft radial bands, and uses a semantic summary from the adjacent coarser level to adapt the spectral magnitude of the current level. Band-mass centering constrains frequency-independent amplification, while Hermitian-weighted per-sample and per-channel energy normalization controls feature-scale drift. Only the change induced by spectral redistribution is written back to the spatial feature through a residual branch. Four spectral bands are used by default, and the deepest level uses itself as context. An optional C2-to-C3 bypass transfers shallow details without retaining C2 as a complete pyramid level.
- **Cross-Scale Adaptive Interaction (CSAI):** Before adjacent FPN/PAN features are aggregated, CSAI flattens the aligned features into spatial sequences, mixes them with forward and reverse depthwise 1D convolutions, and performs bidirectional cross-scale gated modulation. High-level semantics constrain the transmission of low-level details, while low-level spatial cues supplement coarse high-level localization information. This interaction limits indiscriminate propagation of strong background responses such as road textures, shadows, and building edges.
- **Query-Conditioned Attention Gating (QCAG):** A two-layer MLP conditioned on the current object query generates a gate for every query and attention head. The gates act on genuine per-head responses from both self-attention and deformable cross-attention before head concatenation and output projection, reducing irrelevant information written into the query state. QCAG uses an ungated compensation ratio of 0.1 and is initialized to be an exact identity mapping at the beginning of training.

The public names used in the code follow the paper: the model is `SG-DEIM`, the encoder is `SGDEIMEncoder`, and the three core modules are `CGSR`, `CSAI`, and `QCAG`.

## Results

The following results are taken from Tables 1?5 of the paper. All models are initialized with DINOv3 pretrained weights and use the official training and validation/test splits of each dataset.

### VisDrone2019 and AI-TODv1

| Dataset / Input | Model | Params (M) | GFLOPs | AP | AP50 |
| --- | --- | ---: | ---: | ---: | ---: |
| VisDrone2019, 640?40 | SG-DEIM-S | 12.6 | 72.6 | **30.0** | **49.8** |
| VisDrone2019, 800?00 | SG-DEIM-S | 12.6 | 113.1 | **33.6** | **54.8** |
| VisDrone2019, 800?00 | SG-DEIM-M | 24.9 | 254.7 | **35.4** | **57.3** |
| AI-TODv1, 800?00 | SG-DEIM-S | 12.6 | ? | **27.4** | **57.0** |

### COCO

TensorRT latency was measured on a single RTX 4090 using FP16 and a batch size of 1. The reported time covers only the static TensorRT engine forward pass and excludes preprocessing, host-to-device transfer, and postprocessing.

| Model | Latency (ms) | Params (M) | GFLOPs | AP | AP50 | AP75 | AP_S | AP_M | AP_L |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SG-DEIM-S | 4.06 | 11 | 33 | **51.8** | 69.2 | 56.4 | **33.8** | 56.1 | 70.4 |
| SG-DEIM-M | 5.55 | 21 | 65 | **53.3** | 70.6 | 58.0 | **35.7** | 57.8 | 71.5 |
| SG-DEIM-L | 7.18 | 34 | 106 | **56.4** | 74.1 | 61.5 | **38.1** | 61.3 | 74.9 |
| SG-DEIM-X | 8.96 | 53 | 164 | **57.9** | 75.6 | 63.2 | **39.1** | 62.5 | 75.2 |

### TinyPerson

TinyPerson follows its official scale-specific protocol. All metrics below are evaluated at IoU=0.50.

| Model | AP50_tiny | AP50_tiny1 | AP50_tiny2 | AP50_tiny3 | AP50_small |
| --- | ---: | ---: | ---: | ---: | ---: |
| SG-DEIM-S | **52.94** | **36.42** | **56.31** | **63.27** | **68.80** |

Compared with the remeasured DEIMv2-S baseline, the complete SG-DEIM-S improves AP and AP50 by 3.1 and 4.1 points on VisDrone2019, AP and AP_S by 1.1 and 2.7 points on COCO, and AP and AP50 by 3.2 and 4.7 points on AI-TODv1.

## Contents

- [1. Environment Setup](#1-environment-setup)
- [2. Pretrained Weights](#2-pretrained-weights)
- [3. Dataset Preparation](#3-dataset-preparation)
- [4. Available Configurations](#4-available-configurations)
- [5. Training](#5-training)
- [6. Evaluation and Official TinyPerson Evaluation](#6-evaluation-and-official-tinyperson-evaluation)
- [7. Reproduction Settings](#7-reproduction-settings)
- [8. Ablation and Efficiency Analysis](#8-ablation-and-efficiency-analysis)
- [9. Limitations](#9-limitations)
- [Acknowledgments](#acknowledgments)

## 1. Environment Setup

The experiments reported in the paper were conducted on Ubuntu 22.04.5 LTS with Python 3.11, PyTorch 2.5.1, and CUDA 13.0. The hardware platform consisted of an Intel Core i9-14900KF CPU, 64 GB RAM, and two NVIDIA GeForce RTX 4090 GPUs. This repository has also been validated with PyTorch 2.5.1 and torchvision 0.20.1. Select a PyTorch wheel compatible with the driver installed on your machine.

```bash
conda create -n sgdeim python=3.11.14 -y
conda activate sgdeim
```

Clone the repository and install its dependencies:

```bash
git clone https://github.com/wudidekarma/FCSG-DEIM.git SG-DEIM
cd SG-DEIM

pip install -r requirements-SGDEIM.txt
```

If PyTorch must be installed separately, install a driver-compatible PyTorch build first and then install the remaining dependencies. This avoids replacing an already working PyTorch/CUDA combination.

## 2. Pretrained Weights

Download the backbone checkpoint required by the selected model scale:

| SG-DEIM Model | Backbone | Required Checkpoint |
| --- | --- | --- |
| S | DEIMv2 ViT-Tiny | [`vitt_distill.pt`](https://drive.google.com/file/d/1YMTq_woOLjAcZnHSYNTsNg7f0ahj5LPs/view?usp=sharing) |
| M | DEIMv2 ViT-Tiny+ | [`vittplus_distill.pt`](https://drive.google.com/file/d/1COHfjzq5KfnEaXTluVGEOMdhpuVcG6Jt/view?usp=sharing) |
| L | DINOv3 ViT-S/16 | `dinov3_vits16_pretrain_lvd1689m-08c60483.pth` |
| X | DINOv3 ViT-S+/16 | `dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth` |

Obtain the L/X checkpoints by following the [official DINOv3 instructions](https://github.com/facebookresearch/dinov3#pretrained-models). Place all downloaded weights in `ckpts/` under the project root:

```text
SG-DEIM/
??? ckpts/
?   ??? vitt_distill.pt
?   ??? vittplus_distill.pt
?   ??? dinov3_vits16_pretrain_lvd1689m-08c60483.pth
?   ??? dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth
??? configs/
??? train.py
??? requirements-SGDEIM.txt
```

Before training, verify that `weights_path` in the selected `configs/sgdeim/*.yml` file matches the downloaded filename. The S model uses a ViT-Tiny embedding dimension of 192 and therefore sets `num_heads: 3`. The M model uses ViT-Tiny+ with `num_heads: 4`; this difference is intentional.

## 3. Dataset Preparation

The paper evaluates SG-DEIM on COCO, VisDrone2019, AI-TODv1, and TinyPerson.

| Dataset | Source | Purpose |
| --- | --- | --- |
| COCO 2017 | [Official COCO download page](https://cocodataset.org/#download) or [OpenDataLab](https://opendatalab.com/OpenDataLab/COCO_2017) | General detection and small-scale instance evaluation with AP_S |
| VisDrone2019-DET | [Official VisDrone repository](https://github.com/VisDrone/VisDrone-Dataset) | Dense UAV small object detection |
| AI-TODv1 | [Official AI-TOD repository](https://github.com/jwwangchn/AI-TOD) | Aerial tiny object detection |
| TinyPerson | [TinyBenchmark branch of PointTinyBenchmark](https://github.com/ucas-vg/PointTinyBenchmark/tree/TinyBenchmark) | Long-distance pedestrian detection and official tiny-scale evaluation |

A recommended data layout is shown below. The actual paths are controlled by `configs/dataset/*.yml`.

```text
data/
??? coco/
?   ??? train2017/
?   ??? val2017/
?   ??? annotations/
??? visdrone2019/
??? ai-tod-v1/
??? tinyperson/
```

After downloading and converting the annotations, update the corresponding dataset configuration:

```text
configs/dataset/coco_detection.yml
configs/dataset/visdrone_detection.yml
configs/dataset/aitod_detection.yml
configs/dataset/tinyperson_detection.yml
```

Example COCO paths:

```yaml
train_dataloader:
  dataset:
    img_folder: /path/to/data/coco/train2017
    ann_file: /path/to/data/coco/annotations/instances_train2017.json

val_dataloader:
  dataset:
    img_folder: /path/to/data/coco/val2017
    ann_file: /path/to/data/coco/annotations/instances_val2017.json
```

The current data loader expects COCO-style JSON annotations. Convert or split VisDrone2019, AI-TODv1, and TinyPerson into the format expected by their configurations, and set `num_classes` and category IDs correctly. Do not reuse the 80-class COCO mapping for these datasets.

The TinyPerson configuration trains and evaluates on overlapping 640?40 patches. Before official evaluation, patch detections are mapped back to the original images. Both the patch annotation specified by `corner_gt_file` and the official original-image annotation specified by `merged_gt_file` are required.

## 4. Available Configurations

| Dataset / Model | Input Size | Paper Epochs | Configuration |
| --- | ---: | ---: | --- |
| COCO / S | 640?40 | 140 | `configs/sgdeim/sgdeim_dinov3_s_coco.yml` |
| COCO / M | 640?40 | 100 | `configs/sgdeim/sgdeim_dinov3_m_coco.yml` |
| COCO / L | 640?40 | 70 | `configs/sgdeim/sgdeim_dinov3_l_coco.yml` |
| COCO / X | 640?40 | 50 | `configs/sgdeim/sgdeim_dinov3_x_coco.yml` |
| VisDrone2019 / S | 640?40 | 200 | `configs/sgdeim/sgdeim_dinov3_s_visdrone640.yml` |
| VisDrone2019 / S | 800?00 | 200 | `configs/sgdeim/sgdeim_dinov3_s_visdrone800.yml` |
| VisDrone2019 / M | 640?40 | 200 | `configs/sgdeim/sgdeim_dinov3_m_visdrone640.yml` |
| VisDrone2019 / M | 800?00 | 200 | `configs/sgdeim/sgdeim_dinov3_m_visdrone800.yml` |
| AI-TODv1 / S | 800?00 | 140 | `configs/sgdeim/sgdeim_dinov3_s_aitod800.yml` |
| TinyPerson / S | 640?40 | 120 | `configs/sgdeim/sgdeim_dinov3_tinyperson.yml` |

The epoch counts above follow Section 4.3 of the paper. If a local YAML file comes from an earlier experiment and retains a different training schedule, align it with the paper before conducting an official reproduction.

## 5. Training

Two-GPU single-node distributed training is recommended. Run the following command from the project root:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --standalone \
  --nproc_per_node=2 \
  train.py \
  --use-amp \
  --seed 0 \
  -c configs/sgdeim/sgdeim_dinov3_s_coco.yml
```

- `CUDA_VISIBLE_DEVICES` selects the visible GPUs. Its device count must match `--nproc_per_node`.
- `--standalone` initializes single-node distributed communication automatically. It can be replaced with `--master_port=<free-port>` when a fixed port is preferred.
- `--use-amp` enables automatic mixed precision.
- `-c` selects the model and dataset configuration.

To train on another dataset, replace the configuration path:

```bash
# VisDrone2019, SG-DEIM-S, 800?00
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  train.py --use-amp --seed 0 \
  -c configs/sgdeim/sgdeim_dinov3_s_visdrone800.yml

# AI-TODv1, SG-DEIM-S, 800?00
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  train.py --use-amp --seed 0 \
  -c configs/sgdeim/sgdeim_dinov3_s_aitod800.yml

# TinyPerson, SG-DEIM-S, 640?40
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  train.py --use-amp --seed 0 \
  -c configs/sgdeim/sgdeim_dinov3_tinyperson.yml
```

Resume training from a checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  train.py --use-amp --seed 0 \
  -c configs/sgdeim/sgdeim_dinov3_s_coco.yml \
  -r outputs/SG-DEIM_dinov3_s_combined_coco/last.pth
```

## 6. Evaluation and Official TinyPerson Evaluation

Evaluate a COCO, VisDrone2019, or AI-TODv1 checkpoint with:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  train.py --use-amp --test-only \
  -c configs/sgdeim/sgdeim_dinov3_s_coco.yml \
  -r /path/to/checkpoint.pth
```

For TinyPerson, `--test-only` performs patch-level COCO evaluation and then automatically:

1. Streams patch detections from each distributed rank.
2. Maps patch coordinates back to the original images and applies NMS.
3. Normalizes the category ID to the official `person=1` convention.
4. Calls the official `evaluate_tiny.py` implementation to calculate AP/AR under the TinyPerson ignore, uncertain-region, and scale-specific rules.

The official evaluator is included at:

```text
tools/third_party/PointTinyBenchmark_TinyBenchmark/
```

Run official TinyPerson evaluation with:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  train.py --use-amp --seed 0 --test-only \
  -c configs/sgdeim/sgdeim_dinov3_tinyperson.yml \
  -r /path/to/tinyperson_checkpoint.pth \
  --output-dir outputs/SG-DEIM_dinov3_s_tinyperson_official_eval
```

The following files are generated:

```text
outputs/SG-DEIM_dinov3_s_tinyperson_official_eval/tinyperson_official/
??? test_patch_predictions.json
??? test_merged_predictions.json
??? test_official_scores.txt
```

Regular training uses the faster COCO evaluator by default. To calculate the official TinyPerson metric periodically during training, update the TinyPerson configuration:

```yaml
tinyperson_official_eval:
  run_during_training: True
  frequency: 5
```

This setting merges all patch detections and runs the official evaluator every five epochs, which substantially increases CPU time and disk I/O. The default `run_during_training: False` is recommended for ordinary training.

If a patch-level COCO detection JSON is already available, call the wrapper directly:

```bash
python tools/dataset/tinyperson_official_eval.py \
  --res /path/to/patch_predictions.json \
  --official-root tools/third_party/PointTinyBenchmark_TinyBenchmark/tiny_benchmark \
  --corner-gt /path/to/tiny_set_test_sw640_sh512_all.json \
  --gt /path/to/tiny_set_test_all.json \
  --metric ap
```

## 7. Reproduction Settings

| Item | Setting |
| --- | --- |
| Hardware | Intel Core i9-14900KF, 64 GB RAM, 2 ?NVIDIA GeForce RTX 4090 |
| Software | Ubuntu 22.04.5 LTS, Python 3.11, PyTorch 2.5.1, CUDA 13.0 |
| Optimizer | AdamW, ?1=0.9, ?2=0.999 |
| Weight decay | 1e-4 to 5e-4 depending on the dataset and model scale |
| Learning rates | A smaller learning rate for the DINOv3 backbone and larger rates for the encoder, decoder, and detection heads |
| Schedule | Linear warm-up followed by a flat-cosine schedule |
| COCO epochs | S / M / L / X: 140 / 100 / 70 / 50 |
| VisDrone2019 epochs | 200 |
| AI-TODv1 epochs | 140 |
| TinyPerson epochs | 120 |
| Input sizes | COCO and TinyPerson: 640?40; AI-TODv1: 800?00; VisDrone2019: 640?40 or 800?00 |
| Augmentation | Mosaic, random photometric distortion, random scale expansion, random IoU cropping, horizontal flipping, MixUp, and CopyBlend; strong augmentation is disabled during the final stage |
| High-resolution features | COCO uses the C2-to-C3 shallow bypass; VisDrone2019, AI-TODv1, and TinyPerson retain C2 as a complete level by default |

Common switches for the three core modules are shown below:

```yaml
SGDEIMEncoder:
  use_cgsr: True
  cgsr_num_bands: 4
  cgsr_context_source: coarse
  use_csai: True
  csai_position: both

DEIMTransformer:
  use_qcag: True
  qcag_mode: headwise_v2
  qcag_condition_source: query
```

## 8. Ablation and Efficiency Analysis

### 8.1 Core-Module Ablation

Table 6 of the paper uses DEIMv2-S as the baseline and evaluates CGSR, CSAI, and QCAG on VisDrone2019 and COCO. The parameter and GFLOP values differ between the datasets because their experiments use different high-resolution feature settings.

| CGSR | CSAI | QCAG | VisDrone AP | VisDrone AP50 | Vis Params | Vis GFLOPs | COCO AP | COCO AP_S | COCO Params | COCO GFLOPs |
| :---: | :---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
|  |  |  | 26.9 | 45.7 | 10.3 | 43.5 | 50.7 | 31.1 | 9.7 | 25.6 |
| ? |  |  | 27.5 | 46.8 | 10.4 | 43.7 | 50.9 | 31.8 | 9.8 | 25.8 |
|  | ? |  | 27.1 | 46.2 | 12.3 | 71.9 | 51.3 | 32.4 | 11.1 | 32.4 |
|  |  | ? | 27.3 | 46.1 | 10.4 | 43.6 | 50.8 | 31.4 | 9.8 | 25.7 |
| ? | ? |  | 29.3 | 48.8 | 12.5 | 72.5 | 51.6 | 33.5 | 11.2 | 32.5 |
| ? |  | ? | 28.0 | 47.3 | 10.5 | 44.3 | 51.5 | 32.9 | 9.9 | 25.9 |
|  | ? | ? | 28.2 | 47.5 | 12.5 | 72.4 | 51.5 | 32.6 | 11.2 | 32.4 |
| ? | ? | ? | **30.0** | **49.8** | 12.6 | 72.6 | **51.8** | **33.8** | 11.3 | 32.6 |

The individual modules produce limited improvements in isolation, whereas the complete configuration gives a substantially larger gain. This supports the central premise of the paper: decoder-side gating cannot restore fine-grained evidence that has already been lost during upstream feature extraction or cross-scale fusion, so all three stages must work together.

### 8.2 C2, Resolution, and Speed Trade-Off

Table 7 of the paper measures inference speed on a single RTX 4090 with TensorRT 11.0.0.114, FP16, and a batch size of 1. Each static engine is warmed up for 200 iterations and timed for 1,000 consecutive iterations. FPS covers only the engine forward pass.

| Model | Full C2 | Input Size | Params (M) | GFLOPs | AP | AP50 | FPS |
| --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DEIMv2-S | No | 640?40 | 9.7 | 25.6 | 25.3 | 44.2 | 573 |
| SG-DEIM-S | No | 640?40 | 11.3 | 32.6 | 28.4 | 47.4 | 419 |
| DEIMv2-S | Yes | 640?40 | 10.3 | 43.5 | 26.9 | 45.7 | 434 |
| SG-DEIM-S | Yes | 640?40 | 12.6 | 72.6 | 30.0 | 49.8 | 306 |
| SG-DEIM-S | Yes | 800?00 | 12.6 | 113.1 | 33.6 | 54.8 | 231 |
| SG-DEIM-M | Yes | 800?00 | 24.9 | 254.7 | 35.4 | 57.3 | 156 |

The C2-to-C3 bypass offers higher throughput, while retaining full C2 preserves more fine spatial detail. Increasing input resolution and model capacity further improves accuracy at the cost of additional computation and lower throughput.

### 8.3 Structural Validation

- **CGSR context source:** Full spectral routing with adjacent coarser-scale context reaches 30.0 AP / 49.8 AP50. Current-scale context gives 28.9 / 48.6, random-scale context gives 29.2 / 49.0, and adjacent-scale context without spectral routing gives 28.6 / 48.3.
- **CSAI interaction:** Adding CSAI to the CGSR+QCAG configuration improves VisDrone2019 by 2.0 AP / 2.5 AP50 and COCO by 0.3 AP / 0.9 AP_S.
- **QCAG query and head specificity:** No QCAG gives 29.3 / 48.8, a global head gate gives 29.5 / 48.7, and full QCAG reaches 30.0 / 49.8. At a fixed checkpoint, query-mean and head-mean interventions reduce the result to 29.2 / 48.6 and 29.5 / 49.2, respectively, indicating that variation along both query and head dimensions contributes to the gain.

## 9. Limitations

- Computational cost remains relatively high when C2 is retained as a complete high-resolution feature level. The C2-to-C3 bypass is faster but less accurate than full C2 retention.
- Missed detections and duplicate predictions can still occur in crowded scenes with severe occlusion, extremely limited pixel coverage, or strong target-background similarity.
- QCAG depends on the quality of upstream features supplied by CGSR and CSAI. Head-wise gating cannot recover missing object evidence when the encoder and pyramid representations are already insufficient or noisy.

## Acknowledgments

This project builds upon the open-source work of DEIMv2, DEIM, D-FINE, RT-DETR, and DINOv3. We also thank the authors of COCO, VisDrone2019, AI-TOD, and TinyPerson/PointTinyBenchmark for releasing their datasets and evaluation code.

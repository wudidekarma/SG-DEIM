#!/usr/bin/env python3
"""Utilities for converting TinyPerson patch detections to original images."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List

import torch
from torchvision.ops import nms


def load_json(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def dump_json(data, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False)


def _clip_xyxy(box: Iterable[float], width: float, height: float) -> List[float]:
    x1, y1, x2, y2 = box
    x1 = min(max(float(x1), 0.0), width)
    y1 = min(max(float(y1), 0.0), height)
    x2 = min(max(float(x2), 0.0), width)
    y2 = min(max(float(y2), 0.0), height)
    return [x1, y1, x2, y2]


def convert_patch_detections_to_original(
    detections: List[dict],
    patch_ann_file: str,
    original_ann_file: str,
    nms_iou: float = 0.5,
    max_det: int = 1000,
    score_thr: float = 0.001,
    pre_nms_topk: int = 5000,
    nms_cell_size: float = 64.0,
) -> List[dict]:
    """Translate patch-level COCO detections and suppress overlap duplicates.

    ``nms_cell_size`` is accepted for CLI compatibility. Torchvision's compiled
    NMS is used here, so spatial bucketing is unnecessary.
    """
    del nms_cell_size
    patch_gt = load_json(patch_ann_file)
    original_gt = load_json(original_ann_file)

    original_by_name = {image["file_name"]: image for image in original_gt["images"]}
    patch_map = {}
    for image in patch_gt["images"]:
        original = original_by_name[image["file_name"]]
        patch_map[int(image["id"])] = (
            int(original["id"]),
            [float(value) for value in image["corner"]],
            float(original["width"]),
            float(original["height"]),
        )

    grouped = defaultdict(list)
    for detection in detections:
        score = float(detection.get("score", 0.0))
        if score < score_thr:
            continue
        patch_id = int(detection["image_id"])
        if patch_id not in patch_map:
            raise KeyError(f"Patch image id {patch_id} is absent from {patch_ann_file}")
        original_id, corner, width, height = patch_map[patch_id]
        x, y, box_width, box_height = [float(value) for value in detection["bbox"]]
        xyxy = _clip_xyxy(
            [x + corner[0], y + corner[1], x + box_width + corner[0], y + box_height + corner[1]],
            width,
            height,
        )
        if xyxy[2] <= xyxy[0] or xyxy[3] <= xyxy[1]:
            continue
        grouped[(original_id, int(detection.get("category_id", 1)))].append((score, xyxy))

    merged = []
    for (image_id, category_id), candidates in grouped.items():
        candidates.sort(key=lambda item: item[0], reverse=True)
        candidates = candidates[:pre_nms_topk]
        scores = torch.tensor([item[0] for item in candidates], dtype=torch.float32)
        boxes = torch.tensor([item[1] for item in candidates], dtype=torch.float32)
        keep = nms(boxes, scores, nms_iou)[:max_det].tolist()
        for index in keep:
            x1, y1, x2, y2 = boxes[index].tolist()
            merged.append(
                {
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(scores[index]),
                }
            )
    return merged

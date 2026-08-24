#!/usr/bin/env python3
"""
Run the official TinyBenchmark evaluator on TinyPerson detections.

This wrapper intentionally does not reimplement AP/MR. It only normalizes DEIM
detection JSONs to TinyBenchmark's category id and, when needed, merges
patch-level detections back to original images before calling the official
`MyPackage/tools/evaluate/evaluate_tiny.py`.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List

from tinyperson_eval import convert_patch_detections_to_original, dump_json, load_json


DEFAULT_OFFICIAL_ROOT = "/tmp/PointTinyBenchmark_TinyBenchmark/tiny_benchmark"
DEFAULT_GT = "/media/q/data/ys/datasets/TinyPerson/OpenDataLab___TinyPerson/aw1/annotations/task/tiny_set_test_all.json"
DEFAULT_CORNER_GT = "/media/q/data/ys/datasets/TinyPerson/OpenDataLab___TinyPerson/aw1/annotations/corner/task/tiny_set_test_sw640_sh512_all.json"


def write_official_compat(compat_dir: Path) -> Path:
    compat_dir.mkdir(parents=True, exist_ok=True)
    compat_code = r'''
import sys
import types
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, message=".*np\\.object.*")

try:
    import torch
    if not hasattr(torch, "_six"):
        torch._six = types.SimpleNamespace(PY3=True, string_classes=(str,), int_classes=(int,))
except Exception:
    pass

try:
    import numpy as np
    for _name, _value in {"float": float, "int": int, "bool": bool, "object": object}.items():
        if not hasattr(np, _name):
            setattr(np, _name, _value)
    _np_linspace = np.linspace
    def _compat_linspace(start, stop, num=50, *args, **kwargs):
        try:
            if isinstance(num, np.generic):
                num = num.item()
            if isinstance(num, float) and num.is_integer():
                num = int(num)
        except Exception:
            pass
        return _np_linspace(start, stop, num, *args, **kwargs)
    np.linspace = _compat_linspace
except Exception:
    pass

try:
    import faster_coco_eval
    faster_coco_eval.init_as_pycocotools()
except Exception:
    pass

try:
    import matplotlib.pyplot  # noqa: F401
except Exception:
    _matplotlib = types.ModuleType("matplotlib")
    _pyplot = types.ModuleType("matplotlib.pyplot")
    _collections = types.ModuleType("matplotlib.collections")
    _patches = types.ModuleType("matplotlib.patches")
    def _noop(*args, **kwargs):
        return None
    class _DummyPatchCollection:
        def __init__(self, *args, **kwargs):
            pass
    class _DummyPolygon:
        def __init__(self, *args, **kwargs):
            pass
    for _name in ["plot", "imshow", "figure", "show", "savefig", "close", "title", "xlabel", "ylabel"]:
        setattr(_pyplot, _name, _noop)
    _collections.PatchCollection = _DummyPatchCollection
    _patches.Polygon = _DummyPolygon
    _matplotlib.pyplot = _pyplot
    _matplotlib.collections = _collections
    _matplotlib.patches = _patches
    sys.modules.setdefault("matplotlib", _matplotlib)
    sys.modules.setdefault("matplotlib.pyplot", _pyplot)
    sys.modules.setdefault("matplotlib.collections", _collections)
    sys.modules.setdefault("matplotlib.patches", _patches)

try:
    import scipy.io  # noqa: F401
except Exception:
    _scipy = types.ModuleType("scipy")
    _scipy_io = types.ModuleType("scipy.io")
    _scipy.io = _scipy_io
    sys.modules.setdefault("scipy", _scipy)
    sys.modules.setdefault("scipy.io", _scipy_io)

try:
    import cv2  # noqa: F401
except Exception:
    _cv2 = types.ModuleType("cv2")
    def _cv2_missing(*args, **kwargs):
        raise ImportError("cv2 is not installed; this TinyBenchmark AP merge path does not require it.")
    _cv2.imread = _cv2_missing
    _cv2.imwrite = _cv2_missing
    _cv2.cvtColor = _cv2_missing
    _cv2.COLOR_BGR2RGB = 4
    _cv2.COLOR_RGB2BGR = 4
    sys.modules.setdefault("cv2", _cv2)

try:
    import torch

    def _pure_torch_nms(boxes, scores, iou_threshold):
        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=boxes.device)
        x1, y1, x2, y2 = boxes.unbind(1)
        areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        order = scores.argsort(descending=True)
        keep = []
        while order.numel() > 0:
            i = order[0]
            keep.append(i)
            if order.numel() == 1:
                break
            rest = order[1:]
            xx1 = torch.maximum(x1[i], x1[rest])
            yy1 = torch.maximum(y1[i], y1[rest])
            xx2 = torch.minimum(x2[i], x2[rest])
            yy2 = torch.minimum(y2[i], y2[rest])
            inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
            union = areas[i] + areas[rest] - inter
            iou = torch.where(union > 0, inter / union, torch.zeros_like(union))
            order = rest[iou <= iou_threshold]
        return torch.stack(keep) if keep else torch.empty((0,), dtype=torch.long, device=boxes.device)

    try:
        from torchvision.ops import nms as _tv_nms
    except Exception:
        _tv_nms = _pure_torch_nms

    def _safe_nms(boxes, scores, iou_threshold):
        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=boxes.device)
        if boxes.ndim == 1:
            boxes = boxes.reshape(1, 4)
        if scores.ndim == 0:
            scores = scores.reshape(1)
        return _tv_nms(boxes, scores, iou_threshold)

    _layers = types.ModuleType("maskrcnn_benchmark.layers")
    _layers.nms = _safe_nms
    sys.modules.setdefault("maskrcnn_benchmark.layers", _layers)
except Exception:
    pass
'''.lstrip()
    (compat_dir / "sitecustomize.py").write_text(compat_code)
    (compat_dir / "run_evaluate_tiny.py").write_text(
        (
            compat_code
            + r'''
import runpy

if len(sys.argv) < 2:
    raise SystemExit("Usage: run_evaluate_tiny.py /path/to/evaluate_tiny.py [args...]")

_script = sys.argv[1]
sys.argv = [_script] + sys.argv[2:]
runpy.run_path(_script, run_name="__main__")
'''.lstrip()
        )
    )
    return compat_dir


def normalize_category_id(detections: List[dict]) -> List[dict]:
    normalized = []
    for det in detections:
        item = dict(det)
        item["image_id"] = int(item["image_id"])
        item["category_id"] = 1
        item["bbox"] = [float(v) for v in item["bbox"]]
        item["score"] = float(item.get("score", 0.0))
        normalized.append(item)
    return normalized


def with_suffix(path: str, suffix: str) -> str:
    p = Path(path)
    return str(p.with_name(f"{p.stem}{suffix}{p.suffix}"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Call the official TinyBenchmark evaluator.")
    parser.add_argument("--res", required=True, help="Detection JSON. Patch-level by default.")
    parser.add_argument("--input-level", choices=["patch", "original"], default="patch")
    parser.add_argument("--official-root", default=os.environ.get("TINYBENCHMARK_ROOT", DEFAULT_OFFICIAL_ROOT))
    parser.add_argument("--gt", default=DEFAULT_GT, help="Official original-image task annotation JSON.")
    parser.add_argument("--corner-gt", default=DEFAULT_CORNER_GT, help="Official patch/corner annotation JSON.")
    parser.add_argument("--merged-res", help="Where to save merged original-image detections.")
    parser.add_argument("--score-file", help="Where to save the official evaluator log.")
    parser.add_argument("--metric", choices=["all", "ap", "mr"], default="all")
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=1000)
    parser.add_argument("--score-thr", type=float, default=0.001)
    parser.add_argument("--pre-nms-topk", type=int, default=5000)
    parser.add_argument("--nms-cell-size", type=float, default=64.0)
    parser.add_argument(
        "--official-merge",
        action="store_true",
        help="Let TinyBenchmark merge patch results itself. Requires its compiled maskrcnn_benchmark ops.",
    )
    args = parser.parse_args()

    official_root = Path(args.official_root).resolve()
    official_eval = official_root / "MyPackage" / "tools" / "evaluate" / "evaluate_tiny.py"
    if not official_eval.exists():
        raise SystemExit(
            f"Official evaluator not found: {official_eval}\n"
            "Clone it first, for example:\n"
            "  git clone --branch TinyBenchmark https://github.com/ucas-vg/PointTinyBenchmark.git /tmp/PointTinyBenchmark_TinyBenchmark"
        )

    detections = normalize_category_id(load_json(args.res))
    normalized_res = str(Path(with_suffix(args.res, "_category1")).resolve())
    dump_json(detections, normalized_res)

    if args.official_merge:
        res_for_official = normalized_res
        gt_for_official = args.corner_gt
        merge_gt = args.gt
    elif args.input_level == "patch":
        merged_res = str(Path(args.merged_res or with_suffix(args.res, "_official_merged")).resolve())
        merged = convert_patch_detections_to_original(
            detections,
            patch_ann_file=args.corner_gt,
            original_ann_file=args.gt,
            nms_iou=args.nms_iou,
            max_det=args.max_det,
            score_thr=args.score_thr,
            pre_nms_topk=args.pre_nms_topk,
            nms_cell_size=args.nms_cell_size,
        )
        merged = normalize_category_id(merged)
        dump_json(merged, merged_res)
        res_for_official = merged_res
        gt_for_official = args.gt
        merge_gt = ""
    else:
        res_for_official = normalized_res
        gt_for_official = args.gt
        merge_gt = ""

    score_file = str(Path(args.score_file or with_suffix(res_for_official, "_official_scores")).resolve())
    tmp_log = str(Path(with_suffix(score_file, "_tmp")).resolve())
    compat_dir = write_official_compat((Path(score_file).parent / "_tinybenchmark_compat").resolve())

    cmd = [
        sys.executable,
        str(compat_dir / "run_evaluate_tiny.py"),
        str(official_eval),
        "--res",
        str(Path(res_for_official).resolve()),
        "--gt",
        str(Path(gt_for_official).resolve()),
        "--metric",
        args.metric,
        "--score-file",
        str(Path(score_file).resolve()),
        "--tmp-log",
        str(Path(tmp_log).resolve()),
        "--detail",
    ]
    if merge_gt:
        cmd.extend(["--merge-gt", str(Path(merge_gt).resolve())])

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{compat_dir}:{official_root}:{env.get('PYTHONPATH', '')}"

    print("Running official TinyBenchmark evaluator:")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=str(official_eval.parent), env=env, check=True)
    print(f"Official score log: {score_file}")


if __name__ == "__main__":
    main()

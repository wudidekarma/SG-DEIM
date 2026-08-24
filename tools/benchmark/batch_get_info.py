"""
Batch model complexity summary for YAML configs.
"""

import argparse
import contextlib
import csv
import io
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

from calflops import calculate_flops
from engine.core import YAMLConfig


def _find_resize_size(ops):
    if not isinstance(ops, list):
        return None
    for op in ops:
        if isinstance(op, dict) and op.get("type") == "Resize" and "size" in op:
            size = op["size"]
            if isinstance(size, (list, tuple)) and len(size) == 2:
                return int(size[0]), int(size[1])
    return None


def infer_input_size(cfg):
    merged = cfg.global_cfg
    eval_size = merged.get("eval_spatial_size")
    if isinstance(eval_size, (list, tuple)) and len(eval_size) == 2:
        return int(eval_size[0]), int(eval_size[1])

    ops = (
        merged.get("train_dataloader", {})
        .get("dataset", {})
        .get("transforms", {})
        .get("ops")
    )
    resize_size = _find_resize_size(ops)
    if resize_size is not None:
        return resize_size

    base_size = (
        merged.get("train_dataloader", {})
        .get("collate_fn", {})
        .get("base_size")
    )
    if base_size is not None:
        return int(base_size), int(base_size)

    return 640, 640


def summarize(config_path):
    cfg = YAMLConfig(str(config_path), resume=None)
    height, width = infer_input_size(cfg)

    # Keep model initialization chatter out of the CSV/table output.
    with contextlib.redirect_stdout(io.StringIO()):
        model = cfg.model.deploy().eval()
        flops, macs, _ = calculate_flops(
            model=model,
            input_shape=(1, 3, height, width),
            print_results=False,
            print_detailed=False,
            output_as_string=True,
            output_precision=4,
        )

    params = sum(p.numel() for p in model.parameters())
    return {
        "config": str(config_path).replace("\\", "/"),
        "input": f"{height}x{width}",
        "flops": flops,
        "macs": macs,
        "params": params,
        "params_m": round(params / 1_000_000, 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+", help="YAML config files")
    parser.add_argument("--csv", type=Path, help="Optional CSV output path")
    args = parser.parse_args()

    rows = []
    for item in args.configs:
        config_path = Path(item)
        try:
            row = summarize(config_path)
            rows.append(row)
            print(
                f"{row['config']}\t{row['input']}\t{row['flops']}\t"
                f"{row['macs']}\t{row['params']}\t{row['params_m']}"
            )
        except Exception as exc:
            row = {
                "config": str(config_path).replace("\\", "/"),
                "input": "",
                "flops": "",
                "macs": "",
                "params": "",
                "params_m": "",
                "error": repr(exc),
            }
            rows.append(row)
            print(f"{row['config']}\tERROR\t{row['error']}")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["config", "input", "flops", "macs", "params", "params_m", "error"]
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""一键评估 PyTorch 印章 OCR 模型。"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="在真实标注数据上测试模型能力")
    parser.add_argument("--model", required=True, help="训练输出的 best 模型目录")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", help="图片与同名 .txt 标签目录")
    source.add_argument("--manifest", help="训练生成的 data_split.json")
    parser.add_argument("--split", choices=["train", "eval", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0, help="0 表示评估全部")
    parser.add_argument("--device", default="0", help="CUDA 设备编号；留空使用 CPU")
    parser.add_argument("--output", default="reports/evaluation", help="CSV 与汇总 JSON 输出目录")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "seal_ocr.evaluation",
        "--cust_data_init_weights_path",
        args.model,
        "--checkpoint",
        args.model,
        "--CUDA_VISIBLE_DEVICES",
        args.device,
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.workers),
        "--max_samples",
        str(args.max_samples),
        "--report_csv",
        str(output / "predictions.csv"),
        "--summary_json",
        str(output / "summary.json"),
    ]
    if args.manifest:
        command.extend(["--split_manifest", args.manifest, "--manifest_split", args.split])
    else:
        command.extend(["--dataset_dir", args.data])
    subprocess.run(command, check=True)
    print(f"评估结果: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

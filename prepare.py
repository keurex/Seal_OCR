#!/usr/bin/env python3
"""从公司清单与真实标签一键生成词表并初始化 TrOCR。"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

from seal_ocr.data import normalize_company_name


def _labels(company_list: Path, real_data: list[Path]):
    if not company_list.is_file():
        raise FileNotFoundError(f"公司清单不存在: {company_list}")
    for line in company_list.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("#"):
            continue
        label = normalize_company_name(line)
        if label:
            yield label
    for root in real_data:
        if not root.exists():
            raise FileNotFoundError(f"真实数据目录不存在: {root}")
        files = [root] if root.is_file() else root.rglob("*.txt")
        for path in files:
            if path.name.endswith(".meta.txt") or path.stem.endswith("_llm"):
                continue
            label = normalize_company_name(path.read_text(encoding="utf-8"))
            if label:
                yield label


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="一键生成字符词表并初始化模型")
    parser.add_argument("--base-model", required=True, help="基础 TrOCR 模型目录或 Hugging Face 模型名")
    parser.add_argument("--output", default="models/init", help="初始化模型输出目录")
    parser.add_argument("--company-list", default="synthesis/company_names.txt", help="每行一个完整公司名")
    parser.add_argument("--real-data", nargs="*", default=[], help="可选真实标注目录，用于补齐字符")
    parser.add_argument("--image-size", type=int, default=384, help="方形输入尺寸，默认 384")
    parser.add_argument("--allow-unknown-chars", action="store_true", help="允许基础模型未覆盖字符随机初始化")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    labels = list(
        _labels(
            Path(args.company_list).expanduser().resolve(),
            [Path(path).expanduser().resolve() for path in args.real_data],
        )
    )
    characters = sorted({character for label in labels for character in label})
    if not labels or not characters:
        raise ValueError("公司清单和真实标签中没有可用文字")

    with tempfile.TemporaryDirectory(prefix="seal_ocr_vocab_") as temporary_dir:
        vocab_path = Path(temporary_dir) / "vocab.txt"
        vocab_path.write_text("\n".join(characters) + "\n", encoding="utf-8")
        command = [
            sys.executable,
            "-m",
            "seal_ocr.initialize",
            "--cust_vocab",
            str(vocab_path),
            "--pretrain_model",
            args.base_model,
            "--cust_data_init_weights_path",
            str(Path(args.output).expanduser().resolve()),
            "--image_size",
            str(args.image_size),
        ]
        if args.allow_unknown_chars:
            command.append("--allow_unknown_chars")
        subprocess.run(command, check=True)

    print(f"公司名: {len(set(labels))}")
    print(f"业务字符: {len(characters)}")
    print(f"初始化模型: {Path(args.output).expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

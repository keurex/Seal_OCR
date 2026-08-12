"""印章 OCR 数据发现、标签规范化与稳定切分。"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}

TERMINAL_PUNCTUATION = "。.;；,，"
PAREN_TRANSLATION = str.maketrans({"(": "（", ")": "）"})


@dataclass(frozen=True)
class TrainingSample:
    image_path: str
    label: str
    spatial_annotation_path: Optional[str] = None


@dataclass(frozen=True)
class DataIssue:
    issue_type: str
    path: str
    detail: str


def normalize_company_name(text: str) -> str:
    """
    规范公司名标签。

    公司名不应包含空白、NUL、BOM 或句末标点。ASCII 括号统一为当前词表
    已覆盖的全角括号，避免同一个公司名出现两种编码形式。
    """
    text = text.replace("\x00", "").replace("\ufeff", "").replace("\xa0", "")
    text = re.sub(r"\s+", "", text)
    text = text.translate(PAREN_TRANSLATION)
    return text.rstrip(TERMINAL_PUNCTUATION)


def _iter_dataset_files(dataset_path: Path) -> Iterable[Path]:
    if dataset_path.is_file():
        yield dataset_path
        return
    yield from dataset_path.rglob("*")


def _verify_image(image_path: Path) -> Optional[str]:
    try:
        from PIL import Image

        with Image.open(image_path) as image:
            image.verify()
    except Exception as exc:  # Pillow 会给出具体损坏原因
        return f"{type(exc).__name__}: {exc}"
    return None


def _verify_spatial_annotation(
    image_path: Path,
    annotation_path: Path,
) -> Optional[str]:
    try:
        from PIL import Image

        with Image.open(image_path) as image, Image.open(
            annotation_path
        ) as annotation:
            if annotation.mode != "RGBA":
                return f"标注必须为 RGBA，实际为 {annotation.mode}"
            if annotation.size != image.size:
                return (
                    f"标注尺寸 {annotation.size} 与图片尺寸 {image.size} 不一致"
                )
            annotation.verify()
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def discover_samples(
    dataset_paths: Sequence[str],
    verify_images: bool = False,
    spatial_annotation_paths: Optional[Sequence[Optional[str]]] = None,
    auto_spatial_annotations: bool = False,
    require_spatial_annotations: bool = False,
) -> Tuple[List[TrainingSample], List[DataIssue]]:
    """发现图片、文字标签和可选的同路径空间标注，并返回数据问题。

    默认空间目录是图片目录同级的 ``<dataset>_spatial``。只有显式启用
    ``auto_spatial_annotations`` 才会探测该目录；真实人工样本缺少空间标注时
    保持 ``None``，不会被误当成全零负样本。
    """
    if spatial_annotation_paths is not None and len(spatial_annotation_paths) != len(
        dataset_paths
    ):
        raise ValueError(
            "spatial_annotation_paths 数量必须与 dataset_paths 完全一致"
        )
    samples: List[TrainingSample] = []
    issues: List[DataIssue] = []
    seen_images: Set[str] = set()

    for dataset_index, raw_path in enumerate(dataset_paths):
        dataset_path = Path(raw_path).expanduser().resolve()
        if not dataset_path.exists():
            issues.append(DataIssue("dataset_not_found", str(dataset_path), "路径不存在"))
            continue

        raw_spatial_path = (
            spatial_annotation_paths[dataset_index]
            if spatial_annotation_paths is not None
            else None
        )
        spatial_root = None
        if raw_spatial_path:
            spatial_root = Path(raw_spatial_path).expanduser().resolve()
        elif auto_spatial_annotations and dataset_path.is_dir():
            spatial_root = dataset_path.with_name(f"{dataset_path.name}_spatial")
        if spatial_root is not None and not spatial_root.is_dir():
            if require_spatial_annotations:
                issues.append(
                    DataIssue(
                        "spatial_root_not_found",
                        str(spatial_root),
                        "空间标注目录不存在",
                    )
                )
                continue
            spatial_root = None

        files = [path for path in _iter_dataset_files(dataset_path) if path.is_file()]
        images = {
            path.resolve()
            for path in files
            if path.suffix.lower() in IMAGE_SUFFIXES
        }
        label_files = {
            path.resolve()
            for path in files
            if path.suffix.lower() == ".txt" and not path.stem.endswith("_llm")
        }
        image_keys = {(path.parent, path.stem) for path in images}
        labels_by_key = {
            (path.parent, path.stem): path
            for path in label_files
        }

        for label_path in sorted(label_files):
            if (label_path.parent, label_path.stem) not in image_keys:
                issues.append(
                    DataIssue("label_without_image", str(label_path), "找不到同基名图片")
                )

        for image_path in sorted(images):
            image_key = str(image_path)
            if image_key in seen_images:
                issues.append(
                    DataIssue("duplicate_image_path", image_key, "被多个数据集路径重复发现")
                )
                continue
            seen_images.add(image_key)

            label_path = labels_by_key.get((image_path.parent, image_path.stem))
            if label_path is None:
                issues.append(
                    DataIssue("image_without_label", image_key, "找不到同基名 .txt 标签")
                )
                continue

            try:
                raw_label = label_path.read_text(encoding="utf-8")
            except Exception as exc:
                issues.append(
                    DataIssue(
                        "label_read_error",
                        str(label_path),
                        f"{type(exc).__name__}: {exc}",
                    )
                )
                continue

            label = normalize_company_name(raw_label)
            if not label:
                issues.append(
                    DataIssue("empty_label", str(label_path), "规范化后标签为空")
                )
                continue

            if verify_images:
                image_error = _verify_image(image_path)
                if image_error:
                    issues.append(
                        DataIssue("invalid_image", image_key, image_error)
                    )
                    continue

            annotation_path = None
            if spatial_root is not None:
                relative_path = (
                    image_path.relative_to(dataset_path)
                    if dataset_path.is_dir()
                    else Path(image_path.name)
                )
                candidate = (spatial_root / relative_path).with_suffix(".png")
                if candidate.is_file():
                    annotation_path = candidate.resolve()
                elif require_spatial_annotations:
                    issues.append(
                        DataIssue(
                            "spatial_annotation_missing",
                            str(candidate),
                            f"找不到图片 {image_path} 的空间标注",
                        )
                    )
                    continue
            elif require_spatial_annotations:
                issues.append(
                    DataIssue(
                        "spatial_annotation_missing",
                        image_key,
                        "未配置可用的空间标注目录",
                    )
                )
                continue

            if verify_images and annotation_path is not None:
                annotation_error = _verify_spatial_annotation(
                    image_path,
                    annotation_path,
                )
                if annotation_error:
                    issues.append(
                        DataIssue(
                            "invalid_spatial_annotation",
                            str(annotation_path),
                            annotation_error,
                        )
                    )
                    continue

            samples.append(
                TrainingSample(
                    image_path=image_key,
                    label=label,
                    spatial_annotation_path=(
                        str(annotation_path)
                        if annotation_path is not None
                        else None
                    ),
                )
            )

    return sorted(samples, key=lambda sample: sample.image_path), issues


def load_vocabulary(vocab_path: str) -> Set[str]:
    path = Path(vocab_path).expanduser().resolve()
    if path.suffix.lower() == ".json":
        content = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(content, dict):
            raise ValueError(f"JSON 词表必须是 token 到 id 的对象: {path}")
        return set(content)
    return set(path.read_text(encoding="utf-8").splitlines())


def find_missing_characters(
    samples: Sequence[TrainingSample],
    vocabulary: Set[str],
) -> Dict[str, int]:
    missing: Dict[str, int] = defaultdict(int)
    for sample in samples:
        for char in sample.label:
            if char not in vocabulary:
                missing[char] += 1
    return dict(sorted(missing.items(), key=lambda item: (-item[1], item[0])))


@lru_cache(maxsize=None)
def image_content_fingerprint(image_path: str) -> str:
    """返回图片文件内容指纹，用于识别不同路径下的同一张图片。"""
    digest = hashlib.sha256()
    with Path(image_path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def deduplicate_samples_by_image_content(
    samples: Sequence[TrainingSample],
) -> Tuple[List[TrainingSample], List[Tuple[TrainingSample, TrainingSample]]]:
    """按图片字节去重，并拒绝相同图片对应不同标签。

    返回保留样本和 ``(重复样本, 首个保留样本)``。输入通常已经按路径排序，
    因而保留结果稳定；这不会删除磁盘文件，只控制训练/评测实际使用的样本。
    """
    kept: List[TrainingSample] = []
    duplicates: List[Tuple[TrainingSample, TrainingSample]] = []
    first_by_fingerprint: Dict[str, TrainingSample] = {}
    for sample in samples:
        fingerprint = image_content_fingerprint(sample.image_path)
        previous = first_by_fingerprint.get(fingerprint)
        if previous is None:
            first_by_fingerprint[fingerprint] = sample
            kept.append(sample)
            continue
        if previous.label != sample.label:
            raise ValueError(
                "完全相同的图片存在冲突标签: "
                f"{previous.image_path}={previous.label!r}, "
                f"{sample.image_path}={sample.label!r}"
            )
        duplicates.append((sample, previous))
    return kept, duplicates


def find_cross_split_content_duplicates(
    left_samples: Sequence[TrainingSample],
    right_samples: Sequence[TrainingSample],
) -> List[Tuple[TrainingSample, TrainingSample]]:
    """查找两个集合中路径不同、内容完全相同的图片。"""
    left_by_fingerprint: Dict[str, TrainingSample] = {}
    for sample in left_samples:
        left_by_fingerprint.setdefault(
            image_content_fingerprint(sample.image_path),
            sample,
        )
    overlaps = []
    for sample in right_samples:
        previous = left_by_fingerprint.get(
            image_content_fingerprint(sample.image_path)
        )
        if previous is not None:
            overlaps.append((previous, sample))
    return overlaps


def split_samples_by_label(
    samples: Sequence[TrainingSample],
    eval_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[TrainingSample], List[TrainingSample], List[TrainingSample]]:
    """
    按规范化公司名稳定切分。

    哈希切分保证同名公司的所有图片只进入一个集合；后续增加新公司时，已有公司
    的归属不会改变。
    """
    if eval_ratio < 0 or test_ratio < 0 or eval_ratio + test_ratio >= 1:
        raise ValueError("eval_ratio 和 test_ratio 需大于等于 0，且二者之和小于 1")

    train_samples: List[TrainingSample] = []
    eval_samples: List[TrainingSample] = []
    test_samples: List[TrainingSample] = []

    for sample in samples:
        digest = hashlib.sha256(f"{seed}:{sample.label}".encode("utf-8")).digest()
        ratio = int.from_bytes(digest[:8], byteorder="big") / float(2**64)
        if ratio < test_ratio:
            test_samples.append(sample)
        elif ratio < test_ratio + eval_ratio:
            eval_samples.append(sample)
        else:
            train_samples.append(sample)

    if eval_ratio > 0 and not eval_samples:
        raise ValueError("验证集为空；请增加数据或提高 eval_ratio")
    if test_ratio > 0 and not test_samples:
        raise ValueError("测试集为空；请增加数据或提高 test_ratio")
    if not train_samples:
        raise ValueError("训练集为空；请检查切分比例")

    return train_samples, eval_samples, test_samples


def split_samples_by_path(
    samples: Sequence[TrainingSample],
    eval_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[TrainingSample], List[TrainingSample], List[TrainingSample]]:
    """按图片路径稳定切分，适用于同一词典公司名生成多种合成形制。"""
    if eval_ratio < 0 or test_ratio < 0 or eval_ratio + test_ratio >= 1:
        raise ValueError("eval_ratio 和 test_ratio 需大于等于 0，且二者之和小于 1")

    train_samples: List[TrainingSample] = []
    eval_samples: List[TrainingSample] = []
    test_samples: List[TrainingSample] = []
    for sample in samples:
        digest = hashlib.sha256(
            f"{seed}:{sample.image_path}".encode("utf-8")
        ).digest()
        ratio = int.from_bytes(digest[:8], byteorder="big") / float(2**64)
        if ratio < test_ratio:
            test_samples.append(sample)
        elif ratio < test_ratio + eval_ratio:
            eval_samples.append(sample)
        else:
            train_samples.append(sample)

    if eval_ratio > 0 and not eval_samples:
        raise ValueError("验证集为空；请增加数据或提高 eval_ratio")
    if test_ratio > 0 and not test_samples:
        raise ValueError("测试集为空；请增加数据或提高 test_ratio")
    if not train_samples:
        raise ValueError("训练集为空；请检查切分比例")
    return train_samples, eval_samples, test_samples


def limit_samples_per_label(
    samples: Sequence[TrainingSample],
    max_samples_per_label: int,
    seed: int,
) -> List[TrainingSample]:
    if max_samples_per_label <= 0:
        return list(samples)

    grouped: Dict[str, List[TrainingSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.label].append(sample)

    selected: List[TrainingSample] = []
    for label in sorted(grouped):
        candidates = sorted(grouped[label], key=lambda sample: sample.image_path)
        if len(candidates) > max_samples_per_label:
            label_seed = int.from_bytes(
                hashlib.sha256(f"{seed}:{label}".encode("utf-8")).digest()[:8],
                byteorder="big",
            )
            rng = random.Random(label_seed)
            candidates = rng.sample(candidates, max_samples_per_label)
        selected.extend(candidates)
    return sorted(selected, key=lambda sample: sample.image_path)


def write_split_manifest(
    output_path: str,
    train_samples: Sequence[TrainingSample],
    eval_samples: Sequence[TrainingSample],
    test_samples: Sequence[TrainingSample],
    settings: Dict[str, object],
    extra_splits: Optional[Dict[str, Sequence[TrainingSample]]] = None,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    split_samples: Dict[str, Sequence[TrainingSample]] = {
        "train": train_samples,
        "eval": eval_samples,
        "test": test_samples,
    }
    if extra_splits:
        duplicate_names = set(split_samples) & set(extra_splits)
        if duplicate_names:
            raise ValueError(f"额外数据切分名称重复: {sorted(duplicate_names)}")
        split_samples.update(extra_splits)

    payload = {
        "settings": settings,
        "counts": {
            name: len(samples)
            for name, samples in split_samples.items()
        },
        "fingerprints": {
            name: fingerprint_samples(samples)
            for name, samples in split_samples.items()
        },
        "samples": {
            name: [asdict(sample) for sample in samples]
            for name, samples in split_samples.items()
        },
    }
    sample_count = sum(len(samples) for samples in split_samples.values())
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2 if sample_count <= 10000 else None,
        ),
        encoding="utf-8",
    )


def fingerprint_samples(samples: Sequence[TrainingSample]) -> str:
    digest = hashlib.sha256()
    for sample in samples:
        digest.update(sample.image_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sample.label.encode("utf-8"))
        if sample.spatial_annotation_path is not None:
            digest.update(b"\0spatial\0")
            digest.update(sample.spatial_annotation_path.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def summarize_issues(issues: Sequence[DataIssue], limit: int = 30) -> str:
    if not issues:
        return ""
    lines = [
        f"- [{issue.issue_type}] {issue.path}: {issue.detail}"
        for issue in issues[:limit]
    ]
    if len(issues) > limit:
        lines.append(f"- 其余 {len(issues) - limit} 个问题已省略")
    return "\n".join(lines)

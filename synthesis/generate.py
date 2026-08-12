"""生成面向“扫描件印章公司名识别”的合成预训练数据。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from bisect import bisect_left
from collections import Counter, defaultdict
from functools import lru_cache
from math import cos, pi, sin
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont, ImageOps


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_PROCESS_NUM = min(8, os.cpu_count() or 4)
DEFAULT_SEAL_NUM = 300000
DEFAULT_UP_NUM = (4, 38)
DEFAULT_MAKE_PATH = PROJECT_ROOT / "data" / "synthetic"
DEFAULT_BG_PATH = SCRIPT_DIR / "backgrounds"
DEFAULT_FONT_PATH = SCRIPT_DIR / "fonts"
DEFAULT_VOCAB_PATH = PROJECT_ROOT / "models" / "init" / "vocab.json"
DEFAULT_CORPUS_PATH = SCRIPT_DIR / "company_names.txt"
GENERATOR_VERSION = "open_source_v1"
SAMPLE_METADATA_FORMAT = "per_sample_json_v1"
DEFAULT_ELLIPSE_RATIO_RANGE = (0.68, 0.84)
DEFAULT_CIRCLE_STAR_PROBABILITY = 0.82
DEFAULT_LARGE_ROTATION_PROBABILITY = 0.25
LARGE_ROTATION_MIN_DEGREES = 45.0
LARGE_ROTATION_MAX_DEGREES = 180.0
DEFAULT_TEXT_HEATMAP_RADIUS_RATIO = 0.018
DEFAULT_STAMP_HEATMAP_RADIUS_RATIO = 0.025
DEFAULT_CHARACTER_HEATMAP_RADIUS_RATIO = 0.012
SPATIAL_ANNOTATION_FORMAT = "dual_rgba_semantic_maps_v2"
SPATIAL_ANNOTATION_CHANNELS = {
    "primary": {
        "R": "company_text_binary_mask",
        "G": "full_stamp_binary_mask",
        "B": "company_text_gaussian_heatmap",
        "A": "full_stamp_gaussian_heatmap",
    },
    "detail": {
        "R": "company_character_center_gaussian_heatmap",
        "G": "company_character_reading_progress",
        "B": "company_first_character_gaussian_heatmap",
        "A": "company_last_character_gaussian_heatmap",
    },
}
FORBIDDEN_COMPANY_NAME_CHARACTERS = {"?", "？", "�"}
FONT_MISSING_GLYPH_PROBES = ("\u0378", "\u0380", "\uffff", "\U0010ffff")
SUPPORTED_LAYOUTS = {
    "standard",
    "bilingual_ring",
    "oval_service",
    "multiline_center",
}
DEFAULT_LONG_HIGH_RISK_PREFIXES = ("宁波北仑", "宁波")
DEFAULT_LONG_HIGH_RISK_SUFFIXES = ("个体工商户",)


def _equal_ellipse_arc_angles(
    span_degrees: float,
    character_count: int,
    ellipse_ratio: float,
) -> List[float]:
    """返回纵向压缩后沿椭圆弧长等距的角度序列。

    旧实现先按圆的等角间隔排字，再纵向压缩整章，导致椭圆左右两侧字符严重
    拥挤。这里预先按压缩后曲线的弧长反算角度，绘制后再压缩仍保持近似等距。
    """
    if character_count <= 0:
        return []
    if character_count == 1:
        return [0.0]
    sample_count = max(1200, character_count * 120)
    half_span = span_degrees / 2
    sample_angles = [
        -half_span + span_degrees * index / sample_count
        for index in range(sample_count + 1)
    ]
    cumulative = [0.0]
    for left, right in zip(sample_angles, sample_angles[1:]):
        midpoint = (left + right) * pi / 360
        derivative = (
            cos(midpoint) ** 2
            + ellipse_ratio**2 * sin(midpoint) ** 2
        ) ** 0.5
        cumulative.append(
            cumulative[-1] + derivative * (right - left) * pi / 180
        )

    total_length = cumulative[-1]
    ascending_angles = []
    for index in range(character_count):
        target = total_length * index / (character_count - 1)
        right_index = min(
            max(1, bisect_left(cumulative, target)),
            len(cumulative) - 1,
        )
        left_index = right_index - 1
        interval = cumulative[right_index] - cumulative[left_index]
        fraction = (
            0.0
            if interval <= 0
            else (target - cumulative[left_index]) / interval
        )
        ascending_angles.append(
            sample_angles[left_index]
            + fraction
            * (sample_angles[right_index] - sample_angles[left_index])
        )
    return list(reversed(ascending_angles))

# business 贴近现有真实样本；broad/rare 补足困难长尾；ring/long 先用较干净
# 的图学习小字双环形制和长弧形文字，再由前三个池提供退化鲁棒性。
DEFAULT_STYLE = {
    "color_ratios": {"red": 89, "blue": 7, "purple": 1, "black": 3},
    "shape_ratios": {"circle": 90, "ellipse": 7, "rectangle": 3},
    "layout_ratios": {
        "standard": 90,
        "bilingual_ring": 6,
        "oval_service": 2,
        "multiline_center": 2,
    },
    "bilingual_shape_ratios": {"circle": 68, "ellipse": 32},
    "document_interference_probability": 0.18,
    "foreground_occlusion_probability": 0.10,
    "blank_background_probability": 0.58,
    "partial_crop_probability": 0.03,
    "max_partial_crop_ratio": 0.045,
    "large_rotation_probability": DEFAULT_LARGE_ROTATION_PROBABILITY,
    "ink_degradation_probability": 0.72,
    "stamp_size_ratio_range": (0.82, 0.99),
}

DEFAULT_FONT_XRATIO_UP_DICT = {
    "3": 0.70,
    "4": 0.685,
    "5": 0.67,
    "6": 0.655,
    "7": 0.64,
    "8": 0.625,
    "9": 0.61,
    "10": 0.595,
    "11": 0.58,
    "12": 0.565,
    "13": 0.55,
    "14": 0.535,
    "15": 0.52,
    "16": 0.51,
    "17": 0.50,
    "18": 0.49,
    "19": 0.48,
    "20": 0.47,
    "21": 0.46,
    "22": 0.45,
}

# 范围覆盖鲜印、淡印和扫描偏色；模糊、压缩、噪声等退化交给训练加载器动态处理。
INK_COLOR_RANGES = {
    "red": ((195, 255), (0, 55), (0, 55)),
    "blue": ((0, 55), (45, 115), (150, 235)),
    "purple": ((105, 185), (20, 90), (115, 215)),
    # 黑色章必须保持近灰阶，不能让 RGB 三通道独立随机后产生彩色色偏。
    # 轻重墨、褪色和扫描灰度变化再由 alpha 退化与在线增强共同完成。
    "black": ((0, 65), (0, 65), (0, 65)),
}
INK_ALPHA_RANGE = (125, 215)
BLACK_INK_ALPHA_RANGE = (180, 245)

AREAS = [
    "北京",
    "上海",
    "天津",
    "重庆",
    "浙江",
    "江苏",
    "广东",
    "山东",
    "福建",
    "四川",
    "河南",
    "河北",
    "湖北",
    "湖南",
    "安徽",
    "江西",
    "辽宁",
    "陕西",
    "宁波",
    "杭州",
    "苏州",
    "深圳",
    "广州",
    "青岛",
    "温州",
    "嘉兴",
    "绍兴",
    "金华",
    "舟山",
]
BRAND_WORDS = [
    "华信",
    "安泰",
    "恒远",
    "盛达",
    "金源",
    "宏宇",
    "启航",
    "天成",
    "海纳",
    "中科",
    "新联",
    "瑞丰",
    "嘉诚",
    "永兴",
    "博远",
    "万通",
    "鼎盛",
    "鑫隆",
    "德胜",
    "凯润",
    "智创",
    "云联",
    "百川",
    "东升",
    "明辉",
    "康泽",
    "绿源",
]
BRAND_CHAR_POOL = "华信安泰恒远盛达金源宏宇启航天成海纳中科新联瑞丰嘉诚永兴正博万通鼎鑫隆德胜凯润智创云百川东升明辉康泽绿"
INDUSTRIES = [
    "科技",
    "信息技术",
    "物流",
    "国际物流",
    "供应链管理",
    "货运代理",
    "化工",
    "新材料",
    "机械设备",
    "智能制造",
    "电子商务",
    "装饰工程",
    "贸易",
    "实业",
    "生物科技",
    "医药",
    "能源",
    "环保",
    "汽车服务",
    "企业管理",
]
SUFFIXES = [
    "有限公司",
    "有限公司",
    "有限公司",
    "有限责任公司",
    "股份有限公司",
    "集团有限公司",
]
AUXILIARY_TEXTS = [
    "合同专用章",
    "财务专用章",
    "发票专用章",
    "业务专用章",
    "质检专用章",
    "项目部",
]
OVAL_SERVICE_TEXTS = [
    "业务专用章",
    "合同专用章",
    "报关专用章",
    "货运专用章",
    "操作专用章",
]
MULTILINE_CENTER_TEXTS = [
    ("危险货物", "集装箱", "装箱专用章"),
    ("业务办理", "审核", "专用章"),
    ("货物运输", "操作", "专用章"),
    ("合同业务", "受理", "专用章"),
]
ENGLISH_COMPANY_WORDS = [
    "CHINA",
    "SHANGHAI",
    "NINGBO",
    "ZHEJIANG",
    "JIANGSU",
    "INTERNATIONAL",
    "LOGISTICS",
    "FREIGHT",
    "CHEMICALS",
    "ELECTRICAL",
    "INDUSTRY",
    "TRADING",
    "SOURCE",
    "MATERIALS",
    "COMPANY",
]
DOCUMENT_TEXT_SNIPPETS = [
    "检查地点",
    "装箱单位（公章）",
    "签发日期",
    "经办人",
    "审核人",
    "合同编号",
    "货物名称",
    "数量",
    "规格型号",
    "检验结论",
    "备注",
    "Place of Inspection",
    "Packing unit (seal)",
    "Date of Issue",
    "All stated above are correct",
]


@lru_cache(maxsize=256)
def load_font(font_path: str, font_size: int):
    return ImageFont.truetype(font_path, font_size, encoding="utf-8")


def build_font_character_coverage(
    font_paths: Sequence[str],
    characters: Set[str],
    probe_font_size: int = 64,
) -> Dict[str, Set[str]]:
    """
    预先检查每款字体能实际画出的字符。

    部分中文字体遇到未收录的生僻字时不会报错，而是返回空字形或统一的缺字方框；
    如果继续生成，图片和标签就不一致，会直接制造错误监督信号。
    """
    coverage: Dict[str, Set[str]] = {}
    for font_path in font_paths:
        font = load_font(font_path, probe_font_size)
        missing_glyph_signatures = {
            _font_glyph_signature(font, character)
            for character in FONT_MISSING_GLYPH_PROBES
        }
        supported_characters = set()
        for character in characters:
            if character.isspace():
                supported_characters.add(character)
                continue
            glyph_signature = _font_glyph_signature(font, character)
            if (
                glyph_signature[1] is not None
                and glyph_signature not in missing_glyph_signatures
            ):
                supported_characters.add(character)
        coverage[font_path] = supported_characters
    return coverage


def _font_glyph_signature(font, character: str):
    glyph_mask = font.getmask(character, mode="L")
    return (
        glyph_mask.size,
        glyph_mask.getbbox(),
        hashlib.sha256(bytes(glyph_mask)).digest(),
    )


def get_compatible_font_paths(
    text: str,
    font_character_coverage: Dict[str, Set[str]],
) -> List[str]:
    required_characters = {
        character
        for character in text
        if not character.isspace()
    }
    return [
        font_path
        for font_path, supported_characters in font_character_coverage.items()
        if required_characters <= supported_characters
    ]


def get_files_list(path: Path, suffixes: Optional[Set[str]] = None) -> List[str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"资源目录不存在: {path}")
    files = [
        item
        for item in sorted(path.iterdir())
        if item.is_file()
        and (suffixes is None or item.suffix.lower() in suffixes)
    ]
    if not files:
        raise ValueError(f"资源目录为空: {path}")
    return [str(item) for item in files]


def get_recursive_files_list(
    path: Path,
    suffixes: Set[str],
) -> List[str]:
    """递归读取背景图；空目录表示只使用程序生成的纸面背景。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"资源目录不存在: {path}")
    files = [
        str(item)
        for item in sorted(path.rglob("*"))
        if item.is_file() and item.suffix.lower() in suffixes
    ]
    return files


def normalize_company_name(text: str) -> str:
    text = text.replace("\x00", "").replace("\ufeff", "").replace("\xa0", "")
    text = re.sub(r"\s+", "", text)
    text = text.translate(str.maketrans({"(": "（", ")": "）"}))
    return text.rstrip("。.;；,，")


def load_vocabulary(vocab_path: Optional[str]) -> Optional[Set[str]]:
    if not vocab_path:
        return None
    path = Path(vocab_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"词表不存在: {path}")
    if path.suffix.lower() == ".json":
        content = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(content, dict):
            raise ValueError(f"JSON 词表必须是 token 到 id 的对象: {path}")
        return set(content)
    return set(path.read_text(encoding="utf-8").splitlines())


def fingerprint_values(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(values):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def fingerprint_sequence(values: Iterable[int]) -> str:
    """按顺序签名整数序列，区分名称采样计划的排列变化。"""
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def is_long_high_risk_name(
    name: str,
    minimum_length: int = 17,
    prefixes: Sequence[str] = DEFAULT_LONG_HIGH_RISK_PREFIXES,
    suffixes: Sequence[str] = DEFAULT_LONG_HIGH_RISK_SUFFIXES,
) -> bool:
    """判断是否属于容易被模型记成固定模板的长名称组合。

    suffixes 允许传入“个体工商户”这种业务片段；规范化后的公司名通常以
    “（个体工商户）”结束，因此这里同时兼容右括号。前缀或后缀任一命中即纳入，
    以便分别降低宁波前缀和个体工商户后缀，而不只压低两者的交集。
    """
    if len(name) < minimum_length:
        return False
    prefix_match = any(prefix and name.startswith(prefix) for prefix in prefixes)
    suffix_match = any(
        suffix
        and (
            name.endswith(suffix)
            or (
                not suffix.endswith("）")
                and name.endswith(f"{suffix}）")
            )
        )
        for suffix in suffixes
    )
    return prefix_match or suffix_match


def find_long_high_risk_name_indices(
    corpus: Sequence[str],
    minimum_length: int = 17,
    prefixes: Sequence[str] = DEFAULT_LONG_HIGH_RISK_PREFIXES,
    suffixes: Sequence[str] = DEFAULT_LONG_HIGH_RISK_SUFFIXES,
) -> List[int]:
    return [
        index
        for index, name in enumerate(corpus)
        if is_long_high_risk_name(name, minimum_length, prefixes, suffixes)
    ]


def build_name_schedule(
    corpus: Sequence[str],
    samples_per_name: int,
    high_risk_max_samples_per_name: int,
    minimum_length: int = 17,
    prefixes: Sequence[str] = DEFAULT_LONG_HIGH_RISK_PREFIXES,
    suffixes: Sequence[str] = DEFAULT_LONG_HIGH_RISK_SUFFIXES,
) -> List[int]:
    """构造基础采样计划，限制长高风险组合并把空出的样本回填。

    返回空列表表示沿用旧的“每轮按语料顺序循环”逻辑。启用限额后，所有高风险
    名称仍至少出现一次，且每名最多出现指定次数；释放出的样本按轮询方式分配给
    其他名称，因此基础样本总量不变。
    """
    if samples_per_name <= 0:
        raise ValueError("samples_per_name 必须大于 0")
    if high_risk_max_samples_per_name < 0:
        raise ValueError("high_risk_max_samples_per_name 不能小于 0")
    if not corpus or high_risk_max_samples_per_name == 0:
        return []
    if high_risk_max_samples_per_name >= samples_per_name:
        return []

    high_risk_indices = set(
        find_long_high_risk_name_indices(
            corpus,
            minimum_length=minimum_length,
            prefixes=prefixes,
            suffixes=suffixes,
        )
    )
    if not high_risk_indices:
        return []
    normal_indices = [
        index for index in range(len(corpus)) if index not in high_risk_indices
    ]
    if not normal_indices:
        raise ValueError(
            "长名称高风险限额无法回填：语料中没有其他公司名。"
            "请补充正常有限公司/股份有限公司等前后缀，或关闭该限额。"
        )

    capped_samples = min(
        samples_per_name,
        high_risk_max_samples_per_name,
    )
    schedule: List[int] = []
    # 前 capped_samples 轮保留全量名称，确保每个高风险名称仍有覆盖。
    for _ in range(capped_samples):
        schedule.extend(range(len(corpus)))
    # 后续轮次只采样非高风险名称。
    for _ in range(capped_samples, samples_per_name):
        schedule.extend(normal_indices)

    target_count = len(corpus) * samples_per_name
    refill_count = target_count - len(schedule)
    schedule.extend(
        normal_indices[index % len(normal_indices)]
        for index in range(refill_count)
    )
    return schedule


def fingerprint_resources(paths: Sequence[str]) -> str:
    """用资源内容生成签名，避免续跑时静默混入另一批字体或背景。"""
    digest = hashlib.sha256()
    for raw_path in sorted(paths):
        path = Path(raw_path)
        digest.update(str(path.resolve()).encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as resource_file:
            while chunk := resource_file.read(1024 * 1024):
                digest.update(chunk)
        digest.update(b"\n")
    return digest.hexdigest()


def build_rare_character_boost(
    corpus: Sequence[str],
    base_samples_per_name: int,
    minimum_occurrences: int,
    max_extra_samples_per_name: Optional[int] = None,
    base_name_indices: Optional[Sequence[int]] = None,
    avoid_name_indices: Optional[Set[int]] = None,
    max_extra_samples_per_avoided_name: Optional[int] = None,
) -> List[int]:
    """
    返回需要额外生成的公司名索引，使每个字符至少达到指定标签出现次数。

    采用确定性贪心：优先处理缺口最大的字符，并选择一次能覆盖最多剩余稀有字符
    的公司名。若传入实际基础采样计划，则按计划而不是按原始语料等量估算已有出现
    次数；avoid_name_indices 只作为软避让，只有没有其他来源时才会选中这些公司。
    max_extra_samples_per_avoided_name 用于给这类名称保留必要的稀缺字符例外补样，
    不会在有其他字符来源时主动使用。它只增加真实语料中的完整公司名，不拼接或
    截断标签。
    """
    if minimum_occurrences <= 0:
        return []
    if base_samples_per_name <= 0:
        raise ValueError("字符补样需要 samples_per_name 大于 0")
    if not corpus:
        raise ValueError("字符补样需要有效公司名语料")
    if (
        max_extra_samples_per_name is not None
        and max_extra_samples_per_name <= 0
    ):
        raise ValueError("单公司低频字符补样上限必须大于 0")
    if (
        max_extra_samples_per_avoided_name is not None
        and max_extra_samples_per_avoided_name <= 0
    ):
        raise ValueError("被避让公司低频字符补样上限必须大于 0")

    name_character_counts = [Counter(name) for name in corpus]
    character_to_name_indices: Dict[str, List[int]] = defaultdict(list)
    base_occurrences: Counter = Counter()
    for name_index, character_counts in enumerate(name_character_counts):
        for character, count in character_counts.items():
            character_to_name_indices[character].append(name_index)
            if base_name_indices is None:
                base_occurrences[character] += count * base_samples_per_name
    if base_name_indices is not None:
        for name_index in base_name_indices:
            if name_index < 0 or name_index >= len(name_character_counts):
                raise ValueError(
                    f"字符补样基础采样计划包含无效公司索引: {name_index}"
                )
            base_occurrences.update(name_character_counts[name_index])

    deficits = {
        character: minimum_occurrences - count
        for character, count in base_occurrences.items()
        if count < minimum_occurrences
    }
    boost_name_indices: List[int] = []
    extra_counts: Counter = Counter()
    while deficits:
        target_character = max(
            deficits,
            key=lambda character: (deficits[character], character),
        )
        candidates = [
            name_index
            for name_index in character_to_name_indices[target_character]
            if (
                (
                    max_extra_samples_per_avoided_name
                    if (
                        avoid_name_indices
                        and name_index in avoid_name_indices
                        and max_extra_samples_per_avoided_name is not None
                    )
                    else max_extra_samples_per_name
                )
                is None
                or extra_counts[name_index]
                < (
                    max_extra_samples_per_avoided_name
                    if (
                        avoid_name_indices
                        and name_index in avoid_name_indices
                        and max_extra_samples_per_avoided_name is not None
                    )
                    else max_extra_samples_per_name
                )
            )
        ]
        preferred_candidates = [
            name_index
            for name_index in candidates
            if not avoid_name_indices or name_index not in avoid_name_indices
        ]
        if preferred_candidates:
            candidates = preferred_candidates
        if not candidates:
            raise ValueError(
                f"字符 {target_character!r} 无法在单公司最多追加 "
                f"{max_extra_samples_per_name} 张（高风险例外上限 "
                f"{max_extra_samples_per_avoided_name} 张）的约束下达到 "
                f"{minimum_occurrences} 次。请提高 --samples-per-name，"
                "或增加包含该字符的不同公司名；"
                "不建议通过大量重复同一公司解决。"
            )

        def candidate_score(name_index: int):
            character_counts = name_character_counts[name_index]
            coverage = sum(
                min(deficits.get(character, 0), count)
                for character, count in character_counts.items()
            )
            return coverage, -len(corpus[name_index]), -name_index

        selected_index = max(candidates, key=candidate_score)
        boost_name_indices.append(selected_index)
        extra_counts[selected_index] += 1
        for character, count in name_character_counts[selected_index].items():
            if character not in deficits:
                continue
            remaining = deficits[character] - count
            if remaining <= 0:
                del deficits[character]
            else:
                deficits[character] = remaining

    return boost_name_indices


def load_excluded_names(paths: Sequence[str]) -> Set[str]:
    excluded: Set[str] = set()
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"排除标签路径不存在: {path}")
        files: Iterable[Path]
        if path.is_file():
            files = [path]
        else:
            files = path.rglob("*.txt")
        for label_path in files:
            if label_path.stem.endswith("_llm"):
                continue
            try:
                label = normalize_company_name(label_path.read_text(encoding="utf-8"))
            except UnicodeDecodeError:
                continue
            if label:
                excluded.add(label)
    return excluded


def _filter_items_by_vocabulary(
    items: Sequence[str],
    allowed_characters: Optional[Set[str]],
) -> List[str]:
    if allowed_characters is None:
        return list(items)
    return [
        item
        for item in items
        if all(character in allowed_characters for character in item)
    ]


class CompanyNameGenerator:
    """生成结构合理、字符受模型词表约束的公司名。"""

    def __init__(
        self,
        corpus_path: Optional[str],
        allowed_characters: Optional[Set[str]],
        excluded_names: Set[str],
        name_length: Tuple[int, int],
        corpus_probability: float = 0.25,
        excluded_substrings: Sequence[str] = (),
    ) -> None:
        self.allowed_characters = allowed_characters
        self.excluded_names = excluded_names
        self.excluded_substrings = tuple(
            substring for substring in excluded_substrings if substring
        )
        self.min_length, self.max_length = name_length
        self.corpus_probability = corpus_probability

        self.areas = _filter_items_by_vocabulary(AREAS, allowed_characters)
        self.brand_words = _filter_items_by_vocabulary(
            BRAND_WORDS,
            allowed_characters,
        )
        self.industries = _filter_items_by_vocabulary(
            INDUSTRIES,
            allowed_characters,
        )
        self.suffixes = _filter_items_by_vocabulary(SUFFIXES, allowed_characters)
        self.brand_characters = [
            character
            for character in BRAND_CHAR_POOL
            if allowed_characters is None or character in allowed_characters
        ]

        self.corpus: List[str] = []
        self.corpus_total_count = 0
        self.corpus_rejected_count = 0
        self.corpus_duplicate_count = 0
        self.corpus_rejection_reasons: Counter = Counter()
        self.corpus_rejected_examples: Dict[str, List[str]] = defaultdict(list)
        self.corpus_missing_characters: Counter = Counter()
        if corpus_path:
            corpus_file = Path(corpus_path).expanduser().resolve()
            if not corpus_file.exists():
                raise FileNotFoundError(f"公司名语料不存在: {corpus_file}")
            for line in corpus_file.read_text(encoding="utf-8").splitlines():
                if line.lstrip().startswith("#"):
                    continue
                name = normalize_company_name(line)
                if not name:
                    continue
                self.corpus_total_count += 1
                invalid_reasons = self._invalid_reasons(name)
                if not invalid_reasons:
                    self.corpus.append(name)
                else:
                    self.corpus_rejected_count += 1
                    if "out_of_vocabulary" in invalid_reasons:
                        self.corpus_missing_characters.update(
                            character
                            for character in name
                            if (
                                self.allowed_characters is not None
                                and character not in self.allowed_characters
                            )
                        )
                    for reason in invalid_reasons:
                        self.corpus_rejection_reasons[reason] += 1
                        examples = self.corpus_rejected_examples[reason]
                        if len(examples) < 10 and name not in examples:
                            examples.append(name)
            accepted_count = len(self.corpus)
            self.corpus = sorted(set(self.corpus))
            self.corpus_duplicate_count = accepted_count - len(self.corpus)
            if not self.corpus:
                raise ValueError(
                    "公司名语料在标签长度、词表和排除规则过滤后为空"
                )

        if not self.corpus or self.corpus_probability < 1:
            required = {
                "地区": self.areas,
                "字号": self.brand_words,
                "行业": self.industries,
                "组织形式": self.suffixes,
                "字号字符": self.brand_characters,
            }
            empty_groups = [
                name for name, values in required.items() if not values
            ]
            if empty_groups:
                raise ValueError(
                    f"词表过滤后缺少公司名模板组件: {empty_groups}。"
                    "若只使用逐行公司名语料，请设置 --corpus_probability 1。"
                )

    def _is_valid(self, name: str) -> bool:
        return not self._invalid_reasons(name)

    def _invalid_reasons(self, name: str) -> List[str]:
        reasons = []
        if len(name) < self.min_length:
            reasons.append("too_short")
        if len(name) > self.max_length:
            reasons.append("too_long")
        if any(
            character in FORBIDDEN_COMPANY_NAME_CHARACTERS
            for character in name
        ):
            reasons.append("placeholder_character")
        if name in self.excluded_names:
            reasons.append("excluded")
        if any(
            substring in name for substring in self.excluded_substrings
        ):
            reasons.append("excluded_substring")
        if (
            self.allowed_characters is not None
            and any(
                character not in self.allowed_characters
                for character in name
            )
        ):
            reasons.append("out_of_vocabulary")
        return reasons

    def _random_brand(self) -> str:
        if random.random() < 0.55:
            return random.choice(self.brand_words)
        length = random.randint(2, 4)
        return "".join(random.choice(self.brand_characters) for _ in range(length))

    def _template_name(self) -> str:
        area = random.choice(self.areas)
        brand = self._random_brand()
        industry = random.choice(self.industries)
        suffix = random.choice(self.suffixes)
        pattern = random.choices(
            [
                area + brand + industry + suffix,
                area + brand + suffix,
                brand + industry + suffix,
                area + industry + brand + suffix,
            ],
            weights=[52, 16, 12, 20],
            k=1,
        )[0]
        return pattern

    def get_company_name(self, sample_index: Optional[int] = None) -> str:
        if (
            self.corpus
            and self.corpus_probability >= 1
            and sample_index is not None
        ):
            return self.corpus[sample_index % len(self.corpus)]

        for _ in range(1000):
            if self.corpus and random.random() < self.corpus_probability:
                candidate = random.choice(self.corpus)
            else:
                candidate = self._template_name()
            if self._is_valid(candidate):
                return candidate
        raise RuntimeError("连续 1000 次未生成满足长度、词表和排除条件的公司名")

    # 兼容旧调用名称。
    def get_txt_chinese(self, char_len=None) -> str:
        return self.get_company_name()


def pentagram(x: float, y: float, radius: float, degree: float = 0):
    radian = pi / 180
    inner_radius = radius * sin(18 * radian) / cos(36 * radian)
    outer_vertices = [
        (
            x - radius * cos((90 + index * 72 + degree) * radian),
            y - radius * sin((90 + index * 72 + degree) * radian),
        )
        for index in range(5)
    ]
    inner_vertices = [
        (
            x - inner_radius * cos((126 + index * 72 + degree) * radian),
            y - inner_radius * sin((126 + index * 72 + degree) * radian),
        )
        for index in range(5)
    ]
    return [point for pair in zip(outer_vertices, inner_vertices) for point in pair]


def circle(x: float, y: float, radius: float):
    return x - radius, y - radius, x + radius, y + radius


def choose_by_ratio(ratios: Dict[str, float], config_name: str) -> str:
    if not ratios:
        raise ValueError(f"{config_name}不能为空")
    invalid_items = [
        name
        for name, weight in ratios.items()
        if not isinstance(weight, (int, float)) or weight < 0
    ]
    if invalid_items:
        raise ValueError(f"{config_name}包含无效权重: {invalid_items}")
    if sum(ratios.values()) <= 0:
        raise ValueError(f"{config_name}至少需要一个大于 0 的权重")
    return random.choices(
        list(ratios),
        weights=[ratios[name] for name in ratios],
        k=1,
    )[0]


def random_ink_color(color_name: str):
    if color_name not in INK_COLOR_RANGES:
        raise ValueError(
            f"不支持的印章颜色 {color_name!r}，可选值: {sorted(INK_COLOR_RANGES)}"
        )
    if color_name == "black":
        gray = random.randint(0, 58)
        rgb = tuple(
            max(0, min(65, gray + random.randint(-4, 4)))
            for _ in range(3)
        )
        alpha_range = BLACK_INK_ALPHA_RANGE
    else:
        rgb = tuple(
            random.randint(channel_range[0], channel_range[1])
            for channel_range in INK_COLOR_RANGES[color_name]
        )
        alpha_range = INK_ALPHA_RANGE
    return rgb + (random.randint(*alpha_range),)


def get_random_stamp_style(
    color_ratios: Dict[str, float],
    shape_ratios: Dict[str, float],
    ellipse_ratio_range: Tuple[float, float],
):
    unsupported_colors = set(color_ratios) - set(INK_COLOR_RANGES)
    if unsupported_colors:
        raise ValueError(f"颜色比例包含不支持的颜色: {sorted(unsupported_colors)}")
    supported_shapes = {"circle", "ellipse", "rectangle"}
    unsupported_shapes = set(shape_ratios) - supported_shapes
    if unsupported_shapes:
        raise ValueError(f"形状比例包含不支持的形状: {sorted(unsupported_shapes)}")

    min_ratio, max_ratio = ellipse_ratio_range
    if not 0 < min_ratio <= max_ratio <= 1:
        raise ValueError("椭圆短长轴比需满足 0 < 最小值 <= 最大值 <= 1")

    color_name = choose_by_ratio(color_ratios, "color_ratios")
    shape = choose_by_ratio(shape_ratios, "shape_ratios")
    ellipse_ratio = (
        random.uniform(min_ratio, max_ratio) if shape == "ellipse" else 1.0
    )
    return color_name, random_ink_color(color_name), shape, ellipse_ratio


def get_random_stamp_layout(layout_ratios: Dict[str, float]) -> str:
    unsupported_layouts = set(layout_ratios) - SUPPORTED_LAYOUTS
    if unsupported_layouts:
        raise ValueError(
            f"布局比例包含不支持的布局: {sorted(unsupported_layouts)}"
        )
    return choose_by_ratio(layout_ratios, "layout_ratios")


def get_random_english_company_text() -> str:
    word_count = random.randint(2, 4)
    words = random.sample(ENGLISH_COMPANY_WORDS, k=word_count)
    suffix = random.choice(("CO.,LTD.", "COMPANY LTD.", "CORPORATION"))
    return " ".join([*words, suffix])


class Stamp:
    def __init__(self):
        self.edge = 5
        self.radius = 250
        self.border = 13
        self.star_radius = 90
        self.fill = (235, 15, 15, 180)
        self.font_path = ""
        self.company_name = "测试有限公司"
        self.angle_up = 270
        self.font_size_up = 80
        self.font_xratio_up = 0.60
        self.stroke_width_up = 2
        self.background_path = ""
        self.save_path = Path(DEFAULT_MAKE_PATH)
        self.spatial_annotation_path: Optional[Path] = None
        self.save_name = "stamp"
        self.shape = "circle"
        self.layout = "standard"
        self.ellipse_ratio = 0.78
        self.circle_star_probability = DEFAULT_CIRCLE_STAR_PROBABILITY
        self.double_border = False
        self.auxiliary_text = ""
        self.serial_text = ""
        self.english_company_text = ""
        self.center_lines: Tuple[str, ...] = ()
        self.has_star = False
        self.document_interference_probability = 0.29
        self.foreground_occlusion_probability = 0.10
        self.blank_background_probability = 0.58
        self.partial_crop_probability = 0.03
        self.max_partial_crop_ratio = 0.045
        self.large_rotation_probability = DEFAULT_LARGE_ROTATION_PROBABILITY
        self.large_rotation_min_degrees = LARGE_ROTATION_MIN_DEGREES
        self.large_rotation_max_degrees = LARGE_ROTATION_MAX_DEGREES
        self.ink_degradation_probability = 0.72
        self.stamp_size_ratio_range = (0.82, 0.99)
        self.was_partially_cropped = False
        self.actual_partial_crop_ratio = 0.0
        self.actual_rotation_degrees = 0.0
        self.ink_degradation_applied = False
        self.foreground_occlusion_applied = False
        self.document_interference_applied = False
        self.image = None
        self.joined_image = None
        self.company_text_mask = None
        self.stamp_mask = None
        self.company_character_center_mask = None
        self.company_character_order_map = None
        self.company_first_character_center_mask = None
        self.company_last_character_center_mask = None
        self.joined_company_text_mask = None
        self.joined_stamp_mask = None
        self.joined_company_character_center_mask = None
        self.joined_company_character_order_map = None
        self.joined_company_first_character_center_mask = None
        self.joined_company_last_character_center_mask = None
        self.text_heatmap_radius_ratio = DEFAULT_TEXT_HEATMAP_RADIUS_RATIO
        self.stamp_heatmap_radius_ratio = DEFAULT_STAMP_HEATMAP_RADIUS_RATIO
        self.character_heatmap_radius_ratio = (
            DEFAULT_CHARACTER_HEATMAP_RADIUS_RATIO
        )

    @staticmethod
    def _character_progress_value(index: int, count: int) -> int:
        """把字符序号映射到 (0, 255)，避免首字符与背景同为 0。"""
        if count <= 0 or not 0 <= index < count:
            raise ValueError(
                f"非法公司字符序号: index={index}, count={count}"
            )
        return max(1, min(254, round(255 * (index + 1) / (count + 1))))

    def _record_character_geometry(
        self,
        glyph_mask: Image.Image,
        position: Tuple[int, int],
        center: Tuple[float, float],
        character_index: int,
        character_count: int,
    ) -> None:
        """记录单字中心、首尾位置和连续阅读顺序。"""
        if any(
            annotation is None
            for annotation in (
                self.company_character_center_mask,
                self.company_character_order_map,
                self.company_first_character_center_mask,
                self.company_last_character_center_mask,
            )
        ):
            raise RuntimeError("公司字符细节标注尚未初始化")
        progress = self._character_progress_value(
            character_index,
            character_count,
        )
        binary_glyph_mask = glyph_mask.point(
            lambda value: 255 if value >= 16 else 0
        )
        self.company_character_order_map.paste(
            progress,
            position,
            binary_glyph_mask,
        )
        radius = max(2, round(min(glyph_mask.size) * 0.035))
        bounds = (
            round(center[0] - radius),
            round(center[1] - radius),
            round(center[0] + radius),
            round(center[1] + radius),
        )
        ImageDraw.Draw(self.company_character_center_mask).ellipse(
            bounds,
            fill=255,
        )
        if character_index == 0:
            ImageDraw.Draw(self.company_first_character_center_mask).ellipse(
                bounds,
                fill=255,
            )
        if character_index == character_count - 1:
            ImageDraw.Draw(self.company_last_character_center_mask).ellipse(
                bounds,
                fill=255,
            )

    def draw_rotated_text(
        self,
        image,
        angle,
        xy,
        radius,
        character,
        fill,
        font_path,
        font_size,
        font_xratio,
        stroke_width,
        font_flip=False,
        semantic_mask: Optional[Image.Image] = None,
        character_index: Optional[int] = None,
        character_count: Optional[int] = None,
    ):
        font = load_font(font_path, font_size)
        probe = Image.new("L", (font_size * 3, font_size * 3), 0)
        probe_draw = ImageDraw.Draw(probe)
        bounds = probe_draw.textbbox(
            (0, 0),
            character,
            font=font,
            stroke_width=stroke_width,
        )
        padding = stroke_width + 4
        glyph_width = max(1, bounds[2] - bounds[0] + padding * 2)
        glyph_height = max(1, bounds[3] - bounds[1] + padding * 2)
        glyph = Image.new("L", (glyph_width, glyph_height), 0)
        glyph_draw = ImageDraw.Draw(glyph)
        glyph_draw.text(
            (padding - bounds[0], padding - bounds[1]),
            character,
            255,
            font=font,
            stroke_width=stroke_width,
        )
        glyph = glyph.resize(
            (max(1, round(glyph.width * font_xratio)), glyph.height),
            resample=Image.Resampling.LANCZOS,
        )

        rotation = angle + (180 if font_flip else 0)
        rotated_glyph = glyph.rotate(
            rotation,
            resample=Image.Resampling.BICUBIC,
            expand=True,
        )
        theta = angle * pi / 180
        glyph_center_radius = radius - glyph.height / 2
        center_x = xy[0] - sin(theta) * glyph_center_radius
        center_y = xy[1] - cos(theta) * glyph_center_radius
        position = (
            round(center_x - rotated_glyph.width / 2),
            round(center_y - rotated_glyph.height / 2),
        )
        color_image = Image.new("RGBA", rotated_glyph.size, fill)
        image.paste(color_image, position, rotated_glyph)
        if semantic_mask is not None:
            semantic_mask.paste(255, position, rotated_glyph)
        if character_index is not None or character_count is not None:
            if character_index is None or character_count is None:
                raise ValueError("字符细节标注必须同时提供 index 和 count")
            self._record_character_geometry(
                rotated_glyph,
                position,
                (center_x, center_y),
                character_index,
                character_count,
            )

    def _draw_arc_text(
        self,
        image: Image.Image,
        text: str,
        center: Tuple[int, int],
        radius: float,
        span: float,
        font_size: int,
        font_xratio: float,
        stroke_width: int = 0,
        semantic_mask: Optional[Image.Image] = None,
        character_sequence_offset: Optional[int] = None,
        character_sequence_count: Optional[int] = None,
    ) -> None:
        if not text:
            return
        if self.shape == "ellipse":
            angles = _equal_ellipse_arc_angles(
                span,
                len(text),
                self.ellipse_ratio,
            )
        else:
            angle_step = span / max(len(text) - 1, 1)
            angles = [
                span / 2 - index * angle_step
                for index in range(len(text))
            ]
        if (character_sequence_offset is None) != (
            character_sequence_count is None
        ):
            raise ValueError("字符细节标注必须同时提供 offset 和 count")
        for local_index, (character, current_angle) in enumerate(
            zip(text, angles)
        ):
            self.draw_rotated_text(
                image,
                current_angle,
                center,
                radius,
                character,
                self.fill,
                self.font_path,
                font_size,
                font_xratio,
                stroke_width,
                semantic_mask=semantic_mask,
                character_index=(
                    character_sequence_offset + local_index
                    if character_sequence_offset is not None
                    else None
                ),
                character_count=character_sequence_count,
            )

    def _draw_centered_scaled_text(
        self,
        image: Image.Image,
        text: str,
        center: Tuple[int, int],
        max_width: int,
        font_size: int,
        stroke_width: int = 0,
        semantic_mask: Optional[Image.Image] = None,
        character_sequence_offset: Optional[int] = None,
        character_sequence_count: Optional[int] = None,
    ) -> None:
        """横向压缩超长文字，模拟椭圆业务章中的细长公司名。"""
        if (character_sequence_offset is None) != (
            character_sequence_count is None
        ):
            raise ValueError("字符细节标注必须同时提供 offset 和 count")
        font = load_font(self.font_path, font_size)
        probe = Image.new("RGBA", (font_size * max(len(text), 2) * 2, font_size * 3))
        draw = ImageDraw.Draw(probe)
        bounds = draw.textbbox(
            (0, 0),
            text,
            font=font,
            stroke_width=stroke_width,
        )
        padding = stroke_width + 5
        text_image = Image.new(
            "RGBA",
            (
                max(1, bounds[2] - bounds[0] + padding * 2),
                max(1, bounds[3] - bounds[1] + padding * 2),
            ),
            (255, 255, 255, 0),
        )
        text_draw = ImageDraw.Draw(text_image)
        text_draw.text(
            (padding - bounds[0], padding - bounds[1]),
            text,
            font=font,
            fill=self.fill,
            stroke_width=stroke_width,
        )
        detail_center = None
        detail_order = None
        detail_first = None
        detail_last = None
        if character_sequence_offset is not None:
            detail_center = Image.new("L", text_image.size, 0)
            detail_order = Image.new("L", text_image.size, 0)
            detail_first = Image.new("L", text_image.size, 0)
            detail_last = Image.new("L", text_image.size, 0)
            origin_x = padding - bounds[0]
            origin_y = padding - bounds[1]
            running_advance = 0.0
            for local_index, character in enumerate(text):
                character_index = character_sequence_offset + local_index
                # 只测量当前字符及下一字符的 kerning，避免对每个前缀重复布局造成
                # O(n²) 开销；中文公司名通常无 kerning，此公式也兼容少量数字/字母。
                if local_index + 1 < len(text):
                    next_character = text[local_index + 1]
                    character_advance = text_draw.textlength(
                        character + next_character,
                        font=font,
                    ) - text_draw.textlength(next_character, font=font)
                else:
                    character_advance = text_draw.textlength(
                        character,
                        font=font,
                    )
                character_mask = Image.new("L", text_image.size, 0)
                ImageDraw.Draw(character_mask).text(
                    (origin_x + running_advance, origin_y),
                    character,
                    font=font,
                    fill=255,
                    stroke_width=stroke_width,
                )
                character_box = character_mask.getbbox()
                if character_box is None:
                    running_advance += character_advance
                    continue
                progress = self._character_progress_value(
                    character_index,
                    character_sequence_count,
                )
                binary_character_mask = character_mask.point(
                    lambda value: 255 if value >= 16 else 0
                )
                detail_order.paste(
                    progress,
                    (0, 0),
                    binary_character_mask,
                )
                center_x = (character_box[0] + character_box[2]) / 2
                center_y = (character_box[1] + character_box[3]) / 2
                radius = max(
                    2,
                    round(
                        min(
                            character_box[2] - character_box[0],
                            character_box[3] - character_box[1],
                        )
                        * 0.035
                    ),
                )
                center_bounds = (
                    round(center_x - radius),
                    round(center_y - radius),
                    round(center_x + radius),
                    round(center_y + radius),
                )
                ImageDraw.Draw(detail_center).ellipse(
                    center_bounds,
                    fill=255,
                )
                if character_index == 0:
                    ImageDraw.Draw(detail_first).ellipse(
                        center_bounds,
                        fill=255,
                    )
                if character_index == character_sequence_count - 1:
                    ImageDraw.Draw(detail_last).ellipse(
                        center_bounds,
                        fill=255,
                    )
                running_advance += character_advance
        if text_image.width > max_width:
            resized_size = (
                max_width,
                max(1, round(text_image.height * max_width / text_image.width)),
            )
            text_image = text_image.resize(
                resized_size,
                resample=Image.Resampling.LANCZOS,
            )
            if detail_center is not None:
                detail_center = detail_center.resize(
                    resized_size,
                    resample=Image.Resampling.BILINEAR,
                )
                detail_order = detail_order.resize(
                    resized_size,
                    resample=Image.Resampling.BILINEAR,
                )
                detail_first = detail_first.resize(
                    resized_size,
                    resample=Image.Resampling.BILINEAR,
                )
                detail_last = detail_last.resize(
                    resized_size,
                    resample=Image.Resampling.BILINEAR,
                )
        position = (
            round(center[0] - text_image.width / 2),
            round(center[1] - text_image.height / 2),
        )
        image.paste(text_image, position, text_image)
        if semantic_mask is not None:
            semantic_mask.paste(255, position, text_image.getchannel("A"))
        if detail_center is not None:
            for destination, source in (
                (self.company_character_center_mask, detail_center),
                (self.company_character_order_map, detail_order),
                (self.company_first_character_center_mask, detail_first),
                (self.company_last_character_center_mask, detail_last),
            ):
                source_mask = source.point(
                    lambda value: 255 if value > 0 else 0
                )
                destination.paste(source, position, source_mask)

    def _draw_bilingual_ring_stamp(self, image: Image.Image) -> Image.Image:
        draw = ImageDraw.Draw(image)
        center = self.radius + self.edge
        draw.arc(
            circle(center, center, self.radius),
            start=0,
            end=360,
            fill=self.fill,
            width=self.border,
        )
        inner_radius = self.radius - 55
        draw.arc(
            circle(center, center, inner_radius),
            start=0,
            end=360,
            fill=self.fill,
            width=max(5, self.border // 2),
        )
        if self.double_border:
            second_inner_radius = inner_radius - 13
            draw.arc(
                circle(center, center, second_inner_radius),
                start=0,
                end=360,
                fill=self.fill,
                width=max(3, self.border // 3),
            )

        english_text = self.english_company_text or get_random_english_company_text()
        english_size = 27 if len(english_text) > 42 else 31
        self._draw_arc_text(
            image,
            english_text,
            (center, center),
            self.radius - self.border - 7,
            300,
            english_size,
            0.68,
            0,
        )
        chinese_size = max(29, min(58, round(900 / len(self.company_name))))
        self._draw_arc_text(
            image,
            self.company_name,
            (center, center),
            inner_radius - 13,
            260 if len(self.company_name) >= 17 else 235,
            chinese_size,
            min(0.64, _font_xratio(len(self.company_name)) + 0.08),
            self.stroke_width_up,
            semantic_mask=self.company_text_mask,
            character_sequence_offset=0,
            character_sequence_count=len(self.company_name),
        )
        if self.auxiliary_text:
            self._draw_centered_scaled_text(
                image,
                self.auxiliary_text,
                (center, center + 35),
                240,
                34,
            )
        self.has_star = False
        self._apply_ellipse_to_company_annotations()
        return self._apply_ellipse_shape(image)

    def _draw_oval_service_stamp(self, image: Image.Image) -> Image.Image:
        draw = ImageDraw.Draw(image)
        center = self.radius + self.edge
        draw.arc(
            circle(center, center, self.radius),
            start=0,
            end=360,
            fill=self.fill,
            width=self.border,
        )
        for offset, width in ((43, 7), (59, 4)):
            ring_radius = self.radius - offset
            draw.arc(
                circle(center, center, ring_radius),
                start=0,
                end=360,
                fill=self.fill,
                width=width,
            )

        english_text = self.english_company_text or get_random_english_company_text()
        self._draw_arc_text(
            image,
            english_text,
            (center, center),
            self.radius - self.border - 6,
            304,
            27 if len(english_text) > 40 else 30,
            0.66,
        )
        if len(self.company_name) <= 18:
            company_lines = (self.company_name,)
            company_y_positions = (center - 58,)
            company_font_size = 31
        else:
            split_at = (len(self.company_name) + 1) // 2
            company_lines = (
                self.company_name[:split_at],
                self.company_name[split_at:],
            )
            company_y_positions = (center - 82, center - 45)
            company_font_size = 27
        sequence_offset = 0
        for company_line, company_y in zip(company_lines, company_y_positions):
            self._draw_centered_scaled_text(
                image,
                company_line,
                (center, company_y),
                330,
                company_font_size,
                1,
                semantic_mask=self.company_text_mask,
                character_sequence_offset=sequence_offset,
                character_sequence_count=len(self.company_name),
            )
            sequence_offset += len(company_line)
        service_text = self.auxiliary_text or "业务专用章"
        self._draw_centered_scaled_text(
            image,
            service_text,
            (center, center + 28),
            285,
            54,
            1,
        )
        dot_radius = 8
        for x in (center - 174, center + 174):
            draw.ellipse(
                (x - dot_radius, center - dot_radius, x + dot_radius, center + dot_radius),
                fill=self.fill,
            )
        self.has_star = False
        self._apply_ellipse_to_company_annotations()
        return self._apply_ellipse_shape(image)

    def _draw_fitted_text(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        center: Tuple[int, int],
        max_width: int,
        initial_font_size: int,
        stroke_width: int = 1,
        semantic_draw: Optional[ImageDraw.ImageDraw] = None,
    ) -> None:
        font_size = initial_font_size
        while font_size >= 16:
            font = load_font(self.font_path, font_size)
            bounds = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
            if bounds[2] - bounds[0] <= max_width:
                draw.text(
                    center,
                    text,
                    font=font,
                    fill=self.fill,
                    stroke_width=stroke_width,
                    anchor="mm",
                )
                if semantic_draw is not None:
                    semantic_draw.text(
                        center,
                        text,
                        font=font,
                        fill=255,
                        stroke_width=stroke_width,
                        anchor="mm",
                    )
                return
            font_size -= 2
        raise ValueError(f"文字过长，无法放入印章: {text}")

    def _draw_fitted_company_text(
        self,
        image: Image.Image,
        text: str,
        center: Tuple[int, int],
        max_width: int,
        initial_font_size: int,
        stroke_width: int,
        character_sequence_offset: int,
    ) -> None:
        """绘制横排公司文字，并同步生成字符级细节标注。"""
        probe_draw = ImageDraw.Draw(image)
        font_size = initial_font_size
        while font_size >= 16:
            font = load_font(self.font_path, font_size)
            bounds = probe_draw.textbbox(
                (0, 0),
                text,
                font=font,
                stroke_width=stroke_width,
            )
            if bounds[2] - bounds[0] <= max_width:
                self._draw_centered_scaled_text(
                    image,
                    text,
                    center,
                    max_width,
                    font_size,
                    stroke_width,
                    semantic_mask=self.company_text_mask,
                    character_sequence_offset=character_sequence_offset,
                    character_sequence_count=len(self.company_name),
                )
                return
            font_size -= 2
        raise ValueError(f"公司文字过长，无法放入印章: {text}")

    def _draw_rectangle_stamp(self, image: Image.Image) -> Image.Image:
        draw = ImageDraw.Draw(image)
        width, height = image.size
        outer = (28, 112, width - 28, height - 112)
        draw.rounded_rectangle(
            outer,
            radius=24,
            outline=self.fill,
            width=self.border + 4,
        )
        if self.double_border:
            inner = (45, 129, width - 45, height - 129)
            draw.rounded_rectangle(
                inner,
                radius=18,
                outline=self.fill,
                width=max(4, self.border // 2),
            )

        split_at = (len(self.company_name) + 1) // 2
        lines = (
            [self.company_name]
            if len(self.company_name) <= 10
            else [self.company_name[:split_at], self.company_name[split_at:]]
        )
        if len(lines) == 1:
            self._draw_fitted_company_text(
                image,
                lines[0],
                (width // 2, height // 2 - 10),
                width - 110,
                58,
                self.stroke_width_up,
                0,
            )
        else:
            self._draw_fitted_company_text(
                image,
                lines[0],
                (width // 2, height // 2 - 42),
                width - 110,
                50,
                self.stroke_width_up,
                0,
            )
            self._draw_fitted_company_text(
                image,
                lines[1],
                (width // 2, height // 2 + 25),
                width - 110,
                50,
                self.stroke_width_up,
                len(lines[0]),
            )
        if self.auxiliary_text:
            self._draw_fitted_text(
                draw,
                self.auxiliary_text,
                (width // 2, height // 2 + 88),
                width - 150,
                28,
                0,
            )
        self.has_star = False
        return image

    def _apply_ellipse_shape(self, image: Image.Image) -> Image.Image:
        if self.shape != "ellipse":
            return image
        ellipse_height = max(1, round(image.height * self.ellipse_ratio))
        ellipse_image = image.resize(
            (image.width, ellipse_height),
            resample=Image.Resampling.LANCZOS,
        )
        shaped_image = Image.new("RGBA", image.size, (255, 255, 255, 0))
        shaped_image.paste(
            ellipse_image,
            (0, (image.height - ellipse_height) // 2),
        )
        return shaped_image

    def _apply_ellipse_mask(self, mask: Image.Image) -> Image.Image:
        """对语义 mask 使用与椭圆章完全相同的纵向几何变换。"""
        if self.shape != "ellipse":
            return mask
        ellipse_height = max(1, round(mask.height * self.ellipse_ratio))
        ellipse_mask = mask.resize(
            (mask.width, ellipse_height),
            resample=Image.Resampling.BILINEAR,
        )
        shaped_mask = Image.new("L", mask.size, 0)
        shaped_mask.paste(
            ellipse_mask,
            (0, (mask.height - ellipse_height) // 2),
        )
        return shaped_mask

    def _apply_ellipse_to_company_annotations(self) -> None:
        """对公司文字的全部语义图执行与印章一致的椭圆压缩。"""
        annotation_names = (
            "company_text_mask",
            "company_character_center_mask",
            "company_character_order_map",
            "company_first_character_center_mask",
            "company_last_character_center_mask",
        )
        for name in annotation_names:
            annotation = getattr(self, name)
            if annotation is None:
                raise RuntimeError(f"公司空间标注未初始化: {name}")
            setattr(self, name, self._apply_ellipse_mask(annotation))

    def _choose_rotation(self) -> float:
        if random.random() < self.large_rotation_probability:
            magnitude = random.uniform(
                self.large_rotation_min_degrees,
                self.large_rotation_max_degrees,
            )
            return magnitude if random.random() < 0.5 else -magnitude
        if self.shape == "rectangle":
            return random.uniform(-7, 7)
        if self.layout == "oval_service":
            return (
                random.uniform(-10, 10)
                if random.random() < 0.90
                else random.uniform(-25, 25)
            )
        return (
            random.uniform(-8, 8)
            if random.random() < 0.85
            else random.uniform(-18, 18)
        )

    def _apply_ink_degradation(self, image: Image.Image) -> Image.Image:
        """只退化印泥 alpha，模拟断墨、局部褪色和扫描后的不均匀笔画。

        整图模糊/噪声继续由 train.py 在线增强负责；这里必须保持黑色印刷背景
        相对清晰，只让印泥本身退化，才能贴近真实盖章的前景/背景层级。
        """
        self.ink_degradation_applied = False
        if random.random() >= self.ink_degradation_probability:
            return image
        self.ink_degradation_applied = True

        degraded = image.copy()
        alpha = degraded.getchannel("A")
        width, height = alpha.size

        # 低频透明度起伏：避免整枚章使用完全一致的 RGBA alpha。
        variation_size = (random.randint(5, 9), random.randint(5, 9))
        variation = Image.new("L", variation_size)
        variation.putdata(
            [
                random.randint(198, 255)
                for _ in range(variation_size[0] * variation_size[1])
            ]
        )
        variation = variation.resize(
            (width, height),
            resample=Image.Resampling.BICUBIC,
        )
        alpha = ImageChops.multiply(alpha, variation)

        # 稀疏缺墨块和擦痕；面积受控，不大面积抹掉公司名。
        if random.random() < 0.72:
            damage = Image.new("L", (width, height), 255)
            damage_draw = ImageDraw.Draw(damage)
            for _ in range(random.randint(8, 24)):
                x = random.randint(0, width - 1)
                y = random.randint(0, height - 1)
                radius_x = random.randint(2, 11)
                radius_y = random.randint(1, 6)
                damage_draw.ellipse(
                    (
                        x - radius_x,
                        y - radius_y,
                        x + radius_x,
                        y + radius_y,
                    ),
                    fill=random.randint(45, 185),
                )
            for _ in range(random.randint(0, 3)):
                y = random.randint(0, height - 1)
                damage_draw.line(
                    (
                        random.randint(0, width // 3),
                        y,
                        random.randint(width * 2 // 3, width - 1),
                        y + random.randint(-3, 3),
                    ),
                    fill=random.randint(100, 205),
                    width=random.randint(1, 3),
                )
            alpha = ImageChops.multiply(
                alpha,
                damage.filter(ImageFilter.GaussianBlur(random.uniform(0.1, 0.5))),
            )

        if random.random() < 0.42:
            alpha = alpha.filter(
                ImageFilter.GaussianBlur(random.uniform(0.15, 0.65))
            )
        degraded.putalpha(alpha)
        return degraded

    def _make_background(self, width: int, height: int) -> Image.Image:
        """生成纸面，或从用户背景图随机裁片。"""
        if not self.background_path or random.random() < self.blank_background_probability:
            paper = Image.new(
                "RGB",
                (width, height),
                (
                    random.randint(247, 255),
                    random.randint(246, 255),
                    random.randint(243, 255),
                ),
            )
            # 轻微纸张/扫描底噪，避免纯白矢量背景；强扫描噪声仍在线完成。
            noise = Image.effect_noise(
                (width, height), random.uniform(2.0, 7.0)
            ).convert("RGB")
            return Image.blend(paper, noise, random.uniform(0.012, 0.032))

        with Image.open(self.background_path) as source_background:
            return ImageOps.fit(
                source_background.convert("RGB"),
                (width, height),
                method=Image.Resampling.LANCZOS,
                centering=(
                    random.uniform(0.35, 0.65),
                    random.uniform(0.35, 0.65),
                ),
            )

    def _draw_document_interference(self, background: Image.Image) -> None:
        """
        在真实背景随机裁片上叠加黑字和表格线。

        这不是模糊/噪声增强，而是补足“印章压在密集单证文字上”的内容分布。
        印章随后以半透明印泥贴入，因此黑字会自然透过淡印区域。
        """
        self.document_interference_applied = False
        if random.random() >= self.document_interference_probability:
            return
        self.document_interference_applied = True

        width, height = background.size
        draw = ImageDraw.Draw(background, "RGBA")

        if random.random() < 0.58:
            line_color = (15, 15, 15, random.randint(45, 125))
            for _ in range(random.randint(2, 7)):
                y = random.randint(0, height - 1)
                draw.line(
                    (0, y, width, y),
                    fill=line_color,
                    width=random.randint(1, 2),
                )
            for _ in range(random.randint(1, 4)):
                x = random.randint(0, width - 1)
                draw.line(
                    (x, 0, x, height),
                    fill=line_color,
                    width=random.randint(1, 2),
                )

        density_roll = random.random()
        if density_roll < 0.18:
            text_line_count = random.randint(7, 12)
        elif density_roll < 0.55:
            text_line_count = random.randint(4, 7)
        else:
            text_line_count = random.randint(1, 4)

        for _ in range(text_line_count):
            font_size = random.randint(14, 29)
            font = load_font(self.font_path, font_size)
            snippet = random.choice(DOCUMENT_TEXT_SNIPPETS)
            suffix = random.choice(
                [
                    "",
                    f"：{random.randint(1, 9999)}",
                    (
                        f" 20{random.randint(18, 29)}/"
                        f"{random.randint(1, 12):02d}/"
                        f"{random.randint(1, 28):02d}"
                    ),
                ]
            )
            draw.text(
                (
                    random.randint(-width // 8, max(0, width * 3 // 5)),
                    random.randint(-font_size, height - 1),
                ),
                snippet + suffix,
                font=font,
                fill=(
                    random.randint(0, 45),
                    random.randint(0, 45),
                    random.randint(0, 45),
                    random.randint(105, 225),
                ),
                stroke_width=1 if random.random() < 0.16 else 0,
            )

    def _draw_foreground_occlusion(
        self,
        background: Image.Image,
        stamp_box: Tuple[int, int, int, int],
    ) -> None:
        """
        少量模拟盖章后落在印章上层的签字、日期或打印字。

        与背景干扰分开处理，确保存在类似真实样本中“黑色手写内容遮住部分印泥”
        的前后层级；覆盖带保持窄且稀疏，不把公司名大面积抹掉。
        """
        self.foreground_occlusion_applied = False
        if random.random() >= self.foreground_occlusion_probability:
            return
        self.foreground_occlusion_applied = True

        width, height = background.size
        x0, y0, x1, y1 = stamp_box
        x0 = max(0, min(width - 1, x0))
        y0 = max(0, min(height - 1, y0))
        x1 = max(x0 + 1, min(width, x1))
        y1 = max(y0 + 1, min(height, y1))
        stamp_width = x1 - x0
        stamp_height = y1 - y0
        draw = ImageDraw.Draw(background, "RGBA")

        band_y = random.randint(
            y0 + stamp_height // 5,
            max(y0 + stamp_height // 5, y1 - stamp_height // 5),
        )
        occlusion_type = random.random()
        if occlusion_type < 0.56:
            # 签字/手写横划：用数条略带抖动的折线穿过印章的一部分。
            for _ in range(random.randint(1, 3)):
                start_x = random.randint(
                    max(0, x0 - stamp_width // 8),
                    x0 + stamp_width // 4,
                )
                end_x = random.randint(
                    x0 + stamp_width * 2 // 3,
                    min(width, x1 + stamp_width // 8),
                )
                baseline = band_y + random.randint(-12, 12)
                point_count = random.randint(5, 9)
                points = [
                    (
                        round(start_x + (end_x - start_x) * index / (point_count - 1)),
                        baseline + random.randint(-18, 18),
                    )
                    for index in range(point_count)
                ]
                draw.line(
                    points,
                    fill=(5, 5, 8, random.randint(175, 245)),
                    width=random.randint(2, 5),
                    joint="curve",
                )
        elif occlusion_type < 0.86:
            font_size = random.randint(20, 38)
            font = load_font(self.font_path, font_size)
            foreground_text = random.choice(
                (
                    "经办人",
                    "已审核",
                    "签发日期",
                    f"20{random.randint(18, 29)}.{random.randint(1, 12)}."
                    f"{random.randint(1, 28)}",
                    "are correct",
                )
            )
            draw.text(
                (
                    random.randint(max(0, x0 - stamp_width // 10), x0 + stamp_width // 3),
                    band_y - font_size // 2,
                ),
                foreground_text,
                font=font,
                fill=(0, 0, 0, random.randint(180, 245)),
                stroke_width=1 if random.random() < 0.35 else 0,
            )
        else:
            # 少量表格/下划线在章面上层穿过。
            for offset in range(random.randint(1, 2)):
                y = band_y + offset * random.randint(9, 17)
                draw.line(
                    (max(0, x0 - 18), y, min(width, x1 + 18), y),
                    fill=(10, 10, 10, random.randint(145, 220)),
                    width=random.randint(1, 3),
                )

    def draw_stamp(self) -> None:
        self.image = None
        self.joined_image = None
        self.company_text_mask = None
        self.stamp_mask = None
        self.company_character_center_mask = None
        self.company_character_order_map = None
        self.company_first_character_center_mask = None
        self.company_last_character_center_mask = None
        self.joined_company_text_mask = None
        self.joined_stamp_mask = None
        self.joined_company_character_center_mask = None
        self.joined_company_character_order_map = None
        self.joined_company_first_character_center_mask = None
        self.joined_company_last_character_center_mask = None
        size = 2 * (self.radius + self.edge)
        image = Image.new("RGBA", (size, size), (255, 255, 255, 0))
        self.company_text_mask = Image.new("L", (size, size), 0)
        self.company_character_center_mask = Image.new("L", (size, size), 0)
        self.company_character_order_map = Image.new("L", (size, size), 0)
        self.company_first_character_center_mask = Image.new(
            "L", (size, size), 0
        )
        self.company_last_character_center_mask = Image.new(
            "L", (size, size), 0
        )

        if self.layout == "bilingual_ring":
            image = self._draw_bilingual_ring_stamp(image)
        elif self.layout == "oval_service":
            image = self._draw_oval_service_stamp(image)
        elif self.shape == "rectangle":
            image = self._draw_rectangle_stamp(image)
        else:
            draw = ImageDraw.Draw(image)
            center = self.radius + self.edge
            draw.arc(
                circle(center, center, self.radius),
                start=0,
                end=360,
                fill=self.fill,
                width=self.border,
            )
            if self.double_border:
                draw.arc(
                    circle(center, center, self.radius - self.border - 6),
                    start=0,
                    end=360,
                    fill=self.fill,
                    width=max(4, self.border // 2),
                )

            if not 0 <= self.circle_star_probability <= 1:
                raise ValueError("circle_star_probability 需在 0 到 1 之间")
            star_probability = (
                self.circle_star_probability
                if self.shape == "circle"
                else self.circle_star_probability * 0.35
            )
            if self.layout == "multiline_center":
                star_probability = 0
            self.has_star = random.random() < star_probability
            if self.has_star:
                draw.polygon(
                    pentagram(center, center, self.star_radius),
                    fill=self.fill,
                    outline=self.fill,
                )

            angle_per_character = self.angle_up / len(self.company_name)
            effective_span = angle_per_character * (len(self.company_name) - 1)
            if self.shape == "ellipse":
                company_angles = _equal_ellipse_arc_angles(
                    effective_span,
                    len(self.company_name),
                    self.ellipse_ratio,
                )
            else:
                company_angles = [
                    effective_span / 2 - index * angle_per_character
                    for index in range(len(self.company_name))
                ]
            for character_index, (character, current_angle) in enumerate(
                zip(self.company_name, company_angles)
            ):
                self.draw_rotated_text(
                    image,
                    current_angle,
                    (center, center),
                    self.radius - self.border * 2,
                    character,
                    self.fill,
                    self.font_path,
                    self.font_size_up,
                    self.font_xratio_up,
                    self.stroke_width_up,
                    semantic_mask=self.company_text_mask,
                    character_index=character_index,
                    character_count=len(self.company_name),
                )

            draw = ImageDraw.Draw(image)
            if self.layout == "multiline_center":
                line_positions = {
                    1: (15,),
                    2: (-12, 42),
                    3: (-45, 12, 68),
                }
                positions = line_positions.get(
                    len(self.center_lines),
                    tuple(
                        round(-45 + index * 113 / max(len(self.center_lines) - 1, 1))
                        for index in range(len(self.center_lines))
                    ),
                )
                for center_line, offset in zip(self.center_lines, positions):
                    self._draw_fitted_text(
                        draw,
                        center_line,
                        (center, center + offset),
                        285,
                        42,
                        1,
                    )
            elif self.auxiliary_text:
                auxiliary_y = center + (112 if self.has_star else 20)
                self._draw_fitted_text(
                    draw,
                    self.auxiliary_text,
                    (center, auxiliary_y),
                    250,
                    34,
                    0,
                )
            if self.serial_text:
                self._draw_fitted_text(
                    draw,
                    self.serial_text,
                    (center, center + 175),
                    285,
                    25,
                    0,
                )
            image = self._apply_ellipse_shape(image)
            self._apply_ellipse_to_company_annotations()

        # 语义标注描述理想的印章几何：先于断墨/褪色产生，因此即使某些笔画
        # 在输入图中变淡，辅助监督仍能表达公司文字和完整章形应在的位置。
        self.company_text_mask = self.company_text_mask.point(
            lambda value: 255 if value > 0 else 0
        )
        for name in (
            "company_character_center_mask",
            "company_first_character_center_mask",
            "company_last_character_center_mask",
        ):
            annotation = getattr(self, name)
            setattr(
                self,
                name,
                annotation.point(lambda value: 255 if value > 0 else 0),
            )
        self.stamp_mask = ImageChops.lighter(
            image.getchannel("A").point(
                lambda value: 255 if value > 0 else 0
            ),
            self.company_text_mask,
        )
        image = self._apply_ink_degradation(image)
        angle = self._choose_rotation()
        self.actual_rotation_degrees = float(angle)
        self.image = image.rotate(
            angle,
            resample=Image.Resampling.BICUBIC,
            # 椭圆章旋转后包围盒可能大于原始正方形。先完整旋转，再在 join_stamp
            # 中按 alpha 包围盒裁透明边，禁止在这里静默裁掉公司名。
            expand=True,
            fillcolor=(255, 255, 255, 0),
        )
        self.company_text_mask = self.company_text_mask.rotate(
            angle,
            resample=Image.Resampling.BICUBIC,
            expand=True,
            fillcolor=0,
        )
        self.stamp_mask = self.stamp_mask.rotate(
            angle,
            resample=Image.Resampling.BICUBIC,
            expand=True,
            fillcolor=0,
        )
        for name in (
            "company_character_center_mask",
            "company_character_order_map",
            "company_first_character_center_mask",
            "company_last_character_center_mask",
        ):
            annotation = getattr(self, name)
            setattr(
                self,
                name,
                annotation.rotate(
                    angle,
                    resample=Image.Resampling.BICUBIC,
                    expand=True,
                    fillcolor=0,
                ),
            )
        if not (
            self.image.size
            == self.company_text_mask.size
            == self.stamp_mask.size
            == self.company_character_center_mask.size
            == self.company_character_order_map.size
            == self.company_first_character_center_mask.size
            == self.company_last_character_center_mask.size
        ):
            raise AssertionError("印章图像与空间标注旋转后尺寸不一致")

    def join_stamp(self) -> None:
        if (
            self.image is None
            or self.company_text_mask is None
            or self.stamp_mask is None
            or self.company_character_center_mask is None
            or self.company_character_order_map is None
            or self.company_first_character_center_mask is None
            or self.company_last_character_center_mask is None
        ):
            raise RuntimeError("请先绘制印章")

        canvas_width = random.randint(330, 460)
        canvas_height = random.randint(330, 460)
        background = self._make_background(canvas_width, canvas_height)
        self._draw_document_interference(background)

        # 使用退化前的完整章形包围盒，避免断墨恰好抹掉边缘后，图像和语义
        # 标注以不同范围裁切。
        alpha_box = self.stamp_mask.getbbox()
        foreground = self.image.crop(alpha_box) if alpha_box else self.image
        company_text_foreground = (
            self.company_text_mask.crop(alpha_box)
            if alpha_box
            else self.company_text_mask
        )
        stamp_foreground = (
            self.stamp_mask.crop(alpha_box) if alpha_box else self.stamp_mask
        )
        detail_foregrounds = {
            name: (
                getattr(self, name).crop(alpha_box)
                if alpha_box
                else getattr(self, name)
            )
            for name in (
                "company_character_center_mask",
                "company_character_order_map",
                "company_first_character_center_mask",
                "company_last_character_center_mask",
            )
        }
        min_size_ratio, max_size_ratio = self.stamp_size_ratio_range
        target_size = int(
            min(canvas_width, canvas_height)
            * random.uniform(min_size_ratio, max_size_ratio)
        )
        scale = target_size / max(foreground.size)
        resized_size = (
            max(1, round(foreground.width * scale)),
            max(1, round(foreground.height * scale)),
        )
        foreground = foreground.resize(
            resized_size,
            resample=Image.Resampling.LANCZOS,
        )
        company_text_foreground = company_text_foreground.resize(
            resized_size,
            resample=Image.Resampling.BILINEAR,
        )
        stamp_foreground = stamp_foreground.resize(
            resized_size,
            resample=Image.Resampling.BILINEAR,
        )
        detail_foregrounds = {
            name: annotation.resize(
                resized_size,
                resample=Image.Resampling.BILINEAR,
            )
            for name, annotation in detail_foregrounds.items()
        }

        center_x = canvas_width // 2 + int(
            random.uniform(-0.10, 0.10) * canvas_width
        )
        center_y = canvas_height // 2 + int(
            random.uniform(-0.10, 0.10) * canvas_height
        )
        if random.random() < 0.06:
            center_x += int(random.choice([-1, 1]) * 0.12 * canvas_width)
        desired_x = center_x - foreground.width // 2
        desired_y = center_y - foreground.height // 2

        # 默认分支允许章框紧贴裁片边缘，但整枚章必须在画布内，避免把公司名
        # 误裁掉后仍保留完整标签。只有 profile 明确抽中的部分章才允许越界。
        x = min(max(0, desired_x), canvas_width - foreground.width)
        y = min(max(0, desired_y), canvas_height - foreground.height)
        self.was_partially_cropped = (
            random.random() < self.partial_crop_probability
        )
        self.actual_partial_crop_ratio = 0.0
        if self.was_partially_cropped:
            crop_ratio = random.uniform(
                min(0.015, self.max_partial_crop_ratio),
                self.max_partial_crop_ratio,
            )
            overflow = max(1, round(min(foreground.size) * crop_ratio))
            edge = random.choice(("left", "right", "top", "bottom"))
            if edge == "left":
                x = -overflow
            elif edge == "right":
                x = canvas_width - foreground.width + overflow
            elif edge == "top":
                y = -overflow
            else:
                y = canvas_height - foreground.height + overflow
            self.actual_partial_crop_ratio = overflow / min(foreground.size)
        elif not (
            0 <= x
            and 0 <= y
            and x + foreground.width <= canvas_width
            and y + foreground.height <= canvas_height
        ):
            raise AssertionError(
                "普通样本章面发生非预期越界；拒绝写出图片与完整标签"
            )

        background.paste(foreground, (x, y), foreground)
        joined_company_text_mask = Image.new(
            "L", (canvas_width, canvas_height), 0
        )
        joined_stamp_mask = Image.new("L", (canvas_width, canvas_height), 0)
        joined_company_text_mask.paste(company_text_foreground, (x, y))
        joined_stamp_mask.paste(stamp_foreground, (x, y))
        self.joined_company_text_mask = joined_company_text_mask.point(
            lambda value: 255 if value >= 16 else 0
        )
        self.joined_stamp_mask = ImageChops.lighter(
            joined_stamp_mask.point(
                lambda value: 255 if value >= 16 else 0
            ),
            self.joined_company_text_mask,
        )
        detail_destinations = {
            "company_character_center_mask": (
                "joined_company_character_center_mask"
            ),
            "company_character_order_map": "joined_company_character_order_map",
            "company_first_character_center_mask": (
                "joined_company_first_character_center_mask"
            ),
            "company_last_character_center_mask": (
                "joined_company_last_character_center_mask"
            ),
        }
        for source_name, destination_name in detail_destinations.items():
            canvas = Image.new("L", (canvas_width, canvas_height), 0)
            canvas.paste(detail_foregrounds[source_name], (x, y))
            if source_name != "company_character_order_map":
                canvas = canvas.point(
                    lambda value: 255 if value >= 16 else 0
                )
            setattr(self, destination_name, canvas)
        self._draw_foreground_occlusion(
            background,
            (x, y, x + foreground.width, y + foreground.height),
        )
        self.joined_image = background

    @staticmethod
    def _make_normalized_heatmap(
        binary_mask: Image.Image,
        radius_ratio: float,
    ) -> Image.Image:
        """把二值笔画向外扩散成归一化高斯热图。"""
        if binary_mask.mode != "L":
            raise ValueError("空间标注 mask 必须为 L 模式")
        if not 0 < radius_ratio <= 0.25:
            raise ValueError("heatmap radius ratio 需满足 0 < ratio <= 0.25")
        if binary_mask.getbbox() is None:
            return Image.new("L", binary_mask.size, 0)
        radius = max(1.0, min(binary_mask.size) * float(radius_ratio))
        blurred = binary_mask.filter(ImageFilter.GaussianBlur(radius))
        peak = blurred.getextrema()[1]
        if peak <= 0:
            return Image.new("L", binary_mask.size, 0)
        lookup = [min(255, round(value * 255 / peak)) for value in range(256)]
        return blurred.point(lookup)

    def build_spatial_annotation(self) -> Image.Image:
        """打包与 OCR 图片同尺寸的主空间监督 PNG。"""
        if (
            self.joined_image is None
            or self.joined_company_text_mask is None
            or self.joined_stamp_mask is None
        ):
            raise RuntimeError("请先完成印章与背景合成")
        if not (
            self.joined_image.size
            == self.joined_company_text_mask.size
            == self.joined_stamp_mask.size
        ):
            raise AssertionError("OCR 图片与空间标注尺寸不一致")
        text_heatmap = self._make_normalized_heatmap(
            self.joined_company_text_mask,
            self.text_heatmap_radius_ratio,
        )
        stamp_heatmap = self._make_normalized_heatmap(
            self.joined_stamp_mask,
            self.stamp_heatmap_radius_ratio,
        )
        return Image.merge(
            "RGBA",
            (
                self.joined_company_text_mask,
                self.joined_stamp_mask,
                text_heatmap,
                stamp_heatmap,
            ),
        )

    def build_spatial_detail_annotation(self) -> Image.Image:
        """打包字符中心、阅读进度和首尾位置四个连续通道。"""
        detail_maps = (
            self.joined_company_character_center_mask,
            self.joined_company_character_order_map,
            self.joined_company_first_character_center_mask,
            self.joined_company_last_character_center_mask,
        )
        if self.joined_image is None or any(
            detail_map is None for detail_map in detail_maps
        ):
            raise RuntimeError("请先完成印章与字符细节标注合成")
        if any(
            detail_map.size != self.joined_image.size
            for detail_map in detail_maps
        ):
            raise AssertionError("OCR 图片与字符细节标注尺寸不一致")
        center_heatmap = self._make_normalized_heatmap(
            self.joined_company_character_center_mask,
            self.character_heatmap_radius_ratio,
        )
        first_heatmap = self._make_normalized_heatmap(
            self.joined_company_first_character_center_mask,
            self.character_heatmap_radius_ratio,
        )
        last_heatmap = self._make_normalized_heatmap(
            self.joined_company_last_character_center_mask,
            self.character_heatmap_radius_ratio,
        )
        return Image.merge(
            "RGBA",
            (
                center_heatmap,
                self.joined_company_character_order_map,
                first_heatmap,
                last_heatmap,
            ),
        )

    def save_join_stamp(self) -> None:
        if self.joined_image is None:
            raise RuntimeError("合成印章为空，无法保存")
        self.save_path.mkdir(parents=True, exist_ok=True)
        image_path = self.save_path / f"{self.save_name}.jpg"
        temporary_path = self.save_path / f".{self.save_name}.{os.getpid()}.tmp"
        self.joined_image.save(
            temporary_path,
            format="JPEG",
            quality=random.randint(82, 96),
        )
        os.replace(temporary_path, image_path)

    def save_spatial_annotation(self) -> None:
        if self.spatial_annotation_path is None:
            raise RuntimeError("未配置空间标注输出目录")
        annotation = self.build_spatial_annotation()
        detail_annotation = self.build_spatial_detail_annotation()
        self.spatial_annotation_path.mkdir(parents=True, exist_ok=True)
        annotation_path = (
            self.spatial_annotation_path / f"{self.save_name}.png"
        )
        detail_path = (
            self.spatial_annotation_path / f"{self.save_name}.detail.png"
        )
        temporary_path = self.spatial_annotation_path / (
            f".{self.save_name}.{os.getpid()}.png.tmp"
        )
        detail_temporary_path = self.spatial_annotation_path / (
            f".{self.save_name}.{os.getpid()}.detail.png.tmp"
        )
        # 数十万样本时 PNG optimize 的 CPU 代价很高；中等压缩即可保证无损，
        # 同时避免空间标注拖慢主图生成。
        annotation.save(temporary_path, format="PNG", compress_level=4)
        detail_annotation.save(
            detail_temporary_path,
            format="PNG",
            compress_level=4,
        )
        # 主 PNG 最后替换，作为这组双 PNG 完整写出的完成标记。
        os.replace(detail_temporary_path, detail_path)
        os.replace(temporary_path, annotation_path)


def _write_label_atomically(output_path: Path, name: str, label: str) -> None:
    label_path = output_path / f"{name}.txt"
    temporary_path = output_path / f".{name}.{os.getpid()}.txt.tmp"
    temporary_path.write_text(f"{label}\n", encoding="utf-8")
    os.replace(temporary_path, label_path)


def _write_sample_metadata_atomically(
    output_path: Path,
    name: str,
    metadata: Dict[str, object],
) -> None:
    """逐样本写 JSON，避免多生成进程争用一个共享 JSONL 文件。"""
    metadata_path = output_path / f"{name}.meta.json"
    temporary_path = output_path / f".{name}.{os.getpid()}.meta.json.tmp"
    temporary_path.write_text(
        json.dumps(metadata, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    os.replace(temporary_path, metadata_path)


def _font_xratio(name_length: int) -> float:
    return DEFAULT_FONT_XRATIO_UP_DICT.get(
        str(name_length),
        max(0.40, 0.70 - name_length * 0.012),
    )


def generate_worker(pid: int, queue: Queue, config: Dict[str, object]) -> None:
    name_generator = CompanyNameGenerator(
        corpus_path=config["corpus_path"],
        allowed_characters=config["allowed_characters"],
        excluded_names=config["excluded_names"],
        name_length=config["name_length"],
        corpus_probability=config["corpus_probability"],
        excluded_substrings=config["excluded_substrings"],
    )
    stamp = Stamp()
    output_path = Path(config["output_path"])
    spatial_output_path = (
        Path(config["spatial_annotation_output"])
        if config["spatial_annotations"]
        else None
    )
    failure_count = 0

    while True:
        index = queue.get()
        if index is None:
            break

        sample_name = str(index)
        shard_size = int(config["shard_size"])
        stamp.save_path = (
            output_path / f"{index // shard_size:05d}"
            if shard_size > 0
            else output_path
        )
        stamp.spatial_annotation_path = (
            (
                spatial_output_path / f"{index // shard_size:05d}"
                if shard_size > 0
                else spatial_output_path
            )
            if spatial_output_path is not None
            else None
        )
        image_path = stamp.save_path / f"{sample_name}.jpg"
        label_path = stamp.save_path / f"{sample_name}.txt"
        annotation_path = (
            stamp.spatial_annotation_path / f"{sample_name}.png"
            if stamp.spatial_annotation_path is not None
            else None
        )
        detail_annotation_path = (
            stamp.spatial_annotation_path / f"{sample_name}.detail.png"
            if stamp.spatial_annotation_path is not None
            else None
        )
        metadata_path = stamp.save_path / f"{sample_name}.meta.json"
        spatial_annotation_complete = (
            annotation_path is None
            or (
                annotation_path.exists()
                and annotation_path.stat().st_size > 0
                and detail_annotation_path.exists()
                and detail_annotation_path.stat().st_size > 0
            )
        )
        if (
            not config["overwrite"]
            and image_path.exists()
            and label_path.exists()
            and image_path.stat().st_size > 0
            and normalize_company_name(label_path.read_text(encoding="utf-8"))
            and metadata_path.exists()
            and metadata_path.stat().st_size > 0
            and spatial_annotation_complete
        ):
            continue

        random.seed(int(config["seed"]) + int(index))
        try:
            relative_index = index - int(config["start_index"])
            base_schedule_count = int(config["base_schedule_count"])
            base_name_indices = config["base_name_indices"]
            boost_name_indices = config["boost_name_indices"]
            if base_schedule_count > 0 and relative_index < base_schedule_count:
                if base_name_indices:
                    company_name = name_generator.corpus[
                        base_name_indices[relative_index]
                    ]
                else:
                    company_name = name_generator.corpus[
                        relative_index % len(name_generator.corpus)
                    ]
            elif base_schedule_count > 0:
                boost_index = relative_index - base_schedule_count
                company_name = name_generator.corpus[
                    boost_name_indices[boost_index]
                ]
            else:
                company_name = name_generator.get_company_name(
                    sample_index=index
                )
            color_name, fill, shape, ellipse_ratio = get_random_stamp_style(
                config["color_ratios"],
                config["shape_ratios"],
                config["ellipse_ratio_range"],
            )
            layout = get_random_stamp_layout(config["layout_ratios"])
            if layout == "bilingual_ring":
                shape = choose_by_ratio(
                    config["bilingual_shape_ratios"],
                    "bilingual_shape_ratios",
                )
                ellipse_ratio = (
                    random.uniform(0.72, 0.88) if shape == "ellipse" else 1.0
                )
            elif layout == "oval_service":
                color_name = choose_by_ratio(
                    {"red": 25, "blue": 55, "purple": 8, "black": 12},
                    "oval_service_color_ratios",
                )
                fill = random_ink_color(color_name)
                shape = "ellipse"
                ellipse_ratio = random.uniform(0.64, 0.78)
            elif layout == "multiline_center":
                shape = "circle"
                ellipse_ratio = 1.0

            stamp.fill = fill
            stamp.shape = shape
            stamp.layout = layout
            stamp.ellipse_ratio = ellipse_ratio
            stamp.circle_star_probability = config["circle_star_probability"]
            stamp.edge = random.randint(4, 7)
            stamp.radius = random.randint(242, 256)
            stamp.star_radius = random.randint(78, 98)
            stamp.border = random.randint(10, 16)
            stamp.company_name = company_name
            is_long_label = len(company_name) >= 17
            if is_long_label and shape == "ellipse" and layout == "standard":
                stamp.angle_up = random.randint(300, 316)
            elif is_long_label:
                stamp.angle_up = random.randint(282, 310)
            else:
                stamp.angle_up = random.randint(250, 286)
            if is_long_label and layout == "standard":
                stamp.font_size_up = (
                    random.randint(52, 60)
                    if shape == "ellipse"
                    else random.randint(62, 70)
                )
            else:
                stamp.font_size_up = random.randint(74, 86)
            stamp.font_xratio_up = _font_xratio(len(company_name)) * (
                0.82
                if is_long_label and shape == "ellipse" and layout == "standard"
                else 1.0
            )
            stamp.stroke_width_up = random.randint(1, 3)
            compatible_font_paths = get_compatible_font_paths(
                company_name,
                config["font_character_coverage"],
            )
            if not compatible_font_paths:
                raise ValueError(
                    f"没有一款字体能完整绘制公司名: {company_name}"
                )
            stamp.font_path = random.choice(compatible_font_paths)
            stamp.background_path = (
                random.choice(config["background_paths"])
                if config["background_paths"]
                else ""
            )
            stamp.document_interference_probability = config[
                "document_interference_probability"
            ]
            stamp.blank_background_probability = config[
                "blank_background_probability"
            ]
            # 完整公司名监督要求所有字符在图中可见。长弧文字一旦裁边或遮挡，
            # 通常会直接丢失字符，因此长名称只在主池保留轻度成像退化，不再制造
            # “图中缺字、标签完整”的矛盾样本。
            stamp.partial_crop_probability = (
                0.0
                if is_long_label
                else config["partial_crop_probability"]
            )
            stamp.max_partial_crop_ratio = config["max_partial_crop_ratio"]
            stamp.large_rotation_probability = config[
                "large_rotation_probability"
            ]
            stamp.large_rotation_min_degrees = config[
                "large_rotation_min_degrees"
            ]
            stamp.large_rotation_max_degrees = config[
                "large_rotation_max_degrees"
            ]
            stamp.ink_degradation_probability = config[
                "ink_degradation_probability"
            ] * (0.55 if is_long_label else 1.0)
            configured_size_range = tuple(config["stamp_size_ratio_range"])
            stamp.stamp_size_ratio_range = (
                max(0.90, configured_size_range[0]),
                configured_size_range[1],
            ) if is_long_label else configured_size_range
            stamp.text_heatmap_radius_ratio = config[
                "text_heatmap_radius_ratio"
            ]
            stamp.stamp_heatmap_radius_ratio = config[
                "stamp_heatmap_radius_ratio"
            ]
            stamp.character_heatmap_radius_ratio = config[
                "character_heatmap_radius_ratio"
            ]
            foreground_multiplier = {
                "standard": 1.0,
                "bilingual_ring": 1.1,
                "oval_service": 0.7,
                "multiline_center": 1.6,
            }[layout]
            stamp.foreground_occlusion_probability = min(
                1.0,
                config["foreground_occlusion_probability"]
                * foreground_multiplier
                * (0.45 if is_long_label else 1.0),
            )
            stamp.save_name = sample_name
            stamp.english_company_text = ""
            stamp.center_lines = ()
            if layout == "bilingual_ring":
                stamp.double_border = random.random() < 0.25
                stamp.english_company_text = get_random_english_company_text()
                stamp.auxiliary_text = (
                    random.choice(AUXILIARY_TEXTS)
                    if random.random() < 0.28
                    else ""
                )
                stamp.serial_text = ""
            elif layout == "oval_service":
                stamp.double_border = True
                stamp.english_company_text = get_random_english_company_text()
                stamp.auxiliary_text = random.choice(OVAL_SERVICE_TEXTS)
                stamp.serial_text = ""
            elif layout == "multiline_center":
                stamp.double_border = random.random() < 0.10
                stamp.auxiliary_text = ""
                stamp.center_lines = random.choice(MULTILINE_CENTER_TEXTS)
                stamp.serial_text = (
                    f"{random.randint(10, 99)}"
                    f"{random.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}"
                    f"{random.randint(10, 99)}"
                )
            else:
                stamp.double_border = random.random() < 0.14
                stamp.auxiliary_text = (
                    random.choice(AUXILIARY_TEXTS)
                    if random.random() < 0.58
                    else ""
                )
                stamp.serial_text = (
                    ""
                    if is_long_label
                    else (
                        "".join(str(random.randint(0, 9)) for _ in range(13))
                        if random.random() < 0.46 and shape != "rectangle"
                        else ""
                    )
                )

            stamp.draw_stamp()
            stamp.join_stamp()
            stamp.save_join_stamp()
            if config["spatial_annotations"]:
                stamp.save_spatial_annotation()
            _write_sample_metadata_atomically(
                stamp.save_path,
                sample_name,
                {
                    "source": "synthetic",
                    "color": color_name,
                    "shape": shape,
                    "layout": layout,
                    "label_length": len(company_name),
                    "ink_degradation": stamp.ink_degradation_applied,
                    "document_interference": (
                        stamp.document_interference_applied
                    ),
                    "foreground_occlusion": (
                        stamp.foreground_occlusion_applied
                    ),
                    "partially_cropped": stamp.was_partially_cropped,
                    "partial_crop_ratio": round(
                        stamp.actual_partial_crop_ratio,
                        6,
                    ),
                    "rotation_degrees": round(
                        stamp.actual_rotation_degrees,
                        4,
                    ),
                    "rotation_bucket": (
                        "large"
                        if abs(stamp.actual_rotation_degrees)
                        >= stamp.large_rotation_min_degrees
                        else "upright"
                    ),
                },
            )
            _write_label_atomically(stamp.save_path, sample_name, company_name)

            if index % 1000 == 0:
                print(
                    f"进程 {pid}: 已生成 {index}，"
                    f"{color_name}/{shape}/{layout}/星章={stamp.has_star}/"
                    f"部分章={stamp.was_partially_cropped}",
                    flush=True,
                )
        except Exception as exc:
            failure_count += 1
            print(
                f"进程 {pid}: 生成 {index} 失败: {type(exc).__name__}: {exc}",
                flush=True,
            )
    if failure_count:
        raise RuntimeError(f"进程 {pid} 有 {failure_count} 个样本生成失败")


def parse_ratios(value: str) -> Dict[str, float]:
    ratios = {}
    for item in value.split(","):
        if "=" not in item:
            raise argparse.ArgumentTypeError(
                "比例格式应为 name=weight,name=weight"
            )
        name, raw_weight = item.split("=", 1)
        try:
            ratios[name.strip()] = float(raw_weight)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"无效权重: {item}") from exc
    return ratios


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成印章 OCR 合成预训练数据")
    parser.add_argument(
        "--company-list",
        dest="corpus_path",
        default=str(DEFAULT_CORPUS_PATH),
        help="每行一个完整公司名",
    )
    parser.add_argument(
        "--vocab",
        dest="vocab_path",
        default=str(DEFAULT_VOCAB_PATH),
        help="初始化模型的 vocab.json；传空字符串关闭词表校验",
    )
    parser.add_argument(
        "--backgrounds",
        dest="background_path",
        default=str(DEFAULT_BG_PATH),
        help="可选背景图目录；空目录时只生成纸面背景",
    )
    parser.add_argument(
        "--fonts",
        dest="font_path",
        default=str(DEFAULT_FONT_PATH),
        help="字体目录，字体必须完整覆盖每个公司名",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_MAKE_PATH),
        help="合成图片输出目录；空间标注自动写入同级 *_spatial",
    )
    parser.add_argument(
        "--samples-per-name",
        dest="samples_per_name",
        type=int,
        default=14,
        help="每个公司名的基础样本数，默认 14",
    )
    parser.add_argument(
        "--workers",
        dest="processes",
        type=int,
        default=DEFAULT_PROCESS_NUM,
    )
    parser.add_argument(
        "--color-ratios",
        dest="color_ratios",
        type=parse_ratios,
        default=None,
        help="red=89,blue=7,purple=1,black=3",
    )
    parser.add_argument(
        "--shape-ratios",
        dest="shape_ratios",
        type=parse_ratios,
        default=None,
        help="circle=90,ellipse=7,rectangle=3",
    )
    parser.add_argument(
        "--layout-ratios",
        dest="layout_ratios",
        type=parse_ratios,
        default=None,
        help="standard=90,bilingual_ring=6,oval_service=2,multiline_center=2",
    )
    parser.add_argument(
        "--blank-background-ratio",
        dest="blank_background_probability",
        type=float,
        default=None,
        help="纯纸面背景占比，默认 0.58",
    )
    parser.add_argument(
        "--rotation-probability",
        dest="large_rotation_probability",
        type=float,
        default=None,
        help="大角度旋转占比，默认 0.25",
    )
    parser.add_argument(
        "--rotation-min",
        dest="large_rotation_min_degrees",
        type=float,
        default=LARGE_ROTATION_MIN_DEGREES,
        help="大角度旋转绝对值下限，默认 45 度",
    )
    parser.add_argument(
        "--rotation-max",
        dest="large_rotation_max_degrees",
        type=float,
        default=LARGE_ROTATION_MAX_DEGREES,
        help="大角度旋转绝对值上限，默认 180 度",
    )
    parser.add_argument(
        "--exclude-labels",
        dest="exclude_label_path",
        action="append",
        default=[],
        help="排除真实验证/测试标签文件，避免合成数据泄漏；可重复",
    )
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="只校验公司清单、词表、字体和预计数据量",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.set_defaults(
        num=DEFAULT_SEAL_NUM,
        min_character_occurrences=24,
        max_rare_boost_per_name=12,
        max_rare_boost_per_high_risk_name=20,
        high_risk_max_samples_per_name=4,
        high_risk_min_length=17,
        high_risk_prefix=None,
        high_risk_suffix=None,
        start_index=0,
        spatial_annotations=True,
        spatial_annotation_output=None,
        text_heatmap_radius_ratio=DEFAULT_TEXT_HEATMAP_RADIUS_RATIO,
        stamp_heatmap_radius_ratio=DEFAULT_STAMP_HEATMAP_RADIUS_RATIO,
        character_heatmap_radius_ratio=DEFAULT_CHARACTER_HEATMAP_RADIUS_RATIO,
        shard_size=10000,
        ellipse_ratio_min=DEFAULT_ELLIPSE_RATIO_RANGE[0],
        ellipse_ratio_max=DEFAULT_ELLIPSE_RATIO_RANGE[1],
        circle_star_probability=DEFAULT_CIRCLE_STAR_PROBABILITY,
        document_interference_probability=None,
        foreground_occlusion_probability=None,
        partial_crop_probability=None,
        max_partial_crop_ratio=None,
        ink_degradation_probability=None,
        stamp_size_ratio_min=None,
        stamp_size_ratio_max=None,
        min_label_length=DEFAULT_UP_NUM[0],
        max_label_length=DEFAULT_UP_NUM[1],
        corpus_probability=1.0,
        strict_corpus=True,
        exclude_label_substring=[],
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.num <= 0:
        raise ValueError("num 必须大于 0")
    if args.processes <= 0:
        raise ValueError("processes 必须大于 0")
    if args.start_index < 0:
        raise ValueError("start_index 不能小于 0")
    if args.shard_size < 0:
        raise ValueError("shard_size 不能小于 0")
    if args.min_character_occurrences < 0:
        raise ValueError("min_character_occurrences 不能小于 0")
    if args.max_rare_boost_per_name <= 0:
        raise ValueError("max_rare_boost_per_name 必须大于 0")
    if args.max_rare_boost_per_high_risk_name <= 0:
        raise ValueError("max_rare_boost_per_high_risk_name 必须大于 0")
    if args.high_risk_max_samples_per_name < 0:
        raise ValueError("high_risk_max_samples_per_name 不能小于 0")
    if args.high_risk_min_length <= 0:
        raise ValueError("high_risk_min_length 必须大于 0")
    if args.min_label_length < 2 or args.min_label_length > args.max_label_length:
        raise ValueError("标签长度范围无效")
    if not 0 <= args.corpus_probability <= 1:
        raise ValueError("corpus_probability 需在 0 到 1 之间")
    if not args.spatial_annotations and args.spatial_annotation_output:
        raise ValueError(
            "指定 spatial_annotation_output 时不能同时关闭空间标注"
        )
    for name, value in (
        ("text_heatmap_radius_ratio", args.text_heatmap_radius_ratio),
        ("stamp_heatmap_radius_ratio", args.stamp_heatmap_radius_ratio),
        (
            "character_heatmap_radius_ratio",
            args.character_heatmap_radius_ratio,
        ),
    ):
        if not 0 < value <= 0.25:
            raise ValueError(f"{name} 需满足 0 < ratio <= 0.25")

    profile = DEFAULT_STYLE
    color_ratios = args.color_ratios or profile["color_ratios"]
    shape_ratios = args.shape_ratios or profile["shape_ratios"]
    layout_ratios = args.layout_ratios or profile["layout_ratios"]
    bilingual_shape_ratios = profile.get(
        "bilingual_shape_ratios",
        {"circle": 68, "ellipse": 32},
    )
    document_interference_probability = (
        args.document_interference_probability
        if args.document_interference_probability is not None
        else profile["document_interference_probability"]
    )
    if not 0 <= document_interference_probability <= 1:
        raise ValueError("document_interference_probability 需在 0 到 1 之间")
    foreground_occlusion_probability = (
        args.foreground_occlusion_probability
        if args.foreground_occlusion_probability is not None
        else profile["foreground_occlusion_probability"]
    )
    blank_background_probability = (
        args.blank_background_probability
        if args.blank_background_probability is not None
        else profile["blank_background_probability"]
    )
    partial_crop_probability = (
        args.partial_crop_probability
        if args.partial_crop_probability is not None
        else profile["partial_crop_probability"]
    )
    max_partial_crop_ratio = (
        args.max_partial_crop_ratio
        if args.max_partial_crop_ratio is not None
        else profile["max_partial_crop_ratio"]
    )
    large_rotation_probability = (
        args.large_rotation_probability
        if args.large_rotation_probability is not None
        else profile["large_rotation_probability"]
    )
    ink_degradation_probability = (
        args.ink_degradation_probability
        if args.ink_degradation_probability is not None
        else profile["ink_degradation_probability"]
    )
    profile_stamp_size_range = profile["stamp_size_ratio_range"]
    stamp_size_ratio_range = (
        args.stamp_size_ratio_min
        if args.stamp_size_ratio_min is not None
        else profile_stamp_size_range[0],
        args.stamp_size_ratio_max
        if args.stamp_size_ratio_max is not None
        else profile_stamp_size_range[1],
    )
    probability_values = {
        "foreground_occlusion_probability": foreground_occlusion_probability,
        "blank_background_probability": blank_background_probability,
        "partial_crop_probability": partial_crop_probability,
        "large_rotation_probability": large_rotation_probability,
        "ink_degradation_probability": ink_degradation_probability,
    }
    for name, value in probability_values.items():
        if not 0 <= value <= 1:
            raise ValueError(f"{name} 需在 0 到 1 之间")
    if not (
        0 < args.large_rotation_min_degrees
        <= args.large_rotation_max_degrees
        <= 180
    ):
        raise ValueError("rotation-min/max 需满足 0 < min <= max <= 180")
    if not 0 <= max_partial_crop_ratio <= 0.25:
        raise ValueError("max_partial_crop_ratio 需在 0 到 0.25 之间")
    if partial_crop_probability > 0 and max_partial_crop_ratio <= 0:
        raise ValueError("启用部分章时 max_partial_crop_ratio 必须大于 0")
    if not (
        0 < stamp_size_ratio_range[0]
        <= stamp_size_ratio_range[1]
        <= 1
    ):
        raise ValueError(
            "stamp_size_ratio_min/max 需满足 0 < min <= max <= 1"
        )
    ellipse_ratio_range = (
        args.ellipse_ratio_min,
        args.ellipse_ratio_max,
    )
    # 启动进程前统一校验比例。
    random.seed(args.seed)
    get_random_stamp_style(
        color_ratios,
        shape_ratios,
        ellipse_ratio_range,
    )
    get_random_stamp_layout(layout_ratios)
    choose_by_ratio(bilingual_shape_ratios, "bilingual_shape_ratios")

    output_path = Path(args.output).expanduser().resolve()
    spatial_annotation_output = None
    if args.spatial_annotations:
        spatial_annotation_output = (
            Path(args.spatial_annotation_output).expanduser().resolve()
            if args.spatial_annotation_output
            else output_path.with_name(f"{output_path.name}_spatial")
        )
        try:
            spatial_annotation_output.relative_to(output_path)
        except ValueError:
            pass
        else:
            raise ValueError(
                "spatial_annotation_output 必须位于 OCR 图片输出目录之外，"
                "避免数据发现器把 mask 当成训练图片"
            )
    allowed_characters = load_vocabulary(args.vocab_path or None)
    excluded_names = load_excluded_names(args.exclude_label_path)
    corpus_checker = CompanyNameGenerator(
        corpus_path=args.corpus_path,
        allowed_characters=allowed_characters,
        excluded_names=excluded_names,
        name_length=(args.min_label_length, args.max_label_length),
        corpus_probability=args.corpus_probability,
        excluded_substrings=args.exclude_label_substring,
    )
    corpus_name_count = len(corpus_checker.corpus)
    high_risk_prefixes = tuple(
        prefix for prefix in (
            args.high_risk_prefix or DEFAULT_LONG_HIGH_RISK_PREFIXES
        ) if prefix
    )
    high_risk_suffixes = tuple(
        suffix for suffix in (
            args.high_risk_suffix or DEFAULT_LONG_HIGH_RISK_SUFFIXES
        ) if suffix
    )
    if args.high_risk_max_samples_per_name > 0 and (
        not high_risk_prefixes or not high_risk_suffixes
    ):
        raise ValueError(
            "启用长名称高风险限额时，至少需要一个非空前缀和后缀片段"
        )
    if corpus_checker.corpus_rejected_count:
        print(
            f"警告：公司名词典有 {corpus_checker.corpus_rejected_count}/"
            f"{corpus_checker.corpus_total_count} 条记录被过滤，原因: "
            f"{dict(corpus_checker.corpus_rejection_reasons)}。",
            flush=True,
        )
        if corpus_checker.corpus_missing_characters:
            print(
                "词表缺失字符（字符: 出现次数）: "
                f"{dict(corpus_checker.corpus_missing_characters.most_common(80))}",
                flush=True,
            )
        intentional_rejection_reasons = {"excluded", "excluded_substring"}
        blocking_rejection_reasons = {
            reason: count
            for reason, count in corpus_checker.corpus_rejection_reasons.items()
            if reason not in intentional_rejection_reasons
        }
        if args.strict_corpus and blocking_rejection_reasons:
            raise ValueError(
                "strict_corpus 已启用，请先修复语料或更新词表。"
                f"阻断原因: {blocking_rejection_reasons}；示例: "
                f"{dict(corpus_checker.corpus_rejected_examples)}"
            )
    if corpus_checker.corpus_duplicate_count:
        print(
            f"公司名词典规范化去重: 跳过 "
            f"{corpus_checker.corpus_duplicate_count} 条重复记录。",
            flush=True,
        )
    total_num = args.num
    base_schedule_count = 0
    base_name_indices: List[int] = []
    boost_name_indices: List[int] = []
    high_risk_name_indices: List[int] = []
    high_risk_name_index_set: Set[int] = set()
    high_risk_samples_before = 0
    high_risk_samples_after = 0
    high_risk_boost_count = 0
    if args.samples_per_name is not None:
        if args.samples_per_name <= 0:
            raise ValueError("samples_per_name 必须大于 0")
        if not corpus_name_count:
            raise ValueError("samples_per_name 需要有效的 corpus_path")
        if args.corpus_probability < 1:
            raise ValueError(
                "samples_per_name 需要 --corpus_probability 1，"
                "以保证每个公司被稳定覆盖"
            )
        base_schedule_count = corpus_name_count * args.samples_per_name
        if args.high_risk_max_samples_per_name > 0:
            high_risk_name_indices = find_long_high_risk_name_indices(
                corpus_checker.corpus,
                minimum_length=args.high_risk_min_length,
                prefixes=high_risk_prefixes,
                suffixes=high_risk_suffixes,
            )
            high_risk_name_index_set = set(high_risk_name_indices)
            high_risk_samples_before = (
                len(high_risk_name_indices) * args.samples_per_name
            )
            high_risk_samples_after = high_risk_samples_before
            base_name_indices = build_name_schedule(
                corpus_checker.corpus,
                samples_per_name=args.samples_per_name,
                high_risk_max_samples_per_name=(
                    args.high_risk_max_samples_per_name
                ),
                minimum_length=args.high_risk_min_length,
                prefixes=high_risk_prefixes,
                suffixes=high_risk_suffixes,
            )
            if base_name_indices:
                high_risk_samples_after = sum(
                    index in high_risk_name_index_set
                    for index in base_name_indices
                )
                released_samples = (
                    high_risk_samples_before - high_risk_samples_after
                )
                after_ratio = (
                    high_risk_samples_after / base_schedule_count
                    if base_schedule_count
                    else 0.0
                )
                print(
                    "长名称高风险采样限额: 命中 "
                    f"{len(high_risk_name_indices)} 家，单名 "
                    f"{args.samples_per_name} -> "
                    f"{min(args.samples_per_name, args.high_risk_max_samples_per_name)} "
                    f"张；高风险组合基础占比 "
                    f"{high_risk_samples_before / base_schedule_count:.2%} -> "
                    f"{after_ratio:.2%}，释放 {released_samples} 张回填其他公司名。",
                    flush=True,
                )
            else:
                print(
                    "长名称高风险采样限额无需改变当前计划，沿用均匀采样。",
                    flush=True,
                )
        elif args.high_risk_max_samples_per_name == 0:
            high_risk_name_indices = find_long_high_risk_name_indices(
                corpus_checker.corpus,
                minimum_length=args.high_risk_min_length,
                prefixes=high_risk_prefixes,
                suffixes=high_risk_suffixes,
            )
            high_risk_name_index_set = set(high_risk_name_indices)
        boost_name_indices = build_rare_character_boost(
            corpus_checker.corpus,
            base_samples_per_name=args.samples_per_name,
            minimum_occurrences=args.min_character_occurrences,
            max_extra_samples_per_name=args.max_rare_boost_per_name,
            base_name_indices=base_name_indices or None,
            avoid_name_indices=(
                high_risk_name_index_set if base_name_indices else None
            ),
            max_extra_samples_per_avoided_name=(
                args.max_rare_boost_per_high_risk_name
                if base_name_indices
                else None
            ),
        )
        total_num = base_schedule_count + len(boost_name_indices)
        if boost_name_indices:
            high_risk_boost_count = (
                sum(
                    index in high_risk_name_index_set
                    for index in boost_name_indices
                )
                if base_name_indices
                else 0
            )
            print(
                f"稀有字符补样: 追加 {len(boost_name_indices)} 张，"
                f"保证当前采样池每字符至少出现 "
                f"{args.min_character_occurrences} 次。"
                + (
                    f"其中 {high_risk_boost_count} 张来自高风险名称，"
                    "仅用于补回被限额影响的稀缺字符。"
                    if high_risk_boost_count
                    else ""
                ),
                flush=True,
            )
        else:
            high_risk_boost_count = 0
    elif args.min_character_occurrences > 0:
        raise ValueError(
            "min_character_occurrences 需要同时指定 samples_per_name"
        )
    elif args.high_risk_max_samples_per_name > 0:
        raise ValueError(
            "high_risk_max_samples_per_name 需要同时指定 samples_per_name，"
            "这样才能在保持基础池总量不变的前提下回填样本"
        )
    elif (
        args.corpus_path
        and args.corpus_probability >= 1
        and total_num < corpus_name_count
    ):
        print(
            f"警告：num={total_num} 小于有效词典公司数 {corpus_name_count}，"
            "本次不能覆盖全部公司；建议使用 --samples_per_name。",
            flush=True,
        )
    background_paths = get_recursive_files_list(
        Path(args.background_path).expanduser().resolve(),
        {".jpg", ".jpeg", ".png", ".bmp", ".webp"},
    )
    if background_paths:
        print(
            f"背景资源: 递归读取 {len(background_paths)} 张，每张等概率抽样。",
            flush=True,
        )
    else:
        print("背景资源: 空目录，使用程序生成的纸面背景。", flush=True)
    font_paths = get_files_list(
        Path(args.font_path).expanduser().resolve(),
        {".ttf", ".ttc", ".otf"},
    )
    if not font_paths:
        raise FileNotFoundError(
            f"字体目录中没有 TTF/TTC/OTF 文件: {args.font_path}"
        )
    company_name_characters = set("".join(corpus_checker.corpus))
    if not corpus_checker.corpus or args.corpus_probability < 1:
        company_name_characters.update(
            "".join(
                [
                    *corpus_checker.areas,
                    *corpus_checker.brand_words,
                    *corpus_checker.industries,
                    *corpus_checker.suffixes,
                    *corpus_checker.brand_characters,
                ]
            )
        )
    font_character_coverage = build_font_character_coverage(
        font_paths,
        company_name_characters,
    )
    font_coverage_summary = {}
    for font_path, supported_characters in font_character_coverage.items():
        missing_characters = sorted(
            company_name_characters - supported_characters
        )
        font_coverage_summary[Path(font_path).name] = {
            "supported_character_count": len(supported_characters),
            "missing_character_count": len(missing_characters),
            "missing_character_examples": "".join(missing_characters[:80]),
        }
        if missing_characters:
            print(
                f"字体 {Path(font_path).name} 缺少 "
                f"{len(missing_characters)} 个公司名字符；"
                "生成器会在相关公司名上自动避开该字体。"
                f"示例: {''.join(missing_characters[:80])}",
                flush=True,
            )

    characters_supported_by_any_font = set().union(
        *font_character_coverage.values()
    )
    unsupported_characters = sorted(
        company_name_characters - characters_supported_by_any_font
    )
    if unsupported_characters:
        raise ValueError(
            "所有字体都缺少以下公司名字符，请补充字体后再生成: "
            f"{''.join(unsupported_characters)}"
        )
    unrenderable_company_names = [
        company_name
        for company_name in corpus_checker.corpus
        if not get_compatible_font_paths(
            company_name,
            font_character_coverage,
        )
    ]
    if unrenderable_company_names:
        raise ValueError(
            "部分公司名的字符分别存在于不同字体，但没有一款字体能完整绘制；"
            f"请补充覆盖完整名称的字体。示例: {unrenderable_company_names[:10]}"
        )
    font_coverage_fingerprint = fingerprint_values(
        (
            f"{Path(font_path).resolve()}\0"
            f"{''.join(sorted(supported_characters))}"
        )
        for font_path, supported_characters in font_character_coverage.items()
    )

    config = {
        "output_path": str(output_path),
        "spatial_annotations": args.spatial_annotations,
        "spatial_annotation_output": (
            str(spatial_annotation_output)
            if spatial_annotation_output is not None
            else None
        ),
        "text_heatmap_radius_ratio": args.text_heatmap_radius_ratio,
        "stamp_heatmap_radius_ratio": args.stamp_heatmap_radius_ratio,
        "character_heatmap_radius_ratio": (
            args.character_heatmap_radius_ratio
        ),
        "corpus_path": args.corpus_path,
        "corpus_probability": args.corpus_probability,
        "allowed_characters": allowed_characters,
        "excluded_names": excluded_names,
        "excluded_substrings": args.exclude_label_substring,
        "name_length": (args.min_label_length, args.max_label_length),
        "background_paths": background_paths,
        "font_paths": font_paths,
        "font_character_coverage": font_character_coverage,
        "color_ratios": color_ratios,
        "shape_ratios": shape_ratios,
        "layout_ratios": layout_ratios,
        "bilingual_shape_ratios": bilingual_shape_ratios,
        "ellipse_ratio_range": list(ellipse_ratio_range),
        "circle_star_probability": args.circle_star_probability,
        "document_interference_probability": document_interference_probability,
        "foreground_occlusion_probability": foreground_occlusion_probability,
        "blank_background_probability": blank_background_probability,
        "partial_crop_probability": partial_crop_probability,
        "max_partial_crop_ratio": max_partial_crop_ratio,
        "large_rotation_probability": large_rotation_probability,
        "large_rotation_min_degrees": args.large_rotation_min_degrees,
        "large_rotation_max_degrees": args.large_rotation_max_degrees,
        "ink_degradation_probability": ink_degradation_probability,
        "stamp_size_ratio_range": list(stamp_size_ratio_range),
        "seed": args.seed,
        "overwrite": args.overwrite,
        "shard_size": args.shard_size,
        "start_index": args.start_index,
        "base_schedule_count": base_schedule_count,
        "base_name_indices": base_name_indices,
        "boost_name_indices": boost_name_indices,
    }
    high_risk_sampling = {
        "enabled": args.high_risk_max_samples_per_name > 0,
        "minimum_length": args.high_risk_min_length,
        "prefixes": list(high_risk_prefixes),
        "suffixes": list(high_risk_suffixes),
        "max_samples_per_name": args.high_risk_max_samples_per_name,
        "name_count": len(high_risk_name_indices),
        "samples_before": high_risk_samples_before,
        "samples_after": high_risk_samples_after,
        "base_ratio_after": (
            high_risk_samples_after / base_schedule_count
            if base_schedule_count
            else 0.0
        ),
    }
    generation_record = {
        "generator_version": GENERATOR_VERSION,
        "sample_metadata_format": SAMPLE_METADATA_FORMAT,
        "spatial_annotations": args.spatial_annotations,
        "spatial_annotation_format": SPATIAL_ANNOTATION_FORMAT,
        "spatial_annotation_output": (
            str(spatial_annotation_output)
            if spatial_annotation_output is not None
            else None
        ),
        "spatial_annotation_channels": SPATIAL_ANNOTATION_CHANNELS,
        "text_heatmap_radius_ratio": args.text_heatmap_radius_ratio,
        "stamp_heatmap_radius_ratio": args.stamp_heatmap_radius_ratio,
        "character_heatmap_radius_ratio": (
            args.character_heatmap_radius_ratio
        ),
        "num": total_num,
        "base_num": base_schedule_count or total_num,
        "samples_per_name": args.samples_per_name,
        "min_character_occurrences": args.min_character_occurrences,
        "max_rare_boost_per_name": args.max_rare_boost_per_name,
        "max_rare_boost_per_high_risk_name": (
            args.max_rare_boost_per_high_risk_name
        ),
        "high_risk_sampling": high_risk_sampling,
        "name_schedule_fingerprint": (
            fingerprint_sequence(base_name_indices)
            if base_name_indices
            else None
        ),
        "rare_character_boost_samples": len(boost_name_indices),
        "rare_character_boost_high_risk_samples": high_risk_boost_count,
        "rare_character_boost_base_schedule": bool(base_name_indices),
        "rare_character_boost_fingerprint": fingerprint_values(
            map(str, boost_name_indices)
        ),
        "valid_corpus_name_count": corpus_name_count,
        "rejected_corpus_name_count": corpus_checker.corpus_rejected_count,
        "corpus_duplicate_count": corpus_checker.corpus_duplicate_count,
        "corpus_rejection_reasons": dict(
            corpus_checker.corpus_rejection_reasons
        ),
        "corpus_missing_characters": dict(
            corpus_checker.corpus_missing_characters
        ),
        "start_index": args.start_index,
        "processes": args.processes,
        "shard_size": args.shard_size,
        "color_ratios": color_ratios,
        "shape_ratios": shape_ratios,
        "layout_ratios": layout_ratios,
        "bilingual_shape_ratios": bilingual_shape_ratios,
        "ellipse_ratio_range": list(ellipse_ratio_range),
        "circle_star_probability": args.circle_star_probability,
        "document_interference_probability": document_interference_probability,
        "foreground_occlusion_probability": foreground_occlusion_probability,
        "blank_background_probability": blank_background_probability,
        "partial_crop_probability": partial_crop_probability,
        "max_partial_crop_ratio": max_partial_crop_ratio,
        "large_rotation_probability": large_rotation_probability,
        "large_rotation_min_degrees": args.large_rotation_min_degrees,
        "large_rotation_max_degrees": args.large_rotation_max_degrees,
        "ink_degradation_probability": ink_degradation_probability,
        "stamp_size_ratio_range": list(stamp_size_ratio_range),
        "label_length": [args.min_label_length, args.max_label_length],
        "corpus_path": (
            str(Path(args.corpus_path).expanduser().resolve())
            if args.corpus_path
            else None
        ),
        "corpus_probability": args.corpus_probability,
        "corpus_fingerprint": fingerprint_values(corpus_checker.corpus),
        "vocab_path": (
            str(Path(args.vocab_path).expanduser().resolve())
            if args.vocab_path
            else None
        ),
        "vocab_fingerprint": (
            fingerprint_values(allowed_characters)
            if allowed_characters is not None
            else None
        ),
        "excluded_name_count": len(excluded_names),
        "excluded_names_fingerprint": fingerprint_values(excluded_names),
        "excluded_substrings": args.exclude_label_substring,
        "background_fingerprint": fingerprint_resources(background_paths),
        "background_image_count": len(background_paths),
        "background_sampling": "blank_or_uniform_recursive_v2",
        "font_fingerprint": fingerprint_resources(font_paths),
        "font_selection": "full_company_name_glyph_coverage_v1",
        "font_coverage_character_count": len(company_name_characters),
        "font_coverage_summary": font_coverage_summary,
        "font_coverage_fingerprint": font_coverage_fingerprint,
        "seed": args.seed,
    }
    print(
        f"语料审计: 有效唯一公司 {corpus_name_count}，"
            f"基础样本 {base_schedule_count or total_num}，"
            f"低频字符补样 {len(boost_name_indices)}，"
            f"预计总量 {total_num}。"
        + (
            f" 长名称高风险组合 {len(high_risk_name_indices)} 家，"
            f"基础池占比 {high_risk_sampling['base_ratio_after']:.2%}。"
            if high_risk_sampling["enabled"]
            else ""
        ),
        flush=True,
    )
    if args.validate_only:
        if spatial_annotation_output is not None:
            print(
                f"空间标注将写入独立目录: {spatial_annotation_output}，"
                f"格式: {SPATIAL_ANNOTATION_FORMAT}"
            )
        print("validate_only 检查通过；未创建输出目录，未生成图片。")
        return 0

    output_path.mkdir(parents=True, exist_ok=True)
    config_path = output_path / "generation_config.json"
    if config_path.exists() and not args.overwrite:
        existing_record = json.loads(config_path.read_text(encoding="utf-8"))
        signature_keys = [
            "generator_version",
            "sample_metadata_format",
            "spatial_annotations",
            "spatial_annotation_format",
            "spatial_annotation_output",
            "spatial_annotation_channels",
            "text_heatmap_radius_ratio",
            "stamp_heatmap_radius_ratio",
            "character_heatmap_radius_ratio",
            "samples_per_name",
            "min_character_occurrences",
            "max_rare_boost_per_name",
            "max_rare_boost_per_high_risk_name",
            "high_risk_sampling",
            "name_schedule_fingerprint",
            "rare_character_boost_high_risk_samples",
            "rare_character_boost_base_schedule",
            "rare_character_boost_fingerprint",
            "color_ratios",
            "shape_ratios",
            "layout_ratios",
            "bilingual_shape_ratios",
            "ellipse_ratio_range",
            "circle_star_probability",
            "document_interference_probability",
            "foreground_occlusion_probability",
            "blank_background_probability",
            "partial_crop_probability",
            "max_partial_crop_ratio",
            "large_rotation_probability",
            "large_rotation_min_degrees",
            "large_rotation_max_degrees",
            "ink_degradation_probability",
            "stamp_size_ratio_range",
            "label_length",
            "corpus_path",
            "corpus_probability",
            "corpus_fingerprint",
            "vocab_path",
            "vocab_fingerprint",
            "excluded_names_fingerprint",
            "excluded_substrings",
            "background_fingerprint",
            "background_sampling",
            "font_fingerprint",
            "font_selection",
            "font_coverage_fingerprint",
            "seed",
            "shard_size",
        ]
        changed_keys = [
            key
            for key in signature_keys
            if existing_record.get(key) != generation_record.get(key)
        ]
        if changed_keys:
            raise ValueError(
                f"输出目录已有不同生成配置，变化字段: {changed_keys}。"
                "请使用新输出目录；确认重做时才使用 --overwrite。"
            )
    config_path.write_text(
        json.dumps(generation_record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if spatial_annotation_output is not None:
        spatial_annotation_output.mkdir(parents=True, exist_ok=True)
        annotation_config = {
            "format": SPATIAL_ANNOTATION_FORMAT,
            "channels": SPATIAL_ANNOTATION_CHANNELS,
            "files": {
                "primary": "<sample>.png",
                "detail": "<sample>.detail.png",
            },
            "derived_maps": {
                "stamp_graphics_without_company_text": (
                    "primary.G AND (NOT primary.R)"
                )
            },
            "image_mode": "RGBA",
            "file_extension": ".png",
            "alignment": (
                "same shard/name and same pixel size as the paired OCR JPG"
            ),
            "mask_semantics": (
                "ideal pre-degradation geometry; transformed and clipped with "
                "the stamp, not erased by later occlusion or ink fading; "
                "detail.G is nonzero only on company glyph pixels"
            ),
            "text_heatmap_radius_ratio": args.text_heatmap_radius_ratio,
            "stamp_heatmap_radius_ratio": args.stamp_heatmap_radius_ratio,
            "character_heatmap_radius_ratio": (
                args.character_heatmap_radius_ratio
            ),
            "ocr_image_output": str(output_path),
        }
        (spatial_annotation_output / "annotation_config.json").write_text(
            json.dumps(annotation_config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    queue = Queue(maxsize=max(16, args.processes * 8))
    processes = [
        Process(
            target=generate_worker,
            args=(pid, queue, config),
            daemon=False,
        )
        for pid in range(args.processes)
    ]
    for process in processes:
        process.start()

    for index in range(args.start_index, args.start_index + total_num):
        queue.put(index)
    for _ in processes:
        queue.put(None)
    for process in processes:
        process.join()

    failed_processes = [
        process.pid for process in processes if process.exitcode != 0
    ]
    if failed_processes:
        raise RuntimeError(f"生成进程异常退出: {failed_processes}")

    print(f"生成任务结束，输出目录: {output_path}")
    if spatial_annotation_output is not None:
        print(f"空间标注目录: {spatial_annotation_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

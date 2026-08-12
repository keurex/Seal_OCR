"""公司名采样计划的分布回归测试。"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import tempfile
import unittest

from synthesis.generate import (
    DEFAULT_LARGE_ROTATION_PROBABILITY,
    DEFAULT_STYLE,
    build_name_schedule,
    build_rare_character_boost,
    find_long_high_risk_name_indices,
    get_recursive_files_list,
    is_long_high_risk_name,
)


class NameSamplingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.corpus = [
            "宁波北仑甲乙丙丁戊己庚辛壬癸有限公司（个体工商户）",
            "宁波甲乙丙丁戊己庚辛壬癸子丑寅卯有限公司（个体工商户）",
            "宁波甲乙丙丁戊己庚辛壬癸子丑寅卯有限公司",
            "杭州甲乙丙丁戊己庚辛壬癸有限公司",
        ]

    def test_high_risk_definition_covers_either_overrepresented_component(self) -> None:
        self.assertTrue(is_long_high_risk_name(self.corpus[0]))
        self.assertFalse(is_long_high_risk_name("宁波甲乙有限公司（个体工商户）"))
        self.assertTrue(is_long_high_risk_name(self.corpus[2]))
        self.assertTrue(
            is_long_high_risk_name(
                "杭州甲乙丙丁戊己庚辛壬癸有限公司（个体工商户）"
            )
        )
        self.assertEqual(find_long_high_risk_name_indices(self.corpus), [0, 1, 2])

    def test_cap_preserves_total_and_refills_normal_names(self) -> None:
        schedule = build_name_schedule(
            self.corpus,
            samples_per_name=5,
            high_risk_max_samples_per_name=2,
        )
        counts = Counter(schedule)

        self.assertEqual(len(schedule), len(self.corpus) * 5)
        self.assertEqual([counts[index] for index in (0, 1, 2)], [2, 2, 2])
        self.assertGreaterEqual(counts[3], 5)
        self.assertEqual(sum(counts[index] for index in (0, 1, 2)), 6)

    def test_zero_or_non_binding_cap_keeps_legacy_schedule(self) -> None:
        self.assertEqual(
            build_name_schedule(self.corpus, 5, 0),
            [],
        )

    def test_public_default_style_is_stable(self) -> None:
        style = DEFAULT_STYLE

        self.assertEqual(style["color_ratios"], {
            "red": 89, "blue": 7, "purple": 1, "black": 3,
        })
        self.assertEqual(style["shape_ratios"], {
            "circle": 90, "ellipse": 7, "rectangle": 3,
        })
        self.assertEqual(
            style["large_rotation_probability"],
            DEFAULT_LARGE_ROTATION_PROBABILITY,
        )

    def test_empty_background_directory_uses_generated_paper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            self.assertEqual(
                get_recursive_files_list(Path(temporary_dir), {".jpg", ".png"}),
                [],
            )

    def test_rare_boost_uses_capped_schedule_for_base_counts(self) -> None:
        corpus = ["稀A", "普通B"]
        base_schedule = [1, 1, 1, 1, 0]
        boost = build_rare_character_boost(
            corpus,
            base_samples_per_name=5,
            minimum_occurrences=2,
            max_extra_samples_per_name=3,
            base_name_indices=base_schedule,
        )

        self.assertIn(0, boost)
        counts = Counter()
        for name_index in base_schedule + boost:
            counts.update(corpus[name_index])
        self.assertGreaterEqual(counts["稀"], 2)

    def test_rare_boost_prefers_non_high_risk_source_when_available(self) -> None:
        corpus = ["稀A", "稀A", "普通B"]
        base_schedule = [0, 1, 1, 1, 1]
        boost = build_rare_character_boost(
            corpus,
            base_samples_per_name=5,
            minimum_occurrences=6,
            max_extra_samples_per_name=10,
            base_name_indices=base_schedule,
            avoid_name_indices={0},
        )

        self.assertNotIn(0, boost)
        self.assertIn(1, boost)

    def test_rare_boost_allows_necessary_high_risk_exception(self) -> None:
        corpus = ["稀A", "普通B"]
        base_schedule = [0, 1, 1, 1, 1]
        boost = build_rare_character_boost(
            corpus,
            base_samples_per_name=5,
            minimum_occurrences=5,
            max_extra_samples_per_name=1,
            base_name_indices=base_schedule,
            avoid_name_indices={0},
            max_extra_samples_per_avoided_name=4,
        )

        self.assertEqual(boost.count(0), 4)
        self.assertEqual(
            build_name_schedule(self.corpus, 5, 5),
            [],
        )


if __name__ == "__main__":
    unittest.main()

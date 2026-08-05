import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from formatting_sync import (
    AI_FALLBACK_REASON,
    DETERMINISTIC_RELOCATED_REASON,
    apply_formatting_only_change,
    apply_formatting_only_change_with_ai_fallback,
    build_formatting_only_change,
)


class FakeMappingAI:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class FormattingSyncTest(unittest.TestCase):
    def test_rejects_text_changes(self):
        self.assertIsNone(
            build_formatting_only_change("# A\nOld\n", "# A\nNew\n")
        )

    def test_line_ending_only_change_is_not_a_translation_change(self):
        self.assertIsNone(build_formatting_only_change("# A\r\nText.\r\n", "# A\nText.\n"))

    def test_applies_trailing_whitespace_and_preserves_translation(self):
        change = build_formatting_only_change(
            "# A\nTranslated source.   \n",
            "# A\nTranslated source.\n",
        )
        self.assertIsNotNone(change)

        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = Path(tmpdir, "guide.md")
            target_path.write_text("# 甲\n已有译文。   \n", encoding="utf-8")

            success, changed, reason = apply_formatting_only_change(
                "guide.md",
                change,
                tmpdir,
            )

            self.assertTrue(success, reason)
            self.assertTrue(changed)
            self.assertEqual("# 甲\n已有译文。\n", target_path.read_text(encoding="utf-8"))

    def test_rejects_misaligned_target(self):
        change = build_formatting_only_change("# A\n    \n", "# A\n\n")

        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = Path(tmpdir, "guide.md")
            target_path.write_text("# 甲\n正文\n", encoding="utf-8")

            success, changed, reason = apply_formatting_only_change(
                "guide.md",
                change,
                tmpdir,
            )

            self.assertFalse(success)
            self.assertFalse(changed)
            self.assertIn("blank-line layouts differ", reason)
            self.assertEqual("# 甲\n正文\n", target_path.read_text(encoding="utf-8"))

    def test_rejects_same_line_count_when_heading_positions_differ(self):
        change = build_formatting_only_change(
            "# A\nText.   \n## B\nMore.\n",
            "# A\nText.\n## B\nMore.\n",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = Path(tmpdir, "guide.md")
            original_target = "# 甲\n正文。   \n更多。\n## 乙\n"
            target_path.write_text(original_target, encoding="utf-8")

            success, changed, reason = apply_formatting_only_change(
                "guide.md",
                change,
                tmpdir,
            )

            self.assertFalse(success)
            self.assertFalse(changed)
            self.assertIn("heading line numbers or levels differ", reason)
            self.assertEqual(original_target, target_path.read_text(encoding="utf-8"))

    def test_rejects_same_heading_positions_when_blank_layout_differs(self):
        change = build_formatting_only_change(
            "# A\nFirst.   \n\nSecond.\n## B\n",
            "# A\nFirst.\n\nSecond.\n## B\n",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = Path(tmpdir, "guide.md")
            original_target = "# 甲\n\n第一。   \n第二。\n## 乙\n"
            target_path.write_text(original_target, encoding="utf-8")

            success, changed, reason = apply_formatting_only_change(
                "guide.md",
                change,
                tmpdir,
            )

            self.assertFalse(success)
            self.assertFalse(changed)
            self.assertIn("blank-line layouts differ", reason)
            self.assertEqual(original_target, target_path.read_text(encoding="utf-8"))

    def test_rejects_nonblank_change_without_heading_anchors(self):
        change = build_formatting_only_change("Text.   \n", "Text.\n")

        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = Path(tmpdir, "guide.md")
            original_target = "正文。   \n"
            target_path.write_text(original_target, encoding="utf-8")

            success, changed, reason = apply_formatting_only_change(
                "guide.md",
                change,
                tmpdir,
            )

            self.assertFalse(success)
            self.assertFalse(changed)
            self.assertIn("require aligned heading anchors", reason)
            self.assertEqual(original_target, target_path.read_text(encoding="utf-8"))

    def test_allows_blank_line_cleanup_without_heading_anchors(self):
        change = build_formatting_only_change(
            "<CustomContent>\n    \n</CustomContent>\n",
            "<CustomContent>\n\n</CustomContent>\n",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = Path(tmpdir, "guide.md")
            target_path.write_text(
                "<CustomContent>\n    \n</CustomContent>\n",
                encoding="utf-8",
            )

            success, changed, reason = apply_formatting_only_change(
                "guide.md",
                change,
                tmpdir,
            )

            self.assertTrue(success, reason)
            self.assertTrue(changed)
            self.assertEqual(
                "<CustomContent>\n\n</CustomContent>\n",
                target_path.read_text(encoding="utf-8"),
            )

    def test_ai_fallback_maps_shifted_target_line_without_rewriting_text(self):
        change = build_formatting_only_change(
            "# A\n<CustomContent>\n    \n1. Unsubscribe.\n## B\n",
            "# A\n<CustomContent>\n\n1. Unsubscribe.\n## B\n",
        )
        change["source_patch"] = (
            "@@ -2,3 +2,3 @@\n <CustomContent>\n-    \n+\n 1. Unsubscribe."
        )
        ai_client = FakeMappingAI(
            '{"mappings":[{"source_line_number":3,"target_line_number":4}]}'
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = Path(tmpdir, "guide.md")
            original_target = (
                "# 甲\n补充说明。\n<CustomContent>\n    \n"
                "1. 取消订阅。\n<CustomContent>\n    \n"
                "1. 保持订阅。\n## 乙\n"
            )
            target_path.write_text(original_target, encoding="utf-8")

            success, changed, reason, used_ai = (
                apply_formatting_only_change_with_ai_fallback(
                    "guide.md",
                    change,
                    tmpdir,
                    ai_client,
                    "English",
                    "Chinese",
                )
            )

            self.assertTrue(success, reason)
            self.assertTrue(changed)
            self.assertTrue(used_ai)
            self.assertEqual(AI_FALLBACK_REASON, reason)
            self.assertEqual(
                original_target.replace(
                    "<CustomContent>\n    \n",
                    "<CustomContent>\n\n",
                    1,
                ),
                target_path.read_text(encoding="utf-8"),
            )

        self.assertEqual(1, len(ai_client.calls))
        prompt_payload = ai_client.calls[0]["messages"][1]["content"]
        self.assertIn("source_patch", prompt_payload)
        self.assertIn("补充说明。", prompt_payload)

    def test_ai_fallback_rejects_wrong_mapping_and_preserves_target(self):
        change = build_formatting_only_change(
            "# A\n<CustomContent>\n    \n1. Unsubscribe.\n## B\n",
            "# A\n<CustomContent>\n\n1. Unsubscribe.\n## B\n",
        )
        ai_client = FakeMappingAI(
            '{"mappings":[{"source_line_number":3,"target_line_number":2}]}'
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = Path(tmpdir, "guide.md")
            original_target = (
                "# 甲\n补充说明。\n<CustomContent>\n    \n"
                "1. 取消订阅。\n<CustomContent>\n    \n"
                "1. 保持订阅。\n## 乙\n"
            )
            target_path.write_text(original_target, encoding="utf-8")

            success, changed, reason, used_ai = (
                apply_formatting_only_change_with_ai_fallback(
                    "guide.md",
                    change,
                    tmpdir,
                    ai_client,
                    "English",
                    "Chinese",
                )
            )

            self.assertFalse(success)
            self.assertFalse(changed)
            self.assertTrue(used_ai)
            self.assertIn("different content shape", reason)
            self.assertEqual(original_target, target_path.read_text(encoding="utf-8"))

    def test_direct_sync_does_not_call_ai_fallback(self):
        change = build_formatting_only_change(
            "# A\n    \nText.\n",
            "# A\n\nText.\n",
        )
        ai_client = FakeMappingAI(AssertionError("AI must not run"))

        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = Path(tmpdir, "guide.md")
            target_path.write_text("# 甲\n    \n正文。\n", encoding="utf-8")

            success, changed, reason, used_ai = (
                apply_formatting_only_change_with_ai_fallback(
                    "guide.md",
                    change,
                    tmpdir,
                    ai_client,
                    "English",
                    "Chinese",
                )
            )

            self.assertTrue(success, reason)
            self.assertTrue(changed)
            self.assertFalse(used_ai)
            self.assertEqual([], ai_client.calls)

    def test_ai_fallback_batches_large_ambiguous_mapping_and_bounds_patch_input(self):
        change_count = 70
        source_blocks = "".join(
            f"<CustomContent>\n    \n1. Item {index}.\n"
            for index in range(change_count)
        )
        head_blocks = source_blocks.replace("    \n", "\n")
        change = build_formatting_only_change(
            f"# A\n{source_blocks}## End\n",
            f"# A\n{head_blocks}## End\n",
        )
        change["source_patch"] = "x" * 20001

        class BatchedMappingAI:
            def __init__(self):
                self.calls = []

            def chat_completion(self, **kwargs):
                self.calls.append(kwargs)
                payload = json.loads(kwargs["messages"][1]["content"])
                return json.dumps(
                    {
                        "mappings": [
                            {
                                "source_line_number": item["source_line_number"],
                                "target_line_number": item["source_line_number"] + 1,
                            }
                            for item in payload["source_changes"]
                        ]
                    }
                )

        ai_client = BatchedMappingAI()
        target_blocks = "".join(
            f"<CustomContent>\n    \n1. 项目 {index}。\n"
            for index in range(change_count + 1)
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = Path(tmpdir, "guide.md")
            target_path.write_text(
                f"# 甲\n补充说明。\n{target_blocks}## 结束\n",
                encoding="utf-8",
            )

            success, changed, reason, ai_attempted = (
                apply_formatting_only_change_with_ai_fallback(
                    "guide.md",
                    change,
                    tmpdir,
                    ai_client,
                    "English",
                    "Chinese",
                )
            )

            self.assertTrue(success, reason)
            self.assertTrue(changed)
            self.assertTrue(ai_attempted)
            remaining_spaced_blank_lines = [
                line
                for line in target_path.read_text(encoding="utf-8").splitlines()
                if line == "    "
            ]
            self.assertEqual(["    "], remaining_spaced_blank_lines)

        self.assertEqual(3, len(ai_client.calls))
        batch_sizes = []
        for call in ai_client.calls:
            payload = json.loads(call["messages"][1]["content"])
            batch_sizes.append(len(payload["source_changes"]))
            self.assertEqual("", payload["source_patch"])
            self.assertTrue(payload["source_patch_omitted_because_too_large"])
            self.assertNotIn("target_lines", payload)
            self.assertGreaterEqual(call["max_tokens"], 2048)
        self.assertEqual([32, 32, 6], batch_sizes)

    def test_relocated_unique_monotonic_mapping_does_not_call_ai(self):
        source_blocks = "".join(
            f"<CustomContent>\n    \n1. Item {index}.\n"
            for index in range(3)
        )
        change = build_formatting_only_change(
            f"# A\n{source_blocks}## End\n",
            f"# A\n{source_blocks.replace('    \n', '\n')}## End\n",
        )
        ai_client = FakeMappingAI(AssertionError("AI must not run"))
        target_blocks = "".join(
            f"<CustomContent>\n    \n1. 项目 {index}。\n"
            for index in range(3)
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = Path(tmpdir, "guide.md")
            target_path.write_text(
                f"# 甲\n补充说明。\n{target_blocks}## 结束\n",
                encoding="utf-8",
            )

            success, changed, reason, ai_attempted = (
                apply_formatting_only_change_with_ai_fallback(
                    "guide.md",
                    change,
                    tmpdir,
                    ai_client,
                    "English",
                    "Chinese",
                )
            )

            self.assertTrue(success, reason)
            self.assertTrue(changed)
            self.assertFalse(ai_attempted)
            self.assertEqual(DETERMINISTIC_RELOCATED_REASON, reason)
            self.assertNotIn(
                "    \n",
                target_path.read_text(encoding="utf-8"),
            )
            self.assertEqual([], ai_client.calls)

    def test_missing_target_does_not_call_ai(self):
        change = build_formatting_only_change("# A\n    \n", "# A\n\n")
        ai_client = FakeMappingAI(AssertionError("AI must not run"))

        with tempfile.TemporaryDirectory() as tmpdir:
            success, changed, reason, ai_attempted = (
                apply_formatting_only_change_with_ai_fallback(
                    "missing.md",
                    change,
                    tmpdir,
                    ai_client,
                    "English",
                    "Chinese",
                )
            )

        self.assertFalse(success)
        self.assertFalse(changed)
        self.assertFalse(ai_attempted)
        self.assertIn("Target file does not exist", reason)
        self.assertEqual([], ai_client.calls)


if __name__ == "__main__":
    unittest.main()

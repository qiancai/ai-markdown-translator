import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from internal_link_title_finalizer import finalize_internal_link_titles


@dataclass
class DiffFile:
    filename: str
    status: str = "modified"
    patch: str = ""


class InternalLinkTitleFinalizerTest(unittest.TestCase):
    def run_finalizer(self, target_files, before_snapshots, changed_files, source_files):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for path, content in target_files.items():
                full_path = root / path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(content, encoding="utf-8")

            def source_loader(path):
                return source_files.get(path)

            replacements = finalize_internal_link_titles(
                changed_files,
                str(root),
                before_snapshots,
                source_loader,
                printer=lambda *args, **kwargs: None,
            )

            return replacements, {
                path: (root / path).read_text(encoding="utf-8")
                for path in target_files
                if (root / path).exists()
            }

    def test_replaces_ai_link_label_after_all_target_headings_exist(self):
        changed_files = [
            DiffFile(
                filename="guide.md",
                patch=(
                    "@@ -3 +3 @@\n"
                    "+See [Connect](other.md#connect) for details.\n"
                ),
            )
        ]
        before_snapshots = {
            "guide.md": "# Guide\n\nOld line.\n",
        }
        target_files = {
            "guide.md": "# 指南\n\n请参阅 [不准确的译名](other.md#connect) 了解详情。\n",
            "other.md": "# 其他\n\n## 连接 {#connect}\n",
        }
        source_files = {
            "guide.md": "# Guide\n\nSee [Connect](other.md#connect) for details.\n",
            "other.md": "# Other\n\n## Connect {#connect}\n",
        }

        replacements, files = self.run_finalizer(
            target_files,
            before_snapshots,
            changed_files,
            source_files,
        )

        self.assertEqual(1, replacements)
        self.assertIn("[连接](other.md#connect)", files["guide.md"])
        self.assertNotIn("不准确的译名", files["guide.md"])

    def test_keeps_unchanged_target_lines_untouched(self):
        changed_files = [
            DiffFile(
                filename="guide.md",
                patch=(
                    "@@ -4 +4 @@\n"
                    "+New line mentions [Connect](other.md#connect).\n"
                ),
            )
        ]
        before_snapshots = {
            "guide.md": (
                "# Guide\n\n"
                "旧链接 [不准确的译名](other.md#connect)。\n"
                "Old line.\n"
            ),
        }
        target_files = {
            "guide.md": (
                "# Guide\n\n"
                "旧链接 [不准确的译名](other.md#connect)。\n"
                "新行没有链接。\n"
            ),
            "other.md": "# 其他\n\n## 连接 {#connect}\n",
        }
        source_files = {
            "guide.md": "# Guide\n\n旧链接。\nNew line mentions [Connect](other.md#connect).\n",
            "other.md": "# Other\n\n## Connect {#connect}\n",
        }

        replacements, files = self.run_finalizer(
            target_files,
            before_snapshots,
            changed_files,
            source_files,
        )

        self.assertEqual(0, replacements)
        self.assertIn("旧链接 [不准确的译名](other.md#connect)。", files["guide.md"])

    def test_requires_source_link_label_to_match_source_heading(self):
        changed_files = [
            DiffFile(
                filename="guide.md",
                patch=(
                    "@@ -3 +3 @@\n"
                    "+See [Overview](other.md#connect) for details.\n"
                ),
            )
        ]
        before_snapshots = {
            "guide.md": "# Guide\n\nOld line.\n",
        }
        target_files = {
            "guide.md": "# 指南\n\n请参阅 [概览](other.md#connect) 了解详情。\n",
            "other.md": "# 其他\n\n## 连接 {#connect}\n",
        }
        source_files = {
            "guide.md": "# Guide\n\nSee [Overview](other.md#connect) for details.\n",
            "other.md": "# Other\n\n## Connect {#connect}\n",
        }

        replacements, files = self.run_finalizer(
            target_files,
            before_snapshots,
            changed_files,
            source_files,
        )

        self.assertEqual(0, replacements)
        self.assertIn("[概览](other.md#connect)", files["guide.md"])

    def test_ignores_source_links_inside_fence_split_across_hunks(self):
        source_lines = [
            "# Guide", "```", "[Connect](other.md#connect)",
            "a", "b", "c", "d", "e", "f",
            "[Connect](other.md#connect)", "```",
        ]
        changed_files = [DiffFile(
            filename="guide.md",
            patch=(
                "@@ -1,2 +1,3 @@\n"
                " # Guide\n"
                " ```\n"
                "+[Connect](other.md#connect)\n"
                "@@ -9,1 +10,2 @@\n"
                " f\n"
                "+[Connect](other.md#connect)\n"
            ),
        )]
        target_files = {
            "guide.md": "# 指南\n新行 [错译](other.md#connect)\n",
            "other.md": "# Other\n## 连接 {#connect}\n",
        }
        source_files = {
            "guide.md": "\n".join(source_lines) + "\n",
            "other.md": "# Other\n## Connect {#connect}\n",
        }

        replacements, files = self.run_finalizer(
            target_files,
            {"guide.md": "# 指南\n"},
            changed_files,
            source_files,
        )

        self.assertEqual(0, replacements)
        self.assertIn("[错译](other.md#connect)", files["guide.md"])

    def test_skips_changed_target_code_and_replaces_two_links_on_one_line(self):
        changed_files = [DiffFile(
            filename="guide.md",
            patch=(
                "@@ -3 +3 @@\n"
                "+See [Connect](other.md#connect) and [Connect](other.md#connect).\n"
            ),
        )]
        source_files = {
            "guide.md": (
                "# Guide\n\n"
                "See [Connect](other.md#connect) and [Connect](other.md#connect).\n"
            ),
            "other.md": "# Other\n## Connect {#connect}\n",
        }
        target_files = {
            "guide.md": (
                "# 指南\n```md\n[代码示例](other.md#connect)\n```\n"
                "请参阅 [不准确的译名](other.md#connect) 和 [错](other.md#connect)。\n"
            ),
            "other.md": "# Other\n## 连接 {#connect}\n",
        }

        replacements, files = self.run_finalizer(
            target_files,
            {"guide.md": "# 指南\n"},
            changed_files,
            source_files,
        )

        self.assertEqual(2, replacements)
        self.assertIn("[代码示例](other.md#connect)", files["guide.md"])
        self.assertIn(
            "[连接](other.md#connect) 和 [连接](other.md#connect)",
            files["guide.md"],
        )

    def test_preserves_crlf_when_replacing_changed_line(self):
        changed_files = [DiffFile(
            filename="guide.md",
            patch="@@ -3 +3 @@\n+See [Connect](other.md#connect).\n",
        )]
        target_files = {
            "guide.md": "# 指南\r\n\r\n参阅 [错译](other.md#connect)。\r\n",
            "other.md": "# Other\n## 连接 {#connect}\n",
        }
        source_files = {
            "guide.md": "# Guide\n\nSee [Connect](other.md#connect).\n",
            "other.md": "# Other\n## Connect {#connect}\n",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for path, content in target_files.items():
                (root / path).write_bytes(content.encode("utf-8"))
            replacements = finalize_internal_link_titles(
                changed_files,
                str(root),
                {"guide.md": "# 指南\n\n旧行。\n"},
                source_files.get,
                printer=lambda *_: None,
            )
            result = (root / "guide.md").read_bytes()

        self.assertEqual(1, replacements)
        self.assertEqual(
            "# 指南\r\n\r\n参阅 [连接](other.md#connect)。\r\n".encode("utf-8"),
            result,
        )

    def test_new_target_file_treats_all_lines_as_changed(self):
        changed_files = [DiffFile(
            filename="guide.md",
            status="added",
            patch=(
                "@@ -0,0 +1,2 @@\n"
                "+# Guide\n"
                "+See [Connect](other.md#connect).\n"
            ),
        )]
        replacements, files = self.run_finalizer(
            {
                "guide.md": "# 指南\n参阅 [错译](other.md#connect)。\n",
                "other.md": "# Other\n## 连接 {#connect}\n",
            },
            {"guide.md": None},
            changed_files,
            {
                "guide.md": "# Guide\nSee [Connect](other.md#connect).\n",
                "other.md": "# Other\n## Connect {#connect}\n",
            },
        )

        self.assertEqual(1, replacements)
        self.assertIn("[连接](other.md#connect)", files["guide.md"])

    def test_ignores_nested_link_syntax_inside_image_alt_text(self):
        source_files = {
            "guide.md": (
                "# Guide\n\n"
                "See [Connect](other.md#connect) and "
                "![see [Connect](other.md#connect)](img.png).\n"
            ),
            "other.md": "# Other\n## Connect {#connect}\n",
        }
        target_files = {
            "guide.md": (
                "# 指南\n\n"
                "参阅 [错译](other.md#connect) 和 "
                "![see [错译](other.md#connect)](img.png)。\n"
            ),
            "other.md": "# Other\n## 连接 {#connect}\n",
        }
        changed_files = [DiffFile(
            filename="guide.md",
            patch=(
                "@@ -3 +3 @@\n"
                "+See [Connect](other.md#connect) and "
                "![see [Connect](other.md#connect)](img.png).\n"
            ),
        )]

        replacements, files = self.run_finalizer(
            target_files,
            {"guide.md": "# 指南\n\n旧行。\n"},
            changed_files,
            source_files,
        )

        self.assertEqual(1, replacements)
        self.assertIn("[连接](other.md#connect)", files["guide.md"])
        self.assertIn("![see [错译](other.md#connect)](img.png)", files["guide.md"])

    def test_image_alt_link_does_not_create_candidate(self):
        changed_files = [DiffFile(
            filename="guide.md",
            patch=(
                "@@ -3 +3 @@\n"
                "+![see [Connect](other.md#connect)](img.png)\n"
            ),
        )]
        replacements, files = self.run_finalizer(
            {
                "guide.md": "# 指南\n\n参阅 [错译](other.md#connect)。\n",
                "other.md": "# Other\n## 连接 {#connect}\n",
            },
            {"guide.md": "# 指南\n\n旧行。\n"},
            changed_files,
            {
                "guide.md": "# Guide\n\n![see [Connect](other.md#connect)](img.png)\n",
                "other.md": "# Other\n## Connect {#connect}\n",
            },
        )

        self.assertEqual(0, replacements)
        self.assertIn("[错译](other.md#connect)", files["guide.md"])


if __name__ == "__main__":
    unittest.main()

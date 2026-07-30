import sys
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from translation_structure_validator import (
    compare_custom_content_structure,
    compare_expected_heading_anchors,
    compare_heading_structure,
    compare_stable_heading_order,
    compact_heading_levels,
    extract_custom_content_tags,
    extract_heading_levels,
    validate_markdown_heading_structures,
)


class TranslationStructureValidatorTest(unittest.TestCase):
    def test_extract_heading_levels_skips_code_blocks_and_indented_lines(self):
        content = "\n".join(
            [
                "# Title",
                "",
                "    ## Indented example",
                "  ## CommonMark heading",
                "```",
                "## Code heading",
                "```",
                "## Section",
                "~~~sql",
                "### Code heading",
                "~~~",
                "### Child",
            ]
        )

        self.assertEqual([1, 2, 2, 3], extract_heading_levels(content))

    def test_extract_heading_levels_respects_code_fence_length(self):
        content = "\n".join(
            [
                "# Title",
                "````",
                "```",
                "## Not a heading",
                "```",
                "````",
                "## Real heading",
            ]
        )

        self.assertEqual([1, 2], extract_heading_levels(content))

    def test_compare_heading_structure_ignores_translated_heading_text(self):
        issue = compare_heading_structure(
            "guide.md",
            "# Guide\n\n## Configure\n\n### Step\n",
            "# 指南\n\n## 配置\n\n### 步骤\n",
        )

        self.assertIsNone(issue)

    def test_compare_heading_structure_reports_level_mismatch(self):
        issue = compare_heading_structure(
            "guide.md",
            "# Guide\n\n## Configure\n\n### Step\n",
            "# 指南\n\n## 配置\n\n## 步骤\n",
        )

        self.assertIsNotNone(issue)
        self.assertEqual("guide.md", issue.file_path)
        self.assertEqual("heading level sequence differs", issue.reason)
        self.assertEqual("#x1 ##x1 ###x1", issue.source_compact)
        self.assertEqual("#x1 ##x2", issue.target_compact)
        self.assertEqual("heading 3: source ###, target ##", issue.first_difference)

    def test_compare_stable_heading_order_reports_swapped_anchored_sections(self):
        issue = compare_stable_heading_order(
            "guide.md",
            (
                "# Guide\n\n"
                "## Configure database audit logging settings\n\n"
                "## Specify audit filter rules\n\n"
                "## View audit logs\n"
            ),
            (
                "# ガイド\n\n"
                "## 監査フィルタルールを指定する {#specify-audit-filter-rules}\n\n"
                "## データベース監査ログ設定を構成する "
                "{#configure-database-audit-logging-settings}\n\n"
                "## 監査ログを確認する {#view-audit-logs}\n"
            ),
        )

        self.assertIsNotNone(issue)
        self.assertEqual("stable heading relative order differs", issue.reason)
        self.assertEqual(
            "stable heading 1: source configure-database-audit-logging-settings, "
            "target specify-audit-filter-rules",
            issue.first_difference,
        )

    def test_compare_stable_heading_order_ignores_unanchored_translated_titles(self):
        issue = compare_stable_heading_order(
            "guide.md",
            "# Guide\n\n## Configure\n\n## View logs\n",
            "# ガイド\n\n## 構成する\n\n## ログを確認する\n",
        )

        self.assertIsNone(issue)

    def test_compare_expected_heading_anchors_reports_missing_changed_anchor(self):
        issue = compare_expected_heading_anchors(
            "guide.md",
            (
                "# ガイド\n\n"
                "## 監査フィルターイベント {#auditing-filter-events}\n"
            ),
            ["audit-filter-events"],
        )

        self.assertIsNotNone(issue)
        self.assertEqual("expected heading anchors are missing", issue.reason)
        self.assertEqual(
            "expected anchor {#audit-filter-events} is missing from target",
            issue.first_difference,
        )

    def test_compare_expected_heading_anchors_accepts_restored_anchor(self):
        issue = compare_expected_heading_anchors(
            "guide.md",
            "## 監査フィルターイベント {#audit-filter-events}\n",
            ["audit-filter-events"],
        )

        self.assertIsNone(issue)

    def test_extract_custom_content_tags_skips_code_blocks_and_keeps_inline_order(self):
        content = "\n".join(
            [
                "# Guide",
                '<CustomContent plan="dedicated">{{{ .dedicated }}}</CustomContent><CustomContent plan="essential">',
                "```md",
                '<CustomContent plan="ignored">',
                "```",
                "</CustomContent>",
            ]
        )

        tags = extract_custom_content_tags(content)

        self.assertEqual(
            [
                '<CustomContent plan="dedicated">',
                "</CustomContent>",
                '<CustomContent plan="essential">',
                "</CustomContent>",
            ],
            [tag.text for tag in tags],
        )
        self.assertEqual([2, 2, 2, 6], [tag.line_number for tag in tags])

    def test_compare_custom_content_structure_reports_missing_target_wrapper(self):
        issue = compare_custom_content_structure(
            "guide.md",
            '# Guide\n\n<CustomContent plan="essential">\n\n## Section\n\n</CustomContent>\n',
            "# 指南\n\n## Section\n",
        )

        self.assertIsNotNone(issue)
        self.assertEqual("CustomContent tag sequence differs", issue.reason)
        self.assertIn('2 CustomContent tags', issue.source_compact)
        self.assertEqual("0 CustomContent tags", issue.target_compact)
        self.assertIn('<CustomContent plan="essential">', issue.first_difference)

    def test_compare_custom_content_structure_reports_same_count_different_tag(self):
        issue = compare_custom_content_structure(
            "guide.md",
            '<CustomContent plan="essential">\n\n</CustomContent>\n',
            '<CustomContent plan="dedicated">\n\n</CustomContent>\n',
        )

        self.assertIsNotNone(issue)
        self.assertEqual("CustomContent tag sequence differs", issue.reason)
        self.assertIn('plan="essential"', issue.first_difference)
        self.assertIn('plan="dedicated"', issue.first_difference)

    def test_compare_custom_content_structure_reports_unbalanced_target_tags(self):
        issue = compare_custom_content_structure(
            "guide.md",
            '<CustomContent plan="essential">\n\n</CustomContent>\n',
            '<CustomContent plan="essential">\n\n## Section\n',
        )

        self.assertIsNotNone(issue)
        self.assertEqual("target CustomContent tags are unbalanced", issue.reason)
        self.assertIn("opening tag without matching closing tag", issue.first_difference)

    def test_validate_markdown_heading_structures_checks_only_markdown_files(self):
        calls = []

        def source_loader(file_path):
            calls.append(("source", file_path))
            return "# Guide\n\n## Section\n"

        def target_loader(file_path):
            calls.append(("target", file_path))
            return "# 指南\n\n### Section\n"

        issues = validate_markdown_heading_structures(
            ["guide.md", "image.png"],
            source_loader,
            target_loader,
        )

        self.assertEqual(1, len(issues))
        self.assertEqual("guide.md", issues[0].file_path)
        self.assertEqual([("source", "guide.md"), ("target", "guide.md")], calls)

    def test_validate_markdown_heading_structures_includes_custom_content_issues(self):
        def source_loader(file_path):
            return '<CustomContent plan="essential">\n\n# Guide\n\n</CustomContent>\n'

        def target_loader(file_path):
            return "# 指南\n"

        issues = validate_markdown_heading_structures(
            ["guide.md"],
            source_loader,
            target_loader,
        )

        self.assertEqual(["CustomContent tag sequence differs"], [issue.reason for issue in issues])

    def test_validate_markdown_heading_structures_checks_expected_anchors(self):
        issues = validate_markdown_heading_structures(
            ["guide.md"],
            lambda _file_path: "# Guide\n\n## Audit filter events\n",
            lambda _file_path: "# ガイド\n\n## 監査フィルターイベント\n",
            expected_heading_anchors_loader=lambda _file_path: [
                "audit-filter-events"
            ],
        )

        self.assertEqual(
            ["expected heading anchors are missing"],
            [issue.reason for issue in issues],
        )

    def test_validate_markdown_heading_structures_reports_loader_none_and_exceptions(self):
        def source_loader(file_path):
            if file_path == "missing-source.md":
                return None
            if file_path == "source-error.md":
                raise RuntimeError("source unavailable")
            return "# Guide\n"

        def target_loader(file_path):
            if file_path == "missing-target.md":
                return None
            if file_path == "target-error.md":
                raise RuntimeError("target unavailable")
            return "# 指南\n"

        issues = validate_markdown_heading_structures(
            [
                "missing-source.md",
                "missing-target.md",
                "source-error.md",
                "target-error.md",
            ],
            source_loader,
            target_loader,
        )

        reasons = {issue.file_path: issue.reason for issue in issues}
        self.assertEqual("could not read source HEAD content", reasons["missing-source.md"])
        self.assertEqual("could not read translated target content", reasons["missing-target.md"])
        self.assertEqual(
            "could not read source HEAD content: source unavailable",
            reasons["source-error.md"],
        )
        self.assertEqual(
            "could not read translated target content: target unavailable",
            reasons["target-error.md"],
        )

    def test_compact_heading_levels_handles_empty_documents(self):
        self.assertEqual("(no headings)", compact_heading_levels([]))


if __name__ == "__main__":
    unittest.main()

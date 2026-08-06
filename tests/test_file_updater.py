import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from file_updater import (
    REGULAR_TRANSLATION_CHUNK_SIZE,
    TRANSLATION_CHUNK_MAX_SECTIONS,
    TranslationResult,
    _prepare_translation_prompt,
    build_heading_anchor_slug,
    build_translation_chunks,
    filter_diff_for_chunk_sections,
    get_translation_chunk_max_sections,
    get_updated_sections_from_ai,
    preprocess_diff_for_heading_anchor_stability,
    preprocess_source_sections_for_heading_anchor_stability,
    process_single_file,
    update_target_document_from_match_data,
)
from ai_client import CompletionText
from product_specific_handler import (
    get_product_name,
    rewrite_tidb_version_anchors_in_sections,
    rewrite_tidb_version_anchors_in_text,
)


class FileUpdaterRegressionTest(unittest.TestCase):
    def test_incomplete_response_with_missing_key_remains_useful_partial_result(self):
        class FakeAIClient:
            def chat_completion(self, messages, temperature=0.1):
                return CompletionText(
                    json.dumps({"modified_1": "updated one"}),
                    status="incomplete",
                    reason="max_output_tokens",
                )

        prefix = "incomplete-missing-key-unit"
        self._cleanup_chunk_test_outputs(prefix)
        try:
            result = get_updated_sections_from_ai(
                "File: guide.md\n@@ -1,2 +1,2 @@\n-old\n+new",
                {"modified_1": "old one", "modified_2": "old two"},
                {"modified_1": "new one", "modified_2": "new two"},
                FakeAIClient(),
                "English",
                "Chinese",
                f"{prefix}.md",
            )

            self.assertIsInstance(result, TranslationResult)
            self.assertIn("modified_1", result)
            self.assertNotIn("modified_2", result)
            self.assertTrue(
                any("max_output_tokens" in reason for reason in result.partial_reasons)
            )
            self.assertTrue(
                any("modified_2" in reason for reason in result.partial_reasons)
            )
        finally:
            self._cleanup_chunk_test_outputs(prefix)

    def test_unexpected_key_is_ignored_without_discarding_valid_sections(self):
        class FakeAIClient:
            def chat_completion(self, messages, temperature=0.1):
                return json.dumps(
                    {
                        "modified_1": "updated one",
                        "hallucinated_999": "must be ignored",
                    }
                )

        prefix = "unexpected-key-unit"
        self._cleanup_chunk_test_outputs(prefix)
        try:
            result = get_updated_sections_from_ai(
                "File: guide.md\n@@ -1,1 +1,1 @@\n-old\n+new",
                {"modified_1": "old one"},
                {"modified_1": "new one"},
                FakeAIClient(),
                "English",
                "Chinese",
                f"{prefix}.md",
            )

            self.assertIn("modified_1", result)
            self.assertNotIn("hallucinated_999", result)
            self.assertTrue(
                any("hallucinated_999" in reason for reason in result.partial_reasons)
            )
            self.assertFalse(result.failures)
        finally:
            self._cleanup_chunk_test_outputs(prefix)

    def test_only_unexpected_keys_is_a_total_failure(self):
        class FakeAIClient:
            def chat_completion(self, messages, temperature=0.1):
                return json.dumps({"hallucinated_999": "invalid"})

        prefix = "only-unexpected-key-unit"
        self._cleanup_chunk_test_outputs(prefix)
        try:
            result = get_updated_sections_from_ai(
                "File: guide.md\n@@ -1,1 +1,1 @@\n-old\n+new",
                {"modified_1": "old one"},
                {"modified_1": "new one"},
                FakeAIClient(),
                "English",
                "Chinese",
                f"{prefix}.md",
            )

            self.assertFalse(result)
            self.assertTrue(result.failures)
            self.assertIn("no expected section keys", result.failures[0])
        finally:
            self._cleanup_chunk_test_outputs(prefix)

    def _cleanup_chunk_test_outputs(self, prefix):
        temp_dir = SCRIPTS_DIR / "temp_output"
        for path in temp_dir.glob(f"{prefix}_*"):
            path.unlink()

    def _build_system_sections(self, count, term_prefix=""):
        source_sections = {}
        target_sections = {}
        for index in range(1, count + 1):
            key = f"modified_{index}"
            name = f"tidb_chunk_test_{index:03d}"
            term = f" {term_prefix}{index}" if term_prefix else ""
            source_sections[key] = f"### `{name}`\n\nOld English content{term}.\n"
            target_sections[key] = f"### `{name}`\n\n旧中文内容{index}。\n"
        return source_sections, target_sections

    def _build_system_section_diff(
        self,
        count,
        replacement_terms=None,
        group_size=20,
    ):
        replacement_terms = replacement_terms or {}
        lines = ["File: system-variables.md"]
        for start in range(1, count + 1, group_size):
            end = min(start + group_size - 1, count)
            group_count = end - start + 1
            group_replacements = {
                index: replacement
                for index, replacement in replacement_terms.items()
                if start <= index <= end
            }
            if not group_replacements:
                group_replacements = {
                    start: f"updated content {start}",
                }

            lines.append(
                f"@@ -{start},{group_count} +{start},{group_count} @@"
            )
            for index, replacement in sorted(group_replacements.items()):
                lines.extend(
                    [
                        f"-old content {index}",
                        f"+{replacement}",
                    ]
                )
        return "\n".join(lines)

    def test_build_heading_anchor_slug_keeps_visible_text_inside_span(self):
        heading = '`txn-entry-size-limit` <span class="version-mark">New in v4.0.10 and v5.0.0</span>'
        slug = build_heading_anchor_slug(heading)
        self.assertEqual(slug, "txn-entry-size-limit-new-in-v4010-and-v500")

    def test_preprocess_diff_adds_anchor_to_changed_non_top_level_heading(self):
        pr_diff = "\n".join(
            [
                "File: ai/example.md",
                "@@ -10,1 +10,1 @@",
                "-## Example tests",
                "+## Example test",
                "-" * 80,
            ]
        )

        processed = preprocess_diff_for_heading_anchor_stability(
            pr_diff,
            source_language="English",
            target_language="Chinese",
            source_mode="commit",
        )

        self.assertIn("+## Example test {#example-test}", processed)
        self.assertNotIn("-## Example tests {#example-tests}", processed)

    def test_tidb_version_anchor_rewrite_zh_to_en_pr_mode(self):
        text = (
            "See [`tidb_enable_x`](/system-variables.md"
            "#tidb-enable-x-从-v800-版本开始引入)."
        )

        with mock.patch.dict(os.environ, {"PRODUCT": "TiDB"}, clear=False):
            processed = rewrite_tidb_version_anchors_in_text(
                text,
                source_language="Chinese",
                target_language="English",
                source_mode="pr",
            )

        self.assertEqual(
            processed,
            "See [`tidb_enable_x`](/system-variables.md#tidb-enable-x-new-in-v800).",
        )

    def test_tidb_version_anchor_rewrite_en_to_zh_pr_mode(self):
        text = (
            "请参见 [`tidb_enable_x`](/system-variables.md"
            "#tidb-enable-x-new-in-v800)。"
        )

        with mock.patch.dict(os.environ, {"PRODUCT": "TiDB"}, clear=False):
            processed = rewrite_tidb_version_anchors_in_text(
                text,
                source_language="English",
                target_language="Chinese",
                source_mode="pr",
            )

        self.assertEqual(
            processed,
            "请参见 [`tidb_enable_x`](/system-variables.md#tidb-enable-x-从-v800-版本开始引入)。",
        )

    def test_tidb_version_anchor_rewrite_supports_two_digit_versions(self):
        text = (
            "请参见 [`tidb_executor_concurrency`](/system-variables.md"
            "#tidb_executor_concurrency-new-in-v50)。"
        )

        with mock.patch.dict(os.environ, {"PRODUCT": "TiDB"}, clear=False):
            processed = rewrite_tidb_version_anchors_in_text(
                text,
                source_language="English",
                target_language="Chinese",
                source_mode="pr",
            )

        self.assertEqual(
            processed,
            "请参见 [`tidb_executor_concurrency`](/system-variables.md#tidb_executor_concurrency-从-v50-版本开始引入)。",
        )

    def test_tidb_version_anchor_rewrite_skips_commit_mode(self):
        text = (
            "请参见 [`tidb_enable_x`](/system-variables.md"
            "#tidb-enable-x-new-in-v800)。"
        )

        with mock.patch.dict(os.environ, {"PRODUCT": "TiDB"}, clear=False):
            processed = rewrite_tidb_version_anchors_in_text(
                text,
                source_language="English",
                target_language="Chinese",
                source_mode="commit",
            )

        self.assertEqual(processed, text)

    def test_tidb_version_anchor_rewrite_skips_other_products(self):
        text = (
            "See [`tidb_enable_x`](/system-variables.md"
            "#tidb-enable-x-从-v800-版本开始引入)."
        )

        with mock.patch.dict(os.environ, {"PRODUCT": "Other"}, clear=False):
            processed = rewrite_tidb_version_anchors_in_text(
                text,
                source_language="Chinese",
                target_language="English",
                source_mode="pr",
            )

        self.assertEqual(processed, text)

    def test_tidb_version_anchor_rewrite_is_disabled_when_product_is_unset(self):
        text = (
            "See [`tidb_enable_x`](/system-variables.md"
            "#tidb-enable-x-从-v800-版本开始引入)."
        )

        with mock.patch.dict(os.environ, {}, clear=True):
            processed = rewrite_tidb_version_anchors_in_text(
                text,
                source_language="Chinese",
                target_language="English",
                source_mode="pr",
            )

        self.assertEqual(processed, text)

    def test_tidb_version_anchor_rewrite_is_disabled_when_product_is_blank(self):
        text = (
            "See [`tidb_enable_x`](/system-variables.md"
            "#tidb-enable-x-从-v800-版本开始引入)."
        )

        with mock.patch.dict(os.environ, {"PRODUCT": ""}, clear=False):
            processed = rewrite_tidb_version_anchors_in_text(
                text,
                source_language="Chinese",
                target_language="English",
                source_mode="pr",
            )

        self.assertEqual(processed, text)

    def test_product_name_strips_whitespace_and_defaults_to_empty(self):
        with mock.patch.dict(
            os.environ,
            {"PRODUCT": "  TiDB Cloud  "},
            clear=False,
        ):
            self.assertEqual(get_product_name(), "TiDB Cloud")

        with mock.patch.dict(os.environ, {"PRODUCT": "   "}, clear=False):
            self.assertEqual(get_product_name(), "")

        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(get_product_name(), "")

    def test_tidb_version_anchor_rewrite_only_updates_markdown_link_urls(self):
        text = (
            "Anchor text #tidb-enable-x-从-v800-版本开始引入 and "
            "[`tidb_enable_x`](/system-variables.md#tidb-enable-x-从-v800-版本开始引入)."
        )

        with mock.patch.dict(os.environ, {"PRODUCT": "TiDB"}, clear=False):
            processed = rewrite_tidb_version_anchors_in_text(
                text,
                source_language="Chinese",
                target_language="English",
                source_mode="pr",
            )

        self.assertEqual(
            processed,
            "Anchor text #tidb-enable-x-从-v800-版本开始引入 and "
            "[`tidb_enable_x`](/system-variables.md#tidb-enable-x-new-in-v800).",
        )

    def test_tidb_version_anchor_rewrite_requires_markdown_file_anchor_url(self):
        text = (
            "[`tidb_enable_x`](https://example.com/docs#tidb-enable-x-从-v800-版本开始引入) "
            "and [`tidb_enable_y`](/system-variables#tidb-enable-y-从-v800-版本开始引入)."
        )

        with mock.patch.dict(os.environ, {"PRODUCT": "TiDB"}, clear=False):
            processed = rewrite_tidb_version_anchors_in_text(
                text,
                source_language="Chinese",
                target_language="English",
                source_mode="pr",
            )

        self.assertEqual(processed, text)

    def test_tidb_version_anchor_rewrite_skips_image_links(self):
        text = (
            "![alt](/system-variables.md#tidb-enable-x-从-v800-版本开始引入) "
            "[`tidb_enable_x`](/system-variables.md#tidb-enable-x-从-v800-版本开始引入)."
        )

        with mock.patch.dict(os.environ, {"PRODUCT": "TiDB"}, clear=False):
            processed = rewrite_tidb_version_anchors_in_text(
                text,
                source_language="Chinese",
                target_language="English",
                source_mode="pr",
            )

        self.assertEqual(
            processed,
            "![alt](/system-variables.md#tidb-enable-x-从-v800-版本开始引入) "
            "[`tidb_enable_x`](/system-variables.md#tidb-enable-x-new-in-v800).",
        )

    def test_tidb_version_anchor_rewrite_preserves_translation_result_failures(self):
        result = TranslationResult(
            {
                "modified_1": (
                    "See [`tidb_enable_x`](/system-variables.md"
                    "#tidb-enable-x-从-v800-版本开始引入)."
                )
            },
            failures=["chunk failed"],
        )

        with mock.patch.dict(os.environ, {"PRODUCT": "TiDB"}, clear=False):
            processed = rewrite_tidb_version_anchors_in_sections(
                result,
                source_language="Chinese",
                target_language="English",
                source_mode="pr",
            )

        self.assertIsInstance(processed, TranslationResult)
        self.assertEqual(processed.failures, ["chunk failed"])
        self.assertEqual(
            processed["modified_1"],
            "See [`tidb_enable_x`](/system-variables.md#tidb-enable-x-new-in-v800).",
        )

    def test_ai_translation_rewrites_tidb_version_anchor_in_pr_mode(self):
        class FakeAIClient:
            def chat_completion(self, messages, temperature=0.1):
                return json.dumps(
                    {
                        "modified_1": (
                            "See [`tidb_enable_x`](/system-variables.md"
                            "#tidb-enable-x-从-v800-版本开始引入)."
                        )
                    }
                )

        prefix = "tidb-anchor-unit"
        self._cleanup_chunk_test_outputs(prefix)
        try:
            with mock.patch.dict(os.environ, {"PRODUCT": "TiDB"}, clear=False):
                processed = get_updated_sections_from_ai(
                    "\n".join(
                        [
                            "File: system-variables.md",
                            "@@ -1,1 +1,1 @@",
                            "-old content",
                            "+new content",
                        ]
                    ),
                    {
                        "modified_1": "See [`tidb_enable_x`](/system-variables.md#tidb-enable-x-从-v800-版本开始引入).",
                    },
                    {
                        "modified_1": "See [`tidb_enable_x`](/system-variables.md#tidb-enable-x-从-v800-版本开始引入).",
                    },
                    FakeAIClient(),
                    "Chinese",
                    "English",
                    f"{prefix}.md",
                    source_mode="pr",
                )

            self.assertEqual(
                processed["modified_1"],
                "See [`tidb_enable_x`](/system-variables.md#tidb-enable-x-new-in-v800).",
            )
        finally:
            self._cleanup_chunk_test_outputs(prefix)

    def test_translation_prompt_preserves_mdx_component_tags(self):
        prompt, _ = _prepare_translation_prompt(
            "File: example.md\n@@ -1,1 +1,1 @@",
            {
                "added_10": '<CustomContent plan="premium">\n\n## Request units\n\nBody\n\n</CustomContent>',
            },
            {"added_10": ""},
            "English",
            "Chinese",
            "commit",
        )

        self.assertIn("Preserve HTML/MDX component tags exactly", prompt)
        self.assertIn('<CustomContent plan="premium">', prompt)
        self.assertIn("</CustomContent>", prompt)

    def test_translation_prompt_distinguishes_added_and_modified_sections(self):
        prompt, _ = _prepare_translation_prompt(
            "\n".join(
                [
                    "File: example.md",
                    "@@ -1,2 +1,5 @@",
                    "+#### New section",
                    " | Privilege | Scope |",
                    "-Old text.",
                    "+New text.",
                ]
            ),
            {
                "added_10": (
                    "#### New section\n\n"
                    "| Privilege | Scope |\n"
                    "|:--|:--|\n"
                    "| `SELECT` | Tables |\n"
                ),
                "modified_20": "### Existing section\n\nNew text.\n",
            },
            {
                "added_10": "",
                "modified_20": "### Existing section\n\nOld target text.\n",
            },
            "English",
            "Japanese",
            "commit",
        )

        core_principle = "CORE PRINCIPLE:"
        self.assertIn(core_principle, prompt)
        self.assertIn(
            'If a section key begins with "added_", classify it as an added section',
            prompt,
        )
        self.assertIn(
            "regardless of whether its current target section is empty",
            prompt,
        )
        self.assertIn(
            "The key prefix always takes precedence over target content",
            prompt,
        )
        self.assertIn(
            "Otherwise, classify it as a modified section",
            prompt,
        )
        self.assertIn(
            "Added section: Translate the complete latest source section into Japanese",
            prompt,
        )
        self.assertIn(
            'including natural-language content in unchanged context lines that do not begin with "+"',
            prompt,
        )
        self.assertIn(
            "Modified section: Treat the Git diff as the ONLY source of truth",
            prompt,
        )
        self.assertIn(
            "Continue to follow the strict diff-only rules below",
            prompt,
        )
        self.assertIn(
            "Translate added sections completely according to the added-section rule in the CORE PRINCIPLE",
            prompt,
        )
        self.assertIn(
            "Apply minimal edits in Japanese according to specifically changed lines in English",
            prompt,
        )
        self.assertNotIn(
            "Apply minimal edits in Japanese according to specifically changed lines in Japanese",
            prompt,
        )
        operation_detection = "First determine the operation type of each section:"
        operation_rules = "Then apply the corresponding rule:"
        self.assertLess(prompt.index(core_principle), prompt.index(operation_detection))
        self.assertLess(prompt.index(operation_detection), prompt.index(operation_rules))
        self.assertLess(prompt.index(operation_rules), prompt.index("Instructions:"))

    def test_translation_prompt_uses_configured_product_name(self):
        with mock.patch.dict(os.environ, {"PRODUCT": "ExampleDB"}, clear=False):
            prompt, _ = _prepare_translation_prompt(
                "File: example.md\n@@ -1,1 +1,1 @@\n-old\n+new",
                {"modified_1": "New source content."},
                {"modified_1": "Existing target content."},
                "English",
                "Japanese",
                "commit",
            )

        self.assertIn("ExampleDB user documentation is maintained", prompt)
        self.assertIn("**Project** page", prompt)
        self.assertNotIn("TiDB user documentation is maintained", prompt)

    def test_translation_prompt_uses_product_neutral_wording_when_unset(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            prompt, _ = _prepare_translation_prompt(
                "File: example.md\n@@ -1,1 +1,1 @@\n-old\n+new",
                {"modified_1": "New source content."},
                {"modified_1": "Existing target content."},
                "English",
                "Japanese",
                "commit",
            )

        self.assertIn("User documentation is maintained", prompt)
        self.assertNotIn("TiDB user documentation", prompt)
        self.assertNotIn("\n user documentation", prompt)

    def test_added_section_key_takes_precedence_over_nonempty_target_content(self):
        prompt, _ = _prepare_translation_prompt(
            "File: example.md\n@@ -1,0 +1,1 @@\n+## New section",
            {"added_1": "## New section\n\nNew source content.\n"},
            {"added_1": "## Existing target section\n\nStale content.\n"},
            "English",
            "Japanese",
            "commit",
        )

        self.assertIn(
            "The key prefix always takes precedence over target content",
            prompt,
        )
        self.assertIn('"added_1": "## Existing target section', prompt)

    def test_prompt_applies_the_same_changed_heading_anchor_to_source_and_diff(self):
        pr_diff = "\n".join(
            [
                "File: tidb-cloud/tidb-cloud-auditing.md",
                "@@ -40,1 +40,1 @@",
                "-## Auditing filter events",
                "+## Audit filter events",
            ]
        )

        prompt, prompt_diff = _prepare_translation_prompt(
            pr_diff,
            {"modified_40": "## Audit filter events\n\nNew content.\n"},
            {"modified_40": "## 監査フィルターイベント\n\n古い内容。\n"},
            "English",
            "Japanese",
            "commit",
        )

        expected_heading = "## Audit filter events {#audit-filter-events}"
        self.assertIn(f"+{expected_heading}", prompt_diff)
        self.assertIn(expected_heading, prompt)

    def test_source_anchor_preprocessing_skips_fenced_heading_examples(self):
        prompt_diff = "\n".join(
            [
                "File: guide.md",
                "@@ -1,1 +1,1 @@",
                "+## Real heading {#real-heading}",
                "+```md",
                "+## Code heading {#code-heading}",
                "+```",
            ]
        )

        processed = preprocess_source_sections_for_heading_anchor_stability(
            {
                "modified_1": (
                    "## Real heading\n\n"
                    "```md\n"
                    "## Code heading\n"
                    "```\n"
                )
            },
            prompt_diff,
        )

        self.assertIn("## Real heading {#real-heading}", processed["modified_1"])
        self.assertIn("## Code heading\n", processed["modified_1"])
        self.assertNotIn("## Code heading {#code-heading}", processed["modified_1"])

    def test_prompt_diff_uses_source_section_to_reject_mid_fence_heading(self):
        pr_diff = "\n".join(
            [
                "File: guide.md",
                "@@ -20,1 +20,1 @@",
                "-## Old code example",
                "+## New code example",
            ]
        )
        source_sections = {
            "modified_10": (
                "## Real section\n\n"
                "```md\n"
                "## New code example\n"
                "```\n"
            )
        }

        prompt, prompt_diff = _prepare_translation_prompt(
            pr_diff,
            source_sections,
            {
                "modified_10": (
                    "## 実際のセクション\n\n"
                    "```md\n"
                    "## 新しいコード例\n"
                    "```\n"
                )
            },
            "English",
            "Japanese",
            "commit",
        )

        self.assertIn("+## New code example", prompt_diff)
        self.assertNotIn("{#new-code-example}", prompt_diff)
        self.assertNotIn("{#new-code-example}", prompt)

    def test_ai_output_heading_anchor_is_restored_programmatically(self):
        class FakeAIClient:
            def __init__(self):
                self.prompts = []

            def chat_completion(self, messages, temperature=0.1):
                prompt = messages[0]["content"]
                self.prompts.append(prompt)
                return json.dumps(
                    {
                        "modified_40": (
                            "## 監査フィルターイベント\n\n新しい内容。\n"
                        )
                    }
                )

        prefix = "restore-heading-anchor-unit"
        self._cleanup_chunk_test_outputs(prefix)
        try:
            ai_client = FakeAIClient()
            result = get_updated_sections_from_ai(
                "\n".join(
                    [
                        "File: tidb-cloud/tidb-cloud-auditing.md",
                        "@@ -40,1 +40,1 @@",
                        "-## Auditing filter events",
                        "+## Audit filter events",
                    ]
                ),
                {
                    "modified_40": (
                        "## 監査イベント "
                        "{#auditing-filter-events}\n\n古い内容。\n"
                    )
                },
                {
                    "modified_40": (
                        "## Audit filter events\n\nNew content.\n"
                    )
                },
                ai_client,
                "English",
                "Japanese",
                f"{prefix}.md",
                source_mode="commit",
            )

            self.assertIn(
                "## 監査フィルターイベント {#audit-filter-events}",
                result["modified_40"],
            )
            self.assertNotIn("{#auditing-filter-events}", result["modified_40"])
            self.assertEqual(len(ai_client.prompts), 1)
            self.assertFalse(result.partial_reasons)
        finally:
            self._cleanup_chunk_test_outputs(prefix)

    def test_unchanged_main_ai_h1_is_repaired_by_targeted_translation(self):
        class FakeAIClient:
            def __init__(self):
                self.prompts = []

            def chat_completion(self, messages, temperature=0.1):
                prompt = messages[0]["content"]
                self.prompts.append(prompt)
                if "Translate only the changed Markdown headings" in prompt:
                    return json.dumps(
                        {
                            "heading_001": {
                                "status": "updated",
                                "heading": (
                                    "# TiDB Cloud Dedicatedデータベース監査ログ"
                                    "（プレビュー版） "
                                    "{#ai-must-not-control-anchor}"
                                ),
                            }
                        }
                    )
                return json.dumps(
                    {
                        "intro_section": (
                            "# TiDB Cloud Dedicatedデータベース監査ログ\n\n"
                            "更新された導入文。\n"
                        )
                    }
                )

        prefix = "unchanged-modified-h1-unit"
        self._cleanup_chunk_test_outputs(prefix)
        try:
            ai_client = FakeAIClient()

            def glossary_matcher(text, source_language=None):
                if "Preview" not in text:
                    return []
                return [
                    {
                        "en": "Preview",
                        "target": "プレビュー版",
                        "comment": "Use the established Japanese label.",
                    }
                ]

            result = get_updated_sections_from_ai(
                "\n".join(
                    [
                        "File: tidb-cloud/tidb-cloud-auditing.md",
                        "@@ -8,1 +8,1 @@",
                        "-# TiDB Cloud Dedicated database audit logging "
                        "{#tidb-cloud-dedicated-database-audit-logging}",
                        "+# TiDB Cloud Dedicated database audit logging (Preview) "
                        "{#tidb-cloud-dedicated-database-audit-logging}",
                    ]
                ),
                {
                    "intro_section": (
                        "# TiDB Cloud Dedicatedデータベース監査ログ\n\n"
                        "古い導入文。\n"
                    )
                },
                {
                    "intro_section": (
                        "# TiDB Cloud Dedicated database audit logging (Preview) "
                        "{#tidb-cloud-dedicated-database-audit-logging}\n\n"
                        "Updated introduction.\n"
                    )
                },
                ai_client,
                "English",
                "Japanese",
                f"{prefix}.md",
                glossary_matcher=glossary_matcher,
                source_mode="commit",
            )

            self.assertIn(
                "# TiDB Cloud Dedicatedデータベース監査ログ（プレビュー版） "
                "{#tidb-cloud-dedicated-database-audit-logging}",
                result["intro_section"],
            )
            self.assertNotIn(
                "{#ai-must-not-control-anchor}",
                result["intro_section"],
            )
            self.assertEqual(len(ai_client.prompts), 2)
            heading_prompt = ai_client.prompts[1]
            self.assertIn(
                '"old_source_heading": "# TiDB Cloud Dedicated database '
                'audit logging"',
                heading_prompt,
            )
            self.assertIn(
                '"new_source_heading": "# TiDB Cloud Dedicated database '
                'audit logging (Preview)"',
                heading_prompt,
            )
            self.assertIn(
                '"old_target_heading": "# TiDB Cloud '
                'Dedicatedデータベース監査ログ"',
                heading_prompt,
            )
            self.assertNotIn("更新された導入文", heading_prompt)
            self.assertIn("| Preview | プレビュー版 |", heading_prompt)
            self.assertFalse(result.partial_reasons)
        finally:
            self._cleanup_chunk_test_outputs(prefix)

    def test_unchanged_targeted_h1_translation_remains_partial(self):
        class FakeAIClient:
            def chat_completion(self, messages, temperature=0.1):
                prompt = messages[0]["content"]
                if "Translate only the changed Markdown headings" in prompt:
                    return json.dumps(
                        {
                            "heading_001": {
                                "status": "updated",
                                "heading": (
                                    "# TiDB Cloud Dedicatedデータベース監査ログ"
                                ),
                            }
                        }
                    )
                return json.dumps(
                    {
                        "intro_section": (
                            "# TiDB Cloud Dedicatedデータベース監査ログ\n\n"
                            "更新された導入文。\n"
                        )
                    }
                )

        prefix = "unchanged-targeted-h1-unit"
        self._cleanup_chunk_test_outputs(prefix)
        try:
            result = get_updated_sections_from_ai(
                "\n".join(
                    [
                        "File: tidb-cloud/tidb-cloud-auditing.md",
                        "@@ -8,1 +8,1 @@",
                        "-# TiDB Cloud Dedicated database audit logging",
                        "+# TiDB Cloud Dedicated database audit logging (Preview)",
                    ]
                ),
                {
                    "intro_section": (
                        "# TiDB Cloud Dedicatedデータベース監査ログ "
                        "{#tidb-cloud-dedicated-database-audit-logging}\n\n"
                        "古い導入文。\n"
                    )
                },
                {
                    "intro_section": (
                        "# TiDB Cloud Dedicated database audit logging "
                        "(Preview)\n\nUpdated introduction.\n"
                    )
                },
                FakeAIClient(),
                "English",
                "Japanese",
                f"{prefix}.md",
                source_mode="commit",
            )

            self.assertTrue(
                any(
                    "targeted heading translation stayed unchanged"
                    in reason
                    for reason in result.partial_reasons
                )
            )
            self.assertTrue(
                any(
                    "changed source H1 produced an unchanged target H1"
                    in reason
                    for reason in result.partial_reasons
                )
            )
            self.assertIn(
                "# TiDB Cloud Dedicatedデータベース監査ログ "
                "{#tidb-cloud-dedicated-database-audit-logging}",
                result["intro_section"],
            )
        finally:
            self._cleanup_chunk_test_outputs(prefix)

    def test_targeted_h1_can_confirm_existing_translation_is_equivalent(self):
        class FakeAIClient:
            def __init__(self):
                self.prompts = []

            def chat_completion(self, messages, temperature=0.1):
                prompt = messages[0]["content"]
                self.prompts.append(prompt)
                if "Translate only the changed Markdown headings" in prompt:
                    return json.dumps(
                        {
                            "heading_001": {
                                "status": "already_equivalent",
                                "heading": "# 監査ログを設定する",
                            }
                        }
                    )
                return json.dumps(
                    {
                        "intro_section": (
                            "# 監査ログを設定する\n\n更新された導入文。\n"
                        )
                    }
                )

        prefix = "equivalent-modified-h1-unit"
        self._cleanup_chunk_test_outputs(prefix)
        try:
            ai_client = FakeAIClient()
            result = get_updated_sections_from_ai(
                "\n".join(
                    [
                        "File: guide.md",
                        "@@ -1,1 +1,1 @@",
                        "-# Configure auditing logging",
                        "+# Configure audit logging",
                    ]
                ),
                {
                    "intro_section": (
                        "# 監査ログを設定する {#configure-auditing-logging}"
                        "\n\n古い導入文。\n"
                    )
                },
                {
                    "intro_section": (
                        "# Configure audit logging\n\nUpdated introduction.\n"
                    )
                },
                ai_client,
                "English",
                "Japanese",
                f"{prefix}.md",
                source_mode="commit",
            )

            self.assertEqual(len(ai_client.prompts), 2)
            self.assertIn(
                "# 監査ログを設定する {#configure-auditing-logging}",
                result["intro_section"],
            )
            self.assertFalse(result.partial_reasons)
        finally:
            self._cleanup_chunk_test_outputs(prefix)

    def test_missing_changed_h1_is_inserted_by_targeted_translation(self):
        class FakeAIClient:
            def __init__(self):
                self.prompts = []

            def chat_completion(self, messages, temperature=0.1):
                prompt = messages[0]["content"]
                self.prompts.append(prompt)
                if "Translate only the changed Markdown headings" in prompt:
                    return json.dumps(
                        {
                            "heading_001": {
                                "status": "updated",
                                "heading": "# 新しいタイトル {#wrong-anchor}",
                            }
                        }
                    )
                return json.dumps(
                    {"intro_section": "更新された導入文。\n"}
                )

        prefix = "missing-modified-h1-unit"
        self._cleanup_chunk_test_outputs(prefix)
        try:
            ai_client = FakeAIClient()
            result = get_updated_sections_from_ai(
                "\n".join(
                    [
                        "File: guide.md",
                        "@@ -1,1 +1,1 @@",
                        "-# Old title {#stable-title}",
                        "+# New title {#stable-title}",
                    ]
                ),
                {
                    "intro_section": (
                        "# 古いタイトル {#stable-title}\n\n古い導入文。\n"
                    )
                },
                {
                    "intro_section": (
                        "# New title {#stable-title}\n\n"
                        "Updated introduction.\n"
                    )
                },
                ai_client,
                "English",
                "Japanese",
                f"{prefix}.md",
                source_mode="commit",
            )

            self.assertEqual(len(ai_client.prompts), 2)
            self.assertTrue(
                result["intro_section"].startswith(
                    "# 新しいタイトル {#stable-title}\n\n"
                )
            )
            self.assertNotIn("{#wrong-anchor}", result["intro_section"])
            self.assertFalse(result.partial_reasons)
        finally:
            self._cleanup_chunk_test_outputs(prefix)

    def test_heading_retry_does_not_fabricate_section_missing_from_main_response(self):
        class FakeAIClient:
            def __init__(self):
                self.prompts = []

            def chat_completion(self, messages, temperature=0.1):
                self.prompts.append(messages[0]["content"])
                return json.dumps({})

        prefix = "missing-main-section-unit"
        self._cleanup_chunk_test_outputs(prefix)
        try:
            ai_client = FakeAIClient()
            result = get_updated_sections_from_ai(
                "\n".join(
                    [
                        "File: guide.md",
                        "@@ -1,1 +1,1 @@",
                        "-# Old title",
                        "+# New title",
                    ]
                ),
                {
                    "intro_section": (
                        "# 古いタイトル\n\n保持すべき本文。\n"
                    )
                },
                {
                    "intro_section": (
                        "# New title\n\nUpdated body.\n"
                    )
                },
                ai_client,
                "English",
                "Japanese",
                f"{prefix}.md",
                source_mode="commit",
            )

            self.assertEqual(1, len(ai_client.prompts))
            self.assertNotIn("intro_section", result)
            self.assertTrue(
                any(
                    "AI response missing section keys: intro_section" in reason
                    for reason in result.partial_reasons
                )
            )
        finally:
            self._cleanup_chunk_test_outputs(prefix)

    def test_missing_heading_inside_custom_content_is_not_moved_outside(self):
        class FakeAIClient:
            def chat_completion(self, messages, temperature=0.1):
                prompt = messages[0]["content"]
                if "Translate only the changed Markdown headings" in prompt:
                    return json.dumps(
                        {
                            "heading_001": {
                                "status": "updated",
                                "heading": "# 新しいタイトル",
                            }
                        }
                    )
                return json.dumps(
                    {
                        "intro_section": (
                            '<CustomContent plan="premium">\n'
                            "更新された本文。\n"
                            "</CustomContent>\n"
                        )
                    }
                )

        prefix = "custom-content-heading-boundary-unit"
        self._cleanup_chunk_test_outputs(prefix)
        try:
            target_content = (
                '<CustomContent plan="premium">\n'
                "# 古いタイトル\n"
                "古い本文。\n"
                "</CustomContent>\n"
            )
            result = get_updated_sections_from_ai(
                "\n".join(
                    [
                        "File: guide.md",
                        "@@ -2,1 +2,1 @@",
                        "-# Old title",
                        "+# New title",
                    ]
                ),
                {"intro_section": target_content},
                {
                    "intro_section": (
                        '<CustomContent plan="premium">\n'
                        "# New title\n"
                        "Updated body.\n"
                        "</CustomContent>\n"
                    )
                },
                FakeAIClient(),
                "English",
                "Japanese",
                f"{prefix}.md",
                source_mode="commit",
            )

            self.assertEqual(target_content, result["intro_section"])
            self.assertTrue(
                any(
                    "could not be applied" in reason
                    for reason in result.partial_reasons
                )
            )
        finally:
            self._cleanup_chunk_test_outputs(prefix)

    def test_added_technical_heading_can_be_already_equivalent(self):
        class FakeAIClient:
            def __init__(self):
                self.prompts = []

            def chat_completion(self, messages, temperature=0.1):
                prompt = messages[0]["content"]
                self.prompts.append(prompt)
                if "Translate only the changed Markdown headings" in prompt:
                    return json.dumps(
                        {
                            "heading_001": {
                                "status": "already_equivalent",
                                "heading": "## TiDB Cloud",
                            }
                        }
                    )
                return json.dumps(
                    {
                        "added_3": (
                            "## TiDB Cloud\n\nTiDB Cloud の説明。\n"
                        )
                    }
                )

        prefix = "technical-heading-equivalent-unit"
        self._cleanup_chunk_test_outputs(prefix)
        try:
            ai_client = FakeAIClient()
            result = get_updated_sections_from_ai(
                "\n".join(
                    [
                        "File: guide.md",
                        "@@ -3,0 +3,2 @@",
                        "+## TiDB Cloud",
                        "+",
                    ]
                ),
                {"added_3": ""},
                {
                    "added_3": "## TiDB Cloud\n\nDescription.\n",
                },
                ai_client,
                "English",
                "Japanese",
                f"{prefix}.md",
                source_mode="commit",
            )

            self.assertEqual(2, len(ai_client.prompts))
            self.assertIn(
                "## TiDB Cloud {#tidb-cloud}",
                result["added_3"],
            )
            self.assertFalse(result.partial_reasons)
        finally:
            self._cleanup_chunk_test_outputs(prefix)

    def test_anchor_only_h1_change_is_not_reported_as_untranslated(self):
        class FakeAIClient:
            def __init__(self):
                self.prompts = []

            def chat_completion(self, messages, temperature=0.1):
                self.prompts.append(messages[0]["content"])
                return json.dumps(
                    {
                        "intro_section": (
                            "# ガイド {#old-guide}\n\n更新された本文。\n"
                        )
                    }
                )

        prefix = "anchor-only-h1-unit"
        self._cleanup_chunk_test_outputs(prefix)
        try:
            ai_client = FakeAIClient()
            result = get_updated_sections_from_ai(
                "\n".join(
                    [
                        "File: guide.md",
                        "@@ -1,1 +1,1 @@",
                        "-# Guide {#old-guide}",
                        "+# Guide {#new-guide}",
                    ]
                ),
                {
                    "intro_section": (
                        "# ガイド {#old-guide}\n\n古い本文。\n"
                    )
                },
                {
                    "intro_section": (
                        "# Guide {#new-guide}\n\nUpdated body.\n"
                    )
                },
                ai_client,
                "English",
                "Japanese",
                f"{prefix}.md",
                source_mode="commit",
            )

            self.assertEqual(1, len(ai_client.prompts))
            self.assertIn("# ガイド {#old-guide}", result["intro_section"])
            self.assertFalse(result.partial_reasons)
        finally:
            self._cleanup_chunk_test_outputs(prefix)

    def test_ambiguous_multi_heading_repair_preserves_main_output(self):
        class FakeAIClient:
            def __init__(self):
                self.prompts = []

            def chat_completion(self, messages, temperature=0.1):
                self.prompts.append(messages[0]["content"])
                return json.dumps(
                    {
                        "modified_10": (
                            "本文。\n\n### 新しい子見出し\n\n子本文。\n"
                        )
                    }
                )

        prefix = "ambiguous-multi-heading-unit"
        self._cleanup_chunk_test_outputs(prefix)
        try:
            ai_client = FakeAIClient()
            result = get_updated_sections_from_ai(
                "\n".join(
                    [
                        "File: guide.md",
                        "@@ -10,2 +10,2 @@",
                        "-## Old parent",
                        "+## New parent",
                        "-### Old child",
                        "+### New child",
                    ]
                ),
                {
                    "modified_10": (
                        "## 古い親\n\n本文。\n\n"
                        "### 古い子見出し\n\n子本文。\n"
                    )
                },
                {
                    "modified_10": (
                        "## New parent\n\nBody.\n\n"
                        "### New child\n\nChild body.\n"
                    )
                },
                ai_client,
                "English",
                "Japanese",
                f"{prefix}.md",
                source_mode="commit",
            )

            self.assertEqual(1, len(ai_client.prompts))
            self.assertNotIn("## 新しい親", result["modified_10"])
            self.assertIn("### 新しい子見出し", result["modified_10"])
            self.assertTrue(
                any(
                    "ambiguous for multi-heading section modified_10" in reason
                    for reason in result.partial_reasons
                )
            )
        finally:
            self._cleanup_chunk_test_outputs(prefix)

    def test_added_headings_are_translated_in_one_targeted_request(self):
        source_sections = {
            "added_10": "### SQL statement information\n\nSQL body.\n",
            "added_20": "### Connection information\n\nConnection body.\n",
            "added_30": (
                "### Audit operation information\n\nAudit operation body.\n"
            ),
            "added_40": "## Audit logging limitations\n\nLimit body.\n",
            "added_50": (
                "## Legacy database audit logging reference\n\nLegacy body.\n"
            ),
        }
        target_sections = {key: "" for key in source_sections}

        class FakeAIClient:
            def __init__(self):
                self.prompts = []

            def chat_completion(self, messages, temperature=0.1):
                prompt = messages[0]["content"]
                self.prompts.append(prompt)
                if "Translate only the changed Markdown headings" in prompt:
                    return json.dumps(
                        {
                            "heading_001": {
                                "status": "updated",
                                "heading": "### SQL ステートメント情報",
                            },
                            "heading_002": {
                                "status": "updated",
                                "heading": "### 接続情報",
                            },
                            "heading_003": {
                                "status": "updated",
                                "heading": "### 監査操作情報",
                            },
                            "heading_004": {
                                "status": "updated",
                                "heading": "## 監査ログの制限事項",
                            },
                            "heading_005": {
                                "status": "updated",
                                "heading": (
                                    "## 従来のデータベース監査ログのリファレンス"
                                ),
                            },
                        }
                    )
                return json.dumps(source_sections)

        prefix = "added-heading-translation-unit"
        self._cleanup_chunk_test_outputs(prefix)
        try:
            ai_client = FakeAIClient()
            result = get_updated_sections_from_ai(
                "\n".join(
                    [
                        "File: guide.md",
                        "@@ -10,0 +10,2 @@",
                        "+### SQL statement information",
                        "+",
                        "@@ -20,0 +20,2 @@",
                        "+### Connection information",
                        "+",
                        "@@ -30,0 +30,2 @@",
                        "+### Audit operation information",
                        "+",
                        "@@ -40,0 +40,2 @@",
                        "+## Audit logging limitations",
                        "+",
                        "@@ -50,0 +50,2 @@",
                        "+## Legacy database audit logging reference",
                        "+",
                    ]
                ),
                target_sections,
                source_sections,
                ai_client,
                "English",
                "Japanese",
                f"{prefix}.md",
                source_mode="commit",
            )

            self.assertEqual(len(ai_client.prompts), 2)
            self.assertIn(
                "### SQL ステートメント情報 {#sql-statement-information}",
                result["added_10"],
            )
            self.assertIn(
                "### 接続情報 {#connection-information}",
                result["added_20"],
            )
            self.assertIn(
                "### 監査操作情報 {#audit-operation-information}",
                result["added_30"],
            )
            self.assertIn(
                "## 監査ログの制限事項 {#audit-logging-limitations}",
                result["added_40"],
            )
            self.assertIn(
                "## 従来のデータベース監査ログのリファレンス "
                "{#legacy-database-audit-logging-reference}",
                result["added_50"],
            )
            heading_prompt = ai_client.prompts[1]
            self.assertNotIn("SQL body", heading_prompt)
            self.assertNotIn("Connection body", heading_prompt)
            self.assertNotIn("Audit operation body", heading_prompt)
            self.assertNotIn("Limit body", heading_prompt)
            self.assertNotIn("Legacy body", heading_prompt)
            self.assertFalse(result.partial_reasons)
        finally:
            self._cleanup_chunk_test_outputs(prefix)

    def test_heading_translation_ignores_same_hunk_sections_outside_chunk(self):
        class FakeAIClient:
            def chat_completion(self, messages, temperature=0.1):
                prompt = messages[0]["content"]
                if "Translate only the changed Markdown headings" in prompt:
                    return json.dumps(
                        {
                            "heading_001": {
                                "status": "updated",
                                "heading": "## 2 番目の新しい章",
                            }
                        }
                    )
                return json.dumps(
                    {"added_20": "## Second new section\n\nSecond body.\n"}
                )

        prefix = "heading-hunk-chunk-scope-unit"
        self._cleanup_chunk_test_outputs(prefix)
        try:
            result = get_updated_sections_from_ai(
                "\n".join(
                    [
                        "File: guide.md",
                        "@@ -10,0 +10,13 @@",
                        "+## First new section",
                        "+",
                        "+First body.",
                        "+",
                        "+## Second new section",
                        "+",
                        "+Second body.",
                    ]
                ),
                {"added_20": ""},
                {
                    "added_20": (
                        "## Second new section\n\nSecond body.\n"
                    )
                },
                FakeAIClient(),
                "English",
                "Japanese",
                f"{prefix}.md",
                source_mode="commit",
            )

            self.assertIn(
                "## 2 番目の新しい章 {#second-new-section}",
                result["added_20"],
            )
            self.assertFalse(
                any(
                    "First new section" in reason
                    for reason in result.partial_reasons
                )
            )
        finally:
            self._cleanup_chunk_test_outputs(prefix)

    def test_targeted_heading_translation_rejects_wrong_level(self):
        class FakeAIClient:
            def chat_completion(self, messages, temperature=0.1):
                prompt = messages[0]["content"]
                if "Translate only the changed Markdown headings" in prompt:
                    return json.dumps(
                        {
                            "heading_001": {
                                "status": "updated",
                                "heading": "## 間違ったレベル",
                            }
                        }
                    )
                return json.dumps(
                    {
                        "intro_section": (
                            "# 古いタイトル\n\n更新された導入文。\n"
                        )
                    }
                )

        prefix = "wrong-heading-level-unit"
        self._cleanup_chunk_test_outputs(prefix)
        try:
            result = get_updated_sections_from_ai(
                "\n".join(
                    [
                        "File: guide.md",
                        "@@ -1,1 +1,1 @@",
                        "-# Old title",
                        "+# New title",
                    ]
                ),
                {"intro_section": "# 古いタイトル\n\n古い導入文。\n"},
                {"intro_section": "# New title\n\nNew introduction.\n"},
                FakeAIClient(),
                "English",
                "Japanese",
                f"{prefix}.md",
                source_mode="commit",
            )

            self.assertTrue(
                any(
                    "changed heading level" in reason
                    for reason in result.partial_reasons
                )
            )
            self.assertTrue(
                result["intro_section"].startswith("# 古いタイトル\n")
            )
        finally:
            self._cleanup_chunk_test_outputs(prefix)

    def test_targeted_heading_translation_rejects_source_language_title(self):
        class FakeAIClient:
            def chat_completion(self, messages, temperature=0.1):
                prompt = messages[0]["content"]
                if "Translate only the changed Markdown headings" in prompt:
                    return json.dumps(
                        {
                            "heading_001": {
                                "status": "updated",
                                "heading": "# New title",
                            }
                        }
                    )
                return json.dumps(
                    {"intro_section": "# 古いタイトル\n\n更新された導入文。\n"}
                )

        prefix = "source-language-heading-unit"
        self._cleanup_chunk_test_outputs(prefix)
        try:
            result = get_updated_sections_from_ai(
                "\n".join(
                    [
                        "File: guide.md",
                        "@@ -1,1 +1,1 @@",
                        "-# Old title",
                        "+# New title",
                    ]
                ),
                {"intro_section": "# 古いタイトル\n\n古い導入文。\n"},
                {"intro_section": "# New title\n\nNew introduction.\n"},
                FakeAIClient(),
                "English",
                "Japanese",
                f"{prefix}.md",
                source_mode="commit",
            )

            self.assertTrue(
                any(
                    "left heading in the source language" in reason
                    for reason in result.partial_reasons
                )
            )
            self.assertTrue(
                result["intro_section"].startswith("# 古いタイトル\n")
            )
        finally:
            self._cleanup_chunk_test_outputs(prefix)

    def test_old_source_language_heading_is_retried(self):
        class FakeAIClient:
            def __init__(self):
                self.prompts = []

            def chat_completion(self, messages, temperature=0.1):
                prompt = messages[0]["content"]
                self.prompts.append(prompt)
                if "Translate only the changed Markdown headings" in prompt:
                    return json.dumps(
                        {
                            "heading_001": {
                                "status": "updated",
                                "heading": "# 新しいタイトル",
                            }
                        }
                    )
                return json.dumps(
                    {"intro_section": "# Old title\n\n更新された導入文。\n"}
                )

        prefix = "old-source-language-heading-unit"
        self._cleanup_chunk_test_outputs(prefix)
        try:
            ai_client = FakeAIClient()
            result = get_updated_sections_from_ai(
                "\n".join(
                    [
                        "File: guide.md",
                        "@@ -1,1 +1,1 @@",
                        "-# Old title",
                        "+# New title",
                    ]
                ),
                {"intro_section": "# 古いタイトル\n\n古い導入文。\n"},
                {"intro_section": "# New title\n\nNew introduction.\n"},
                ai_client,
                "English",
                "Japanese",
                f"{prefix}.md",
                source_mode="commit",
            )

            self.assertEqual(len(ai_client.prompts), 2)
            self.assertTrue(
                result["intro_section"].startswith("# 新しいタイトル\n")
            )
            self.assertFalse(result.partial_reasons)
        finally:
            self._cleanup_chunk_test_outputs(prefix)

    def test_preprocess_diff_keeps_existing_explicit_anchor(self):
        pr_diff = "\n".join(
            [
                "File: ai/example.md",
                "@@ -10,1 +10,1 @@",
                "+## {{{ .starter }}} {#starter}",
                "-" * 80,
            ]
        )

        processed = preprocess_diff_for_heading_anchor_stability(
            pr_diff,
            source_language="English",
            target_language="Chinese",
            source_mode="commit",
        )

        self.assertIn("+## {{{ .starter }}} {#starter}", processed)
        self.assertEqual(processed.count("{#starter}"), 1)

    def test_preprocess_diff_is_disabled_for_pr_mode(self):
        pr_diff = "\n".join(
            [
                "File: ai/example.md",
                "@@ -10,1 +10,1 @@",
                "+## Example test",
                "-" * 80,
            ]
        )

        processed = preprocess_diff_for_heading_anchor_stability(
            pr_diff,
            source_language="English",
            target_language="Chinese",
            source_mode="pr",
        )

        self.assertEqual(processed, pr_diff)

    def test_preprocess_diff_rewrites_tidb_cloud_links_in_pr_mode(self):
        pr_diff = "\n".join(
            [
                "File: ai/example.md",
                "@@ -10,1 +10,1 @@",
                "+See [Private Endpoints](/tidb-cloud/test/set-up-private-endpoint-connections-serverless3.md#examples).",
                "-" * 80,
            ]
        )

        processed = preprocess_diff_for_heading_anchor_stability(
            pr_diff,
            source_language="English",
            target_language="Chinese",
            source_mode="pr",
        )

        self.assertIn(
            "+See [Private Endpoints](https://docs.pingcap.com/tidbcloud/set-up-private-endpoint-connections-serverless3#examples).",
            processed,
        )

    def test_preprocess_diff_does_not_add_anchor_for_heading_level_only_change(self):
        pr_diff = "\n".join(
            [
                "File: ai/example.md",
                "@@ -10,1 +10,1 @@",
                "-## Example test",
                "+### Example test",
                "-" * 80,
            ]
        )

        processed = preprocess_diff_for_heading_anchor_stability(
            pr_diff,
            source_language="English",
            target_language="Chinese",
            source_mode="commit",
        )

        self.assertEqual(processed, pr_diff)

    def test_preprocess_diff_adds_anchor_to_newly_added_heading(self):
        pr_diff = "\n".join(
            [
                "File: tidb-cloud/releases/tidb-cloud-release-notes.md",
                "@@ -5,0 +5,3 @@",
                "+## June 16, 2026",
                "+",
                "+Some new release content.",
                "-" * 80,
            ]
        )

        processed = preprocess_diff_for_heading_anchor_stability(
            pr_diff,
            source_language="English",
            target_language="Japanese",
            source_mode="commit",
        )

        self.assertIn("+## June 16, 2026 {#june-16-2026}", processed)

    def test_preprocess_diff_adds_anchor_to_added_heading_after_non_heading_removal(self):
        pr_diff = "\n".join(
            [
                "File: ai/example.md",
                "@@ -10,1 +10,2 @@",
                "-Some old text",
                "+## New Section",
                "+New content here.",
                "-" * 80,
            ]
        )

        processed = preprocess_diff_for_heading_anchor_stability(
            pr_diff,
            source_language="English",
            target_language="Chinese",
            source_mode="commit",
        )

        self.assertIn("+## New Section {#new-section}", processed)

    def test_preprocess_diff_does_not_add_anchor_to_added_top_level_heading(self):
        pr_diff = "\n".join(
            [
                "File: ai/example.md",
                "@@ -1,0 +1,2 @@",
                "+# Top Level Title",
                "+Some content.",
                "-" * 80,
            ]
        )

        processed = preprocess_diff_for_heading_anchor_stability(
            pr_diff,
            source_language="English",
            target_language="Chinese",
            source_mode="commit",
        )

        self.assertNotIn("{#", processed)

    def test_preprocess_diff_skips_heading_inside_fenced_code_block(self):
        pr_diff = "\n".join(
            [
                "File: ai/example.md",
                "@@ -5,0 +5,5 @@",
                "+```markdown",
                "+## This is a code example",
                "+",
                "+Some content inside code block.",
                "+```",
                "-" * 80,
            ]
        )

        processed = preprocess_diff_for_heading_anchor_stability(
            pr_diff,
            source_language="English",
            target_language="Japanese",
            source_mode="commit",
        )

        self.assertNotIn("{#", processed)
        self.assertIn("+## This is a code example", processed)

    def test_preprocess_diff_adds_anchor_after_code_block_ends(self):
        pr_diff = "\n".join(
            [
                "File: ai/example.md",
                "@@ -5,0 +5,6 @@",
                "+```",
                "+## Inside code block",
                "+```",
                "+## Real heading outside",
                "+",
                "+Content here.",
                "-" * 80,
            ]
        )

        processed = preprocess_diff_for_heading_anchor_stability(
            pr_diff,
            source_language="English",
            target_language="Japanese",
            source_mode="commit",
        )

        self.assertNotIn("+## Inside code block {#", processed)
        self.assertIn("+## Real heading outside {#real-heading-outside}", processed)

    def test_preprocess_diff_skips_heading_inside_context_code_block(self):
        """Context lines (no +/-) can open a code block that spans added lines."""
        pr_diff = "\n".join(
            [
                "File: ai/example.md",
                "@@ -5,3 +5,4 @@",
                " ```",
                "+## Heading inside existing code block",
                " some code",
                " ```",
                "-" * 80,
            ]
        )

        processed = preprocess_diff_for_heading_anchor_stability(
            pr_diff,
            source_language="English",
            target_language="Chinese",
            source_mode="commit",
        )

        self.assertNotIn("{#", processed)

    def test_preprocess_diff_skips_heading_inside_buffered_code_block(self):
        """A - line followed by + lines that open a code block should not anchor headings inside."""
        pr_diff = "\n".join(
            [
                "File: ai/example.md",
                "@@ -10,1 +10,4 @@",
                "-old paragraph text",
                "+```markdown",
                "+## Example heading in code",
                "+```",
                "+## Real heading after code",
                "-" * 80,
            ]
        )

        processed = preprocess_diff_for_heading_anchor_stability(
            pr_diff,
            source_language="English",
            target_language="Japanese",
            source_mode="commit",
        )

        self.assertNotIn("+## Example heading in code {#", processed)
        self.assertIn("+## Real heading after code {#real-heading-after-code}", processed)

    def test_preprocess_diff_adds_zh_prefix_to_added_aliases(self):
        pr_diff = "\n".join(
            [
                "File: ai/example.md",
                "@@ -1,1 +1,1 @@",
                "+aliases: ['/tidb/stable/saas-best-practices/','/zh/tidb/dev/saas-best-practices/']",
                "-" * 80,
            ]
        )

        processed = preprocess_diff_for_heading_anchor_stability(
            pr_diff,
            source_language="English",
            target_language="Chinese",
            source_mode="commit",
        )

        self.assertIn(
            "+aliases: ['/zh/tidb/stable/saas-best-practices/','/zh/tidb/dev/saas-best-practices/']",
            processed,
        )

    def test_preprocess_diff_rewrites_tidb_cloud_links_only_for_ai_commit_scope(self):
        pr_diff = "\n".join(
            [
                "File: docs/example.md",
                "@@ -1,1 +1,1 @@",
                "+See [Private Endpoints](/tidb-cloud/test/set-up-private-endpoint-connections-serverless2.md).",
                "-" * 80,
            ]
        )

        with mock.patch.dict(os.environ, {"SOURCE_FOLDER": "docs"}, clear=False):
            processed = preprocess_diff_for_heading_anchor_stability(
                pr_diff,
                source_language="English",
                target_language="Chinese",
                source_mode="commit",
            )

        self.assertEqual(processed, pr_diff)

    def test_preprocess_diff_does_not_rewrite_tidb_cloud_links_for_cloud_commit_scope(self):
        pr_diff = "\n".join(
            [
                "File: tidb-cloud/example.md",
                "@@ -1,1 +1,1 @@",
                "+See [Private Endpoints](/tidb-cloud/test/set-up-private-endpoint-connections-serverless2.md).",
                "-" * 80,
            ]
        )

        with mock.patch.dict(
            os.environ,
            {"SOURCE_FOLDER": "", "SOURCE_FILES": "tidb-cloud/example.md"},
            clear=False,
        ):
            processed = preprocess_diff_for_heading_anchor_stability(
                pr_diff,
                source_language="English",
                target_language="Chinese",
                source_mode="commit",
            )

        self.assertEqual(processed, pr_diff)

    def test_preprocess_diff_rewrites_tidb_cloud_links_for_ai_commit_scope(self):
        pr_diff = "\n".join(
            [
                "File: ai/example.md",
                "@@ -1,1 +1,1 @@",
                "+See [Private Endpoints](/tidb-cloud/test/set-up-private-endpoint-connections-serverless2.md).",
                "-" * 80,
            ]
        )

        with mock.patch.dict(os.environ, {"SOURCE_FOLDER": "ai"}, clear=False):
            processed = preprocess_diff_for_heading_anchor_stability(
                pr_diff,
                source_language="English",
                target_language="Chinese",
                source_mode="commit",
            )

        self.assertIn(
            "+See [Private Endpoints](https://docs.pingcap.com/tidbcloud/set-up-private-endpoint-connections-serverless2).",
            processed,
        )

    def test_insert_preserves_unmodified_line_endings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            target_file = tmp_path / "system-variables.md"
            match_file = tmp_path / "system-variables-match_source_diff_to_target.json"

            with open(target_file, "w", encoding="utf-8", newline="") as f:
                f.write("# System Variables\n\n")
                f.write("## Variable reference\n\n")
                f.write("### tidb_enable_tso_follower_proxy <span class=\"version-mark\">New in v5.3.0</span>\n\n")
                f.write("- Scope: GLOBAL\r\n")
                f.write("- Persists to cluster: Yes\r\n")
                f.write("- Type: Boolean\n")

            match_file.write_text(
                json.dumps(
                    {
                        "added_2490": {
                            "source_operation": "added",
                            "insertion_type": "before_reference",
                            "target_line": "5",
                            "target_hierarchy": "## Variable reference > ### tidb_enable_tso_follower_proxy <span class=\"version-mark\">New in v5.3.0</span>",
                            "target_new_content": "### `tidb_enable_ts_validation` <span class=\"version-mark\">New in v9.0.0</span>\n\n- Scope: GLOBAL\n- Persists to cluster: Yes\n",
                        }
                    }
                ),
                encoding="utf-8",
            )

            success = update_target_document_from_match_data(
                str(match_file), str(tmp_path), "system-variables.md"
            )

            self.assertTrue(success)

            with open(target_file, "r", encoding="utf-8", newline="") as f:
                updated_content = f.read()

            self.assertIn("### `tidb_enable_ts_validation`", updated_content)
            self.assertIn(
                "- Scope: GLOBAL\r\n- Persists to cluster: Yes\r\n- Type: Boolean\n",
                updated_content,
            )
            self.assertEqual(updated_content.count("\r\n"), 2)

    def test_update_trims_extra_blank_lines_at_eof(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            target_file = tmp_path / "example.md"
            match_file = tmp_path / "example-match_source_diff_to_target.json"

            target_file.write_text("# Example\n\n## Last section\n\nOld content\n", encoding="utf-8")
            match_file.write_text(
                json.dumps(
                    {
                        "modified_3": {
                            "source_operation": "modified",
                            "target_line": "3",
                            "target_hierarchy": "## Last section",
                            "target_new_content": "## Last section\n\nNew content\n\n",
                        }
                    }
                ),
                encoding="utf-8",
            )

            success = update_target_document_from_match_data(
                str(match_file), str(tmp_path), "example.md"
            )

            self.assertTrue(success)
            updated_content = target_file.read_text(encoding="utf-8")
            self.assertTrue(updated_content.endswith("New content\n"))
            self.assertFalse(updated_content.endswith("New content\n\n"))

    def test_update_preserves_missing_final_newline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            target_file = tmp_path / "example.md"
            match_file = tmp_path / "example-match_source_diff_to_target.json"

            target_file.write_text("# Example\n\n## Last section\n\nOld content", encoding="utf-8")
            match_file.write_text(
                json.dumps(
                    {
                        "modified_3": {
                            "source_operation": "modified",
                            "target_line": "3",
                            "target_hierarchy": "## Last section",
                            "target_new_content": "## Last section\n\nNew content\n\n",
                        }
                    }
                ),
                encoding="utf-8",
            )

            success = update_target_document_from_match_data(
                str(match_file), str(tmp_path), "example.md"
            )

            self.assertTrue(success)
            updated_content = target_file.read_text(encoding="utf-8")
            self.assertEqual(updated_content, "# Example\n\n## Last section\n\nNew content")

    def test_update_preserves_existing_final_newline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            target_file = tmp_path / "example.md"
            match_file = tmp_path / "example-match_source_diff_to_target.json"

            target_file.write_text("# Example\n\n## Last section\n\nOld content\n", encoding="utf-8")
            match_file.write_text(
                json.dumps(
                    {
                        "modified_3": {
                            "source_operation": "modified",
                            "target_line": "3",
                            "target_hierarchy": "## Last section",
                            "target_new_content": "## Last section\n\nNew content",
                        }
                    }
                ),
                encoding="utf-8",
            )

            success = update_target_document_from_match_data(
                str(match_file), str(tmp_path), "example.md"
            )

            self.assertTrue(success)
            updated_content = target_file.read_text(encoding="utf-8")
            self.assertEqual(updated_content, "# Example\n\n## Last section\n\nNew content\n")

    def test_update_resolves_stale_target_line_by_leaf_heading(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            target_file = tmp_path / "example.md"
            match_file = tmp_path / "example-match_source_diff_to_target.json"

            target_file.write_text(
                "# Example\n\n## Parent\n\nParent intro\n\n### Child\n\nOld content\n",
                encoding="utf-8",
            )
            match_file.write_text(
                json.dumps(
                    {
                        "modified_20": {
                            "source_operation": "modified",
                            "target_line": "3",
                            "target_hierarchy": "## Parent > ### Child",
                            "target_new_content": "### Child\n\nNew content\n",
                        }
                    }
                ),
                encoding="utf-8",
            )

            success = update_target_document_from_match_data(
                str(match_file), str(tmp_path), "example.md"
            )

            self.assertTrue(success)
            updated_content = target_file.read_text(encoding="utf-8")
            self.assertEqual(
                updated_content,
                "# Example\n\n## Parent\n\nParent intro\n\n### Child\n\nNew content\n",
            )

    def test_large_system_sections_are_translated_in_chunks_and_merged(self):
        class FakeAIClient:
            def __init__(self):
                self.prompts = []

            def chat_completion(self, messages, temperature=0.1):
                prompt = messages[0]["content"]
                self.prompts.append(prompt)
                keys = list(dict.fromkeys(re.findall(r'"(modified_\d+)"\s*:', prompt)))
                return json.dumps({key: f"translated {key}" for key in keys})

        prefix = "chunk-test-unit"
        self._cleanup_chunk_test_outputs(prefix)
        try:
            source_sections, target_sections = self._build_system_sections(25)
            ai_client = FakeAIClient()

            result = get_updated_sections_from_ai(
                "File: system-variables.md\n@@ -1,1 +1,1 @@",
                target_sections,
                source_sections,
                ai_client,
                "English",
                "Chinese",
                "chunk-test-unit.md",
            )

            self.assertIsInstance(result, TranslationResult)
            self.assertEqual(len(ai_client.prompts), 2)
            self.assertEqual(set(result.keys()), set(source_sections.keys()))
            self.assertFalse(result.failures)

            temp_dir = SCRIPTS_DIR / "temp_output"
            self.assertTrue((temp_dir / f"{prefix}_updated_sections_from_ai.part-001.json").exists())
            self.assertTrue((temp_dir / f"{prefix}_updated_sections_from_ai.part-002.json").exists())
            merged_file = temp_dir / f"{prefix}_updated_sections_from_ai.json"
            self.assertTrue(merged_file.exists())
            merged = json.loads(merged_file.read_text(encoding="utf-8"))
            self.assertEqual(set(merged.keys()), set(source_sections.keys()))
        finally:
            self._cleanup_chunk_test_outputs(prefix)

    def test_translation_chunks_use_character_budget(self):
        source_sections = {
            "modified_1": "x" * 60,
            "modified_2": "x" * 60,
            "modified_3": "x" * 20,
        }

        with mock.patch("file_updater.TRANSLATION_CHUNK_MAX_SECTIONS", 20), mock.patch(
            "file_updater.TRANSLATION_CHUNK_CHAR_LIMIT",
            100,
        ):
            chunks = build_translation_chunks(source_sections)

        self.assertEqual([chunk["keys"] for chunk in chunks], [["modified_1"], ["modified_2", "modified_3"]])

    def test_translation_chunks_include_target_content_in_character_budget(self):
        source_sections = {
            "modified_1": "x" * 40,
            "modified_2": "x" * 40,
            "modified_3": "x" * 20,
        }
        target_sections = {
            "modified_1": "y" * 20,
            "modified_2": "y" * 20,
            "modified_3": "y" * 10,
        }

        with mock.patch("file_updater.TRANSLATION_CHUNK_MAX_SECTIONS", 20), mock.patch(
            "file_updater.TRANSLATION_CHUNK_CHAR_LIMIT",
            100,
        ):
            chunks = build_translation_chunks(source_sections, target_sections)

        self.assertEqual([chunk["keys"] for chunk in chunks], [["modified_1"], ["modified_2", "modified_3"]])

    def test_default_translation_chunk_size_uses_regular_doc_limit(self):
        self.assertEqual(TRANSLATION_CHUNK_MAX_SECTIONS, REGULAR_TRANSLATION_CHUNK_SIZE)
        self.assertEqual(get_translation_chunk_max_sections("guide.md"), REGULAR_TRANSLATION_CHUNK_SIZE)
        self.assertEqual(get_translation_chunk_max_sections("system-variables.md"), 20)

        source_sections = {
            f"modified_{index}": f"section {index}"
            for index in range(REGULAR_TRANSLATION_CHUNK_SIZE + 1)
        }

        with mock.patch("file_updater.TRANSLATION_CHUNK_CHAR_LIMIT", 100000):
            chunks = build_translation_chunks(source_sections)

        self.assertEqual(
            [len(chunk["keys"]) for chunk in chunks],
            [REGULAR_TRANSLATION_CHUNK_SIZE, 1],
        )

    def test_chunk_diff_keeps_prefix_and_numeric_section_hunks(self):
        pr_diff = "\n".join(
            [
                "File: tidb-cloud/tidb-cloud-auditing.md",
                "diff --git a/auditing.md b/auditing.md",
                "--- a/auditing.md",
                "+++ b/auditing.md",
                "@@ -1,20 +1,27 @@",
                "-title: Audit Logging",
                "+title: Audit Logging (Preview)",
                "-Old intro.",
                "+New public preview intro.",
                "@@ -33,7 +40,7 @@",
                "-Old enable text.",
                "+New enable text.",
                "@@ -93,7 +100,7 @@",
                "-Old unrelated text.",
                "+New unrelated text.",
            ]
        )

        filtered = filter_diff_for_chunk_sections(
            pr_diff,
            ["frontmatter", "intro_section", "modified_40"],
            [
                "frontmatter",
                "intro_section",
                "modified_40",
                "modified_100",
            ],
        )

        self.assertIn("@@ -1,20 +1,27 @@", filtered)
        self.assertIn("+title: Audit Logging (Preview)", filtered)
        self.assertIn("+New public preview intro.", filtered)
        self.assertIn("@@ -33,7 +40,7 @@", filtered)
        self.assertIn("+New enable text.", filtered)
        self.assertNotIn("@@ -93,7 +100,7 @@", filtered)
        self.assertNotIn("+New unrelated text.", filtered)

    def test_chunk_diff_falls_back_when_modified_section_has_no_hunk(self):
        pr_diff = "\n".join(
            [
                "File: guide.md",
                "diff --git a/guide.md b/guide.md",
                "--- a/guide.md",
                "+++ b/guide.md",
                "@@ -38,5 +40,5 @@",
                "-Old section 40.",
                "+New section 40.",
                "@@ -118,5 +120,5 @@",
                "-Old unrelated section.",
                "+New unrelated section.",
            ]
        )

        filtered = filter_diff_for_chunk_sections(
            pr_diff,
            ["modified_40", "modified_80"],
            ["modified_40", "modified_80", "modified_120"],
        )

        self.assertEqual(pr_diff, filtered)

    def test_chunk_glossary_matching_uses_chunk_filtered_diff(self):
        class FakeAIClient:
            def __init__(self):
                self.prompts = []

            def chat_completion(self, messages, temperature=0.1):
                prompt = messages[0]["content"]
                self.prompts.append(prompt)
                keys = list(dict.fromkeys(re.findall(r'"(modified_\d+)"\s*:', prompt)))
                return json.dumps({key: f"translated {key}" for key in keys})

        prefix = "chunk-glossary-unit"
        self._cleanup_chunk_test_outputs(prefix)
        try:
            source_sections, target_sections = self._build_system_sections(25)
            seen_glossary_inputs = []

            def glossary_matcher(text, source_language=None):
                seen_glossary_inputs.append(text)
                return []

            get_updated_sections_from_ai(
                self._build_system_section_diff(
                    25,
                    {
                        1: "ChunkDiffOnlyTerm1",
                        25: "ChunkDiffOnlyTerm25",
                    },
                ),
                target_sections,
                source_sections,
                FakeAIClient(),
                "English",
                "Chinese",
                "chunk-glossary-unit.md",
                glossary_matcher=glossary_matcher,
            )

            self.assertEqual(len(seen_glossary_inputs), 2)
            self.assertIn("ChunkDiffOnlyTerm1", seen_glossary_inputs[0])
            self.assertNotIn("ChunkDiffOnlyTerm25", seen_glossary_inputs[0])
            self.assertIn("ChunkDiffOnlyTerm25", seen_glossary_inputs[1])
            self.assertNotIn("ChunkDiffOnlyTerm1", seen_glossary_inputs[1])
        finally:
            self._cleanup_chunk_test_outputs(prefix)

    def test_modified_glossary_matching_ignores_source_and_target_sections(self):
        class FakeAIClient:
            def chat_completion(self, messages, temperature=0.1):
                return json.dumps({"modified_1": "translated modified_1"})

        prefix = "modified-glossary-unit"
        self._cleanup_chunk_test_outputs(prefix)
        seen_glossary_inputs = []

        def glossary_matcher(text, source_language=None):
            seen_glossary_inputs.append(text)
            return []

        try:
            get_updated_sections_from_ai(
                "\n".join([
                    "File: example.md",
                    "@@ -1,3 +1,3 @@",
                    "-old content",
                    "+DiffOnlyTerm",
                    " context",
                ]),
                {
                    "modified_1": "### TargetOnlyTerm\n\n旧中文内容。",
                },
                {
                    "modified_1": "### SourceOnlyTerm\n\nOld English content.",
                },
                FakeAIClient(),
                "English",
                "Chinese",
                "modified-glossary-unit.md",
                glossary_matcher=glossary_matcher,
            )

            self.assertEqual(len(seen_glossary_inputs), 1)
            self.assertIn("DiffOnlyTerm", seen_glossary_inputs[0])
            self.assertNotIn("SourceOnlyTerm", seen_glossary_inputs[0])
            self.assertNotIn("TargetOnlyTerm", seen_glossary_inputs[0])
        finally:
            self._cleanup_chunk_test_outputs(prefix)

    def test_chunk_glossary_matching_ignores_chunk_source_and_target_sections(self):
        class FakeAIClient:
            def chat_completion(self, messages, temperature=0.1):
                prompt = messages[0]["content"]
                keys = list(dict.fromkeys(re.findall(r'"(modified_\d+)"\s*:', prompt)))
                return json.dumps({key: f"translated {key}" for key in keys})

        prefix = "chunk-glossary-ignore-unit"
        self._cleanup_chunk_test_outputs(prefix)
        try:
            source_sections, target_sections = self._build_system_sections(25)
            source_sections["modified_1"] += "\nSourceOnlyTerm1\n"
            target_sections["modified_1"] += "\nTargetOnlyTerm1\n"
            source_sections["modified_25"] += "\nSourceOnlyTerm25\n"
            target_sections["modified_25"] += "\nTargetOnlyTerm25\n"
            seen_glossary_inputs = []

            def glossary_matcher(text, source_language=None):
                seen_glossary_inputs.append(text)
                return []

            get_updated_sections_from_ai(
                self._build_system_section_diff(
                    25,
                    {
                        1: "DiffOnlyTerm1",
                        25: "DiffOnlyTerm25",
                    },
                ),
                target_sections,
                source_sections,
                FakeAIClient(),
                "English",
                "Chinese",
                "chunk-glossary-ignore-unit.md",
                glossary_matcher=glossary_matcher,
            )

            self.assertEqual(len(seen_glossary_inputs), 2)
            self.assertIn("DiffOnlyTerm1", seen_glossary_inputs[0])
            self.assertNotIn("DiffOnlyTerm25", seen_glossary_inputs[0])
            self.assertNotIn("SourceOnlyTerm1", seen_glossary_inputs[0])
            self.assertNotIn("TargetOnlyTerm1", seen_glossary_inputs[0])
            self.assertIn("DiffOnlyTerm25", seen_glossary_inputs[1])
            self.assertNotIn("DiffOnlyTerm1", seen_glossary_inputs[1])
            self.assertNotIn("SourceOnlyTerm25", seen_glossary_inputs[1])
            self.assertNotIn("TargetOnlyTerm25", seen_glossary_inputs[1])
        finally:
            self._cleanup_chunk_test_outputs(prefix)

    def test_chunk_failure_with_useful_output_is_attached_as_partial_result(self):
        class FakeAIClient:
            def __init__(self):
                self.prompts = []

            def chat_completion(self, messages, temperature=0.1):
                prompt = messages[0]["content"]
                self.prompts.append(prompt)
                if len(self.prompts) == 2:
                    return "not json"
                keys = list(dict.fromkeys(re.findall(r'"(modified_\d+)"\s*:', prompt)))
                return json.dumps({key: f"translated {key}" for key in keys})

        prefix = "chunk-failure-unit"
        self._cleanup_chunk_test_outputs(prefix)
        try:
            source_sections, target_sections = self._build_system_sections(25)

            result = get_updated_sections_from_ai(
                "File: system-variables.md\n@@ -1,1 +1,1 @@",
                target_sections,
                source_sections,
                FakeAIClient(),
                "English",
                "Chinese",
                "chunk-failure-unit.md",
            )

            self.assertIsInstance(result, TranslationResult)
            self.assertEqual(len(result), 20)
            self.assertFalse(result.failures)
            self.assertTrue(result.partial_reasons)
            self.assertTrue(
                any(
                    "failed to translate chunk 2/2" in reason
                    and "tidb_chunk_test_021" in reason
                    for reason in result.partial_reasons
                )
            )
        finally:
            self._cleanup_chunk_test_outputs(prefix)

    def test_enhanced_modified_section_prompt_falls_back_to_new_content_when_old_missing(self):
        class FakeAIClient:
            def __init__(self):
                self.prompts = []

            def chat_completion(self, messages, temperature=0.1):
                prompt = messages[0]["content"]
                self.prompts.append(prompt)
                if "Translate only the changed Markdown headings" in prompt:
                    return json.dumps(
                        {
                            "heading_001": {
                                "status": "updated",
                                "heading": "### 新标题",
                            }
                        }
                    )
                return json.dumps({"modified_10": "### 新标题\n\n新内容"})

        ai_client = FakeAIClient()
        success, updated_sections = process_single_file(
            "example.md",
            {
                "type": "enhanced_sections",
                "sections": {
                    "modified_10": {
                        "source_operation": "modified",
                        "source_old_content": "",
                        "source_new_content": "### New heading\n\nNew content",
                        "target_content": "### 旧标题\n\n旧内容",
                    }
                },
            },
            "File: example.md\n@@ -10,1 +10,1 @@\n-### Old heading\n+### New heading",
            {"mode": "commit"},
            github_client=None,
            ai_client=ai_client,
            repo_config={
                "source_language": "English",
                "target_language": "Chinese",
            },
        )

        self.assertTrue(success)
        self.assertEqual(
            updated_sections["modified_10"],
            "### 新标题 {#new-heading}\n\n新内容",
        )
        self.assertIn("### New heading", ai_client.prompts[0])
        self.assertIn("New content", ai_client.prompts[0])
        self.assertIn("after applying the diff", ai_client.prompts[0])
        self.assertIn("modified_10", ai_client.prompts[0])
        self.assertNotIn('"modified_10": ""', ai_client.prompts[0])

    def test_enhanced_modified_section_prompt_uses_new_content(self):
        """Prompt should contain post-change (new) source content so the AI
        sees the full final state and does not miss added paragraphs."""
        class FakeAIClient:
            def __init__(self):
                self.prompts = []

            def chat_completion(self, messages, temperature=0.1):
                prompt = messages[0]["content"]
                self.prompts.append(prompt)
                if "Translate only the changed Markdown headings" in prompt:
                    return json.dumps(
                        {
                            "heading_001": {
                                "status": "updated",
                                "heading": "### 新标题",
                            }
                        }
                    )
                return json.dumps({"modified_10": "### 新标题\n\n新内容"})

        ai_client = FakeAIClient()
        success, updated_sections = process_single_file(
            "example.md",
            {
                "type": "enhanced_sections",
                "sections": {
                    "modified_10": {
                        "source_operation": "modified",
                        "source_old_content": "### Old heading\n\nOld content",
                        "source_new_content": "### New heading\n\nNew content",
                        "target_content": "### 旧标题\n\n旧内容",
                    }
                },
            },
            "File: example.md\n@@ -10,1 +10,1 @@\n-### Old heading\n+### New heading",
            {"mode": "commit"},
            github_client=None,
            ai_client=ai_client,
            repo_config={
                "source_language": "English",
                "target_language": "Chinese",
            },
        )

        self.assertTrue(success)
        self.assertEqual(
            updated_sections["modified_10"],
            "### 新标题 {#new-heading}\n\n新内容",
        )
        self.assertIn("New heading", ai_client.prompts[0])
        self.assertIn("New content", ai_client.prompts[0])
        self.assertIn("post-change", ai_client.prompts[0])

    # ------------------------------------------------------------------
    # Cross-parent move detection tests
    # ------------------------------------------------------------------
    def test_cross_parent_move_splits_into_delete_and_insert(self):
        """When a modified section has empty old_content, a completely
        different heading, and an added parent section exists, the entry
        should be split into a delete + insert."""
        from file_updater import _detect_and_fix_cross_parent_moves

        sections = [
            ("added_328", {
                "source_operation": "added",
                "source_new_content": "## Manage instance access\n",
                "target_new_content": "## 管理实例访问\n",
                "target_hierarchy": "## 管理用户资料",
                "insertion_type": "before_reference",
            }, 327),
            ("modified_354", {
                "source_operation": "modified",
                "source_old_content": "",
                "source_new_content": "### Remove instance access\n\nSome content.\n",
                "source_original_hierarchy": "### Modify project roles",
                "target_hierarchy": "## 管理项目访问 > ### 修改项目角色",
                "target_new_content": "### 移除实例访问权限\n\n一些内容。\n",
            }, 299),
        ]

        result = _detect_and_fix_cross_parent_moves(sections)

        keys = [k for k, _, _ in result]
        self.assertIn("added_328", keys)
        self.assertIn("modified_354_delete", keys)
        self.assertIn("modified_354_insert", keys)
        self.assertNotIn("modified_354", keys)

        for key, data, line in result:
            if key == "modified_354_delete":
                self.assertEqual(data["source_operation"], "deleted")
                self.assertEqual(line, 299)
            elif key == "modified_354_insert":
                self.assertEqual(data["insertion_type"], "before_reference")
                self.assertEqual(line, 327)
                self.assertIn("移除实例访问权限", data["target_new_content"])

    def test_cross_parent_move_no_false_positive_for_heading_rename(self):
        """A heading rename (old content present, same parent) must NOT
        be treated as a cross-parent move."""
        from file_updater import _detect_and_fix_cross_parent_moves

        sections = [
            ("modified_100", {
                "source_operation": "modified",
                "source_old_content": "### Old heading\n\nContent.",
                "source_new_content": "### New heading\n\nContent.",
                "source_original_hierarchy": "## Parent > ### Old heading",
                "target_hierarchy": "## 父级 > ### 旧标题",
                "target_new_content": "### 新标题\n\n内容。",
            }, 50),
        ]

        result = _detect_and_fix_cross_parent_moves(sections)
        keys = [k for k, _, _ in result]
        self.assertEqual(keys, ["modified_100"])

    def test_cross_parent_move_no_split_without_added_parent(self):
        """If there is no added parent H2, the section should not be split
        even if the heading changed completely."""
        from file_updater import _detect_and_fix_cross_parent_moves

        sections = [
            ("modified_354", {
                "source_operation": "modified",
                "source_old_content": "",
                "source_new_content": "### Totally new heading\n\nNew stuff.\n",
                "source_original_hierarchy": "### Old heading",
                "target_hierarchy": "## 父级 > ### 旧标题",
                "target_new_content": "### 全新标题\n\n新内容。\n",
            }, 299),
        ]

        result = _detect_and_fix_cross_parent_moves(sections)
        keys = [k for k, _, _ in result]
        self.assertEqual(keys, ["modified_354"])


if __name__ == "__main__":
    unittest.main()

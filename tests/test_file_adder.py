import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from file_adder import (
    _join_translated_batches,
    create_section_batches,
    ensure_blank_lines_before_headings,
    preprocess_added_file_batch_for_heading_anchor_stability,
    process_added_files,
    strip_ai_markdown_wrapper,
    translate_file_batch,
)
from ai_client import CompletionText


class FileAdderRegressionTest(unittest.TestCase):
    def test_translation_prompt_uses_configured_product_name(self):
        class CapturingAIClient:
            def __init__(self):
                self.prompt = ""

            def chat_completion(self, messages, temperature=0.1):
                self.prompt = messages[0]["content"]
                return "翻译后的内容。"

        ai_client = CapturingAIClient()
        with mock.patch.dict(os.environ, {"PRODUCT": "ExampleDB"}, clear=False):
            translate_file_batch(
                "Source content.\n",
                ai_client,
                source_language="English",
                target_language="Chinese",
            )

        self.assertIn("the following ExampleDB document content", ai_client.prompt)
        self.assertNotIn("the following document content", ai_client.prompt)

    def test_translation_prompt_uses_product_neutral_wording_when_unset(self):
        class CapturingAIClient:
            def __init__(self):
                self.prompt = ""

            def chat_completion(self, messages, temperature=0.1):
                self.prompt = messages[0]["content"]
                return "翻译后的内容。"

        ai_client = CapturingAIClient()
        with mock.patch.dict(os.environ, {}, clear=True):
            translate_file_batch(
                "Source content.\n",
                ai_client,
                source_language="English",
                target_language="Chinese",
            )

        self.assertIn("the following document content", ai_client.prompt)
        self.assertNotIn("the following  document content", ai_client.prompt)

    def test_trailing_batch_newlines_do_not_trigger_truncation_warning(self):
        class FakeAIClient:
            def chat_completion(self, messages, temperature=0.1):
                return "翻译后的单行段落。"

        with mock.patch("file_adder.thread_safe_print") as print_mock:
            translated = translate_file_batch(
                "A single-line source paragraph.\n\n",
                FakeAIClient(),
                source_language="English",
                target_language="Chinese",
            )

        self.assertEqual([], translated.partial_reasons)
        self.assertFalse(
            any(
                "significantly fewer lines" in str(call)
                for call in print_mock.call_args_list
            )
        )

    def test_real_line_count_drop_still_triggers_truncation_warning(self):
        class FakeAIClient:
            def chat_completion(self, messages, temperature=0.1):
                return "只有一行。"

        with mock.patch("file_adder.thread_safe_print") as print_mock:
            translated = translate_file_batch(
                "Line one.\nLine two.\nLine three.\nLine four.\nLine five.\n",
                FakeAIClient(),
                source_language="English",
                target_language="Chinese",
            )

        self.assertTrue(
            any(
                "1 lines versus 5 source lines" in reason
                for reason in translated.partial_reasons
            )
        )
        self.assertTrue(
            any(
                "significantly fewer lines" in str(call)
                for call in print_mock.call_args_list
            )
        )

    def test_incomplete_added_file_response_is_written_and_reported_as_partial(self):
        class FakeAIClient:
            def chat_completion(self, messages, temperature=0.1):
                return CompletionText(
                    "# 已翻译\n\n有用的部分输出\n",
                    status="incomplete",
                    reason="max_output_tokens",
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            outcomes = process_added_files(
                {"guide.md": "# Guide\n\nUseful source content\n"},
                {"mode": "commit"},
                object(),
                FakeAIClient(),
                {
                    "target_local_path": tmpdir,
                    "source_language": "English",
                    "target_language": "Chinese",
                },
                return_outcomes=True,
            )

            self.assertEqual(outcomes["guide.md"]["status"], "partial")
            self.assertIn("max_output_tokens", outcomes["guide.md"]["reason"])
            self.assertIn(
                "有用的部分输出",
                Path(tmpdir, "guide.md").read_text(encoding="utf-8"),
            )
    def test_preprocess_added_file_batch_adds_anchors_for_non_top_level_headings(self):
        batch = "\n".join(
            [
                "# Title",
                "",
                "## Example test",
                "",
                "### txn-entry-size-limit <span class=\"version-mark\">New in v4.0.10 and v5.0.0</span>",
            ]
        )

        processed = preprocess_added_file_batch_for_heading_anchor_stability(
            batch,
            source_language="English",
            target_language="Chinese",
            source_mode="commit",
        )

        self.assertIn("## Example test {#example-test}", processed)
        self.assertIn(
            "### txn-entry-size-limit <span class=\"version-mark\">New in v4.0.10 and v5.0.0</span> {#txn-entry-size-limit-new-in-v4010-and-v500}",
            processed,
        )
        self.assertIn("# Title", processed)

    def test_preprocess_added_file_batch_skips_pr_mode(self):
        batch = "## Example test"

        processed = preprocess_added_file_batch_for_heading_anchor_stability(
            batch,
            source_language="English",
            target_language="Chinese",
            source_mode="pr",
        )

        self.assertEqual(processed, batch)

    def test_preprocess_added_file_batch_adds_zh_prefix_to_aliases(self):
        batch = "\n".join(
            [
                "---",
                "aliases: ['/tidb/stable/saas-best-practices/','/zh/tidb/dev/saas-best-practices/']",
                "---",
            ]
        )

        processed = preprocess_added_file_batch_for_heading_anchor_stability(
            batch,
            source_language="English",
            target_language="Chinese",
            source_mode="commit",
        )

        self.assertIn(
            "aliases: ['/zh/tidb/stable/saas-best-practices/','/zh/tidb/dev/saas-best-practices/']",
            processed,
        )

    def test_preprocess_added_file_batch_rewrites_tidb_cloud_links_in_pr_mode(self):
        batch = "See [Private Endpoints](/tidb-cloud/test/set-up-private-endpoint-connections-serverless2.md)."

        processed = preprocess_added_file_batch_for_heading_anchor_stability(
            batch,
            source_language="English",
            target_language="Chinese",
            source_mode="pr",
        )

        self.assertEqual(
            processed,
            "See [Private Endpoints](https://docs.pingcap.com/tidbcloud/set-up-private-endpoint-connections-serverless2).",
        )

    def test_preprocess_added_file_batch_rewrites_tidb_cloud_links_only_for_ai_commit_scope(self):
        batch = "See [Private Endpoints](/tidb-cloud/test/set-up-private-endpoint-connections-serverless2.md)."

        with mock.patch.dict(os.environ, {"SOURCE_FOLDER": "docs"}, clear=False):
            processed = preprocess_added_file_batch_for_heading_anchor_stability(
                batch,
                source_language="English",
                target_language="Chinese",
                source_mode="commit",
            )

        self.assertEqual(processed, batch)

    def test_preprocess_added_file_batch_rewrites_tidb_cloud_links_for_ai_commit_scope(self):
        batch = "See [Private Endpoints](/tidb-cloud/test/set-up-private-endpoint-connections-serverless2.md)."

        with mock.patch.dict(os.environ, {"SOURCE_FOLDER": "ai"}, clear=False):
            processed = preprocess_added_file_batch_for_heading_anchor_stability(
                batch,
                source_language="English",
                target_language="Chinese",
                source_mode="commit",
            )

        self.assertEqual(
            processed,
            "See [Private Endpoints](https://docs.pingcap.com/tidbcloud/set-up-private-endpoint-connections-serverless2).",
        )

    def test_translate_file_batch_rewrites_tidb_version_anchor_in_pr_mode(self):
        class FakeAIClient:
            def chat_completion(self, messages, temperature=0.1):
                return (
                    "See [`tidb_enable_x`](/system-variables.md"
                    "#tidb-enable-x-从-v800-版本开始引入)."
                )

        with mock.patch.dict(os.environ, {"PRODUCT": "TiDB"}, clear=False):
            processed = translate_file_batch(
                "请参见 [`tidb_enable_x`](/system-variables.md#tidb-enable-x-从-v800-版本开始引入)。",
                FakeAIClient(),
                source_language="Chinese",
                target_language="English",
                source_mode="pr",
            )

        self.assertEqual(
            processed,
            "See [`tidb_enable_x`](/system-variables.md#tidb-enable-x-new-in-v800).",
        )

    def test_process_added_files_can_overwrite_existing_target_file(self):
        repo_config = {
            "target_local_path": "",
            "source_language": "English",
            "target_language": "Chinese",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_config["target_local_path"] = tmpdir
            target_file = Path(tmpdir, "guide.md")
            target_file.write_text("old target", encoding="utf-8")

            with mock.patch("file_adder.translate_file_batch", return_value="translated target"):
                success, failures = process_added_files(
                    {"guide.md": "# Guide\n\nSource content"},
                    {"mode": "commit"},
                    object(),
                    object(),
                    repo_config,
                    return_details=True,
                    overwrite_existing=True,
                )

            self.assertTrue(success)
            self.assertEqual(failures, {})
            self.assertEqual(target_file.read_text(encoding="utf-8"), "translated target")

    def test_process_added_files_preserves_related_resources_when_called_directly(self):
        repo_config = {
            "target_local_path": "",
            "source_language": "English",
            "target_language": "Chinese",
        }
        source_content = "\n".join(
            [
                "# Guide",
                "",
                "## Usage",
                "",
                "Use TiDB.",
                "",
                "## Related resources",
                "",
                "<RelatedResources>",
                '  <ResourceCard title="Example" type="blog" link="https://example.com" />',
                "</RelatedResources>",
                "",
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_config["target_local_path"] = tmpdir
            target_file = Path(tmpdir, "guide.md")

            with mock.patch(
                "file_adder.translate_file_batch",
                side_effect=lambda batch, *args, **kwargs: batch,
            ):
                success, failures = process_added_files(
                    {"guide.md": source_content},
                    {"mode": "commit"},
                    object(),
                    object(),
                    repo_config,
                    return_details=True,
                )

            self.assertTrue(success)
            self.assertEqual(failures, {})
            translated = target_file.read_text(encoding="utf-8")
            self.assertIn("## Usage", translated)
            self.assertIn("RelatedResources", translated)
            self.assertIn("ResourceCard", translated)

    def test_process_added_files_still_rejects_existing_target_file_by_default(self):
        repo_config = {
            "target_local_path": "",
            "source_language": "English",
            "target_language": "Chinese",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_config["target_local_path"] = tmpdir
            target_file = Path(tmpdir, "guide.md")
            target_file.write_text("old target", encoding="utf-8")

            success, failures = process_added_files(
                {"guide.md": "# Guide\n\nSource content"},
                {"mode": "commit"},
                object(),
                object(),
                repo_config,
                return_details=True,
            )

            self.assertFalse(success)
            self.assertIn("guide.md", failures)
            self.assertEqual(target_file.read_text(encoding="utf-8"), "old target")


class TestCreateSectionBatches(unittest.TestCase):
    """Tests for create_section_batches, especially the last-batch oversizing fix."""

    def test_simple_file_single_batch(self):
        content = "# Title\n\nSome content.\n\n## Section\n\nMore content."
        batches = create_section_batches(content, max_lines_per_batch=200)
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0], content)

    def test_splits_at_heading_boundaries(self):
        lines = ["# Title", ""]
        for i in range(3):
            lines.append(f"## Section {i}")
            lines.extend([f"Line {j}" for j in range(80)])
            lines.append("")
        content = "\n".join(lines)
        batches = create_section_batches(content, max_lines_per_batch=100)
        self.assertGreater(len(batches), 1)
        rejoined = "\n".join(batches)
        self.assertEqual(rejoined, content)

    def test_last_batch_not_oversized(self):
        """Reproduces the release-8.5.7.md bug: last batch was 236 lines."""
        lines = ["---", "title: Test", "---", "", "# Release Notes", ""]
        for i in range(3):
            lines.append(f"## Section {i}")
            lines.extend([f"Content line {j}" for j in range(40)])
            lines.append("")
        lines.append("## Improvements")
        lines.extend([f"Improvement {j}" for j in range(90)])
        lines.append("")
        lines.append("## Bug fixes")
        lines.extend([f"Bug fix {j}" for j in range(140)])
        content = "\n".join(lines)

        batches = create_section_batches(content, max_lines_per_batch=200)
        for i, batch in enumerate(batches):
            line_count = len(batch.split('\n'))
            self.assertLessEqual(
                line_count, 250,
                f"Batch {i} has {line_count} lines, expected <=250",
            )
        rejoined = "\n".join(batches)
        self.assertEqual(rejoined, content)

    def test_large_file_without_headings_is_not_split_by_line_count(self):
        lines = [f"Line {i}" for i in range(500)]
        content = "\n".join(lines)
        batches = create_section_batches(content, max_lines_per_batch=200)
        self.assertEqual(batches, [content])

    def test_large_file_without_headings_splits_at_blank_lines(self):
        lines = []
        for paragraph in range(10):
            lines.extend([f"Paragraph {paragraph}, line {i}" for i in range(50)])
            lines.append("")
        content = "\n".join(lines)

        batches = create_section_batches(content, max_lines_per_batch=200)

        self.assertGreater(len(batches), 1)
        self.assertEqual("\n".join(batches), content)
        self.assertTrue(all(batch.endswith("\n") for batch in batches[:-1]))

    def test_single_large_section_is_not_split_by_line_count(self):
        content = "\n".join(["# Large section", *[f"Line {i}" for i in range(500)]])
        batches = create_section_batches(content, max_lines_per_batch=200)
        self.assertEqual(batches, [content])

    def test_single_large_section_splits_at_blank_lines(self):
        lines = ["# Large section", ""]
        for paragraph in range(10):
            lines.extend([f"Paragraph {paragraph}, line {i}" for i in range(50)])
            lines.append("")
        content = "\n".join(lines)

        batches = create_section_batches(content, max_lines_per_batch=200)

        self.assertGreater(len(batches), 1)
        self.assertEqual("\n".join(batches), content)

    def test_heading_like_lines_in_code_fences_are_not_section_boundaries(self):
        lines = ["# Title", "", "## First section", "", "```bash"]
        lines.extend([f"# shell comment {i}" for i in range(220)])
        lines.extend(["```", "", "## Second section", "", "Body"])
        content = "\n".join(lines)

        batches = create_section_batches(content, max_lines_per_batch=100)

        self.assertEqual("\n".join(batches), content)
        self.assertTrue(any(batch.startswith("## Second section") for batch in batches))
        self.assertFalse(any(batch.startswith("# shell comment") for batch in batches))

    def test_blank_lines_inside_code_fences_are_not_split_boundaries(self):
        lines = ["# Large code sample", "", "```text"]
        for i in range(250):
            lines.append("")
            lines.append(f"code line {i}")
        lines.append("```")
        content = "\n".join(lines)

        batches = create_section_batches(content, max_lines_per_batch=100)

        self.assertEqual("\n".join(batches), content)
        self.assertEqual(len(batches), 2)
        self.assertTrue(batches[1].startswith("```text"))
        self.assertFalse(any(batch.startswith("code line") for batch in batches))

    def test_content_preserved_after_roundtrip(self):
        """Joining all batches with \\n must reproduce the original content."""
        lines = ["---", "title: RN", "---", ""]
        for i in range(10):
            lines.append(f"## Heading {i}")
            lines.extend([f"Text {i}-{j}" for j in range(25)])
            lines.append("")
        content = "\n".join(lines)
        batches = create_section_batches(content, max_lines_per_batch=100)
        self.assertEqual("\n".join(batches), content)

    def test_release_notes_like_structure(self):
        """Simulate the actual release-8.5.7.md structure (~400 lines)."""
        lines = ["---", "title: TiDB 8.5.7 Release Notes", "summary: ...", "---", ""]
        lines.append("# TiDB 8.5.7 Release Notes")
        lines.extend([""] + ["Intro line. " * 8] * 5 + [""])
        lines.append("## Features")
        lines.append("")
        for sub in ["### Performance", "### Reliability", "### SQL",
                     "### Observability", "### Data migration"]:
            lines.append(sub)
            lines.extend([""] + ["Feature text. " * 15] * 12 + [""])
        lines.append("## Compatibility changes")
        lines.extend([""] + ["Change text. " * 10] * 30 + [""])
        lines.append("## Improvements")
        lines.extend([""] + ["+ " + "Improvement. " * 15] * 50 + [""])
        lines.append("## Bug fixes")
        lines.extend([""] + ["+ " + "Bug fix. " * 15] * 80)
        content = "\n".join(lines)

        batches = create_section_batches(content, max_lines_per_batch=200)

        self.assertGreater(len(batches), 1, "Should split into multiple batches")
        for i, batch in enumerate(batches):
            batch_lines = len(batch.split('\n'))
            self.assertLessEqual(
                batch_lines, 250,
                f"Batch {i} has {batch_lines} lines, expected <=250",
            )
        self.assertEqual("\n".join(batches), content)

class TestEnsureBlankLinesBeforeHeadings(unittest.TestCase):

    def test_adds_blank_line_before_heading(self):
        text = "Some text.\n## Heading"
        result = ensure_blank_lines_before_headings(text)
        self.assertEqual(result, "Some text.\n\n## Heading")

    def test_preserves_existing_blank_line(self):
        text = "Some text.\n\n## Heading"
        result = ensure_blank_lines_before_headings(text)
        self.assertEqual(result, "Some text.\n\n## Heading")

    def test_first_line_heading_no_extra_blank(self):
        text = "# Title\n\nContent"
        result = ensure_blank_lines_before_headings(text)
        self.assertEqual(result, "# Title\n\nContent")

    def test_multiple_headings(self):
        text = "Text\n## H1\nMore text\n### H2\nEnd"
        result = ensure_blank_lines_before_headings(text)
        self.assertEqual(result, "Text\n\n## H1\nMore text\n\n### H2\nEnd")

    def test_batch_boundary_simulation(self):
        """Simulates what happens when .strip() removes trailing blank line at batch boundary."""
        batch1_output = "Content of batch 1"
        batch2_output = "## Improvements\n\nImprovement details."
        joined = "\n".join([batch1_output, batch2_output])
        result = ensure_blank_lines_before_headings(joined)
        self.assertIn("\n\n## Improvements", result)

    def test_empty_content(self):
        self.assertEqual(ensure_blank_lines_before_headings(""), "")
        self.assertEqual(ensure_blank_lines_before_headings(None), None)

    def test_no_headings(self):
        text = "Just plain text\nNo headings here"
        result = ensure_blank_lines_before_headings(text)
        self.assertEqual(result, text)

    def test_code_block_hash_not_treated_as_heading(self):
        text = "Text\n```\n# not a heading\n```\n## Real heading"
        result = ensure_blank_lines_before_headings(text)
        self.assertEqual(
            result,
            "Text\n```\n# not a heading\n```\n\n## Real heading",
        )


class TestJoinTranslatedBatches(unittest.TestCase):

    def test_preserves_blank_boundary_stripped_by_ai(self):
        source_batches = ["First paragraph\n", "Second paragraph"]
        translated_batches = ["第一段", "第二段"]

        result = _join_translated_batches(source_batches, translated_batches)

        self.assertEqual(result, "第一段\n\n第二段")

    def test_does_not_add_extra_blank_line_when_translation_preserves_it(self):
        source_batches = ["First paragraph\n", "Second paragraph"]
        translated_batches = ["第一段\n", "\n第二段"]

        result = _join_translated_batches(source_batches, translated_batches)

        self.assertEqual(result, "第一段\n\n第二段")

    def test_section_boundary_without_source_blank_uses_one_newline(self):
        source_batches = ["First section", "## Second section"]
        translated_batches = ["第一节", "## 第二节"]

        result = _join_translated_batches(source_batches, translated_batches)

        self.assertEqual(result, "第一节\n## 第二节")


class TestStripAiMarkdownWrapper(unittest.TestCase):

    def test_strips_markdown_fence(self):
        text = "```markdown\n# Title\n\nContent\n```"
        result = strip_ai_markdown_wrapper(text)
        self.assertEqual(result, "# Title\n\nContent")

    def test_strips_md_fence(self):
        text = "```md\n# Title\n\nContent\n```"
        result = strip_ai_markdown_wrapper(text)
        self.assertEqual(result, "# Title\n\nContent")

    def test_strips_bare_fence(self):
        text = "```\n# Title\n\nContent\n```"
        result = strip_ai_markdown_wrapper(text)
        self.assertEqual(result, "# Title\n\nContent")

    def test_preserves_non_markdown_fence(self):
        text = "```python\nprint('hello')\n```"
        result = strip_ai_markdown_wrapper(text)
        self.assertEqual(result, text)

    def test_preserves_plain_text(self):
        text = "# Title\n\nContent"
        result = strip_ai_markdown_wrapper(text)
        self.assertEqual(result, text)

    def test_empty_input(self):
        self.assertEqual(strip_ai_markdown_wrapper(""), "")
        self.assertEqual(strip_ai_markdown_wrapper(None), None)


class TestProcessAddedFilesBlankLineNormalization(unittest.TestCase):

    def test_blank_lines_normalized_after_batch_join(self):
        """AI strips trailing blank lines from each batch; joining must still produce
        correct blank lines before headings."""
        repo_config = {
            "target_local_path": "",
            "source_language": "English",
            "target_language": "Chinese",
        }
        source_content = "\n".join([
            "# Title",
            "",
            "Content before section.",
            "",
            "## Section A",
            "",
            "Section A content.",
            "",
            "## Section B",
            "",
            "Section B content.",
        ])

        def fake_translate(batch, *args, **kwargs):
            return batch.strip()

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_config["target_local_path"] = tmpdir
            target_file = Path(tmpdir, "doc.md")

            with mock.patch("file_adder.translate_file_batch", side_effect=fake_translate):
                success, failures = process_added_files(
                    {"doc.md": source_content},
                    {"mode": "commit"},
                    object(),
                    object(),
                    repo_config,
                    return_details=True,
                )

            self.assertTrue(success)
            result = target_file.read_text(encoding="utf-8")
            for heading in ["## Section A", "## Section B"]:
                idx = result.index(heading)
                before = result[:idx]
                self.assertTrue(
                    before.endswith("\n\n"),
                    f"Expected blank line before '{heading}', got: ...{before[-20:]!r}",
                )


if __name__ == "__main__":
    unittest.main()

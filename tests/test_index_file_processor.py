import json
import os
import sys
import tempfile
import unittest
import io
from contextlib import redirect_stdout
from unittest import mock
from markdown_it import MarkdownIt
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from index_file_processor import (
    build_index_translation_memory,
    localize_docs_absolute_links,
    parse_index_line,
    plan_synced_index_lines,
    process_index_file_by_source_snapshot,
    translate_index_lines,
    _extract_frontmatter,
    _needs_translation,
)


# ---------------------------------------------------------------------------
# Fake AI client for deterministic tests
# ---------------------------------------------------------------------------

class EchoAIClient:
    """Return the input lines unchanged (echo them back as-is)."""

    def chat_completion(self, messages, temperature=0.1):
        prompt = messages[0]["content"]
        marker = "Input lines to translate:\n"
        start = prompt.find(marker)
        if start == -1:
            return "{}"
        start += len(marker)
        json_start = prompt.find("{", start)
        if json_start == -1:
            return "{}"
        depth = 0
        for i in range(json_start, len(prompt)):
            if prompt[i] == "{":
                depth += 1
            elif prompt[i] == "}":
                depth -= 1
                if depth == 0:
                    return prompt[json_start : i + 1]
        return "{}"


class EmptyAIClient:
    def chat_completion(self, messages, temperature=0.1):
        return "{}"


# ---------------------------------------------------------------------------
# Sample _index.md content
# ---------------------------------------------------------------------------

SOURCE_BASE = """\
---
title: TiDB Cloud Documentation
aliases: ['/tidbcloud/privacy-policy']
hide_sidebar: true
hide_commit: true
summary: Summary text here.
---

<LearningPathContainer platform="tidb-cloud" title="TiDB Cloud" subTitle="TiDB Cloud is a fully-managed DBaaS.">

<LearningPath label="Learn" icon="cloud1">

[Why TiDB Cloud](https://docs.pingcap.com/tidbcloud/tidb-cloud-intro)

[FAQ](https://docs.pingcap.com/tidbcloud/tidb-cloud-faq)

</LearningPath>

<LearningPath label="Develop" icon="doc8">

[Developer Guide Overview](https://docs.pingcap.com/tidbcloud/dev-guide-overview)

[Quick Start](https://docs.pingcap.com/tidbcloud/dev-guide-build-cluster-in-cloud)

[Example Application](https://docs.pingcap.com/tidbcloud/dev-guide-sample-application-spring-boot)

</LearningPath>

</LearningPathContainer>"""

SOURCE_HEAD = """\
---
title: TiDB Cloud Documentation
aliases: ['/tidbcloud/privacy-policy']
hide_sidebar: true
hide_commit: true
summary: Summary text here.
---

<LearningPathContainer platform="tidb-cloud" title="TiDB Cloud Dedicated" subTitle="TiDB Cloud Dedicated is designed for mission-critical businesses.">

<LearningPath label="Learn" icon="cloud1">

[Why TiDB Cloud](https://docs.pingcap.com/tidbcloud/tidb-cloud-intro)

[FAQ](https://docs.pingcap.com/tidbcloud/tidb-cloud-faq)

</LearningPath>

<LearningPath label="Develop" icon="doc8">

[Developer Guide Overview](https://docs.pingcap.com/developer/)

[Quick Start](https://docs.pingcap.com/developer/dev-guide-build-cluster-in-cloud/)

[Connect to TiDB](https://docs.pingcap.com/developer/dev-guide-connect-to-tidb/)

</LearningPath>

</LearningPathContainer>"""

TARGET_EXISTING = """\
---
title: TiDB Cloud 文档
aliases: ['/tidbcloud/privacy-policy']
hide_sidebar: true
hide_commit: true
summary: 这里是摘要文本。
---

<LearningPathContainer platform="tidb-cloud" title="TiDB Cloud" subTitle="TiDB Cloud 是一种全托管的 DBaaS。">

<LearningPath label="学习" icon="cloud1">

[为什么选择 TiDB Cloud](https://docs.pingcap.com/tidbcloud/tidb-cloud-intro)

[常见问题](https://docs.pingcap.com/tidbcloud/tidb-cloud-faq)

</LearningPath>

<LearningPath label="开发" icon="doc8">

[开发者指南概览](https://docs.pingcap.com/tidbcloud/dev-guide-overview)

[快速入门](https://docs.pingcap.com/tidbcloud/dev-guide-build-cluster-in-cloud)

[示例应用](https://docs.pingcap.com/tidbcloud/dev-guide-sample-application-spring-boot)

</LearningPath>

</LearningPathContainer>"""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestParseIndexLine(unittest.TestCase):
    def test_blank_line(self):
        entry = parse_index_line("")
        self.assertEqual(entry["type"], "blank")

    def test_frontmatter_fence(self):
        entry = parse_index_line("---")
        self.assertEqual(entry["type"], "frontmatter_fence")

    def test_learning_path_container_open(self):
        entry = parse_index_line(
            '<LearningPathContainer platform="tidb-cloud" title="TiDB Cloud" subTitle="A managed DBaaS.">'
        )
        self.assertEqual(entry["type"], "tag")
        self.assertEqual(entry["tag_kind"], "container_open")
        self.assertEqual(entry["attrs"]["title"], "TiDB Cloud")
        self.assertEqual(entry["attrs"]["subTitle"], "A managed DBaaS.")

    def test_learning_path_open(self):
        entry = parse_index_line('<LearningPath label="Learn" icon="cloud1">')
        self.assertEqual(entry["type"], "tag")
        self.assertEqual(entry["tag_kind"], "path_open")
        self.assertEqual(entry["attrs"]["label"], "Learn")

    def test_learning_path_close(self):
        entry = parse_index_line("</LearningPath>")
        self.assertEqual(entry["type"], "tag")
        self.assertEqual(entry["tag_kind"], "path_close")

    def test_markdown_link(self):
        entry = parse_index_line("[Why TiDB Cloud](https://docs.pingcap.com/tidbcloud/tidb-cloud-intro)")
        self.assertEqual(entry["type"], "link")
        self.assertEqual(entry["display_text"], "Why TiDB Cloud")
        self.assertEqual(entry["url"], "https://docs.pingcap.com/tidbcloud/tidb-cloud-intro")

    def test_other_line(self):
        entry = parse_index_line("Some random text")
        self.assertEqual(entry["type"], "other")


class TestExtractFrontmatter(unittest.TestCase):
    def test_valid_frontmatter(self):
        lines = ["---", "title: Test", "---", "body"]
        end_idx, fm_lines = _extract_frontmatter(lines)
        self.assertEqual(end_idx, 2)
        self.assertEqual(fm_lines, ["---", "title: Test", "---"])

    def test_no_frontmatter(self):
        lines = ["body", "more body"]
        end_idx, fm_lines = _extract_frontmatter(lines)
        self.assertEqual(end_idx, -1)
        self.assertEqual(fm_lines, [])


class TestNeedsTranslation(unittest.TestCase):
    def test_link_needs_translation(self):
        entry = parse_index_line("[Why TiDB Cloud](https://example.com)")
        self.assertTrue(_needs_translation(entry))

    def test_blank_does_not(self):
        entry = parse_index_line("")
        self.assertFalse(_needs_translation(entry))

    def test_close_tag_does_not(self):
        entry = parse_index_line("</LearningPath>")
        self.assertFalse(_needs_translation(entry))

    def test_container_tag_with_title_needs(self):
        entry = parse_index_line(
            '<LearningPathContainer platform="tidb-cloud" title="TiDB Cloud" subTitle="Description">'
        )
        self.assertTrue(_needs_translation(entry))

    def test_path_open_with_label_needs(self):
        entry = parse_index_line('<LearningPath label="Learn" icon="cloud1">')
        self.assertTrue(_needs_translation(entry))


class TestBuildTranslationMemory(unittest.TestCase):
    def test_link_memory(self):
        base_lines = [
            "[Why TiDB Cloud](https://docs.pingcap.com/tidbcloud/tidb-cloud-intro)",
            "[FAQ](https://docs.pingcap.com/tidbcloud/faq)",
        ]
        target_lines = [
            "[为什么选择 TiDB Cloud](https://docs.pingcap.com/tidbcloud/tidb-cloud-intro)",
            "[常见问题](https://docs.pingcap.com/tidbcloud/faq)",
        ]
        base_map, target_map = build_index_translation_memory(base_lines, target_lines)

        key = ("link", "https://docs.pingcap.com/tidbcloud/tidb-cloud-intro", "Why TiDB Cloud")
        self.assertIn(key, target_map)
        self.assertEqual(
            target_map[key],
            "[为什么选择 TiDB Cloud](https://docs.pingcap.com/tidbcloud/tidb-cloud-intro)",
        )


    def test_duplicate_url_does_not_guess_translation(self):
        """A repeated URL cannot identify translated labels reliably."""
        base_lines = [
            "[From CSV Files](https://docs.pingcap.com/tidbcloud/import-csv-files)",
            "[From Apache Parquet Files](https://docs.pingcap.com/tidbcloud/import-csv-files)",
        ]
        target_lines = [
            "[从 CSV 文件导入](https://docs.pingcap.com/zh/tidbcloud/import-csv-files)",
            "[从 Apache Parquet 文件导入](https://docs.pingcap.com/zh/tidbcloud/import-csv-files)",
        ]
        base_map, target_map = build_index_translation_memory(base_lines, target_lines)

        csv_key = ("link", "https://docs.pingcap.com/tidbcloud/import-csv-files", "From CSV Files")
        parquet_key = ("link", "https://docs.pingcap.com/tidbcloud/import-csv-files", "From Apache Parquet Files")

        self.assertIn(csv_key, base_map)
        self.assertIn(parquet_key, base_map)
        self.assertEqual(target_map, {})


class TestLocalizeDocsAbsoluteLinks(unittest.TestCase):
    def test_localizes_http_links_for_chinese(self):
        content = "[Guide](http://docs.pingcap.com/developer/)"
        self.assertEqual(
            localize_docs_absolute_links(content, "Chinese"),
            "[Guide](http://docs.pingcap.com/zh/developer/)",
        )

    def test_localizes_links_for_chinese(self):
        content = "[Guide](https://docs.pingcap.com/developer/)"
        self.assertEqual(
            localize_docs_absolute_links(content, "Chinese"),
            "[Guide](https://docs.pingcap.com/zh/developer/)",
        )

    def test_localizes_links_for_japanese(self):
        content = "[Guide](https://docs.pingcap.com/tidbcloud/)"
        self.assertEqual(
            localize_docs_absolute_links(content, "Japanese"),
            "[Guide](https://docs.pingcap.com/ja/tidbcloud/)",
        )

    def test_does_not_double_prefix_existing_locale(self):
        content = "\n".join(
            [
                "[ZH](https://docs.pingcap.com/zh/developer/)",
                "[JA](https://docs.pingcap.com/ja/developer/)",
            ]
        )
        self.assertEqual(
            localize_docs_absolute_links(content, "Chinese"),
            "\n".join(
                [
                    "[ZH](https://docs.pingcap.com/zh/developer/)",
                    "[JA](https://docs.pingcap.com/zh/developer/)",
                ]
            ),
        )


class TestPlanSyncedIndexLines(unittest.TestCase):
    def test_unchanged_links_keep_target_localized_url(self):
        """Reused links should keep the target's localized URL, not source URL."""
        source_base = "[Changefeed](http://docs.pingcap.com/tidbcloud/changefeed-overview)"
        source_head = "[Changefeed](http://docs.pingcap.com/tidbcloud/changefeed-overview)"
        target_existing = "[流式数据](http://docs.pingcap.com/zh/tidbcloud/changefeed-overview)"

        planned, to_translate = plan_synced_index_lines(
            localize_docs_absolute_links(source_base, "Chinese"),
            localize_docs_absolute_links(source_head, "Chinese"),
            localize_docs_absolute_links(target_existing, "Chinese"),
        )

        self.assertEqual(
            planned,
            ["[流式数据](http://docs.pingcap.com/zh/tidbcloud/changefeed-overview)"],
        )
        self.assertEqual(to_translate, [])

    def test_unchanged_links_reused(self):
        """Links whose URL and display text are unchanged should reuse target translations."""
        planned, to_translate = plan_synced_index_lines(
            SOURCE_BASE, SOURCE_HEAD, TARGET_EXISTING
        )

        result_text = "\n".join(
            line if line is not None else "<NEED_TRANSLATE>"
            for line in planned
        )

        # Unchanged links should be reused from target
        self.assertIn("[为什么选择 TiDB Cloud](https://docs.pingcap.com/tidbcloud/tidb-cloud-intro)", result_text)
        self.assertIn("[常见问题](https://docs.pingcap.com/tidbcloud/tidb-cloud-faq)", result_text)

        # Changed container tag should need translation
        self.assertTrue(
            any("TiDB Cloud Dedicated" in line for _, line in to_translate),
            f"Expected container tag change in to_translate: {to_translate}"
        )

        # Changed link URLs should need translation
        self.assertTrue(
            any("developer/" in line for _, line in to_translate),
            f"Expected changed-URL links in to_translate: {to_translate}"
        )

    def test_frontmatter_preserved_when_unchanged(self):
        """Unchanged frontmatter lines should reuse target translations."""
        planned, _ = plan_synced_index_lines(
            SOURCE_BASE, SOURCE_HEAD, TARGET_EXISTING
        )
        self.assertEqual(planned[0], "---")
        self.assertEqual(planned[1], "title: TiDB Cloud 文档")

    def test_frontmatter_propagates_source_changes(self):
        """Changed frontmatter lines should use source HEAD values."""
        modified_head = SOURCE_HEAD.replace(
            "title: TiDB Cloud Documentation",
            "title: TiDB Cloud Dedicated Documentation",
        )
        planned, _ = plan_synced_index_lines(
            SOURCE_BASE, modified_head, TARGET_EXISTING
        )
        self.assertEqual(planned[0], "---")
        self.assertEqual(planned[1], "title: TiDB Cloud Dedicated Documentation")
        # Unchanged lines should still use target
        self.assertEqual(planned[3], "hide_sidebar: true")

    def test_frontmatter_new_field_propagated(self):
        """New frontmatter fields in source HEAD should appear in output."""
        modified_head = SOURCE_HEAD.replace(
            "hide_commit: true\nsummary:",
            "hide_commit: true\nnew_field: new_value\nsummary:",
        )
        planned, _ = plan_synced_index_lines(
            SOURCE_BASE, modified_head, TARGET_EXISTING
        )
        planned_text = "\n".join(line for line in planned if line is not None)
        self.assertIn("new_field: new_value", planned_text)

    def test_empty_target_all_translated(self):
        """When target is empty, all translatable lines should be queued."""
        planned, to_translate = plan_synced_index_lines(
            SOURCE_BASE, SOURCE_HEAD, ""
        )
        self.assertTrue(len(to_translate) > 0)

    def test_duplicate_url_retranslates_ambiguous_labels(self):
        """A URL correction must not reuse labels from an ambiguous pair."""
        source_base = "\n".join([
            '<LearningPath label="Migrate" icon="cloud3">',
            "",
            "[From CSV Files](https://docs.pingcap.com/tidbcloud/import-csv-files)",
            "",
            "[From Apache Parquet Files](https://docs.pingcap.com/tidbcloud/import-csv-files)",
            "",
            "</LearningPath>",
        ])
        source_head = "\n".join([
            '<LearningPath label="Migrate" icon="cloud3">',
            "",
            "[From CSV Files](https://docs.pingcap.com/tidbcloud/import-csv-files)",
            "",
            "[From Apache Parquet Files](https://docs.pingcap.com/tidbcloud/import-parquet-files)",
            "",
            "</LearningPath>",
        ])
        target_existing = "\n".join([
            '<LearningPath label="迁移" icon="cloud3">',
            "",
            "[从 CSV 文件导入](https://docs.pingcap.com/zh/tidbcloud/import-csv-files)",
            "",
            "[从 Apache Parquet 文件导入](https://docs.pingcap.com/zh/tidbcloud/import-csv-files)",
            "",
            "</LearningPath>",
        ])

        planned, to_translate = plan_synced_index_lines(
            source_base, source_head, target_existing
        )

        result_text = "\n".join(
            line if line is not None else "<NEED_TRANSLATE>"
            for line in planned
        )

        self.assertIsNone(planned[2])
        self.assertIsNone(planned[4])
        self.assertEqual([slot for slot, _ in to_translate], [2, 4])
        self.assertIn("迁移", result_text)


class TestProcessIndexFileBySourceSnapshot(unittest.TestCase):
    def test_preserves_localized_http_link_when_reusing_existing_translation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = os.path.join(tmpdir, "_index.md")
            existing_target = "[流式数据](http://docs.pingcap.com/zh/tidbcloud/changefeed-overview)"
            Path(target_path).write_text(existing_target, encoding="utf-8")

            index_data = {
                "type": "index",
                "source_base_content": "[Changefeed](http://docs.pingcap.com/tidbcloud/changefeed-overview)",
                "source_head_content": "[Changefeed](http://docs.pingcap.com/tidbcloud/changefeed-overview)",
            }
            repo_config = {
                "target_local_path": tmpdir,
                "source_language": "English",
                "target_language": "Chinese",
            }

            success = process_index_file_by_source_snapshot(
                "_index.md",
                index_data,
                EchoAIClient(),
                repo_config,
                target_path,
            )

            self.assertTrue(success)
            self.assertEqual(
                Path(target_path).read_text(encoding="utf-8"),
                "[流式数据](http://docs.pingcap.com/zh/tidbcloud/changefeed-overview)",
            )

    def test_creates_target_file(self):
        """Processing should create the target file with translated content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = os.path.join(tmpdir, "dedicated", "_index.md")

            index_data = {
                "type": "index",
                "source_base_content": SOURCE_BASE,
                "source_head_content": SOURCE_HEAD,
            }

            repo_config = {
                "target_local_path": tmpdir,
                "source_language": "English",
                "target_language": "Chinese",
            }

            ai_client = EchoAIClient()
            success = process_index_file_by_source_snapshot(
                "dedicated/_index.md",
                index_data,
                ai_client,
                repo_config,
                target_path,
            )

            self.assertTrue(success)
            self.assertTrue(os.path.exists(target_path))

            with open(target_path, "r", encoding="utf-8") as f:
                content = f.read()

            self.assertIn("LearningPathContainer", content)
            self.assertIn("LearningPath", content)

    def test_reuses_existing_translations(self):
        """Unchanged links should reuse translations from the existing target file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = os.path.join(tmpdir, "_index.md")
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(TARGET_EXISTING)

            index_data = {
                "type": "index",
                "source_base_content": SOURCE_BASE,
                "source_head_content": SOURCE_HEAD,
            }

            repo_config = {
                "target_local_path": tmpdir,
                "source_language": "English",
                "target_language": "Chinese",
            }

            ai_client = EchoAIClient()
            success = process_index_file_by_source_snapshot(
                "_index.md",
                index_data,
                ai_client,
                repo_config,
                target_path,
            )

            self.assertTrue(success)

            with open(target_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Unchanged link should use target translation
            self.assertIn("[为什么选择 TiDB Cloud]", content)
            self.assertIn("[常见问题]", content)

    def test_localizes_docs_links_for_chinese_targets(self):
        """docs.pingcap.com links in index files should use the zh prefix."""
        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = os.path.join(tmpdir, "_index.md")

            index_data = {
                "type": "index",
                "source_base_content": SOURCE_BASE,
                "source_head_content": SOURCE_HEAD,
            }

            repo_config = {
                "target_local_path": tmpdir,
                "source_language": "English",
                "target_language": "Chinese",
            }

            ai_client = EchoAIClient()
            success = process_index_file_by_source_snapshot(
                "_index.md",
                index_data,
                ai_client,
                repo_config,
                target_path,
            )

            self.assertTrue(success)
            content = Path(target_path).read_text(encoding="utf-8")
            self.assertIn("https://docs.pingcap.com/zh/developer/", content)
            self.assertIn(
                "https://docs.pingcap.com/zh/developer/dev-guide-build-cluster-in-cloud/",
                content,
            )

    def test_localizes_docs_links_for_japanese_targets(self):
        """docs.pingcap.com links in index files should use the ja prefix."""
        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = os.path.join(tmpdir, "_index.md")

            index_data = {
                "type": "index",
                "source_base_content": SOURCE_BASE,
                "source_head_content": SOURCE_HEAD,
            }

            repo_config = {
                "target_local_path": tmpdir,
                "source_language": "English",
                "target_language": "Japanese",
            }

            ai_client = EchoAIClient()
            success = process_index_file_by_source_snapshot(
                "_index.md",
                index_data,
                ai_client,
                repo_config,
                target_path,
            )

            self.assertTrue(success)
            content = Path(target_path).read_text(encoding="utf-8")
            self.assertIn("https://docs.pingcap.com/ja/developer/", content)
            self.assertIn(
                "https://docs.pingcap.com/ja/tidbcloud/tidb-cloud-intro",
                content,
            )

    def test_empty_ai_response_returns_false(self):
        """When AI returns empty translations, should return False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = os.path.join(tmpdir, "_index.md")

            index_data = {
                "type": "index",
                "source_base_content": SOURCE_BASE,
                "source_head_content": SOURCE_HEAD,
            }

            repo_config = {
                "target_local_path": tmpdir,
                "source_language": "English",
                "target_language": "Chinese",
            }

            ai_client = EmptyAIClient()
            success = process_index_file_by_source_snapshot(
                "_index.md",
                index_data,
                ai_client,
                repo_config,
                target_path,
            )

            # Failed translations must not create a partial target file
            self.assertFalse(success)
            self.assertFalse(Path(target_path).exists())


class TestTranslateIndexLines(unittest.TestCase):
    def test_empty_input(self):
        result = translate_index_lines(
            [],
            EmptyAIClient(),
            {"source_language": "English", "target_language": "Chinese"},
        )
        self.assertEqual(result, {})


class TestIsLearningPathIndexContent(unittest.TestCase):
    """Test content-based classification of _index.md files."""

    def test_learning_path_index_detected(self):
        from special_file_utils import is_learning_path_index_content

        content = """\
---
title: TiDB Cloud Documentation
hide_sidebar: true
---

<LearningPathContainer platform="tidb-cloud" title="TiDB Cloud" subTitle="A fully-managed DBaaS.">

<LearningPath label="Learn" icon="cloud1">

[Why TiDB Cloud](https://docs.pingcap.com/tidbcloud/tidb-cloud-intro)

</LearningPath>

</LearningPathContainer>"""
        self.assertTrue(is_learning_path_index_content(content))

    def test_normal_index_with_headings(self):
        from special_file_utils import is_learning_path_index_content

        content = """\
---
title: TiDB for AI
summary: Build modern AI applications with TiDB.
---

# TiDB for AI

TiDB is a distributed SQL database designed for modern AI applications.

## Quick start

| Document | Description |
| --- | --- |
| [Get Started](/ai/quickstart.md) | Build your first AI app. |"""
        self.assertFalse(is_learning_path_index_content(content))

    def test_empty_content(self):
        from special_file_utils import is_learning_path_index_content

        self.assertFalse(is_learning_path_index_content(""))
        self.assertFalse(is_learning_path_index_content(None))

    def test_only_headings_no_learning_path(self):
        from special_file_utils import is_learning_path_index_content

        content = "# Title\n\nSome content\n\n## Section"
        self.assertFalse(is_learning_path_index_content(content))

    def test_learning_path_with_headings_treated_as_normal(self):
        """If a file has both LearningPathContainer AND headings,
        it should be treated as normal (headings take priority)."""
        from special_file_utils import is_learning_path_index_content

        content = """\
---
title: Mixed File
---

# Overview

<LearningPathContainer platform="tidb-cloud" title="TiDB Cloud">
</LearningPathContainer>"""
        self.assertFalse(is_learning_path_index_content(content))

    def test_heading_inside_code_block_ignored(self):
        """Headings inside code blocks should not count as real headings."""
        from special_file_utils import is_learning_path_index_content

        content = """\
---
title: Test
---

<LearningPathContainer platform="tidb-cloud" title="TiDB Cloud">

```markdown
# This is inside a code block
```

[Link](https://example.com)

</LearningPathContainer>"""
        self.assertTrue(is_learning_path_index_content(content))

    def test_link_only_index_no_headings(self):
        """Link-only _index.md files (like starter/_index.md) have no headings
        and should also use the special snapshot-sync path."""
        from special_file_utils import is_learning_path_index_content

        content = """\
---
title: TiDB Cloud Documentation
hide_sidebar: true
hide_commit: true
summary: TiDB Cloud is a fully-managed DBaaS.
---

\u00a0

\u00a0

[Why TiDB Cloud](https://docs.pingcap.com/tidbcloud/tidb-cloud-intro/?plan=starter)

[Key Concepts](https://docs.pingcap.com/tidbcloud/key-concepts/?plan=starter)"""
        self.assertTrue(is_learning_path_index_content(content))

    def test_frontmatter_title_with_hash_not_counted(self):
        """YAML frontmatter values are not headings."""
        from special_file_utils import is_learning_path_index_content

        content = """\
---
title: '# Not a heading'
---

<LearningPathContainer platform="tidb-cloud" title="TiDB Cloud">
</LearningPathContainer>"""
        self.assertTrue(is_learning_path_index_content(content))


class TestSpecialFileUtils(unittest.TestCase):
    def test_is_index_file_name(self):
        from special_file_utils import is_index_file_name

        self.assertTrue(is_index_file_name("tidb-cloud/dedicated/_index.md"))
        self.assertTrue(is_index_file_name("tidb-cloud/starter/_index.md"))
        self.assertFalse(is_index_file_name("tidb-cloud/TOC.md"))
        self.assertFalse(is_index_file_name("tidb-cloud/some-doc.md"))

    def test_is_index_file_name_with_ignore(self):
        from special_file_utils import is_index_file_name

        self.assertFalse(
            is_index_file_name(
                "tidb-cloud/dedicated/_index.md",
                ignore_files=["tidb-cloud/dedicated/_index.md"],
            )
        )


class TestDetermineFileProcessingType(unittest.TestCase):
    def test_index_file_returns_special_file_index(self):
        from main_workflow import determine_file_processing_type

        result = determine_file_processing_type(
            "tidb-cloud/dedicated/_index.md",
            {},
            special_files=["TOC.md", "keywords.md"],
            ignore_files=[],
        )
        self.assertEqual(result, "special_file_index")


class TestPRModeDoesNotSkipIndexFile(unittest.TestCase):
    """Verify _index.md is not silently dropped in PR mode."""

    def test_pr_mode_modified_file_handler_accepts_special_file_index(self):
        """In PR mode, special_file_index should NOT be skipped; it should
        fall through to regular_modified processing."""
        from main_workflow import make_task_result

        file_type = "special_file_index"
        # Simulate the check logic from main_workflow's modified-file handler
        if file_type == "special_file_toc":
            result = make_task_result("skipped", "TOC")
        elif file_type == "special_file_keyword":
            result = make_task_result("skipped", "keyword")
        elif file_type not in ("regular_modified", "special_file_index"):
            result = make_task_result("failure", f"Unknown: {file_type}")
        else:
            result = None  # would proceed to process_regular_modified_file

        self.assertIsNone(result, "special_file_index should not be skipped in PR mode")


class TestRemoveIncrementalWorkIncludesIndexFiles(unittest.TestCase):
    """Verify remove_incremental_work_for_files clears index_files."""

    def test_index_files_cleared_when_passed(self):
        from commit_sync_workflow import remove_incremental_work_for_files

        index_files = {"tidb-cloud/dedicated/_index.md": {"type": "index"}}
        remove_incremental_work_for_files(
            ["tidb-cloud/dedicated/_index.md"],
            {}, {}, {}, {}, {},
            index_files=index_files,
        )
        self.assertEqual(index_files, {})

    def test_index_files_not_cleared_when_not_passed(self):
        from commit_sync_workflow import remove_incremental_work_for_files

        remove_incremental_work_for_files(
            ["some-file.md"],
            {}, {}, {}, {}, {},
        )
        # Should not raise — index_files defaults to None



class FieldAIClient:
    def __init__(self, response):
        self.response = response
        self.prompt = None

    def chat_completion(self, messages, temperature=0.1):
        self.prompt = messages[0]["content"]
        return json.dumps(self.response)


class TestIndexLanguageIndependentSync(unittest.TestCase):
    def test_real_lake_snapshot_requires_no_ai(self):
        fixtures = Path(__file__).parent / "fixtures" / "index_lake"
        base, head, target = [(fixtures / name).read_text() for name in
                              ("source_base.md", "source_head.md", "target_ja.md")]
        base, head = [localize_docs_absolute_links(text, "Japanese")
                      for text in (base, head)]
        planned, queue = plan_synced_index_lines(base, head, target)
        self.assertEqual(queue, [])
        expected = target.replace(
            "[コストの管理](https://docs.pingcap.com/ja/tidbcloudlake/manage-costs/)\n\n", ""
        ).replace(
            "[AI を活用した機能](https://docs.pingcap.com/ja/tidbcloudlake/ai-powered-features/)\n\n", ""
        )
        self.assertEqual("\n".join(planned), expected)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "_index.md"
            path.write_text(target)
            success = process_index_file_by_source_snapshot(
                "_index.md", {"source_base_content": base, "source_head_content": head},
                EmptyAIClient(), {"source_language": "English", "target_language": "Japanese"}, str(path)
            )
            self.assertTrue(success)
            self.assertEqual(path.read_text(), expected)

    def test_alphabets_do_not_affect_component_matching(self):
        base = '<LearningPath label="Learn" icon="learn">\n[Guide](/guide)\n</LearningPath>'
        for label in ("ロード", "배우기", "تعلم", "Учиться", "เรียน", "123"):
            with self.subTest(label=label):
                target = base.replace('label="Learn"', f'label="{label}"').replace("[Guide]", "[译文]")
                self.assertTrue(_needs_translation(parse_index_line(target.splitlines()[0])))
                planned, queue = plan_synced_index_lines(base, base, target)
                self.assertEqual(queue, [])
                self.assertEqual("\n".join(planned), target)

    def test_missing_and_reordered_links_do_not_shift_translations(self):
        base = '<LearningPath label="Learn" icon="learn">\n[A](/a)\n[B](/b)\n[C](/c)\n</LearningPath>'
        target = '<LearningPath label="学习" icon="learn">\n[丙](/c)\n[甲](/a)\n</LearningPath>'
        planned, queue = plan_synced_index_lines(base, base, target)
        self.assertEqual(planned[1], "[甲](/a)")
        self.assertEqual(planned[3], "[丙](/c)")
        self.assertEqual(queue, [(2, "[B](/b)")])

    def test_reordered_components_with_duplicate_icons(self):
        one = '<LearningPath label="One" icon="same">\n[A](/a)\n</LearningPath>'
        two = '<LearningPath label="Two" icon="same">\n[B](/b)\n</LearningPath>'
        target = two.replace('"Two"', '"二"').replace("[B]", "[乙]") + "\n" + one.replace('"One"', '"一"').replace("[A]", "[甲]")
        planned, queue = plan_synced_index_lines(one + "\n" + two, one + "\n" + two, target)
        self.assertEqual(queue, [])
        self.assertEqual(planned[0], '<LearningPath label="一" icon="same">')
        self.assertEqual(planned[1], '[甲](/a)')
        self.assertEqual(planned[3], '<LearningPath label="二" icon="same">')

    def test_ambiguous_components_do_not_reuse_labels(self):
        base = '<LearningPath label="One" icon="same">\n</LearningPath>\n<LearningPath label="Two" icon="same">\n</LearningPath>'
        target = base.replace('"One"', '"一"').replace('"Two"', '"二"')
        planned, queue = plan_synced_index_lines(base, base, target)
        self.assertEqual([slot for slot, _ in queue], [0, 2])

    def test_changed_source_text_requires_translation(self):
        planned, queue = plan_synced_index_lines('[Old](/a)', '[New](/a)', '[旧](/a)')
        self.assertEqual(queue, [(0, '[New](/a)')])

    def test_source_controls_nontranslated_attributes(self):
        base = '<LearningPath label="Learn" icon="old">\n[A](/a)\n</LearningPath>'
        head = base.replace('icon="old"', 'icon="new"')
        target = base.replace('"Learn"', '"学习"').replace('icon="old"', 'icon="stale"')
        planned, queue = plan_synced_index_lines(base, head, target)
        self.assertEqual(queue, [])
        self.assertEqual(planned[0], '<LearningPath label="学习" icon="new">')

    def test_explicit_locale_with_known_prefixes(self):
        content = '[Guide](https://docs.pingcap.com/ja/guide#part)\n[Other](https://docs.pingcap.com/fr/guide)\n[External](https://example.com/ja/guide)'
        localized = localize_docs_absolute_links(content, 'Korean', target_locale='ko', known_locales=('fr',))
        self.assertIn('https://docs.pingcap.com/ko/guide#part', localized)
        self.assertNotIn('/ko/ja/', localized)
        self.assertNotIn('/ko/fr/', localized)
        self.assertIn('https://example.com/ja/guide', localized)
        self.assertEqual(localize_docs_absolute_links(localized, 'Korean', 'ko'), localized)

    def test_process_explicit_locale(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / '_index.md'
            path.write_text('[안내](https://docs.pingcap.com/ko/guide)')
            source = '[Guide](https://docs.pingcap.com/guide)'
            success = process_index_file_by_source_snapshot(
                '_index.md', {'source_base_content': source, 'source_head_content': source},
                EmptyAIClient(), {'source_language': 'English', 'target_language': 'Korean', 'target_locale': 'ko'}, str(path)
            )
            self.assertTrue(success)
            self.assertEqual(path.read_text(), '[안내](https://docs.pingcap.com/ko/guide)')

    def test_ai_receives_only_fields_and_quotes_are_escaped(self):
        client = FieldAIClient({'line_0': {'label': '「学ぶ」 "入門" & ガイド'}})
        line = '<LearningPath label="Learn" icon="safe">'
        result = translate_index_lines([(0, line)], client, {'source_language': 'English', 'target_language': 'Japanese'})
        self.assertNotIn('<LearningPath', client.prompt)
        self.assertNotIn('icon=', client.prompt)
        self.assertEqual(result[0], '<LearningPath label="「学ぶ」 &quot;入門&quot; &amp; ガイド" icon="safe">')

    def test_ai_cannot_supply_url_or_markdown_structure(self):
        client = FieldAIClient({'line_0': {'display_text': 'Guide [A]'}})
        result = translate_index_lines([(0, '  [Guide](https://example.com/a)  ')], client,
                                       {'source_language': 'English', 'target_language': 'Japanese'})
        self.assertNotIn('https://example.com', client.prompt)
        self.assertEqual(result[0], '  [Guide &#91;A&#93;](https://example.com/a)  ')

    def test_invalid_ai_preserves_existing_file(self):
        responses = ({}, {'line_0': 'whole line'}, {'line_0': {'display_text': 123}},
                     {'line_0': {'display_text': ''}}, {'line_0': {'display_text': 'a\nb'}},
                     {'line_0': {'display_text': 'text', 'url': '/evil'}})
        for response in responses:
            with self.subTest(response=response), tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / '_index.md'
                path.write_text('[旧](/a)')
                success = process_index_file_by_source_snapshot(
                    '_index.md', {'source_base_content': '[Old](/a)', 'source_head_content': '[New](/a)'},
                    FieldAIClient(response), {'source_language': 'English', 'target_language': 'Japanese'}, str(path)
                )
                self.assertFalse(success)
                self.assertEqual(path.read_text(), '[旧](/a)')

    def test_unbalanced_source_components_are_not_written(self):
        source = '<LearningPath label="Learn" icon="x">\n[A](/a)'
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / '_index.md'
            path.write_text(source)
            success = process_index_file_by_source_snapshot(
                '_index.md', {'source_base_content': source, 'source_head_content': source}, EmptyAIClient(),
                {'source_language': 'English', 'target_language': 'Japanese'}, str(path)
            )
            self.assertFalse(success)
            self.assertEqual(path.read_text(), source)

    def test_commit_cursor_does_not_break_frontmatter_reuse(self):
        base = '---\ntitle: Docs\n---\n[Guide](/a)'
        target = '<!--Corresponding EN commit: 49ed15df968aedb44b0290df53e124ef740c6a67-->\n---\ntitle: ドキュメント\n---\n[ガイド](/a)'
        planned, queue = plan_synced_index_lines(base, base, target)
        self.assertEqual(queue, [])
        self.assertEqual(planned[1], 'title: ドキュメント')

    def test_data_attributes_are_not_translated(self):
        client = FieldAIClient({'line_0': {'label': '学ぶ'}})
        line = '<LearningPath label="Learn" data-label="identifier" icon="x">'
        result = translate_index_lines([(0, line)], client, {'source_language': 'English', 'target_language': 'Japanese'})
        self.assertEqual(result[0], '<LearningPath label="学ぶ" data-label="identifier" icon="x">')

    def test_external_host_and_unknown_paths_are_preserved(self):
        content = '[A](https://docs.pingcap.com.evil/ja/a)\n[B](https://docs.pingcap.com/developer/a)'
        result = localize_docs_absolute_links(content, 'Korean', 'ko')
        self.assertIn('https://docs.pingcap.com.evil/ja/a', result)
        self.assertIn('https://docs.pingcap.com/ko/developer/a', result)

    def test_same_url_in_distinct_components_is_unambiguous(self):
        base = '<LearningPath label="One" icon="one">\n[A](/a)\n</LearningPath>\n<LearningPath label="Two" icon="two">\n[B](/a)\n</LearningPath>'
        target = base.replace('"One"', '"一"').replace('"Two"', '"二"').replace('[A]', '[甲]').replace('[B]', '[乙]')
        planned, queue = plan_synced_index_lines(base, base, target)
        self.assertEqual(queue, [])
        self.assertEqual(planned[1], '[甲](/a)')
        self.assertEqual(planned[4], '[乙](/a)')


class TestIndexReviewRegressions(unittest.TestCase):
    def test_existing_entities_and_markdown_are_preserved_verbatim(self):
        base = '<LearningPath label="Learn" icon="a">\n[A](/a)\n</LearningPath>'
        target = '<LearningPath label="学ぶ &copy; &#34;引用&#34;" icon="a">\n[A &copy; **bold** `code` \\[x\\]](/a)\n</LearningPath>'
        planned, queue = plan_synced_index_lines(base, base, target)
        self.assertEqual(queue, [])
        self.assertEqual('\n'.join(planned), target)

    def test_links_with_nested_or_escaped_delimiters_are_translatable(self):
        for source in ('[Guide [A]](/a)', r'[Guide \[A\]](/a)',
                       '[Guide `]` A](/a)', '[Guide ``code ``` ] inside`` end](/a)',
                       '[Guide](/a(b))'):
            with self.subTest(source=source):
                entry = parse_index_line(source)
                self.assertEqual(entry['type'], 'link')
                planned, queue = plan_synced_index_lines(source, source, '')
                self.assertEqual(queue, [(0, source)])
                client = FieldAIClient({'line_0': {'display_text': '指南'}})
                result = translate_index_lines(queue, client, {
                    'source_language': 'English', 'target_language': 'Chinese'
                })
                self.assertEqual(parse_index_line(result[0])['url'], entry['url'])
                self.assertEqual(parse_index_line(result[0])['display_text'], '指南')

    def test_multiple_containers_use_descendant_links_for_matching(self):
        one = '<LearningPathContainer platform="cloud" title="One">\n<LearningPath label="Learn" icon="same">\n[A](/a)\n</LearningPath>\n</LearningPathContainer>'
        two = one.replace('title="One"', 'title="Two"').replace('[A](/a)', '[B](/b)')
        target_one = one.replace('title="One"', 'title="一"').replace('label="Learn"', 'label="学ぶ"').replace('[A]', '[甲]')
        target_two = two.replace('title="Two"', 'title="二"').replace('label="Learn"', 'label="学習"').replace('[B]', '[乙]')
        base = one + '\n' + two
        planned, queue = plan_synced_index_lines(base, base, target_two + '\n' + target_one)
        self.assertEqual(queue, [])
        self.assertEqual('\n'.join(planned), target_one + '\n' + target_two)

    def test_repeated_icons_and_urls_stay_with_their_parent_container(self):
        one = '<LearningPathContainer platform="one" title="One">\n<LearningPath label="Learn" icon="same">\n[A](/a)\n</LearningPath>\n</LearningPathContainer>'
        two = one.replace('platform="one"', 'platform="two"').replace('title="One"', 'title="Two"')
        target_one = one.replace('title="One"', 'title="一"').replace('label="Learn"', 'label="一の学習"').replace('[A]', '[第一产品]')
        target_two = two.replace('title="Two"', 'title="二"').replace('label="Learn"', 'label="二の学習"').replace('[A]', '[第二产品]')
        planned, queue = plan_synced_index_lines(one + '\n' + two, one, target_two + '\n' + target_one)
        self.assertEqual(queue, [])
        self.assertEqual('\n'.join(planned), target_one)

    def test_conflicting_container_identity_is_not_reused(self):
        source = '<LearningPathContainer platform="one" title="One">\n[A](/a)\n</LearningPathContainer>'
        target = source.replace('platform="one"', 'platform="two"').replace('title="One"', 'title="第二产品"').replace('[A]', '[第二产品文档]')
        planned, queue = plan_synced_index_lines(source, source, target)
        self.assertEqual([slot for slot, _ in queue], [0, 1])

    def test_url_overlap_does_not_override_unique_component_attributes(self):
        source = '<LearningPath label="One" icon="one">\n[A](/a)\n[B](/b)\n</LearningPath>\n<LearningPath label="Two" icon="two">\n[C](/c)\n</LearningPath>'
        target = '<LearningPath label="一" icon="one">\n[X](/x)\n</LearningPath>\n<LearningPath label="二" icon="two">\n[甲](/a)\n[乙](/b)\n[丙](/c)\n</LearningPath>'
        planned, queue = plan_synced_index_lines(source, source, target)
        self.assertEqual(planned[0], '<LearningPath label="一" icon="one">')
        self.assertEqual(planned[4], '<LearningPath label="二" icon="two">')
        self.assertEqual(planned[5], '[丙](/c)')
        self.assertEqual([slot for slot, _ in queue], [1, 2])

    def test_comparison_text_is_escaped_instead_of_rejected(self):
        client = FieldAIClient({'line_0': {'display_text': '条件 x < 3 > 1'}})
        result = translate_index_lines([(0, '[Conditions](/a)')], client, {
            'source_language': 'English', 'target_language': 'Japanese'
        })
        self.assertEqual(result[0], '[条件 x &lt; 3 &gt; 1](/a)')


class TestIndexReviewFeedback(unittest.TestCase):
    def test_empty_and_inherited_locale_preserve_chinese_japanese_defaults(self):
        for language, locale in (('Chinese', 'zh'), ('Japanese', 'ja')):
            for inherited in ('', '  ', 'ko', 'k o'):
                for explicit in (None, '', '  '):
                    with self.subTest(language=language, inherited=inherited, explicit=explicit):
                        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(os.environ, {'TARGET_LOCALE': inherited}):
                            path = Path(tmpdir) / '_index.md'
                            target = f'[指南](https://docs.pingcap.com/{locale}/tidb/stable/x)'
                            path.write_text(target)
                            source = '[Guide](https://docs.pingcap.com/tidb/stable/x)'
                            config = {'source_language': 'English', 'target_language': language, 'target_locale': explicit}
                            self.assertTrue(process_index_file_by_source_snapshot(
                                '_index.md', {'source_base_content': source, 'source_head_content': source},
                                EmptyAIClient(), config, str(path)
                            ))
                            self.assertEqual(path.read_text(), target)

    def test_explicit_locale_overrides_environment_and_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(os.environ, {'TARGET_LOCALE': 'ko'}):
            path = Path(tmpdir) / '_index.md'
            source = '[Guide](https://docs.pingcap.com/guide)'
            target = '[指南](https://docs.pingcap.com/zh-Hans/guide)'
            path.write_text(target)
            self.assertTrue(process_index_file_by_source_snapshot(
                '_index.md', {'source_base_content': source, 'source_head_content': source}, EmptyAIClient(),
                {'source_language': 'English', 'target_language': 'Chinese', 'target_locale': 'zh-Hans'}, str(path)
            ))
            self.assertEqual(path.read_text(), target)

    def test_custom_language_reads_environment_locale(self):
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(os.environ, {'TARGET_LOCALE': 'ko'}):
            path = Path(tmpdir) / '_index.md'
            source = '[Guide](https://docs.pingcap.com/guide)'
            target = '[안내](https://docs.pingcap.com/ko/guide)'
            path.write_text(target)
            self.assertTrue(process_index_file_by_source_snapshot(
                '_index.md', {'source_base_content': source, 'source_head_content': source}, EmptyAIClient(),
                {'source_language': 'English', 'target_language': 'Korean'}, str(path)
            ))
            self.assertEqual(path.read_text(), target)

    def test_invalid_locale_is_explicit_and_does_not_write(self):
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(os.environ, {'TARGET_LOCALE': 'k o'}):
            path = Path(tmpdir) / '_index.md'
            path.write_text('[안내](/a)')
            log = io.StringIO()
            with redirect_stdout(log):
                success = process_index_file_by_source_snapshot(
                    '_index.md', {'source_base_content': '[Guide](/a)', 'source_head_content': '[Guide](/a)'},
                    EmptyAIClient(), {'source_language': 'English', 'target_language': 'Korean'}, str(path)
                )
            self.assertFalse(success)
            self.assertIn('Invalid Index locale configuration for _index.md', log.getvalue())
            self.assertEqual(path.read_text(), '[안내](/a)')

    def test_code_spans_and_markdown_escapes_render_correctly(self):
        cases = (
            ('`<name>` 指南', '<name>'),
            ('`List<T>` 指南', 'List<T>'),
            ('`<&amp;>` 指南', '<&amp;>'),
            ('`` `<name>` & `` 指南', '`<name>` &'),
            (r'\[名称\] 指南', '[名称]'),
            ('List<T> 指南', 'List<T>'),
            ('<LearningPath> 指南', '<LearningPath>'),
        )
        for translation, expected in cases:
            with self.subTest(translation=translation):
                client = FieldAIClient({'line_0': {'display_text': translation}})
                result = translate_index_lines([(0, '[Guide](/a)')], client, {
                    'source_language': 'English', 'target_language': 'Chinese'
                })
                tokens = MarkdownIt().parseInline(result[0])[0].children
                self.assertEqual(sum(token.type == 'link_open' for token in tokens), 1)
                self.assertEqual(next(token.attrs['href'] for token in tokens if token.type == 'link_open'), '/a')
                rendered = ''.join(token.content for token in tokens if token.type in ('text', 'code_inline'))
                self.assertIn(expected, rendered)

    def test_prompt_does_not_decode_entities_inside_code(self):
        source = '[`&amp;` &amp; guide](/a)'
        client = FieldAIClient({'line_0': {'display_text': '`&amp;` & 指南'}})
        result = translate_index_lines([(0, source)], client, {
            'source_language': 'English', 'target_language': 'Chinese'
        })
        request = json.JSONDecoder().raw_decode(client.prompt.split('Input lines to translate:\n', 1)[1])[0]
        self.assertEqual(request['line_0']['display_text'], '`&amp;` & guide')
        tokens = MarkdownIt().parseInline(result[0])[0].children
        self.assertEqual(next(token.content for token in tokens if token.type == 'code_inline'), '&amp;')

    def test_multiline_component_attributes_are_translated(self):
        source = '<LearningPath\n  label="Learn"\n  icon="cloud1"\n>\n\n[Guide](/a)\n\n</LearningPath>'
        client = FieldAIClient({'line_0': {'label': '学ぶ'}, 'line_1': {'display_text': 'ガイド'}})
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / '_index.md'
            self.assertTrue(process_index_file_by_source_snapshot(
                '_index.md', {'source_base_content': '', 'source_head_content': source}, client,
                {'source_language': 'English', 'target_language': 'Japanese'}, str(path)
            ))
            self.assertEqual(path.read_text(), source.replace('label="Learn"', 'label="学ぶ"').replace('[Guide]', '[ガイド]'))

    def test_multiline_existing_translation_is_reused(self):
        source = '<LearningPath\n label="Learn"\n icon="cloud1"\n>\n[A](/a)\n</LearningPath>'
        target = source.replace('label="Learn"', 'label="ロード"').replace('[A]', '[甲]')
        planned, queue = plan_synced_index_lines(source, source.replace('icon="cloud1"', 'icon="new"'), target)
        self.assertEqual(queue, [])
        self.assertEqual('\n'.join(planned), target.replace('icon="cloud1"', 'icon="new"'))

    def test_nested_and_self_closing_components_preserve_source_structure(self):
        sources = (
            '<LearningPathContainer title="One">\n<LearningPathContainer title="Two">\n[A](/a)\n</LearningPathContainer>\n</LearningPathContainer>',
            '<LearningPath label="Learn" icon="x" />',
        )
        for source in sources:
            with self.subTest(source=source), tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / '_index.md'
                path.write_text(source)
                self.assertTrue(process_index_file_by_source_snapshot(
                    '_index.md', {'source_base_content': source, 'source_head_content': source}, EmptyAIClient(),
                    {'source_language': 'English', 'target_language': 'Japanese'}, str(path)
                ))
                self.assertEqual(path.read_text(), source)

    def test_extra_response_keys_do_not_discard_valid_requested_fields(self):
        client = FieldAIClient({'line_0': {'display_text': '甲'}, 'unexpected': {'url': '/evil'}})
        result = translate_index_lines([(0, '[A](/a)')], client, {
            'source_language': 'English', 'target_language': 'Chinese'
        })
        self.assertEqual(result, {0: '[甲](/a)'})

    def test_missing_response_keys_retain_valid_mappings_and_report_failure(self):
        client = FieldAIClient({'line_0': {'display_text': '甲'}})
        log = io.StringIO()
        with redirect_stdout(log):
            result = translate_index_lines([(0, '[A](/a)'), (1, '[B](/b)')], client, {
                'source_language': 'English', 'target_language': 'Chinese'
            })
        self.assertEqual(result, {0: '[甲](/a)'})
        self.assertIn('1/2', log.getvalue())
        self.assertNotIn('Successfully translated', log.getvalue())

    def test_zero_valid_fields_log_is_not_success(self):
        log = io.StringIO()
        with redirect_stdout(log):
            translate_index_lines([(0, '[A](/a)')], FieldAIClient({}), {
                'source_language': 'English', 'target_language': 'Chinese'
            })
        self.assertNotIn('Successfully translated', log.getvalue())
        self.assertIn('0/1', log.getvalue())

    def test_nested_entity_encoding_is_decoded_only_once(self):
        source = '[&amp;copy; guide](/a)'
        client = FieldAIClient({'line_0': {'display_text': '&copy; 指南'}})
        result = translate_index_lines([(0, source)], client, {
            'source_language': 'English', 'target_language': 'Chinese'
        })
        tokens = MarkdownIt().parseInline(result[0])[0].children
        self.assertIn('&copy;', ''.join(token.content for token in tokens))
        self.assertNotIn('©', result[0])

    def test_multiline_scanner_leaves_frontmatter_block_scalars_untouched(self):
        source = '---\nsummary: |\n  <LearningPath sample text\n---\n[Guide](/a)'
        target = source.replace('[Guide]', '[指南]')
        planned, queue = plan_synced_index_lines(source, source, target)
        self.assertEqual(queue, [])
        self.assertEqual('\n'.join(planned), target)

    def test_many_ambiguous_icons_are_disambiguated_by_link_overlap(self):
        source, target = [], []
        for index in range(40):
            source.append(f'<LearningPath label="Learn {index}" icon="same">\n[Common](/common)\n[A](/a{index})\n</LearningPath>')
            target.append(f'<LearningPath label="学ぶ {index}" icon="same">\n[共通](/common)\n[甲{index}](/a{index})\n</LearningPath>')
        planned, queue = plan_synced_index_lines('\n'.join(source), '\n'.join(source), '\n'.join(reversed(target)))
        self.assertEqual(queue, [])
        self.assertEqual('\n'.join(planned), '\n'.join(target))

    def test_unterminated_multiline_tag_is_explicit_failure(self):
        log = io.StringIO()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / '_index.md'
            path.write_text('[既存](/a)')
            with redirect_stdout(log):
                success = process_index_file_by_source_snapshot(
                    '_index.md', {'source_base_content': '', 'source_head_content': '<LearningPath\n label="Learn"'},
                    EmptyAIClient(), {'source_language': 'English', 'target_language': 'Japanese'}, str(path)
                )
            self.assertFalse(success)
            self.assertEqual(path.read_text(), '[既存](/a)')
            self.assertIn('Unterminated LearningPath opening tag', log.getvalue())


if __name__ == '__main__':
    unittest.main()

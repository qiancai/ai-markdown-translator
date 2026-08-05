import sys
import subprocess
import tempfile
import unittest
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from diff_analyzer import (
    analyze_diff_operations,
    analyze_source_changes,
    build_commit_diff_context,
    build_local_commit_diff_context,
    build_diff_text,
    build_pr_diff_context,
    build_hierarchy_dict,
    build_source_diff_dict,
    collect_added_heading_prefix_lines,
    detect_heading_level_only_file,
    detect_restructured_file,
    filter_changed_files_to_pr_scope,
    filter_related_resources_resource_card_diff,
    get_pr_diff,
    get_target_file_content,
    get_target_hierarchy_and_content,
    is_heading_level_only_change,
    maybe_use_normalized_snapshot_operations,
    parse_pr_commit_range_url,
    parse_pr_url,
    preserve_base_related_resources_resource_card_sections,
    should_skip_temporary_japanese_release_alias_change,
)
from formatting_sync import apply_formatting_only_change, build_formatting_only_change
from special_file_utils import path_resource_key


class FakeContent:
    def __init__(self, text):
        self.decoded_content = text.encode("utf-8")


class FakePR:
    def __init__(self, files, title, base_sha, head_sha):
        self._files = files
        self.title = title
        self.base = SimpleNamespace(sha=base_sha, ref="master")
        self.head = SimpleNamespace(sha=head_sha, ref="feature/test")

    def get_files(self):
        return self._files


class FakeComparison:
    def __init__(self, files):
        self.files = files


class FakeRepository:
    def __init__(self, source_files, pr, comparison_files):
        self.default_branch = "master"
        self._source_files = source_files
        self._pr = pr
        self._comparison_files = comparison_files

    def get_pull(self, number):
        return self._pr

    def compare(self, base, head):
        return FakeComparison(self._comparison_files[(base, head)])

    def get_contents(self, path, ref):
        return FakeContent(self._source_files[(path, ref)])


class FakeGithub:
    def __init__(self, repositories):
        self._repositories = repositories

    def get_repo(self, repo_name):
        return self._repositories[repo_name]


def make_full_rewrite_patch(base_content, head_content):
    """Build a GitHub API-style patch body without file header lines."""
    base_lines = base_content.replace("\r\n", "\n").splitlines()
    head_lines = head_content.replace("\r\n", "\n").splitlines()
    patch_lines = [f"@@ -1,{len(base_lines)} +1,{len(head_lines)} @@"]
    patch_lines.extend(f"-{line}" for line in base_lines)
    patch_lines.extend(f"+{line}" for line in head_lines)
    return "\n".join(patch_lines)


class DiffAnalyzerContextTest(unittest.TestCase):
    def setUp(self):
        self.temp_output_dir = SCRIPTS_DIR / "temp_output"
        self.generated_file = self.temp_output_dir / "guide-source-diff-dict.json"
        if self.generated_file.exists():
            self.generated_file.unlink()

        patch = "\n".join(
            [
                "@@ -1,4 +1,4 @@",
                " # Title",
                " ",
                " ## Section",
                "-Old text",
                "+New text",
            ]
        )
        self.changed_file = SimpleNamespace(
            filename="guide.md",
            status="modified",
            patch=patch,
            previous_filename=None,
        )
        base_sha = "base123"
        head_sha = "head123"
        base_content = "# Title\n\n## Section\nOld text\n"
        head_content = "# Title\n\n## Section\nNew text\n"
        self.repo_configs = {
            "acme/docs": {
                "target_repo": "acme/docs-cn",
                "target_local_path": "/tmp/target",
                "prefer_local_target_for_read": False,
                "source_language": "English",
                "target_language": "Chinese",
            }
        }
        repository = FakeRepository(
            {
                ("guide.md", base_sha): base_content,
                ("guide.md", head_sha): head_content,
            },
            FakePR([self.changed_file], "Update guide", base_sha, head_sha),
            {(base_sha, head_sha): [self.changed_file]},
        )
        self.github = FakeGithub({"acme/docs": repository})
        self.pr_url = "https://github.com/acme/docs/pull/123"

    def tearDown(self):
        if self.generated_file.exists():
            self.generated_file.unlink()

    def test_pr_and_commit_contexts_produce_same_analysis(self):
        pr_context = build_pr_diff_context(self.pr_url, self.github, self.repo_configs)
        commit_context = build_commit_diff_context(
            "acme/docs",
            "acme/docs-cn",
            "base123",
            "head123",
            self.github,
            self.repo_configs,
        )

        self.assertEqual(
            build_diff_text(pr_context["changed_files"]),
            build_diff_text(commit_context["changed_files"]),
        )

        pr_result = analyze_source_changes(
            pr_context,
            self.github,
            special_files=["TOC.md", "keywords.md"],
            ignore_files=[],
            repo_configs=self.repo_configs,
        )
        commit_result = analyze_source_changes(
            commit_context,
            self.github,
            special_files=["TOC.md", "keywords.md"],
            ignore_files=[],
            repo_configs=self.repo_configs,
        )

        self.assertEqual(pr_result, commit_result)
        self.assertEqual(12, len(pr_result))
        self.assertTrue(self.generated_file.exists())
        source_diff = self.generated_file.read_text(encoding="utf-8")
        self.assertIn('"operation": "modified"', source_diff)
        self.assertIn("New text", source_diff)

    def test_japanese_translation_skips_release_aliases_only_change(self):
        file_path = "releases/release-3.0.7.md"
        patch = "\n".join(
            [
                "@@ -1,6 +1,7 @@",
                " ---",
                " title: TiDB 3.0.7 Release Notes",
                "+aliases: ['/tidb/dev/release-3.0.7/']",
                " ---",
                " # TiDB 3.0.7 Release Notes",
            ]
        )
        changed_file = SimpleNamespace(
            filename=file_path,
            status="modified",
            patch=patch,
            previous_filename=None,
        )
        repo_configs = {
            "acme/docs": {
                "target_repo": "acme/docs-ja",
                "target_local_path": "/tmp/target",
                "source_language": "English",
                "target_language": "Japanese",
            }
        }
        # Deliberately omit file snapshots: the temporary rule must skip the
        # file before source content is fetched or any translation queue is built.
        repository = FakeRepository(
            {},
            FakePR([changed_file], "Add release aliases", "base123", "head123"),
            {("base123", "head123"): [changed_file]},
        )
        github = FakeGithub({"acme/docs": repository})
        context = build_commit_diff_context(
            "acme/docs",
            "acme/docs-ja",
            "base123",
            "head123",
            github,
            repo_configs,
        )

        result = analyze_source_changes(
            context,
            github,
            special_files=["TOC.md", "keywords.md"],
            ignore_files=[],
            repo_configs=repo_configs,
            include_formatting_files=True,
        )

        expected = (
            {}, {}, {}, {}, [], {}, {}, [], [], [], set(), {}, {}
        )
        self.assertEqual(expected, result)

    def test_temporary_japanese_release_rule_requires_aliases_only_diff(self):
        aliases_patch = "\n".join(
            [
                "@@ -1,4 +1,4 @@",
                " ---",
                "-aliases: ['/old/']",
                "+aliases: ['/new/']",
                " ---",
            ]
        )
        aliases_only_file = SimpleNamespace(
            filename="releases/release-8.5.0.md",
            status="modified",
            patch=aliases_patch,
        )

        self.assertTrue(
            should_skip_temporary_japanese_release_alias_change(
                aliases_only_file,
                "Japanese",
            )
        )
        self.assertFalse(
            should_skip_temporary_japanese_release_alias_change(
                aliases_only_file,
                "Chinese",
            )
        )

        aliases_and_summary_file = SimpleNamespace(
            filename="releases/release-8.5.0.md",
            status="modified",
            patch=aliases_patch + "\n-summary: Old\n+summary: New",
        )
        self.assertFalse(
            should_skip_temporary_japanese_release_alias_change(
                aliases_and_summary_file,
                "Japanese",
            )
        )

        non_release_file = SimpleNamespace(
            filename="release-notes/release-8.5.0.md",
            status="modified",
            patch=aliases_patch,
        )
        self.assertFalse(
            should_skip_temporary_japanese_release_alias_change(
                non_release_file,
                "Japanese",
            )
        )

    def test_formatting_only_blank_line_change_syncs_target_without_regular_queue(self):
        file_path = "tidb-cloud/monitor-alert-slack.md"
        base_content = (
            "# Subscribe via Slack\n\n"
            "## Unsubscribe from alert notifications\n\n"
            '<CustomContent plan="dedicated">\n'
            "    \n"
            "1. Unsubscribe.\n\n"
            "</CustomContent>\n"
        )
        head_content = base_content.replace(
            '<CustomContent plan="dedicated">\n    \n',
            '<CustomContent plan="dedicated">\n\n',
        )
        target_content = (
            "# 通过 Slack 订阅\n\n"
            "## 取消报警通知订阅\n\n"
            '<CustomContent plan="dedicated">\n'
            "    \n"
            "1. 取消订阅。\n\n"
            "</CustomContent>\n"
        )
        patch = "\n".join(
            [
                "@@ -3,7 +3,7 @@",
                " ## Unsubscribe from alert notifications",
                " ",
                ' <CustomContent plan="dedicated">',
                "-    ",
                "+",
                " 1. Unsubscribe.",
                " ",
                " </CustomContent>",
            ]
        )
        changed_file = SimpleNamespace(
            filename=file_path,
            status="modified",
            patch=patch,
            previous_filename=None,
        )
        source_repo = FakeRepository(
            {
                (file_path, "base123"): base_content,
                (file_path, "head123"): head_content,
            },
            FakePR([changed_file], "Remove trailing spaces", "base123", "head123"),
            {("base123", "head123"): [changed_file]},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = Path(tmpdir, file_path)
            target_path.parent.mkdir(parents=True)
            target_path.write_text(target_content, encoding="utf-8")
            repo_configs = {
                "acme/docs": {
                    "target_repo": "acme/docs-cn",
                    "target_local_path": tmpdir,
                    "source_language": "English",
                    "target_language": "Chinese",
                }
            }
            github = FakeGithub({"acme/docs": source_repo})
            context = build_commit_diff_context(
                "acme/docs",
                "acme/docs-cn",
                "base123",
                "head123",
                github,
                repo_configs,
            )

            result = analyze_source_changes(
                context,
                github,
                special_files=["TOC.md", "keywords.md"],
                ignore_files=[],
                repo_configs=repo_configs,
                include_formatting_files=True,
            )

            modified_sections = result[1]
            formatting_files = result[12]
            self.assertEqual({}, modified_sections)
            self.assertEqual([file_path], list(formatting_files))
            self.assertEqual(patch, formatting_files[file_path]["source_patch"])
            self.assertFalse(
                (
                    self.temp_output_dir
                    / f"{path_resource_key(file_path)}-source-diff-dict.json"
                ).exists()
            )

            success, changed, reason = apply_formatting_only_change(
                file_path,
                formatting_files[file_path],
                tmpdir,
            )
            self.assertTrue(success, reason)
            self.assertTrue(changed)
            self.assertIn(
                '<CustomContent plan="dedicated">\n\n1. 取消订阅。',
                target_path.read_text(encoding="utf-8"),
            )

    def test_empty_source_diff_does_not_enter_regular_modified_queue(self):
        file_path = "guide.md"
        base_content = "# Guide\n\n## Section\n    \nText.\n"
        head_content = "# Guide\n\n## Section\n\nText.\n"
        changed_file = SimpleNamespace(
            filename=file_path,
            status="modified",
            patch="@@ -3,3 +3,3 @@\n ## Section\n-    \n+\n Text.",
            previous_filename=None,
        )
        repository = FakeRepository(
            {
                (file_path, "base123"): base_content,
                (file_path, "head123"): head_content,
            },
            FakePR([changed_file], "Whitespace", "base123", "head123"),
            {("base123", "head123"): [changed_file]},
        )
        github = FakeGithub({"acme/docs": repository})
        context = build_commit_diff_context(
            "acme/docs",
            "acme/docs-cn",
            "base123",
            "head123",
            github,
            self.repo_configs,
        )

        with mock.patch(
            "diff_analyzer.build_formatting_only_change",
            return_value=None,
        ), mock.patch("builtins.print") as print_mock:
            result = analyze_source_changes(
                context,
                github,
                special_files=["TOC.md", "keywords.md"],
                ignore_files=[],
                repo_configs=self.repo_configs,
                include_formatting_files=True,
            )

        self.assertEqual({}, result[1])
        self.assertEqual({}, result[12])
        rendered_log = "\n".join(
            " ".join(str(arg) for arg in call.args)
            for call in print_mock.call_args_list
        )
        self.assertIn("Canonical no-op modified files: 1 files", rendered_log)

    def test_toc_whitespace_only_change_stays_on_toc_path(self):
        file_path = "TOC-test.md"
        base_content = "- [Guide](/guide.md)   \n"
        head_content = "- [Guide](/guide.md)\n"
        changed_file = SimpleNamespace(
            filename=file_path,
            status="modified",
            patch="@@ -1 +1 @@\n-- [Guide](/guide.md)   \n+- [Guide](/guide.md)",
            previous_filename=None,
        )
        repo_configs = {
            "acme/docs": {
                "target_repo": "acme/docs-cn",
                "target_local_path": "/tmp/target",
                "prefer_local_target_for_read": False,
                "source_language": "English",
                "target_language": "Chinese",
            }
        }
        source_repo = FakeRepository(
            {
                (file_path, "base123"): base_content,
                (file_path, "head123"): head_content,
            },
            FakePR([changed_file], "TOC whitespace", "base123", "head123"),
            {("base123", "head123"): [changed_file]},
        )
        target_repo = FakeRepository(
            {(file_path, "master"): "- [指南](/guide.md)\n"},
            FakePR([], "Empty", "base123", "head123"),
            {},
        )
        github = FakeGithub(
            {"acme/docs": source_repo, "acme/docs-cn": target_repo}
        )
        context = build_commit_diff_context(
            "acme/docs",
            "acme/docs-cn",
            "base123",
            "head123",
            github,
            repo_configs,
        )

        result = analyze_source_changes(
            context,
            github,
            special_files=["TOC.md", "keywords.md"],
            ignore_files=[],
            repo_configs=repo_configs,
            include_formatting_files=True,
        )

        self.assertIn(file_path, result[5])
        self.assertEqual({}, result[12])

    def test_keyword_and_index_whitespace_changes_do_not_use_formatting_path(self):
        cases = {
            "keywords.md": "# Keywords\n    \nText.\n",
            "docs/_index.md": "<LearningPathContainer>\n    \n</LearningPathContainer>\n",
        }

        for file_path, base_content in cases.items():
            with self.subTest(file_path=file_path):
                head_content = base_content.replace("    \n", "\n")
                changed_file = SimpleNamespace(
                    filename=file_path,
                    status="modified",
                    patch="@@ -1,3 +1,3 @@\n context\n-    \n+\n context",
                    previous_filename=None,
                )
                repository = FakeRepository(
                    {
                        (file_path, "base123"): base_content,
                        (file_path, "head123"): head_content,
                    },
                    FakePR(
                        [changed_file],
                        "Special whitespace",
                        "base123",
                        "head123",
                    ),
                    {("base123", "head123"): [changed_file]},
                )
                github = FakeGithub({"acme/docs": repository})
                context = build_commit_diff_context(
                    "acme/docs",
                    "acme/docs-cn",
                    "base123",
                    "head123",
                    github,
                    self.repo_configs,
                )

                with mock.patch(
                    "diff_analyzer.build_formatting_only_change",
                    wraps=build_formatting_only_change,
                ) as formatting_detector:
                    result = analyze_source_changes(
                        context,
                        github,
                        special_files=["TOC.md", "keywords.md"],
                        ignore_files=[],
                        repo_configs=self.repo_configs,
                        include_formatting_files=True,
                    )

                formatting_detector.assert_not_called()
                self.assertEqual({}, result[12])
                if file_path.endswith("_index.md"):
                    self.assertIn(file_path, result[11])

    def test_normal_index_whitespace_change_uses_formatting_path(self):
        file_path = "docs/_index.md"
        base_content = "# Overview\n\n## Usage\n    \nText.\n"
        head_content = base_content.replace("    \n", "\n")
        changed_file = SimpleNamespace(
            filename=file_path,
            status="modified",
            patch="@@ -3,3 +3,3 @@\n ## Usage\n-    \n+\n Text.",
            previous_filename=None,
        )
        repository = FakeRepository(
            {
                (file_path, "base123"): base_content,
                (file_path, "head123"): head_content,
            },
            FakePR([changed_file], "Index whitespace", "base123", "head123"),
            {("base123", "head123"): [changed_file]},
        )
        github = FakeGithub({"acme/docs": repository})
        context = build_commit_diff_context(
            "acme/docs",
            "acme/docs-cn",
            "base123",
            "head123",
            github,
            self.repo_configs,
        )

        result = analyze_source_changes(
            context,
            github,
            special_files=["TOC.md", "keywords.md"],
            ignore_files=[],
            repo_configs=self.repo_configs,
            include_formatting_files=True,
        )

        self.assertEqual({}, result[11])
        self.assertIn(file_path, result[12])

    def test_blank_line_insertion_remains_section_translation_work(self):
        file_path = "guide.md"
        base_content = "# Guide\n\n## Section\nText.\n"
        head_content = "# Guide\n\n## Section\n\nText.\n"
        changed_file = SimpleNamespace(
            filename=file_path,
            status="modified",
            patch="@@ -3,2 +3,3 @@\n ## Section\n+\n Text.",
            previous_filename=None,
        )
        repository = FakeRepository(
            {
                (file_path, "base123"): base_content,
                (file_path, "head123"): head_content,
            },
            FakePR([changed_file], "Add blank line", "base123", "head123"),
            {("base123", "head123"): [changed_file]},
        )
        github = FakeGithub({"acme/docs": repository})
        context = build_commit_diff_context(
            "acme/docs",
            "acme/docs-cn",
            "base123",
            "head123",
            github,
            self.repo_configs,
        )

        result = analyze_source_changes(
            context,
            github,
            special_files=["TOC.md", "keywords.md"],
            ignore_files=[],
            repo_configs=self.repo_configs,
            include_formatting_files=True,
        )

        self.assertIn(file_path, result[1])
        self.assertEqual({}, result[12])

    def test_fenced_hash_comments_are_regular_section_content_changes(self):
        base_content = (
            "# Guide\n\n"
            "## Configuration file description\n\n"
            "```toml\n"
            "# Old explanatory comment.\n"
            "value = 1\n"
            "```\n\n"
            "## End\n\nStable.\n"
        )
        head_content = (
            "# Guide\n\n"
            "## Configuration file description\n\n"
            "```toml\n"
            "# New explanatory comment.\n"
            "# Additional explanatory comment.\n"
            "value = 1\n"
            "```\n\n"
            "## End\n\nStable.\n"
        )
        patch = "\n".join(
            [
                "@@ -3,7 +3,8 @@",
                " ## Configuration file description",
                " ",
                " ```toml",
                "-# Old explanatory comment.",
                "+# New explanatory comment.",
                "+# Additional explanatory comment.",
                " value = 1",
                " ```",
                " ",
            ]
        )
        changed_file = SimpleNamespace(
            filename="guide.md",
            status="modified",
            patch=patch,
            previous_filename=None,
        )
        repository = FakeRepository(
            {
                ("guide.md", "base-fence"): base_content,
                ("guide.md", "head-fence"): head_content,
            },
            FakePR([changed_file], "Update config comments", "base-fence", "head-fence"),
            {("base-fence", "head-fence"): [changed_file]},
        )
        github = FakeGithub({"acme/docs": repository})
        context = build_commit_diff_context(
            "acme/docs",
            "acme/docs-cn",
            "base-fence",
            "head-fence",
            github,
            self.repo_configs,
        )

        analyze_source_changes(
            context,
            github,
            special_files=["TOC.md", "keywords.md"],
            ignore_files=[],
            repo_configs=self.repo_configs,
        )

        source_diff = json.loads(self.generated_file.read_text(encoding="utf-8"))
        configuration_changes = [
            change
            for change in source_diff.values()
            if change.get("original_hierarchy") == "## Configuration file description"
        ]
        self.assertEqual(1, len(configuration_changes))
        self.assertEqual("modified", configuration_changes[0]["operation"])
        self.assertIn(
            "# Additional explanatory comment.",
            configuration_changes[0]["new_content"],
        )

    def test_preserve_base_related_resources_replaces_only_ignored_section(self):
        base = (
            "# Guide\n\n## Content\n\nOld.\n\n"
            "## Related resources\n\n"
            "<RelatedResources>\n"
            '  <ResourceCard title="Base card" />\n'
            "</RelatedResources>\n"
        )
        head = (
            "# Guide\n\n## Content\n\nNew.\n\n"
            "## Related resources\n\n"
            "<RelatedResources>\n"
            '  <ResourceCard title="Changed card" />\n'
            "</RelatedResources>\n"
        )

        effective_head, preserved_sections = (
            preserve_base_related_resources_resource_card_sections(base, head)
        )

        self.assertIn("New.", effective_head)
        self.assertIn('title="Base card"', effective_head)
        self.assertNotIn("Changed card", effective_head)
        self.assertEqual(1, len(preserved_sections))
        self.assertTrue(effective_head.endswith("\n"))
        self.assertFalse(effective_head.endswith("\n\n"))

        without_trailing_newline, _ = (
            preserve_base_related_resources_resource_card_sections(
                base.rstrip("\n"),
                head.rstrip("\n"),
            )
        )
        self.assertFalse(without_trailing_newline.endswith("\n"))

    def test_preserve_base_related_resources_rejects_missing_snapshot(self):
        content = "# Guide\n"

        self.assertEqual(
            (None, []),
            preserve_base_related_resources_resource_card_sections(None, content),
        )
        self.assertEqual(
            (None, []),
            preserve_base_related_resources_resource_card_sections(content, None),
        )

    def test_snapshot_heading_scope_does_not_pair_fenced_comment_with_real_heading(self):
        base = (
            "# Guide\n\n```toml\n# Removed code comment.\n```\n\nBody.\n"
        )
        head = (
            "# Guide\n\n```toml\n```\n\n## Added section\n\nBody.\n"
        )
        patch = "\n".join(
            [
                "@@ -1,7 +1,8 @@",
                " # Guide",
                " ",
                " ```toml",
                "-# Removed code comment.",
                " ```",
                " ",
                "+## Added section",
                "+",
                " Body.",
            ]
        )
        changed_file = SimpleNamespace(filename="guide.md", patch=patch)

        operations = analyze_diff_operations(
            changed_file,
            base_content=base,
            head_content=head,
        )

        self.assertEqual([], operations["modified_lines"])
        self.assertEqual(["## Added section", ""], [
            entry["content"] for entry in operations["added_lines"]
        ])
        self.assertTrue(operations["added_lines"][0]["is_header"])
        self.assertEqual(
            ["# Removed code comment."],
            [entry["content"] for entry in operations["deleted_lines"]],
        )
        self.assertFalse(operations["deleted_lines"][0]["is_header"])

    def test_parse_pr_url_accepts_plain_and_files_range_urls(self):
        plain_url = "https://github.com/acme/docs/pull/123/"
        range_url = "https://github.com/acme/docs/pull/123/files/base123..head123?plain=1"

        self.assertEqual(parse_pr_url(plain_url), ("acme", "docs", 123))
        self.assertEqual(parse_pr_url(range_url), ("acme", "docs", 123))
        self.assertIsNone(parse_pr_commit_range_url(plain_url))
        self.assertEqual(
            parse_pr_commit_range_url(range_url),
            ("acme", "docs", 123, "base123", "head123"),
        )

    def test_pr_files_range_context_filters_compare_diff_to_pr_scope(self):
        full_pr_file = SimpleNamespace(
            filename="range.md",
            status="modified",
            patch="@@ -1 +1 @@\n-full\n+full pr",
            previous_filename=None,
        )
        range_file = SimpleNamespace(
            filename="range.md",
            status="modified",
            patch="@@ -1 +1 @@\n-old\n+range only",
            previous_filename=None,
        )
        upstream_file = SimpleNamespace(
            filename="releases/release-8.2.0.md",
            status="modified",
            patch="@@ -1 +1 @@\n-old\n+upstream only",
            previous_filename=None,
        )
        repository = FakeRepository(
            {},
            FakePR([full_pr_file], "Update range", "prbase", "prhead"),
            {("base456", "head456"): [range_file, upstream_file]},
        )
        github = FakeGithub({"acme/docs": repository})

        context = build_pr_diff_context(
            "https://github.com/acme/docs/pull/123/files/base456..head456/",
            github,
            self.repo_configs,
        )

        self.assertEqual(context["mode"], "pr")
        self.assertEqual(context["base_ref"], "base456")
        self.assertEqual(context["head_ref"], "head456")
        self.assertEqual([file.filename for file in context["changed_files"]], ["range.md"])
        self.assertIn("commit range base456..head456", context["source_description"])
        self.assertEqual(context["range_compare_file_count"], 2)
        self.assertEqual(context["range_pr_file_count"], 1)
        self.assertEqual(context["range_matched_file_count"], 1)
        rendered_diff = build_diff_text(context["changed_files"])
        self.assertIn("+range only", rendered_diff)
        self.assertNotIn("full pr", rendered_diff)
        self.assertNotIn("upstream only", rendered_diff)

    def test_range_file_scope_filter_matches_previous_filenames(self):
        pr_files = [
            SimpleNamespace(
                filename="current.md",
                status="renamed",
                patch=None,
                previous_filename="original.md",
            )
        ]
        changed_files = [
            SimpleNamespace(
                filename="current.md",
                status="modified",
                patch="@@ -1 +1 @@\n-old\n+current path",
                previous_filename=None,
            ),
            SimpleNamespace(
                filename="intermediate.md",
                status="renamed",
                patch="@@ -1 +1 @@\n-old\n+previous path",
                previous_filename="original.md",
            ),
            SimpleNamespace(
                filename="unrelated.md",
                status="modified",
                patch="@@ -1 +1 @@\n-old\n+unrelated",
                previous_filename=None,
            ),
        ]

        matched_files = filter_changed_files_to_pr_scope(changed_files, pr_files)

        self.assertEqual(
            [file.filename for file in matched_files],
            ["current.md", "intermediate.md"],
        )

    def test_get_pr_diff_range_filters_compare_diff_to_current_pr_files(self):
        range_file = SimpleNamespace(
            filename="range.md",
            status="modified",
            patch="@@ -1 +1 @@\n-old\n+range only",
            previous_filename=None,
        )
        upstream_file = SimpleNamespace(
            filename="releases/release-8.2.0.md",
            status="modified",
            patch="@@ -1 +1 @@\n-old\n+upstream only",
            previous_filename=None,
        )
        repository = FakeRepository(
            {},
            FakePR([range_file], "Update range", "prbase", "prhead"),
            {("base456", "head456"): [range_file, upstream_file]},
        )
        github = FakeGithub({"acme/docs": repository})

        rendered_diff = get_pr_diff(
            "https://github.com/acme/docs/pull/123/files/base456..head456/",
            github,
        )

        self.assertIn("+range only", rendered_diff)
        self.assertNotIn("upstream only", rendered_diff)

    def test_pr_files_range_context_only_uses_files_changed_in_range(self):
        pr_files = [
            SimpleNamespace(
                filename=f"file-{index}.md",
                status="modified",
                patch=f"@@ -1 +1 @@\n-old\n+full pr {index}",
                previous_filename=None,
            )
            for index in range(10)
        ]
        range_files = [
            SimpleNamespace(
                filename=f"file-{index}.md",
                status="modified",
                patch=f"@@ -1 +1 @@\n-old\n+range {index}",
                previous_filename=None,
            )
            for index in [1, 3, 5, 7]
        ]
        repository = FakeRepository(
            {},
            FakePR(pr_files, "Update ten files", "prbase", "prhead"),
            {("base456", "head456"): range_files},
        )
        github = FakeGithub({"acme/docs": repository})

        context = build_pr_diff_context(
            "https://github.com/acme/docs/pull/123/files/base456..head456/",
            github,
            self.repo_configs,
        )

        self.assertEqual(
            [file.filename for file in context["changed_files"]],
            ["file-1.md", "file-3.md", "file-5.md", "file-7.md"],
        )
        rendered_diff = build_diff_text(context["changed_files"])
        self.assertIn("+range 1", rendered_diff)
        self.assertIn("+range 7", rendered_diff)
        self.assertNotIn("+full pr 0", rendered_diff)
        self.assertNotIn("+full pr 9", rendered_diff)

    def test_commit_related_resources_added_section_is_filtered(self):
        file_path = "guide.md"
        base_content = "# Guide\n\n## Usage\n\nUse TiDB.\n"
        head_content = "\n".join(
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
                "  <ResourceCard",
                '    title="Example"',
                '    type="blog"',
                '    link="https://example.com"',
                "  />",
                "</RelatedResources>",
                "",
            ]
        )
        patch = "\n".join(
            [
                "@@ -3,3 +3,13 @@",
                " ## Usage",
                " ",
                " Use TiDB.",
                "+",
                "+## Related resources",
                "+",
                "+<RelatedResources>",
                "+  <ResourceCard",
                '+    title="Example"',
                '+    type="blog"',
                '+    link="https://example.com"',
                "+  />",
                "+</RelatedResources>",
            ]
        )
        changed_file = SimpleNamespace(
            filename=file_path,
            status="modified",
            patch=patch,
            previous_filename=None,
        )
        repository = FakeRepository(
            {
                (file_path, "base123"): base_content,
                (file_path, "head123"): head_content,
            },
            FakePR([changed_file], "Add related resources", "base123", "head123"),
            {("base123", "head123"): [changed_file]},
        )
        github = FakeGithub({"acme/docs": repository})
        context = build_commit_diff_context(
            "acme/docs",
            "acme/docs-cn",
            "base123",
            "head123",
            github,
            self.repo_configs,
        )

        result = analyze_source_changes(
            context,
            github,
            special_files=["TOC.md", "keywords.md"],
            ignore_files=[],
            repo_configs=self.repo_configs,
        )

        self.assertEqual(result, ({}, {}, {}, {}, [], {}, {}, [], [], [], set(), {}))
        self.assertFalse(self.generated_file.exists())

    def test_commit_related_resources_modified_section_is_filtered(self):
        file_path = "guide.md"
        base_content = "\n".join(
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
                "  <ResourceCard",
                '    title="Old example"',
                '    type="blog"',
                '    link="https://example.com"',
                "  />",
                "</RelatedResources>",
                "",
            ]
        )
        head_content = base_content.replace("Old example", "New example")
        patch = "\n".join(
            [
                "@@ -7,9 +7,9 @@",
                " ## Related resources",
                " ",
                " <RelatedResources>",
                "   <ResourceCard",
                '-    title="Old example"',
                '+    title="New example"',
                '     type="blog"',
                '     link="https://example.com"',
                "   />",
                " </RelatedResources>",
            ]
        )
        changed_file = SimpleNamespace(
            filename=file_path,
            status="modified",
            patch=patch,
            previous_filename=None,
        )
        repository = FakeRepository(
            {
                (file_path, "base123"): base_content,
                (file_path, "head123"): head_content,
            },
            FakePR([changed_file], "Update related resources", "base123", "head123"),
            {("base123", "head123"): [changed_file]},
        )
        github = FakeGithub({"acme/docs": repository})
        context = build_commit_diff_context(
            "acme/docs",
            "acme/docs-cn",
            "base123",
            "head123",
            github,
            self.repo_configs,
        )

        result = analyze_source_changes(
            context,
            github,
            special_files=["TOC.md", "keywords.md"],
            ignore_files=[],
            repo_configs=self.repo_configs,
        )

        self.assertEqual(result, ({}, {}, {}, {}, [], {}, {}, [], [], [], set(), {}))
        self.assertFalse(self.generated_file.exists())

    def test_commit_related_resources_deleted_section_is_filtered(self):
        file_path = "guide.md"
        base_content = "\n".join(
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
                "  <ResourceCard",
                '    title="Example"',
                '    type="blog"',
                '    link="https://example.com"',
                "  />",
                "</RelatedResources>",
                "",
            ]
        )
        head_content = "# Guide\n\n## Usage\n\nUse TiDB.\n"
        patch = "\n".join(
            [
                "@@ -3,13 +3,3 @@",
                " ## Usage",
                " ",
                " Use TiDB.",
                "-",
                "-## Related resources",
                "-",
                "-<RelatedResources>",
                "-  <ResourceCard",
                '-    title="Example"',
                '-    type="blog"',
                '-    link="https://example.com"',
                "-  />",
                "-</RelatedResources>",
            ]
        )
        changed_file = SimpleNamespace(
            filename=file_path,
            status="modified",
            patch=patch,
            previous_filename=None,
        )
        repository = FakeRepository(
            {
                (file_path, "base123"): base_content,
                (file_path, "head123"): head_content,
            },
            FakePR([changed_file], "Delete related resources", "base123", "head123"),
            {("base123", "head123"): [changed_file]},
        )
        github = FakeGithub({"acme/docs": repository})
        context = build_commit_diff_context(
            "acme/docs",
            "acme/docs-cn",
            "base123",
            "head123",
            github,
            self.repo_configs,
        )

        result = analyze_source_changes(
            context,
            github,
            special_files=["TOC.md", "keywords.md"],
            ignore_files=[],
            repo_configs=self.repo_configs,
        )

        self.assertEqual(result, ({}, {}, {}, {}, [], {}, {}, [], [], [], set(), {}))
        self.assertFalse(self.generated_file.exists())

    def test_pr_related_resources_section_is_not_filtered(self):
        file_path = "guide.md"
        base_content = "# Guide\n\n## Usage\n\nUse TiDB.\n"
        head_content = "\n".join(
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
                "  <ResourceCard",
                '    title="Example"',
                "  />",
                "</RelatedResources>",
                "",
            ]
        )
        patch = "\n".join(
            [
                "@@ -3,3 +3,11 @@",
                " ## Usage",
                " ",
                " Use TiDB.",
                "+",
                "+## Related resources",
                "+",
                "+<RelatedResources>",
                "+  <ResourceCard",
                '+    title="Example"',
                "+  />",
                "+</RelatedResources>",
            ]
        )
        changed_file = SimpleNamespace(
            filename=file_path,
            status="modified",
            patch=patch,
            previous_filename=None,
        )
        repository = FakeRepository(
            {
                (file_path, "base123"): base_content,
                (file_path, "head123"): head_content,
            },
            FakePR([changed_file], "Add related resources", "base123", "head123"),
            {("base123", "head123"): [changed_file]},
        )
        github = FakeGithub({"acme/docs": repository})
        context = build_pr_diff_context(self.pr_url, github, self.repo_configs)

        analyze_source_changes(
            context,
            github,
            special_files=["TOC.md", "keywords.md"],
            ignore_files=[],
            repo_configs=self.repo_configs,
        )

        source_diff = json.loads(self.generated_file.read_text(encoding="utf-8"))
        self.assertIn("added_7", source_diff)
        self.assertIn("<RelatedResources>", source_diff["added_7"]["new_content"])

    def test_pr_analysis_honors_source_files_before_processing(self):
        included_file = SimpleNamespace(
            filename="guide.md",
            status="modified",
            patch="\n".join(
                [
                    "@@ -1,4 +1,4 @@",
                    " # Title",
                    " ",
                    " ## Section",
                    "-Old text",
                    "+New text",
                ]
            ),
            previous_filename=None,
        )
        skipped_file = SimpleNamespace(
            filename="other.md",
            status="modified",
            patch="\n".join(
                [
                    "@@ -1,4 +1,4 @@",
                    " # Other",
                    " ",
                    " ## Section",
                    "-Old text",
                    "+New text",
                ]
            ),
            previous_filename=None,
        )
        repository = FakeRepository(
            {
                ("guide.md", "base123"): "# Title\n\n## Section\nOld text\n",
                ("guide.md", "head123"): "# Title\n\n## Section\nNew text\n",
                ("other.md", "base123"): "# Other\n\n## Section\nOld text\n",
                ("other.md", "head123"): "# Other\n\n## Section\nNew text\n",
            },
            FakePR([included_file, skipped_file], "Update guide", "base123", "head123"),
            {("base123", "head123"): [included_file, skipped_file]},
        )
        content_calls = []

        original_get_contents = repository.get_contents

        def tracking_get_contents(path, ref):
            content_calls.append((path, ref))
            return original_get_contents(path, ref)

        repository.get_contents = tracking_get_contents
        github = FakeGithub({"acme/docs": repository})
        context = build_pr_diff_context(self.pr_url, github, self.repo_configs)

        result = analyze_source_changes(
            context,
            github,
            special_files=["TOC.md", "keywords.md"],
            ignore_files=[],
            repo_configs=self.repo_configs,
            source_files="./guide.md, guide.md",
        )

        self.assertIn("guide.md", result[1])
        self.assertNotIn("other.md", result[1])
        self.assertTrue(content_calls)
        self.assertTrue(all(path == "guide.md" for path, _ in content_calls))

    def test_commit_related_resources_filter_can_be_disabled(self):
        file_path = "guide.md"
        base_content = "# Guide\n\n## Usage\n\nUse TiDB.\n"
        head_content = "\n".join(
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
                "  <ResourceCard",
                '    title="Example"',
                "  />",
                "</RelatedResources>",
                "",
            ]
        )
        patch = "\n".join(
            [
                "@@ -3,3 +3,11 @@",
                " ## Usage",
                " ",
                " Use TiDB.",
                "+",
                "+## Related resources",
                "+",
                "+<RelatedResources>",
                "+  <ResourceCard",
                '+    title="Example"',
                "+  />",
                "+</RelatedResources>",
            ]
        )
        changed_file = SimpleNamespace(
            filename=file_path,
            status="modified",
            patch=patch,
            previous_filename=None,
        )
        repository = FakeRepository(
            {
                (file_path, "base123"): base_content,
                (file_path, "head123"): head_content,
            },
            FakePR([changed_file], "Add related resources", "base123", "head123"),
            {("base123", "head123"): [changed_file]},
        )
        github = FakeGithub({"acme/docs": repository})
        context = build_commit_diff_context(
            "acme/docs",
            "acme/docs-cn",
            "base123",
            "head123",
            github,
            self.repo_configs,
        )
        context["repo_config"]["ignore_resource_card_section"] = False

        analyze_source_changes(
            context,
            github,
            special_files=["TOC.md", "keywords.md"],
            ignore_files=[],
            repo_configs=self.repo_configs,
        )

        source_diff = json.loads(self.generated_file.read_text(encoding="utf-8"))
        self.assertIn("added_7", source_diff)
        self.assertIn("<RelatedResources>", source_diff["added_7"]["new_content"])
        self.assertIn("<ResourceCard", source_diff["added_7"]["new_content"])

    def test_commit_related_resources_inside_code_block_is_not_filtered(self):
        file_path = "guide.md"
        base_content = "# Guide\n\n## Examples\n\nOld examples.\n"
        head_content = "\n".join(
            [
                "# Guide",
                "",
                "## Examples",
                "",
                "Old examples.",
                "",
                "```html",
                "<RelatedResources>",
                '  <ResourceCard title="Demo" />',
                "</RelatedResources>",
                "```",
                "",
            ]
        )
        patch = "\n".join(
            [
                "@@ -3,3 +3,10 @@",
                " ## Examples",
                " ",
                " Old examples.",
                "+",
                "+```html",
                "+<RelatedResources>",
                '+  <ResourceCard title="Demo" />',
                "+</RelatedResources>",
                "+```",
            ]
        )
        changed_file = SimpleNamespace(
            filename=file_path,
            status="modified",
            patch=patch,
            previous_filename=None,
        )
        repository = FakeRepository(
            {
                (file_path, "base123"): base_content,
                (file_path, "head123"): head_content,
            },
            FakePR([changed_file], "Add example", "base123", "head123"),
            {("base123", "head123"): [changed_file]},
        )
        github = FakeGithub({"acme/docs": repository})
        context = build_commit_diff_context(
            "acme/docs",
            "acme/docs-cn",
            "base123",
            "head123",
            github,
            self.repo_configs,
        )

        analyze_source_changes(
            context,
            github,
            special_files=["TOC.md", "keywords.md"],
            ignore_files=[],
            repo_configs=self.repo_configs,
        )

        source_diff = json.loads(self.generated_file.read_text(encoding="utf-8"))
        self.assertIn("modified_3", source_diff)
        self.assertIn("<RelatedResources>", source_diff["modified_3"]["new_content"])

    def test_related_resources_diff_filter_keeps_other_section_changes(self):
        base_content = "\n".join(
            [
                "# Guide",
                "",
                "## Usage",
                "",
                "Old usage.",
                "",
                "## Related resources",
                "",
                "<RelatedResources>",
                '  <ResourceCard title="Old" />',
                "</RelatedResources>",
                "",
            ]
        )
        head_content = base_content.replace("Old usage.", "New usage.").replace(
            'title="Old"',
            'title="New"',
        )
        diff_text = "\n".join(
            [
                "File: guide.md",
                "@@ -3,9 +3,9 @@",
                " ## Usage",
                " ",
                "-Old usage.",
                "+New usage.",
                " ",
                " ## Related resources",
                " ",
                " <RelatedResources>",
                '-  <ResourceCard title="Old" />',
                '+  <ResourceCard title="New" />',
                " </RelatedResources>",
            ]
        )

        filtered_diff = filter_related_resources_resource_card_diff(
            diff_text,
            base_content,
            head_content,
        )

        self.assertIn("-Old usage.", filtered_diff)
        self.assertIn("+New usage.", filtered_diff)
        self.assertNotIn("Related resources", filtered_diff)
        self.assertNotIn("RelatedResources", filtered_diff)
        self.assertNotIn("ResourceCard", filtered_diff)

    def test_commit_added_file_strips_multiple_related_resources_sections(self):
        file_path = "new-guide.md"
        file_content = "\n".join(
            [
                "# New Guide",
                "",
                "## Related resources",
                "",
                "<RelatedResources>",
                '  <ResourceCard title="Example A" type="blog" link="https://example.com/a" />',
                "</RelatedResources>",
                "",
                "## Usage",
                "",
                "Use TiDB.",
                "",
                "## More resources",
                "",
                "<RelatedResources>",
                '  <ResourceCard title="Example B" type="blog" link="https://example.com/b" />',
                "</RelatedResources>",
                "",
            ]
        )
        changed_file = SimpleNamespace(
            filename=file_path,
            status="added",
            patch="",
            previous_filename=None,
        )
        repository = FakeRepository(
            {
                (file_path, "head123"): file_content,
            },
            FakePR([changed_file], "Add guide", "base123", "head123"),
            {("base123", "head123"): [changed_file]},
        )
        github = FakeGithub({"acme/docs": repository})
        context = build_commit_diff_context(
            "acme/docs",
            "acme/docs-cn",
            "base123",
            "head123",
            github,
            self.repo_configs,
        )

        (
            added_sections,
            modified_sections,
            deleted_sections,
            added_files,
            deleted_files,
            toc_files,
            keyword_files,
            added_images,
            modified_images,
            deleted_images,
            _restructured,
            _index_files,
        ) = analyze_source_changes(
            context,
            github,
            special_files=["TOC.md", "keywords.md"],
            ignore_files=[],
            repo_configs=self.repo_configs,
        )

        self.assertEqual(added_sections, {})
        self.assertEqual(modified_sections, {})
        self.assertEqual(deleted_sections, {})
        self.assertIn(file_path, added_files)
        self.assertIn("## Usage", added_files[file_path])
        self.assertNotIn("RelatedResources", added_files[file_path])
        self.assertNotIn("ResourceCard", added_files[file_path])
        self.assertEqual(deleted_files, [])
        self.assertEqual(toc_files, {})
        self.assertEqual(keyword_files, {})
        self.assertEqual(added_images, [])
        self.assertEqual(modified_images, [])
        self.assertEqual(deleted_images, [])

    def test_commit_renamed_file_strips_related_resources_section(self):
        file_path = "new-guide.md"
        file_content = "\n".join(
            [
                "# New Guide",
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
        changed_file = SimpleNamespace(
            filename=file_path,
            status="renamed",
            patch="",
            previous_filename="old-guide.md",
        )
        repository = FakeRepository(
            {
                (file_path, "head123"): file_content,
            },
            FakePR([changed_file], "Rename guide", "base123", "head123"),
            {("base123", "head123"): [changed_file]},
        )
        github = FakeGithub({"acme/docs": repository})
        context = build_commit_diff_context(
            "acme/docs",
            "acme/docs-cn",
            "base123",
            "head123",
            github,
            self.repo_configs,
        )

        (
            added_sections,
            modified_sections,
            deleted_sections,
            added_files,
            deleted_files,
            toc_files,
            keyword_files,
            added_images,
            modified_images,
            deleted_images,
            _restructured,
            _index_files,
        ) = analyze_source_changes(
            context,
            github,
            special_files=["TOC.md", "keywords.md"],
            ignore_files=[],
            repo_configs=self.repo_configs,
        )

        self.assertEqual(added_sections, {})
        self.assertEqual(modified_sections, {})
        self.assertEqual(deleted_sections, {})
        self.assertEqual(deleted_files, ["old-guide.md"])
        self.assertIn(file_path, added_files)
        self.assertIn("## Usage", added_files[file_path])
        self.assertNotIn("RelatedResources", added_files[file_path])
        self.assertNotIn("ResourceCard", added_files[file_path])
        self.assertEqual(toc_files, {})
        self.assertEqual(keyword_files, {})
        self.assertEqual(added_images, [])
        self.assertEqual(modified_images, [])
        self.assertEqual(deleted_images, [])

    def test_bottom_modified_keeps_matching_hierarchy_in_source_diff_dict(self):
        base_content = "# Title\n\n## Section\nOld text\n"
        head_content = "# Title\n\n## Section\nNew text\n"
        base_hierarchy = build_hierarchy_dict(base_content)
        head_hierarchy = build_hierarchy_dict(head_content)

        source_diff_dict = build_source_diff_dict(
            modified_sections={"3": "## Section"},
            added_sections={},
            deleted_sections={},
            all_hierarchy_dict=head_hierarchy,
            base_hierarchy_dict=base_hierarchy,
            operations={"modified_lines": [{"line_number": 4, "is_header": False}]},
            file_content=head_content,
            base_file_content=base_content,
        )

        self.assertEqual(source_diff_dict["modified_3"]["original_hierarchy"], "bottom-modified-3")
        self.assertEqual(source_diff_dict["modified_3"]["matching_hierarchy"], "## Section")

    def test_crlf_wrapper_before_added_heading_routes_to_structural_reconciliation(self):
        file_path = "tidb-cloud/architecture-concepts.md"
        output_file = self.temp_output_dir / "tidb-cloud-architecture-concepts-source-diff-dict.json"
        if output_file.exists():
            output_file.unlink()

        base_content = "\n".join(
            [
                "# Architecture",
                "",
                "## Nodes",
                "",
                "### TiFlash node",
                "",
                "Old tail",
            ]
        )
        head_content_lf = "\n".join(
            [
                "# Architecture",
                "",
                "## Nodes",
                "",
                "### TiFlash node",
                "",
                "Old tail",
                "",
                '<CustomContent plan="premium">',
                "",
                "## Request units and capacity in {{{ .premium }}} {#request-units-and-capacity-in-premium}",
                "",
                "New details.",
                "",
                "</CustomContent>",
            ]
        )
        head_content = head_content_lf.replace("\n", "\r\n")
        changed_file = SimpleNamespace(
            filename=file_path,
            status="modified",
            patch=make_full_rewrite_patch(base_content, head_content_lf),
            previous_filename=None,
        )
        repository = FakeRepository(
            {
                (file_path, "base123"): base_content,
                (file_path, "head123"): head_content,
            },
            FakePR([changed_file], "Add premium section", "base123", "head123"),
            {("base123", "head123"): [changed_file]},
        )
        github = FakeGithub({"acme/docs": repository})
        context = build_commit_diff_context(
            "acme/docs",
            "acme/docs-cn",
            "base123",
            "head123",
            github,
            self.repo_configs,
        )

        result = analyze_source_changes(
            context,
            github,
            special_files=["TOC.md", "keywords.md"],
            ignore_files=[],
            repo_configs=self.repo_configs,
        )

        added_sections, modified_sections, _, added_files = result[:4]
        restructured_files = result[10]
        self.assertNotIn(file_path, added_sections)
        self.assertNotIn(file_path, modified_sections)
        self.assertEqual(added_files[file_path], head_content)
        self.assertIn(file_path, restructured_files)
        self.assertFalse(output_file.exists())

    def test_regular_paragraph_before_added_heading_still_modifies_previous_section(self):
        file_path = "guide.md"
        base_content = "# Guide\n\n## Existing\n\nOld\n"
        head_content = "# Guide\n\n## Existing\n\nOld\n\nNew paragraph.\n\n## New Section\n\nBody\n"
        changed_file = SimpleNamespace(
            filename=file_path,
            status="modified",
            patch=make_full_rewrite_patch(base_content, head_content),
            previous_filename=None,
        )
        repository = FakeRepository(
            {
                (file_path, "base123"): base_content,
                (file_path, "head123"): head_content,
            },
            FakePR([changed_file], "Add section with intro paragraph", "base123", "head123"),
            {("base123", "head123"): [changed_file]},
        )
        github = FakeGithub({"acme/docs": repository})
        context = build_commit_diff_context(
            "acme/docs",
            "acme/docs-cn",
            "base123",
            "head123",
            github,
            self.repo_configs,
        )

        analyze_source_changes(
            context,
            github,
            special_files=["TOC.md", "keywords.md"],
            ignore_files=[],
            repo_configs=self.repo_configs,
        )

        source_diff = json.loads(self.generated_file.read_text(encoding="utf-8"))
        self.assertIn("modified_3", source_diff)
        self.assertIn("added_9", source_diff)
        self.assertIn("New paragraph.", source_diff["modified_3"]["new_content"])

    def test_deleted_mdx_wrapper_routes_to_structural_reconciliation(self):
        file_path = "guide.md"
        base_content = "\n".join(
            [
                "# Guide",
                "",
                "## Limitations",
                "",
                "### Filtered out and deleted databases",
                "",
                "Filtered body.",
                "",
                '<CustomContent plan="essential">',
                "",
                "### Limitations of Alibaba Cloud RDS",
                "",
                "RDS body.",
                "",
                "### Limitations of Alibaba Cloud PolarDB-X",
                "",
                "PolarDB-X body.",
                "",
                "</CustomContent>",
                "",
                "### Limitations of existing data migration",
                "",
                "Existing data body.",
            ]
        )
        head_content = "\n".join(
            [
                "# Guide",
                "",
                "## Limitations",
                "",
                "### Filtered out and deleted databases",
                "",
                "Filtered body.",
                "",
                "### Limitations of Alibaba Cloud RDS",
                "",
                "RDS body.",
                "",
                "### Limitations of Alibaba Cloud PolarDB-X",
                "",
                "PolarDB-X body.",
                "",
                "### Limitations of existing data migration",
                "",
                "Existing data body.",
            ]
        )
        patch = "\n".join(
            [
                "@@ -5,17 +5,13 @@",
                " ### Filtered out and deleted databases",
                " ",
                " Filtered body.",
                " ",
                '-<CustomContent plan="essential">',
                "-",
                " ### Limitations of Alibaba Cloud RDS",
                " ",
                " RDS body.",
                " ",
                " ### Limitations of Alibaba Cloud PolarDB-X",
                " ",
                " PolarDB-X body.",
                " ",
                "-</CustomContent>",
                "-",
                " ### Limitations of existing data migration",
            ]
        )
        changed_file = SimpleNamespace(
            filename=file_path,
            status="modified",
            patch=patch,
            previous_filename=None,
        )
        repository = FakeRepository(
            {
                (file_path, "base123"): base_content,
                (file_path, "head123"): head_content,
            },
            FakePR([changed_file], "Remove essential wrapper", "base123", "head123"),
            {("base123", "head123"): [changed_file]},
        )
        github = FakeGithub({"acme/docs": repository})
        context = build_commit_diff_context(
            "acme/docs",
            "acme/docs-cn",
            "base123",
            "head123",
            github,
            self.repo_configs,
        )

        result = analyze_source_changes(
            context,
            github,
            special_files=["TOC.md", "keywords.md"],
            ignore_files=[],
            repo_configs=self.repo_configs,
        )

        added_sections, modified_sections, deleted_sections, added_files = result[:4]
        restructured_files = result[10]
        self.assertNotIn(file_path, added_sections)
        self.assertNotIn(file_path, modified_sections)
        self.assertNotIn(file_path, deleted_sections)
        self.assertEqual(added_files[file_path], head_content)
        self.assertIn(file_path, restructured_files)
        self.assertFalse(self.generated_file.exists())

    def test_deleted_line_uses_head_position_for_section_mapping(self):
        file_path = "guide.md"
        base_content = "# Guide\n\n## First\n\nkeep\n\n## Second\n\nold line\nmore\n"
        head_content = "# Guide\n\n## First\n\nkeep\n\n## Inserted\n\ninserted body\n\n## Second\n\nmore\n"
        patch = "\n".join(
            [
                "@@ -3,8 +3,11 @@",
                " ## First",
                " ",
                " keep",
                " ",
                "+## Inserted",
                "+",
                "+inserted body",
                "+",
                " ## Second",
                " ",
                "-old line",
                " more",
            ]
        )
        changed_file = SimpleNamespace(
            filename=file_path,
            status="modified",
            patch=patch,
            previous_filename=None,
        )
        repository = FakeRepository(
            {
                (file_path, "base123"): base_content,
                (file_path, "head123"): head_content,
            },
            FakePR([changed_file], "Add section and delete line", "base123", "head123"),
            {("base123", "head123"): [changed_file]},
        )
        github = FakeGithub({"acme/docs": repository})
        context = build_commit_diff_context(
            "acme/docs",
            "acme/docs-cn",
            "base123",
            "head123",
            github,
            self.repo_configs,
        )

        analyze_source_changes(
            context,
            github,
            special_files=["TOC.md", "keywords.md"],
            ignore_files=[],
            repo_configs=self.repo_configs,
        )

        source_diff = json.loads(self.generated_file.read_text(encoding="utf-8"))
        self.assertIn("added_7", source_diff)
        self.assertIn("modified_11", source_diff)
        self.assertNotIn("modified_7", source_diff)
        self.assertIn("old line", source_diff["modified_11"]["old_content"])

    def test_blank_heading_prefix_scan_is_bounded(self):
        file_lines = ["# Guide", "", "Old", "", "", "", "## New"]
        operations = {
            "added_lines": [
                {"line_number": 4, "is_header": False},
                {"line_number": 5, "is_header": False},
                {"line_number": 6, "is_header": False},
                {"line_number": 7, "is_header": True},
            ],
            "modified_lines": [],
            "deleted_lines": [],
        }

        prefixes, ignored_lines = collect_added_heading_prefix_lines(file_lines, operations)

        self.assertEqual({}, prefixes)
        self.assertEqual({6}, ignored_lines)

    def test_bare_cr_headings_use_normalized_line_numbers(self):
        self.assertEqual(
            {
                1: "# Guide",
                3: "## Section",
            },
            build_hierarchy_dict("# Guide\r\r## Section\rBody\r"),
        )

    def test_snapshot_diff_is_only_used_for_line_ending_style_changes(self):
        operations = {
            "added_lines": [{"line_number": 4, "is_header": False, "content": "New"}],
            "modified_lines": [],
            "deleted_lines": [],
        }

        self.assertIs(
            operations,
            maybe_use_normalized_snapshot_operations(
                operations,
                "# Guide\n\n## Section\nOld\n",
                "# Guide\n\n## Section\nNew\n",
            ),
        )

    def test_numbered_heading_rename_is_treated_as_modified(self):
        patch = "\n".join(
            [
                "@@ -12,9 +12,9 @@",
                " Intro text",
                "-## Step 1: Create a TiDB cluster",
                "+## Step 1: Create a {{{ .starter }}} instance {#step-1-create-a-starter-instance}",
                " ",
                "-Old body",
                "+New body",
                " ## Step 2: Try AI-assisted SQL Editor",
            ]
        )
        changed_file = SimpleNamespace(filename="quickstart.md", status="modified", patch=patch)

        operations = analyze_diff_operations(changed_file)

        self.assertEqual([], [line["content"] for line in operations["added_lines"] if line["is_header"]])
        self.assertEqual([], [line["content"] for line in operations["deleted_lines"] if line["is_header"]])
        self.assertEqual(
            ["## Step 1: Create a {{{ .starter }}} instance {#step-1-create-a-starter-instance}"],
            [line["content"] for line in operations["modified_lines"] if line["is_header"]],
        )
        self.assertEqual(
            "## Step 1: Create a TiDB cluster",
            next(line for line in operations["modified_lines"] if line["is_header"])["original_content"],
        )

    def test_keyword_overlapping_heading_rename_is_treated_as_modified(self):
        patch = "\n".join(
            [
                "@@ -144,8 +198,8 @@",
                " Some context",
                "-### Invite an organization member",
                "+### Invite a user to your organization",
                " ",
                "-Old body",
                "+New body",
                " ### Remove an organization member",
            ]
        )
        changed_file = SimpleNamespace(filename="manage-user-access.md", status="modified", patch=patch)

        operations = analyze_diff_operations(changed_file)

        self.assertEqual([], [line["content"] for line in operations["added_lines"] if line["is_header"]])
        self.assertEqual([], [line["content"] for line in operations["deleted_lines"] if line["is_header"]])
        self.assertEqual(
            ["### Invite a user to your organization"],
            [line["content"] for line in operations["modified_lines"] if line["is_header"]],
        )
        self.assertEqual(
            "### Invite an organization member",
            next(line for line in operations["modified_lines"] if line["is_header"])["original_content"],
        )

    def test_short_heading_rename_with_shared_leading_keyword_is_modified(self):
        patch = "\n".join(
            [
                "@@ -43,9 +43,9 @@",
                " ### External storage",
                " ",
                "-### Downstream cluster",
                "+### Downstream TiDB Cloud",
                " ",
                " The sharded schemas and tables are merged into the table `store.sales`.",
            ]
        )
        changed_file = SimpleNamespace(filename="migrate-sql-shards.md", status="modified", patch=patch)

        operations = analyze_diff_operations(changed_file)

        self.assertEqual([], [line["content"] for line in operations["added_lines"] if line["is_header"]])
        self.assertEqual([], [line["content"] for line in operations["deleted_lines"] if line["is_header"]])
        self.assertEqual(
            ["### Downstream TiDB Cloud"],
            [line["content"] for line in operations["modified_lines"] if line["is_header"]],
        )
        self.assertEqual(
            "### Downstream cluster",
            next(line for line in operations["modified_lines"] if line["is_header"])["original_content"],
        )

    def test_adjacent_same_level_heading_rename_without_keyword_overlap_is_modified(self):
        patch = "\n".join(
            [
                "@@ -240,8 +240,8 @@",
                " ## Incremental replication",
                " ",
                "-### Prerequisites",
                "+### Before you begin",
                " ",
                "-Old body",
                "+New body",
            ]
        )
        changed_file = SimpleNamespace(filename="guide.md", status="modified", patch=patch)

        operations = analyze_diff_operations(changed_file)

        self.assertEqual([], [line["content"] for line in operations["added_lines"] if line["is_header"]])
        self.assertEqual([], [line["content"] for line in operations["deleted_lines"] if line["is_header"]])
        self.assertEqual(
            ["### Before you begin"],
            [line["content"] for line in operations["modified_lines"] if line["is_header"]],
        )
        self.assertEqual(
            "### Prerequisites",
            next(line for line in operations["modified_lines"] if line["is_header"])["original_content"],
        )

    def test_adjacent_delete_add_heading_blocks_are_not_paired_as_renames(self):
        patch = "\n".join(
            [
                "@@ -20,8 +20,8 @@",
                " ## Parent",
                "-### Old one",
                "-### Old two",
                "+### New three",
                "+### New four",
                " ## Next",
            ]
        )
        changed_file = SimpleNamespace(filename="guide.md", status="modified", patch=patch)

        operations = analyze_diff_operations(changed_file)

        self.assertEqual(
            ["### New three", "### New four"],
            [line["content"] for line in operations["added_lines"] if line["is_header"]],
        )
        self.assertEqual(
            ["### Old one", "### Old two"],
            [line["content"] for line in operations["deleted_lines"] if line["is_header"]],
        )
        self.assertEqual([], [line["content"] for line in operations["modified_lines"] if line["is_header"]])

    def test_renamed_markdown_is_treated_as_delete_and_add(self):
        renamed_file = SimpleNamespace(
            filename="new-guide.md",
            status="renamed",
            patch="",
            previous_filename="guide.md",
        )
        head_sha = "head123"
        repository = FakeRepository(
            {
                ("new-guide.md", head_sha): "# Title\n\n## Section\nRenamed text\n",
            },
            FakePR([renamed_file], "Rename guide", "base123", head_sha),
            {("base123", head_sha): [renamed_file]},
        )
        github = FakeGithub({"acme/docs": repository})
        commit_context = build_commit_diff_context(
            "acme/docs",
            "acme/docs-cn",
            "base123",
            head_sha,
            github,
            self.repo_configs,
        )

        (
            added_sections,
            modified_sections,
            deleted_sections,
            added_files,
            deleted_files,
            toc_files,
            keyword_files,
            added_images,
            modified_images,
            deleted_images,
            _restructured,
            _index_files,
        ) = analyze_source_changes(
            commit_context,
            github,
            special_files=["TOC.md", "keywords.md"],
            ignore_files=[],
            repo_configs=self.repo_configs,
        )

        self.assertEqual(added_sections, {})
        self.assertEqual(modified_sections, {})
        self.assertEqual(deleted_sections, {})
        self.assertEqual(added_files, {"new-guide.md": "# Title\n\n## Section\nRenamed text\n"})
        self.assertEqual(deleted_files, ["guide.md"])
        self.assertEqual(toc_files, {})
        self.assertEqual(keyword_files, {})
        self.assertEqual(added_images, [])
        self.assertEqual(modified_images, [])
        self.assertEqual(deleted_images, [])

    def test_target_hierarchy_prefers_local_target_checkout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target_file = Path(tmpdir) / "guide.md"
            target_file.write_text("# Local title\n\n## Local section\n", encoding="utf-8")

            repository = FakeRepository(
                {("guide.md", "i18n-zh-release-8.5"): "# Remote title\n"},
                FakePR([], "Empty", "base123", "head123"),
                {},
            )
            github = FakeGithub({"acme/docs-cn": repository})

            hierarchy, lines = get_target_hierarchy_and_content(
                "guide.md",
                github,
                "acme/docs-cn",
                target_local_path=tmpdir,
                prefer_local_target_for_read=True,
                target_ref="i18n-zh-release-8.5",
            )

        self.assertIn(1, hierarchy)
        self.assertEqual(hierarchy[1], "# Local title")
        self.assertEqual(lines[0], "# Local title")

    def test_target_hierarchy_uses_target_ref_before_default_branch(self):
        repository = FakeRepository(
            {
                ("guide.md", "master"): "# Master title\n",
                ("guide.md", "i18n-zh-release-8.5"): "# Target title\n",
            },
            FakePR([], "Empty", "base123", "head123"),
            {},
        )
        github = FakeGithub({"acme/docs-cn": repository})

        hierarchy, lines = get_target_hierarchy_and_content(
            "guide.md",
            github,
            "acme/docs-cn",
            target_ref="i18n-zh-release-8.5",
        )

        self.assertEqual(hierarchy[1], "# Target title")
        self.assertEqual(lines[0], "# Target title")

    def test_target_hierarchy_does_not_fallback_to_default_branch_with_explicit_target_ref(self):
        repository = FakeRepository(
            {
                ("guide.md", "master"): "# Master title\n",
            },
            FakePR([], "Empty", "base123", "head123"),
            {},
        )
        github = FakeGithub({"acme/docs-cn": repository})

        hierarchy, lines = get_target_hierarchy_and_content(
            "guide.md",
            github,
            "acme/docs-cn",
            target_ref="i18n-zh-release-8.5",
        )

        self.assertEqual(hierarchy, {})
        self.assertEqual(lines, [])

    def test_target_file_content_uses_default_branch_without_explicit_target_ref(self):
        repository = FakeRepository(
            {
                ("guide.md", "master"): "# Master title\n",
            },
            FakePR([], "Empty", "base123", "head123"),
            {},
        )
        github = FakeGithub({"acme/docs-cn": repository})

        content, source = get_target_file_content("guide.md", github, "acme/docs-cn")

        self.assertEqual(content, "# Master title\n")
        self.assertEqual(source, "remote:acme/docs-cn@master")

    def test_keyword_tabs_fallbacks_to_remote_when_local_blocks_are_empty(self):
        patch = "\n".join(
            [
                "@@ -5,1 +5,1 @@",
                "- - apple",
                "+ - apple updated",
            ]
        )
        changed_file = SimpleNamespace(
            filename="keywords.md",
            status="modified",
            patch=patch,
            previous_filename=None,
        )
        base_content = "# Keywords\n\n<TabsPanel>\n<a id=\"A\" class=\"letter\" href=\"#A\">A</a>\n- apple\n"
        head_content = "# Keywords\n\n<TabsPanel>\n<a id=\"A\" class=\"letter\" href=\"#A\">A</a>\n- apple updated\n"
        target_content = "# 关键字\n\n<TabsPanel>\n<a id=\"A\" class=\"letter\" href=\"#A\">A</a>\n- 苹果\n"

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "keywords.md").write_text("# 关键字\n\nNo tabs yet\n", encoding="utf-8")
            repo_configs = {
                "acme/docs": {
                    "target_repo": "acme/docs-cn",
                    "target_local_path": tmpdir,
                    "prefer_local_target_for_read": True,
                    "target_ref": "i18n",
                    "source_language": "English",
                    "target_language": "Chinese",
                }
            }
            source_repo = FakeRepository(
                {
                    ("keywords.md", "base123"): base_content,
                    ("keywords.md", "head123"): head_content,
                },
                FakePR([changed_file], "Update keywords", "base123", "head123"),
                {("base123", "head123"): [changed_file]},
            )
            target_repo = FakeRepository(
                {
                    ("keywords.md", "i18n"): target_content,
                },
                FakePR([], "Empty", "base123", "head123"),
                {},
            )
            github = FakeGithub({"acme/docs": source_repo, "acme/docs-cn": target_repo})
            commit_context = build_commit_diff_context(
                "acme/docs",
                "acme/docs-cn",
                "base123",
                "head123",
                github,
                repo_configs,
            )

            result = analyze_source_changes(
                commit_context,
                github,
                special_files=["keywords.md"],
                ignore_files=[],
                repo_configs=repo_configs,
            )

        keyword_files = result[6]
        self.assertIn("keywords.md", keyword_files)
        self.assertIn("- 苹果", keyword_files["keywords.md"]["tabs_changes"]["A"]["target_old_block"])

    def test_toc_snapshot_sync_is_commit_only(self):
        patch = "\n".join(
            [
                "@@ -1,1 +1,2 @@",
                " - [Old](/old.md)",
                "+- [New](/new.md)",
            ]
        )
        changed_file = SimpleNamespace(
            filename="TOC-test.md",
            status="modified",
            patch=patch,
            previous_filename=None,
        )
        repo_configs = {
            "acme/docs": {
                "target_repo": "acme/docs-cn",
                "target_local_path": "/tmp/target",
                "prefer_local_target_for_read": False,
                "source_language": "English",
                "target_language": "Chinese",
            }
        }
        source_repo = FakeRepository(
            {
                ("TOC-test.md", "base123"): "- [Old](/old.md)\n",
                ("TOC-test.md", "head123"): "- [Old](/old.md)\n- [New](/new.md)\n",
            },
            FakePR([changed_file], "Update toc", "base123", "head123"),
            {("base123", "head123"): [changed_file]},
        )
        target_repo = FakeRepository(
            {
                ("TOC-test.md", "master"): "- [旧](/old.md)\n- [仅目标端](/target-only.md)\n",
            },
            FakePR([], "Empty", "base123", "head123"),
            {},
        )
        github = FakeGithub({"acme/docs": source_repo, "acme/docs-cn": target_repo})

        pr_context = build_pr_diff_context("https://github.com/acme/docs/pull/123", github, repo_configs)
        commit_context = build_commit_diff_context(
            "acme/docs",
            "acme/docs-cn",
            "base123",
            "head123",
            github,
            repo_configs,
        )

        pr_result = analyze_source_changes(
            pr_context,
            github,
            special_files=["TOC.md", "keywords.md"],
            ignore_files=[],
            repo_configs=repo_configs,
        )
        commit_result = analyze_source_changes(
            commit_context,
            github,
            special_files=["TOC.md", "keywords.md"],
            ignore_files=[],
            repo_configs=repo_configs,
        )

        pr_toc = pr_result[5]["TOC-test.md"]
        commit_toc = commit_result[5]["TOC-test.md"]

        self.assertTrue(pr_toc["operations"])
        self.assertNotIn("source_base_content", pr_toc)
        self.assertEqual([], commit_toc["operations"])
        self.assertEqual("- [Old](/old.md)\n", commit_toc["source_base_content"])
        self.assertEqual("- [Old](/old.md)\n- [New](/new.md)\n", commit_toc["source_head_content"])

    def test_local_commit_context_reads_all_changed_files_from_git_diff(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            subprocess.run(["git", "init"], cwd=repo, check=True, stdout=subprocess.DEVNULL)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)

            (repo / "guide.md").write_text("# Guide\n\nOld\n", encoding="utf-8")
            (repo / "old-name.md").write_text("# Old\n", encoding="utf-8")
            subprocess.run(["git", "add", "."], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-m", "base"], cwd=repo, check=True, stdout=subprocess.DEVNULL)
            base_ref = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()

            (repo / "guide.md").write_text("# Guide\n\nNew\n", encoding="utf-8")
            (repo / "new-name.md").write_text((repo / "old-name.md").read_text(encoding="utf-8"), encoding="utf-8")
            (repo / "old-name.md").unlink()
            subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-m", "head"], cwd=repo, check=True, stdout=subprocess.DEVNULL)
            head_ref = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()

            context = build_local_commit_diff_context(
                "acme/docs",
                "acme/docs-cn",
                base_ref,
                head_ref,
                str(repo),
                self.repo_configs,
            )

        files = {file.filename: file for file in context["changed_files"]}
        self.assertEqual(files["guide.md"].status, "modified")
        self.assertIn("+New", files["guide.md"].patch)
        self.assertEqual(files["new-name.md"].status, "renamed")
        self.assertEqual(files["new-name.md"].previous_filename, "old-name.md")


class DetectRestructuredFileTest(unittest.TestCase):
    """Tests for detect_restructured_file()."""

    def _make_operations(self, base_content, head_content):
        patch = make_full_rewrite_patch(base_content, head_content)
        file = SimpleNamespace(
            filename="doc.md", status="modified", patch=patch, previous_filename=None,
        )
        return analyze_diff_operations(file)

    def _make_operations_from_patch(self, patch):
        file = SimpleNamespace(
            filename="doc.md", status="modified", patch=patch, previous_filename=None,
        )
        return analyze_diff_operations(file)

    def test_all_headings_changed_is_restructured(self):
        base = "# Doc\n\n## Old A\n\nText A\n\n## Old B\n\nText B\n"
        head = "# Doc\n\n## New X\n\nText X\n\n## New Y\n\nText Y\n\n## New Z\n\nText Z\n"
        ops = self._make_operations(base, head)
        self.assertTrue(detect_restructured_file(head, base, ops))

    def test_one_heading_unchanged_is_not_restructured(self):
        base = "# Doc\n\n## Same\n\nOld text\n\n## Old B\n\nText B\n"
        head = "# Doc\n\n## Same\n\nNew text\n\n## New Y\n\nText Y\n"
        ops = self._make_operations(base, head)
        self.assertFalse(detect_restructured_file(head, base, ops))

    def test_too_few_headings_is_not_restructured(self):
        base = "# Doc\n\n## Only One\n\nText\n"
        head = "# Doc\n\n## Changed\n\nText\n"
        ops = self._make_operations(base, head)
        self.assertFalse(detect_restructured_file(head, base, ops))

    def test_no_sub_headings_is_not_restructured(self):
        base = "# Doc\n\nBody\n"
        head = "# Doc\n\nNew body\n"
        ops = self._make_operations(base, head)
        self.assertFalse(detect_restructured_file(head, base, ops))

    def test_added_section_with_existing_unchanged_is_not_restructured(self):
        base = "# Doc\n\n## Existing\n\nOld\n"
        head = "# Doc\n\n## Existing\n\nOld\n\n## New Section\n\nBody\n"
        ops = self._make_operations(base, head)
        self.assertFalse(detect_restructured_file(head, base, ops))

    def test_all_headings_renamed_is_restructured(self):
        base = "# Guide\n\n## Step 1 old\n\nA\n\n## Step 2 old\n\nB\n\n## Step 3 old\n\nC\n"
        head = "# Guide\n\n## Step 1 new\n\nA\n\n## Step 2 new\n\nB\n\n## Step 3 new\n\nC\n"
        ops = self._make_operations(base, head)
        self.assertTrue(detect_restructured_file(head, base, ops))

    def test_headings_inside_code_blocks_are_ignored(self):
        base = "# Doc\n\n## Real\n\nText\n\n```\n## Not Real\n```\n"
        head = "# Doc\n\n## Real\n\nText\n\n```\n## Not Real\n```\n"
        ops = self._make_operations(base, head)
        self.assertFalse(detect_restructured_file(head, base, ops))

    def test_heading_only_renames_below_ratio_is_not_restructured(self):
        """All headings renamed in a long doc but changed lines < 50%."""
        body = "\n".join(f"Line {i}" for i in range(1, 16))
        base = f"# Doc\n\n## Alpha\n\n{body}\n\n## Beta\n\n{body}\n"
        head = f"# Doc\n\n## Gamma\n\n{body}\n\n## Delta\n\n{body}\n"
        patch = "\n".join([
            "@@ -1,5 +1,5 @@",
            " # Doc",
            " ",
            "-## Alpha",
            "+## Gamma",
            " ",
            " Line 1",
            "@@ -19,5 +19,5 @@",
            " Line 15",
            " ",
            "-## Beta",
            "+## Delta",
            " ",
            " Line 1",
        ])
        ops = self._make_operations_from_patch(patch)
        self.assertFalse(detect_restructured_file(head, base, ops))

    def test_partial_diff_with_unchanged_context_heading(self):
        """Heading appears as context line in a realistic partial diff."""
        base = "# Doc\n\n## Unchanged\n\nBody A\n\n## Old\n\nBody B\n"
        head = "# Doc\n\n## Unchanged\n\nBody A\n\n## New\n\nBody B\n"
        patch = "\n".join([
            "@@ -5,5 +5,5 @@",
            " Body A",
            " ",
            "-## Old",
            "+## New",
            " ",
            " Body B",
        ])
        ops = self._make_operations_from_patch(patch)
        self.assertFalse(detect_restructured_file(head, base, ops))

    def test_duplicate_base_headings_one_unchanged_is_not_restructured(self):
        """Base has two identical headings; only one deleted → not restructured."""
        base = (
            "# Doc\n\n## Examples\n\nFirst block\n\n"
            "## Other\n\nMiddle block\n\n"
            "## Examples\n\nSecond block\n"
        )
        head = (
            "# Doc\n\n## New A\n\nFirst block\n\n"
            "## New B\n\nMiddle block\n\n"
            "## Examples\n\nSecond block\n"
        )
        ops = self._make_operations(base, head)
        self.assertFalse(detect_restructured_file(head, base, ops))

    def test_commit_mode_routes_restructured_to_added_files(self):
        """Integration: restructured file in commit mode goes to added_files."""
        file_path = "overview.md"
        base_content = "# Overview\n\n## A\n\nText A\n\n## B\n\nText B\n"
        head_content = "# Overview\n\n## X\n\nText X\n\n## Y\n\nText Y\n\n## Z\n\nText Z\n"
        changed_file = SimpleNamespace(
            filename=file_path,
            status="modified",
            patch=make_full_rewrite_patch(base_content, head_content),
            previous_filename=None,
        )
        repository = FakeRepository(
            {
                (file_path, "base1"): base_content,
                (file_path, "head1"): head_content,
            },
            FakePR([changed_file], "Restructure overview", "base1", "head1"),
            {("base1", "head1"): [changed_file]},
        )
        github = FakeGithub({"acme/docs": repository})
        repo_configs = {
            "acme/docs": {
                "source_repo": "acme/docs",
                "target_repo": "acme/docs-cn",
                "source_language": "English",
                "target_language": "Chinese",
            },
        }
        context = build_commit_diff_context(
            "acme/docs", "acme/docs-cn", "base1", "head1", github, repo_configs,
        )

        result = analyze_source_changes(
            context, github,
            special_files=["TOC.md", "keywords.md"],
            ignore_files=[],
            repo_configs=repo_configs,
        )
        added_sections, modified_sections, deleted_sections, added_files = result[:4]
        restructured_files = result[10]

        self.assertIn(file_path, added_files)
        self.assertEqual(added_files[file_path], head_content)
        self.assertNotIn(file_path, modified_sections)
        self.assertNotIn(file_path, added_sections)
        self.assertNotIn(file_path, deleted_sections)
        self.assertIn(file_path, restructured_files)

    def test_pr_mode_routes_restructured_to_added_files(self):
        """In PR mode, restructured files are rerouted to added_files for full translation."""
        file_path = "overview.md"
        base_content = "# Overview\n\n## A\n\nText A\n\n## B\n\nText B\n"
        head_content = "# Overview\n\n## X\n\nText X\n\n## Y\n\nText Y\n\n## Z\n\nText Z\n"
        changed_file = SimpleNamespace(
            filename=file_path,
            status="modified",
            patch=make_full_rewrite_patch(base_content, head_content),
            previous_filename=None,
        )
        repository = FakeRepository(
            {
                (file_path, "base1"): base_content,
                (file_path, "head1"): head_content,
            },
            FakePR([changed_file], "Restructure overview", "base1", "head1"),
            {("base1", "head1"): [changed_file]},
        )
        github = FakeGithub({"acme/docs": repository})
        repo_configs = {
            "acme/docs": {
                "source_repo": "acme/docs",
                "target_repo": "acme/docs-cn",
                "source_language": "English",
                "target_language": "Chinese",
            }
        }

        result = analyze_source_changes(
            f"https://github.com/acme/docs/pull/1",
            github,
            special_files=["TOC.md", "keywords.md"],
            ignore_files=[],
            repo_configs=repo_configs,
        )
        added_files = result[3]
        restructured_files = result[10]

        self.assertIn(file_path, added_files)
        self.assertEqual(added_files[file_path], head_content)
        self.assertIn(file_path, restructured_files)


class HeadingLevelOnlyChangeTest(unittest.TestCase):
    """Tests for is_heading_level_only_change() and detect_heading_level_only_file()."""

    def test_basic_level_change(self):
        new = "### log.file\nSome content here\nMore content"
        old = "## log.file\nSome content here\nMore content"
        result, old_level, new_level = is_heading_level_only_change(new, old)
        self.assertTrue(result)
        self.assertEqual(old_level, 2)
        self.assertEqual(new_level, 3)

    def test_same_level_returns_false(self):
        new = "## log.file\nSome content"
        old = "## log.file\nSome content"
        result, _, _ = is_heading_level_only_change(new, old)
        self.assertFalse(result)

    def test_title_change_returns_false(self):
        new = "### log.file-new\nSome content"
        old = "## log.file\nSome content"
        result, _, _ = is_heading_level_only_change(new, old)
        self.assertFalse(result)

    def test_body_change_returns_false(self):
        new = "### log.file\nNew content here"
        old = "## log.file\nOld content here"
        result, _, _ = is_heading_level_only_change(new, old)
        self.assertFalse(result)

    def test_empty_content(self):
        result, _, _ = is_heading_level_only_change("", "## title\ncontent")
        self.assertFalse(result)
        result, _, _ = is_heading_level_only_change("## title\ncontent", "")
        self.assertFalse(result)

    def test_non_heading_first_line(self):
        new = "Not a heading\nSome content"
        old = "## title\nSome content"
        result, _, _ = is_heading_level_only_change(new, old)
        self.assertFalse(result)

    def test_heading_with_anchor(self):
        new = "### log.file {#log-file}\nContent"
        old = "## log.file {#log-file}\nContent"
        result, old_level, new_level = is_heading_level_only_change(new, old)
        self.assertTrue(result)
        self.assertEqual(old_level, 2)
        self.assertEqual(new_level, 3)

    def test_detect_file_all_heading_level_changes(self):
        operations = {
            'added_lines': [],
            'deleted_lines': [],
            'modified_lines': [
                {'line_number': 5, 'is_header': True, 'content': '### log.file', 'original_content': '## log.file'},
                {'line_number': 20, 'is_header': True, 'content': '### log.reporter', 'original_content': '## log.reporter'},
            ]
        }
        result = detect_heading_level_only_file("", "", operations)
        self.assertTrue(result)

    def test_detect_file_with_non_header_modification(self):
        operations = {
            'added_lines': [],
            'deleted_lines': [],
            'modified_lines': [
                {'line_number': 5, 'is_header': True, 'content': '### log.file', 'original_content': '## log.file'},
                {'line_number': 10, 'is_header': False, 'content': 'new text', 'original_content': 'old text'},
            ]
        }
        result = detect_heading_level_only_file("", "", operations)
        self.assertFalse(result)

    def test_detect_file_with_content_added_lines(self):
        operations = {
            'added_lines': [
                {'line_number': 15, 'is_header': False, 'content': 'new paragraph'},
            ],
            'deleted_lines': [],
            'modified_lines': [
                {'line_number': 5, 'is_header': True, 'content': '### log.file', 'original_content': '## log.file'},
            ]
        }
        result = detect_heading_level_only_file("", "", operations)
        self.assertFalse(result)

    def test_detect_file_add_delete_pairs(self):
        """Heading level changes may appear as add/delete pairs in unified diffs."""
        operations = {
            'added_lines': [
                {'line_number': 5, 'is_header': True, 'content': '### log.file'},
            ],
            'deleted_lines': [
                {'line_number': 5, 'is_header': True, 'content': '## log.file'},
            ],
            'modified_lines': []
        }
        result = detect_heading_level_only_file("", "", operations)
        self.assertTrue(result)

    def test_detect_file_add_delete_pairs_title_mismatch(self):
        operations = {
            'added_lines': [
                {'line_number': 5, 'is_header': True, 'content': '### new-section'},
            ],
            'deleted_lines': [
                {'line_number': 5, 'is_header': True, 'content': '## old-section'},
            ],
            'modified_lines': []
        }
        result = detect_heading_level_only_file("", "", operations)
        self.assertFalse(result)

    def test_build_source_diff_dict_marks_heading_level_only(self):
        """build_source_diff_dict should mark heading-level-only entries."""
        base_content = "# Title\n\n## log\n\nSome intro.\n\n## log.file\n\nFile config here.\n"
        head_content = "# Title\n\n## log\n\nSome intro.\n\n### log.file\n\nFile config here.\n"

        base_hierarchy = build_hierarchy_dict(base_content)
        head_hierarchy = build_hierarchy_dict(head_content)

        # Heading is at line 7 in both files
        modified_sections = {7: head_hierarchy[7]}
        added_sections = {}
        deleted_sections = {}

        operations = {
            'added_lines': [],
            'deleted_lines': [],
            'modified_lines': [
                {'line_number': 7, 'is_header': True, 'content': '### log.file', 'original_content': '## log.file'},
            ]
        }

        result = build_source_diff_dict(
            modified_sections, added_sections, deleted_sections,
            head_hierarchy, base_hierarchy, operations,
            head_content, base_content
        )

        self.assertIn("modified_7", result)
        entry = result["modified_7"]
        self.assertTrue(entry.get("heading_level_change_only"))
        self.assertEqual(entry["old_heading_level"], 2)
        self.assertEqual(entry["new_heading_level"], 3)

    def test_added_section_anchors_before_immediately_renamed_section(self):
        base_content = (
            "# Audit\n\n"
            "## Specify auditing filter rules\n\n"
            "Old filter text.\n\n"
            "## View audit logs\n\n"
            "View text.\n"
        )
        head_content = (
            "# Audit\n\n"
            "## Configure database audit logging settings\n\n"
            "Configure text.\n\n"
            "## Specify audit filter rules\n\n"
            "New filter text.\n\n"
            "## View audit logs\n\n"
            "View text.\n"
        )
        base_hierarchy = build_hierarchy_dict(base_content)
        head_hierarchy = build_hierarchy_dict(head_content)

        result = build_source_diff_dict(
            modified_sections={7: head_hierarchy[7]},
            added_sections={3: head_hierarchy[3]},
            deleted_sections={},
            all_hierarchy_dict=head_hierarchy,
            base_hierarchy_dict=base_hierarchy,
            operations={
                "added_lines": [
                    {
                        "line_number": 3,
                        "is_header": True,
                        "content": "## Configure database audit logging settings",
                    },
                ],
                "deleted_lines": [],
                "modified_lines": [
                    {
                        "line_number": 7,
                        "is_header": True,
                        "content": "## Specify audit filter rules",
                        "original_content": "## Specify auditing filter rules",
                    },
                ],
            },
            file_content=head_content,
            base_file_content=base_content,
        )

        self.assertEqual(
            "## Specify auditing filter rules",
            result["added_3"]["original_hierarchy"],
        )

    def test_added_section_uses_parent_path_for_duplicate_renamed_anchor(self):
        base_content = (
            "# Guide\n\n"
            "## First workflow\n\n"
            "### Existing step\n\n"
            "## Second workflow\n\n"
            "### Existing step\n"
        )
        head_content = (
            "# Guide\n\n"
            "## First workflow\n\n"
            "### Existing step\n\n"
            "## Second workflow\n\n"
            "### New prerequisite\n\n"
            "### Renamed step\n"
        )
        base_hierarchy = build_hierarchy_dict(base_content)
        head_hierarchy = build_hierarchy_dict(head_content)

        result = build_source_diff_dict(
            modified_sections={},
            added_sections={9: head_hierarchy[9]},
            deleted_sections={},
            all_hierarchy_dict=head_hierarchy,
            base_hierarchy_dict=base_hierarchy,
            operations={
                "added_lines": [
                    {
                        "line_number": 9,
                        "is_header": True,
                        "content": "### New prerequisite",
                    },
                ],
                "deleted_lines": [],
                "modified_lines": [
                    {
                        "line_number": 11,
                        "is_header": True,
                        "content": "### Renamed step",
                        "original_content": "### Existing step",
                    },
                ],
            },
            file_content=head_content,
            base_file_content=base_content,
        )

        self.assertEqual(
            "## Second workflow > ### Existing step",
            result["added_9"]["original_hierarchy"],
        )

    def test_modified_anchor_fallback_uses_head_titles_for_ordinal_matching(self):
        base_content = (
            "# Guide\n"
            "## X\n"
            "### B\n"
            "#### New\n"
            "## Y\n"
            "### B\n"
            "#### New\n"
            "## U\n"
            "### Old\n"
            "## V\n"
            "### Old\n"
        )
        head_content = (
            "# Guide\n"
            "## X\n"
            "### B\n"
            "#### New\n"
            "## A\n"
            "### B\n"
            "#### Added\n"
            "#### New\n"
            "## Y\n"
            "### B\n"
            "#### New\n"
            "## U\n"
            "## V\n"
        )
        base_hierarchy = build_hierarchy_dict(base_content)
        head_hierarchy = build_hierarchy_dict(head_content)

        result = build_source_diff_dict(
            modified_sections={},
            added_sections={7: head_hierarchy[7]},
            deleted_sections={},
            all_hierarchy_dict=head_hierarchy,
            base_hierarchy_dict=base_hierarchy,
            operations={
                "added_lines": [
                    {
                        "line_number": 7,
                        "is_header": True,
                        "content": "#### Added",
                    },
                ],
                "deleted_lines": [],
                "modified_lines": [
                    {
                        "line_number": 8,
                        "is_header": True,
                        "content": "#### New",
                        "original_content": "#### Old",
                    },
                ],
            },
            file_content=head_content,
            base_file_content=base_content,
        )

        self.assertEqual(
            "## Y > ### B > #### New",
            result["added_7"]["original_hierarchy"],
        )

    def test_duplicate_anchor_tie_uses_nearest_global_base_ordinal(self):
        base_content = (
            "# Guide\n"
            "## Alpha\n"
            "### Shared\n"
            "#### Step\n"
            "## Other\n"
            "### Different\n"
            "#### Step\n"
            "#### Step\n"
            "## Beta\n"
            "### Shared\n"
            "#### Step\n"
        )
        head_content = (
            "# Guide\n"
            "## Other\n"
            "### Different\n"
            "#### Step\n"
            "## Current\n"
            "### Shared\n"
            "#### Added\n"
            "#### Step\n"
        )
        base_hierarchy = build_hierarchy_dict(base_content)
        head_hierarchy = build_hierarchy_dict(head_content)

        with mock.patch("builtins.print") as print_mock:
            result = build_source_diff_dict(
                modified_sections={},
                added_sections={7: head_hierarchy[7]},
                deleted_sections={},
                all_hierarchy_dict=head_hierarchy,
                base_hierarchy_dict=base_hierarchy,
                operations={
                    "added_lines": [
                        {
                            "line_number": 7,
                            "is_header": True,
                            "content": "#### Added",
                        },
                    ],
                    "deleted_lines": [],
                    "modified_lines": [],
                },
                file_content=head_content,
                base_file_content=base_content,
            )

        self.assertEqual(
            "## Alpha > ### Shared > #### Step",
            result["added_7"]["original_hierarchy"],
        )
        log_output = "\n".join(
            " ".join(str(arg) for arg in call.args)
            for call in print_mock.call_args_list
        )
        self.assertIn(
            "candidates tied at parent-hierarchy score 1",
            log_output,
        )
        self.assertIn("HEAD ordinal 1, BASE ordinal 0", log_output)

    def test_build_source_diff_dict_collapses_add_delete_heading_level_pair(self):
        """Added+deleted heading-level pairs should become one modified entry."""
        base_content = "# Title\n\n## Parent\n\n## Child\n\nBody text.\n"
        head_content = "# Title\n\n## Parent\n\n### Child\n\nBody text.\n"

        base_hierarchy = build_hierarchy_dict(base_content)
        head_hierarchy = build_hierarchy_dict(head_content)

        result = build_source_diff_dict(
            modified_sections={},
            added_sections={5: head_hierarchy[5]},
            deleted_sections={5: base_hierarchy[5]},
            all_hierarchy_dict=head_hierarchy,
            base_hierarchy_dict=base_hierarchy,
            operations={
                'added_lines': [
                    {'line_number': 5, 'is_header': True, 'content': '### Child'},
                ],
                'deleted_lines': [
                    {'line_number': 5, 'is_header': True, 'content': '## Child'},
                ],
                'modified_lines': [],
            },
            file_content=head_content,
            base_file_content=base_content,
        )

        self.assertEqual(list(result.keys()), ["modified_5"])
        entry = result["modified_5"]
        self.assertEqual(entry["operation"], "modified")
        self.assertTrue(entry["heading_level_change_only"])
        self.assertEqual(entry["old_heading_level"], 2)
        self.assertEqual(entry["new_heading_level"], 3)
        self.assertEqual(entry["old_content"], "## Child\n\nBody text.\n")
        self.assertEqual(entry["new_content"], "### Child\n\nBody text.\n")


class ApplyHeadingLevelChangeTest(unittest.TestCase):
    """Tests for apply_heading_level_change_to_target()."""

    def setUp(self):
        sys.path.insert(0, str(SCRIPTS_DIR))
        from file_updater import apply_heading_level_change_to_target
        self.apply_fn = apply_heading_level_change_to_target

    def test_basic_level_increase(self):
        target = "## log.file\n\n配置日志文件。"
        result = self.apply_fn(target, old_level=2, new_level=3)
        self.assertEqual(result, "### log.file\n\n配置日志文件。")

    def test_level_decrease(self):
        target = "### log.file\n\nContent here."
        result = self.apply_fn(target, old_level=3, new_level=2)
        self.assertEqual(result, "## log.file\n\nContent here.")

    def test_multi_level_jump(self):
        target = "## section\n\nBody text"
        result = self.apply_fn(target, old_level=2, new_level=4)
        self.assertEqual(result, "#### section\n\nBody text")

    def test_preserves_body_content(self):
        target = "## tikv-client.copr-cache\n\nCoprocessor cache config.\n\n- `capacity`: 1GB"
        result = self.apply_fn(target, old_level=2, new_level=3)
        expected = "### tikv-client.copr-cache\n\nCoprocessor cache config.\n\n- `capacity`: 1GB"
        self.assertEqual(result, expected)

    def test_empty_content(self):
        self.assertEqual(self.apply_fn("", 2, 3), "")
        self.assertIsNone(self.apply_fn(None, 2, 3))

    def test_non_heading_content(self):
        target = "Not a heading\nSome content"
        result = self.apply_fn(target, old_level=2, new_level=3)
        self.assertEqual(result, target)

    def test_clamp_to_valid_levels(self):
        target = "# Title\nContent"
        result = self.apply_fn(target, old_level=2, new_level=1)
        self.assertEqual(result, "# Title\nContent")

    def test_heading_with_anchor_preserved(self):
        target = "## log.file {#log-file}\n\nContent"
        result = self.apply_fn(target, old_level=2, new_level=3)
        self.assertEqual(result, "### log.file {#log-file}\n\nContent")

    def test_direct_level_not_delta(self):
        """Target already at new_level should not be shifted further (Comment 2 fix)."""
        target = "### log.file\n\nContent"
        # Source changed from 2→3, target already at 3. Should stay at 3, not go to 4.
        result = self.apply_fn(target, old_level=2, new_level=3)
        self.assertEqual(result, "### log.file\n\nContent")

    def test_target_at_different_level(self):
        """Target at an unexpected level still gets set to new_level."""
        target = "#### section\n\nContent"
        result = self.apply_fn(target, old_level=2, new_level=3)
        self.assertEqual(result, "### section\n\nContent")


class CollapseHeadingLevelPairsTest(unittest.TestCase):
    """Tests for _collapse_heading_level_pairs() and the add+delete pairing path."""

    def test_collapse_simple_pair(self):
        """An added+deleted pair with same title/body becomes a modified entry."""
        source_diff_dict = {
            "deleted_7": {
                "new_line_number": 7,
                "original_hierarchy": "## log.file",
                "operation": "deleted",
                "new_content": None,
                "old_content": "## log.file\n\nContent here.",
            },
            "added_7": {
                "new_line_number": 7,
                "original_hierarchy": "## log > ### log.file",
                "operation": "added",
                "new_content": "### log.file\n\nContent here.",
                "old_content": None,
            },
        }
        from diff_analyzer import _collapse_heading_level_pairs
        _collapse_heading_level_pairs(source_diff_dict)

        self.assertNotIn("deleted_7", source_diff_dict)
        self.assertNotIn("added_7", source_diff_dict)
        self.assertIn("modified_7", source_diff_dict)
        entry = source_diff_dict["modified_7"]
        self.assertTrue(entry["heading_level_change_only"])
        self.assertEqual(entry["old_heading_level"], 2)
        self.assertEqual(entry["new_heading_level"], 3)
        self.assertEqual(entry["original_hierarchy"], "## log.file")

    def test_no_collapse_different_titles(self):
        """Pairs with different titles should not be collapsed."""
        source_diff_dict = {
            "deleted_7": {
                "new_line_number": 7,
                "original_hierarchy": "## old-section",
                "operation": "deleted",
                "new_content": None,
                "old_content": "## old-section\n\nContent.",
            },
            "added_7": {
                "new_line_number": 7,
                "original_hierarchy": "### new-section",
                "operation": "added",
                "new_content": "### new-section\n\nContent.",
                "old_content": None,
            },
        }
        from diff_analyzer import _collapse_heading_level_pairs
        _collapse_heading_level_pairs(source_diff_dict)

        self.assertIn("deleted_7", source_diff_dict)
        self.assertIn("added_7", source_diff_dict)
        self.assertNotIn("modified_7", source_diff_dict)

    def test_no_collapse_body_differs(self):
        """Pairs with same title but different body should not be collapsed."""
        source_diff_dict = {
            "deleted_7": {
                "new_line_number": 7,
                "original_hierarchy": "## section",
                "operation": "deleted",
                "new_content": None,
                "old_content": "## section\n\nOld content.",
            },
            "added_7": {
                "new_line_number": 7,
                "original_hierarchy": "### section",
                "operation": "added",
                "new_content": "### section\n\nNew content.",
                "old_content": None,
            },
        }
        from diff_analyzer import _collapse_heading_level_pairs
        _collapse_heading_level_pairs(source_diff_dict)

        self.assertIn("deleted_7", source_diff_dict)
        self.assertIn("added_7", source_diff_dict)

    def test_are_headers_similar_allows_level_change(self):
        """are_headers_similar should pair headings with same title but different level."""
        from diff_analyzer import analyze_diff_operations
        patch = (
            "@@ -5,3 +5,3 @@\n"
            " some context\n"
            "-## log.file\n"
            "+### log.file\n"
            " more context\n"
        )
        changed_file = SimpleNamespace(
            filename="config.md", status="modified", patch=patch, previous_filename=None
        )
        operations = analyze_diff_operations(changed_file)
        # The heading should be paired as a modified line
        modified_headers = [
            m for m in operations['modified_lines'] if m.get('is_header')
        ]
        self.assertTrue(
            len(modified_headers) >= 1,
            f"Expected heading-level change to be paired as modified; "
            f"got added={len(operations['added_lines'])}, "
            f"deleted={len(operations['deleted_lines'])}, "
            f"modified={len(operations['modified_lines'])}"
        )
        mod = modified_headers[0]
        self.assertIn("### log.file", mod['content'])
        self.assertIn("## log.file", mod.get('original_content', ''))


if __name__ == "__main__":
    unittest.main()

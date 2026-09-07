import subprocess
import sys
import tempfile
import unittest
import json
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import commit_sync_workflow as workflow
from formatting_sync import build_formatting_only_change
from translation_structure_validator import StructureValidationIssue


class CommitSyncWorkflowHelpersTest(unittest.TestCase):
    def test_full_translation_paths_are_excluded_from_marker_groups(self):
        marker_groups = {
            "base-a": {"bootstrap.md", "incremental-a.md"},
            "base-b": {"bootstrap.md"},
            "base-c": {"incremental-c.md"},
        }

        filtered = workflow.exclude_paths_from_marker_groups(
            marker_groups,
            {"bootstrap.md"},
        )

        self.assertEqual(
            filtered,
            {
                "base-a": {"incremental-a.md"},
                "base-c": {"incremental-c.md"},
            },
        )

    def test_restructured_reconciliation_failure_never_falls_back_to_overwrite(self):
        stats = workflow.TranslationStats()
        added_files = {"guide.md": "# New source\n"}
        full_translation_paths = {"guide.md"}

        with mock.patch.object(
            workflow, "TARGET_REPO_PATH", "/tmp/target"
        ), mock.patch.object(
            workflow, "read_target_file_content", return_value="# 现有译文\n"
        ), mock.patch.object(
            workflow, "get_source_ref_content", return_value="# Source\n"
        ), mock.patch.object(
            workflow, "reconcile_restructured_file", return_value=None
        ):
            reconciled = workflow.reconcile_restructured_files(
                {"guide.md"},
                added_files,
                full_translation_paths,
                {"mode": "commit"},
                object(),
                object(),
                {"ignore_resource_card_section": False},
                None,
                stats,
                set(),
            )

        self.assertEqual(reconciled, set())
        self.assertNotIn("guide.md", added_files)
        self.assertNotIn("guide.md", full_translation_paths)
        self.assertEqual(1, len(stats.failed))
        self.assertIn("declined", stats.failed[0][1])

    def test_restructured_reconciliation_preserves_ignored_target_related_resources(self):
        stats = workflow.TranslationStats()
        added_files = {"guide.md": "filtered source used only by full-translation fallback"}
        full_translation_paths = {"guide.md"}
        successful_paths = set()
        base_source = (
            "# Guide\n\n## Content\n\nOld body.\n\n"
            "## Related resources\n\n"
            "<RelatedResources>\n"
            '  <ResourceCard title="Old source card" />\n'
            "</RelatedResources>\n"
        )
        head_source = (
            "# Guide\n\n## Content\n\nNew body.\n\n"
            "## Related resources\n\n"
            "<RelatedResources>\n"
            '  <ResourceCard title="New source card that must be ignored" />\n'
            "</RelatedResources>\n"
        )
        existing_target = (
            "# ガイド\n\n## 内容\n\n古い本文。\n\n"
            "## 関連リソース\n\n"
            "<RelatedResources>\n"
            '  <ResourceCard title="既存の翻訳済みカード" />\n'
            "</RelatedResources>\n"
        )

        def source_at_ref(_file_path, _context, _github, ref_name):
            return head_source if ref_name == "head_ref" else base_source

        def assert_canonical_inputs(
            _file_path,
            effective_head,
            effective_base,
            target,
            *_args,
            **_kwargs,
        ):
            self.assertEqual(base_source, effective_base)
            self.assertIn('title="Old source card"', effective_head)
            self.assertNotIn("New source card that must be ignored", effective_head)
            self.assertEqual(existing_target, target)
            return existing_target

        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.object(
            workflow, "TARGET_REPO_PATH", tmpdir
        ), mock.patch.object(
            workflow, "read_target_file_content", return_value=existing_target
        ), mock.patch.object(
            workflow, "get_source_ref_content", side_effect=source_at_ref
        ), mock.patch.object(
            workflow, "reconcile_restructured_file", side_effect=assert_canonical_inputs
        ), mock.patch.object(
            workflow, "git_add_changes"
        ):
            reconciled = workflow.reconcile_restructured_files(
                {"guide.md"},
                added_files,
                full_translation_paths,
                {"mode": "commit"},
                object(),
                object(),
                {"ignore_resource_card_section": True},
                None,
                stats,
                successful_paths,
            )

            written = Path(tmpdir, "guide.md").read_text(encoding="utf-8")

        self.assertEqual({"guide.md"}, reconciled)
        self.assertEqual(existing_target, written)
        self.assertEqual({"guide.md"}, successful_paths)
        self.assertNotIn("guide.md", added_files)
        self.assertNotIn("guide.md", full_translation_paths)
        self.assertEqual([], stats.failed)

    def test_restructured_reconciliation_rejects_missing_source_snapshot(self):
        stats = workflow.TranslationStats()
        added_files = {"guide.md": "# Head\n"}
        full_translation_paths = {"guide.md"}

        def source_at_ref(_file_path, _context, _github, ref_name):
            return "# Head\n" if ref_name == "head_ref" else None

        with mock.patch.object(
            workflow, "TARGET_REPO_PATH", "/tmp/target"
        ), mock.patch.object(
            workflow, "read_target_file_content", return_value="# 既存\n"
        ), mock.patch.object(
            workflow, "get_source_ref_content", side_effect=source_at_ref
        ), mock.patch.object(
            workflow, "reconcile_restructured_file"
        ) as reconcile:
            reconciled = workflow.reconcile_restructured_files(
                {"guide.md"},
                added_files,
                full_translation_paths,
                {"mode": "commit"},
                object(),
                object(),
                {"ignore_resource_card_section": True},
                None,
                stats,
                set(),
            )

        self.assertEqual(set(), reconciled)
        reconcile.assert_not_called()
        self.assertNotIn("guide.md", added_files)
        self.assertNotIn("guide.md", full_translation_paths)
        self.assertEqual(1, len(stats.failed))
        self.assertIn("BASE source snapshot is unavailable", stats.failed[0][1])

    def test_restructured_reconciliation_rejects_related_resources_count_drift(self):
        stats = workflow.TranslationStats()
        added_files = {"guide.md": "# Head\n"}
        full_translation_paths = {"guide.md"}
        base_source = (
            "# Guide\n\n## Related resources\n\n"
            "<RelatedResources>\n"
            '  <ResourceCard title="Base card" />\n'
            "</RelatedResources>\n"
        )
        head_source = "# Guide\n\n## Content\n\nNew.\n"

        def source_at_ref(_file_path, _context, _github, ref_name):
            return head_source if ref_name == "head_ref" else base_source

        with mock.patch.object(
            workflow, "TARGET_REPO_PATH", "/tmp/target"
        ), mock.patch.object(
            workflow, "read_target_file_content", return_value="# 既存\n"
        ), mock.patch.object(
            workflow, "get_source_ref_content", side_effect=source_at_ref
        ), mock.patch.object(
            workflow, "reconcile_restructured_file"
        ) as reconcile:
            reconciled = workflow.reconcile_restructured_files(
                {"guide.md"},
                added_files,
                full_translation_paths,
                {"mode": "commit"},
                object(),
                object(),
                {"ignore_resource_card_section": True},
                None,
                stats,
                set(),
            )

        self.assertEqual(set(), reconciled)
        reconcile.assert_not_called()
        self.assertNotIn("guide.md", added_files)
        self.assertNotIn("guide.md", full_translation_paths)
        self.assertEqual(1, len(stats.failed))
        self.assertIn("section count changed", stats.failed[0][1])

    def test_normalize_source_files_prefixes_folder(self):
        normalized = workflow.normalize_source_files("foo.md, ai/bar.md ,baz/qux.md", "ai")
        self.assertEqual(normalized, {"ai/foo.md", "ai/bar.md", "ai/baz/qux.md"})

    def test_filter_changed_files_matches_folder_and_previous_filename(self):
        changed_files = [
            SimpleNamespace(filename="ai/new.md", previous_filename=None),
            SimpleNamespace(filename="other/skip.md", previous_filename=None),
            SimpleNamespace(filename="docs/renamed.md", previous_filename="ai/old.md"),
        ]

        filtered = workflow.filter_changed_files(changed_files, source_folder="ai")
        self.assertEqual([item.filename for item in filtered], ["ai/new.md", "docs/renamed.md"])

        explicitly_filtered = workflow.filter_changed_files(
            changed_files,
            source_folder="ai",
            source_files="old.md",
        )
        self.assertEqual([item.filename for item in explicitly_filtered], ["docs/renamed.md"])

    def test_source_files_translation_mode_defaults_to_incremental(self):
        self.assertEqual(
            workflow.normalize_source_files_translation_mode(""),
            "incremental",
        )
        self.assertEqual(
            workflow.normalize_source_files_translation_mode("diff"),
            "incremental",
        )

    def test_source_files_translation_mode_accepts_full_aliases(self):
        self.assertEqual(
            workflow.normalize_source_files_translation_mode("full"),
            "full",
        )
        self.assertEqual(
            workflow.normalize_source_files_translation_mode("full_translation"),
            "full",
        )

    def test_source_files_translation_mode_rejects_unknown_value(self):
        with self.assertRaises(ValueError):
            workflow.normalize_source_files_translation_mode("partial")

    def test_full_translation_mode_only_applies_with_source_files(self):
        self.assertEqual(
            workflow.get_effective_source_files_translation_mode("full", ""),
            "incremental",
        )
        self.assertEqual(
            workflow.get_effective_source_files_translation_mode("full", "guide.md"),
            "full",
        )

    def test_commits_match_accepts_seven_char_prefixes(self):
        self.assertTrue(
            workflow.commits_match(
                "abcdef1",
                "abcdef1234567890",
            )
        )
        self.assertFalse(
            workflow.commits_match(
                "abcdef",
                "abcdef1234567890",
            )
        )
        self.assertFalse(
            workflow.commits_match(
                "abcdef1",
                "abcdef2234567890",
            )
        )

    def test_get_safe_target_file_path_blocks_symlink_escape(self):
        with tempfile.TemporaryDirectory() as tmpdir, tempfile.TemporaryDirectory() as outside:
            symlink_path = Path(tmpdir, "outside-link")
            try:
                symlink_path.symlink_to(outside, target_is_directory=True)
            except OSError as e:
                self.skipTest(f"symlink creation is unavailable: {e}")

            self.assertIsNone(
                workflow.get_safe_target_file_path(
                    tmpdir,
                    "outside-link/escaped.md",
                )
            )

    def test_retry_cursor_ledger_preserves_incomplete_range_and_clears_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger = Path(tmpdir, workflow.RETRY_LEDGER_FILE)
            ledger.write_text(
                json.dumps(
                    {
                        "target": "main",
                        "sha": "abcdef1234567890",
                        "pending": {"done.md": "1111111", "old.md": "2222222"},
                    }
                ),
                encoding="utf-8",
            )
            stats = workflow.TranslationStats()
            stats.mark_partial("partial.md", "AI output incomplete")
            stats.set_retry_ref("partial.md", "3333333")

            self.assertTrue(
                workflow.update_retry_cursor_ledger(
                    tmpdir,
                    stats,
                    {"done.md"},
                    source_branch="release-8.5",
                    source_head_ref="4444444",
                )
            )
            updated = json.loads(ledger.read_text(encoding="utf-8"))
            self.assertEqual(updated["target"], "release-8.5")
            self.assertEqual(updated["sha"], "4444444")
            self.assertEqual(
                updated["pending"],
                {"old.md": "2222222", "partial.md": "3333333"},
            )
            groups, paths = workflow.load_retry_cursor_groups(tmpdir)
            self.assertEqual(groups["3333333"], {"partial.md"})
            self.assertEqual(paths, {"old.md", "partial.md"})

    def test_retry_cursor_ledger_is_initialized_when_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stats = workflow.TranslationStats()
            stats.mark_failure("failed.md", "AI unavailable")
            stats.set_retry_ref("failed.md", "1111111")

            self.assertTrue(
                workflow.update_retry_cursor_ledger(
                    tmpdir,
                    stats,
                    set(),
                    source_branch="main",
                    source_head_ref="2222222",
                )
            )
            ledger = json.loads(
                Path(tmpdir, workflow.RETRY_LEDGER_FILE).read_text(encoding="utf-8")
            )

        self.assertEqual(ledger["target"], "main")
        self.assertEqual(ledger["sha"], "2222222")
        self.assertEqual(ledger["pending"], {"failed.md": "1111111"})

    def test_unsafe_modified_target_path_is_explicitly_rejected(self):
        with mock.patch.object(workflow, "get_safe_target_file_path", return_value=None):
            with self.assertRaisesRegex(ValueError, "Unsafe target file path"):
                workflow.should_process_modified_file_as_added(
                    "../escaped.md",
                    {"target_local_path": "/tmp/target"},
                    object(),
                )

    def test_partial_task_is_staged_but_not_counted_as_complete(self):
        stats = workflow.TranslationStats()
        result = {
            "file_path": "guide.md",
            "ok": True,
            "result": {"status": "partial", "reason": "AI response incomplete"},
            "error": None,
        }

        self.assertTrue(workflow._record_translation_task_result(result, stats))
        self.assertEqual(stats.succeeded, [])
        self.assertEqual(stats.partial, [("guide.md", "AI response incomplete")])

    def test_upsert_corresponding_en_commit_adds_marker_after_h1(self):
        content = "---\ntitle: 系统变量\n---\n\n# 系统变量\n\nBody\n"

        updated, changed = workflow.upsert_corresponding_en_commit(
            content,
            "abcdef1234567890",
            add_if_missing=True,
        )

        self.assertTrue(changed)
        self.assertIn(
            "# 系统变量 <!--Corresponding EN commit: abcdef1234567890-->",
            updated,
        )

    def test_upsert_corresponding_en_commit_inserts_first_line_without_h1(self):
        updated, changed = workflow.upsert_corresponding_en_commit(
            "Body\n",
            "abcdef1234567890",
            add_if_missing=True,
        )

        self.assertTrue(changed)
        self.assertTrue(
            updated.startswith("<!--Corresponding EN commit: abcdef1234567890-->\n")
        )

    def test_remove_corresponding_en_commit_removes_inline_marker(self):
        content = "# 系统变量 <!--Corresponding EN commit: abcdef1234567890-->\n\nBody\n"

        updated, changed = workflow.remove_corresponding_en_commit(content)

        self.assertTrue(changed)
        self.assertEqual(updated, "# 系统变量\n\nBody\n")

    def test_split_changed_files_uses_per_file_marker_only_when_mismatched(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "guide.md").write_text(
                "# Guide <!--Corresponding EN commit: bbbbbbb222222222-->\n",
                encoding="utf-8",
            )
            Path(tmpdir, "global.md").write_text(
                "# Global <!--Corresponding EN commit: aaaaaaa111111111-->\n",
                encoding="utf-8",
            )
            Path(tmpdir, "plain.md").write_text("# Plain\n", encoding="utf-8")
            changed_files = [
                SimpleNamespace(filename="guide.md"),
                SimpleNamespace(filename="global.md"),
                SimpleNamespace(filename="plain.md"),
            ]

            global_files, marker_groups, marker_file_paths = workflow.split_changed_files_by_corresponding_en_commit(
                changed_files,
                tmpdir,
                "aaaaaaa111111111",
            )

        self.assertEqual([file.filename for file in global_files], ["global.md", "plain.md"])
        self.assertEqual(marker_groups, {"bbbbbbb222222222": {"guide.md"}})
        self.assertEqual(marker_file_paths, {"guide.md", "global.md"})

    def test_update_corresponding_en_commit_rejects_conflicting_modes(self):
        with self.assertRaises(ValueError):
            workflow.update_corresponding_en_commit_for_files(
                {"guide.md"},
                "/tmp/target",
                "ccccccc333333333",
                add_if_missing=True,
                remove_if_present=True,
            )

    def test_update_corresponding_en_commit_removes_marker_for_scheduled_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "guide.md").write_text(
                "# Guide <!--Corresponding EN commit: bbbbbbb222222222-->\n\nBody\n",
                encoding="utf-8",
            )
            Path(tmpdir, "plain.md").write_text("# Plain\n", encoding="utf-8")

            updated_files, failures = workflow.update_corresponding_en_commit_for_files(
                {"guide.md", "plain.md"},
                tmpdir,
                "ccccccc333333333",
                add_if_missing=False,
                remove_if_present=True,
            )

            self.assertEqual(updated_files, ["guide.md"])
            self.assertEqual(failures, {})
            self.assertNotIn(
                "Corresponding EN commit",
                Path(tmpdir, "guide.md").read_text(encoding="utf-8"),
            )
            self.assertNotIn(
                "Corresponding EN commit",
                Path(tmpdir, "plain.md").read_text(encoding="utf-8"),
            )

    def test_full_translation_source_content_prefers_local_source_repo_path(self):
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="# Local head content\n",
            stderr="warning: ignored stderr\n",
        )

        with mock.patch(
            "subprocess.run",
            return_value=completed,
        ) as run, mock.patch.object(
            workflow, "get_source_file_content"
        ) as get_source_file_content:
            content = workflow.get_full_translation_source_content(
                "guide.md",
                {
                    "head_ref": "head-sha",
                    "source_repo_path": "/tmp/source",
                },
                object(),
            )

        self.assertEqual(content, "# Local head content\n")
        run.assert_called_once_with(
            ["git", "-C", "/tmp/source", "show", "head-sha:guide.md"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        get_source_file_content.assert_not_called()

    def test_full_translation_source_content_redacts_local_source_path_on_git_failure(self):
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=128,
            stdout="",
            stderr="fatal: cannot change to '/tmp/private-source': No such file or directory",
        )

        with mock.patch("subprocess.run", return_value=completed):
            with self.assertRaises(RuntimeError) as context:
                workflow.get_full_translation_source_content(
                    "missing.md",
                    {
                        "head_ref": "head-sha",
                        "source_repo_path": "/tmp/private-source",
                    },
                    object(),
                )

        self.assertIn("[SOURCE_REPO_PATH]", str(context.exception))
        self.assertNotIn("/tmp/private-source", str(context.exception))
        self.assertNotIn("git -C", str(context.exception))

    def test_missing_env_message_omits_source_base_ref_in_full_mode(self):
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(workflow, "SOURCE_REPO", ""))
            stack.enter_context(mock.patch.object(workflow, "TARGET_REPO", "acme/docs-cn"))
            stack.enter_context(mock.patch.object(workflow, "GITHUB_TOKEN", "token"))
            stack.enter_context(mock.patch.object(workflow, "TARGET_REPO_PATH", "/tmp/target"))
            stack.enter_context(mock.patch.object(workflow, "SOURCE_BASE_REF", ""))
            stack.enter_context(mock.patch.object(workflow, "SOURCE_HEAD_REF", "head"))
            stack.enter_context(mock.patch.object(workflow, "SOURCE_FILES", "guide.md"))
            stack.enter_context(mock.patch.object(workflow, "SOURCE_FILES_TRANSLATION_MODE", "full"))
            thread_safe_print = stack.enter_context(mock.patch.object(workflow, "thread_safe_print"))

            result = workflow.main()

        printed = "\n".join(str(call.args[0]) for call in thread_safe_print.call_args_list)
        self.assertEqual(result, 1)
        self.assertNotIn("SOURCE_BASE_REF:", printed)
        self.assertIn("SOURCE_HEAD_REF:", printed)

    def test_resolve_compare_range_requires_explicit_refs(self):
        with mock.patch.object(workflow, "SOURCE_BASE_REF", ""), mock.patch.object(
            workflow, "SOURCE_HEAD_REF", ""
        ):
            with self.assertRaises(ValueError):
                workflow.resolve_compare_range()

    def test_resolve_compare_range_returns_explicit_refs(self):
        with mock.patch.object(workflow, "SOURCE_BASE_REF", "abc123"), mock.patch.object(
            workflow, "SOURCE_HEAD_REF", "def456"
        ):
            self.assertEqual(workflow.resolve_compare_range(), ("abc123", "def456"))

    def test_build_exclude_folders_keeps_explicit_commit_target_folder(self):
        repo_config = {"target_language": "Chinese"}

        with mock.patch.object(
            workflow,
            "COMMIT_BASED_MODE_IGNORE_FOLDERS",
            ("ai", "tidb-cloud"),
        ), mock.patch.object(workflow, "SOURCE_FOLDER", "ai"), mock.patch.object(
            workflow, "SOURCE_FILES", ""
        ):
            exclude_folders = workflow.build_exclude_folders(repo_config)

        self.assertNotIn("ai", exclude_folders)
        self.assertIn("tidb-cloud", exclude_folders)

    def test_commit_ignore_folders_do_not_inherit_pr_mode_defaults(self):
        with mock.patch.object(
            workflow,
            "COMMIT_BASED_MODE_IGNORE_FOLDERS",
            ("custom-folder",),
        ):
            self.assertEqual(workflow.get_commit_ignore_folders(), ["custom-folder"])

    def test_commit_repo_config_includes_target_ref_and_local_read_preference(self):
        with mock.patch.object(workflow, "SOURCE_REPO", "pingcap/docs"), mock.patch.object(
            workflow, "TARGET_REPO", "pingcap/docs"
        ), mock.patch.object(workflow, "TARGET_REPO_PATH", "/tmp/docs"), mock.patch.object(
            workflow, "TARGET_REF", "i18n-zh-release-8.5"
        ), mock.patch.object(
            workflow, "PREFER_LOCAL_TARGET_FOR_READ", True
        ), mock.patch.object(
            workflow, "IGNORE_RESOURCE_CARD_SECTION", "No"
        ):
            repo_config = workflow.get_commit_repo_config()

        self.assertEqual(repo_config["target_ref"], "i18n-zh-release-8.5")
        self.assertTrue(repo_config["prefer_local_target_for_read"])
        self.assertEqual(repo_config["target_local_path"], "/tmp/docs")
        self.assertFalse(repo_config["ignore_resource_card_section"])

    def test_commit_repo_config_ignores_resource_card_sections_by_default(self):
        with mock.patch.object(workflow, "SOURCE_REPO", "pingcap/docs"), mock.patch.object(
            workflow, "TARGET_REPO", "pingcap/docs-cn"
        ), mock.patch.object(workflow, "IGNORE_RESOURCE_CARD_SECTION", "Yes"):
            repo_config = workflow.get_commit_repo_config()

        self.assertTrue(repo_config["ignore_resource_card_section"])

    def test_commit_ignore_files_do_not_inherit_pr_mode_defaults(self):
        with mock.patch.object(
            workflow,
            "COMMIT_BASED_MODE_IGNORE_FILES",
            ("custom.md",),
        ):
            ignore_files = workflow.get_commit_ignore_files()

        self.assertEqual(ignore_files, ["custom.md"])

    def test_commit_mode_treats_pr_ignored_toc_files_as_special_toc_files(self):
        ignore_files = workflow.get_commit_ignore_files()

        self.assertEqual(
            workflow.determine_file_processing_type(
                "docs/TOC-tidb-cloud.md",
                {},
                workflow.SPECIAL_FILES,
                ignore_files,
            ),
            "special_file_toc",
        )

    def test_process_modified_file_as_added_when_target_file_is_missing(self):
        pr_diff = "\n".join(
            [
                "File: guide.md",
                "@@ -1,1 +1,1 @@",
                "-Old",
                "+New",
                "-" * 80,
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.object(
            workflow,
            "get_source_file_content",
            return_value="# Guide\n\nNew content\n",
        ) as get_source_file_content, mock.patch.object(
            workflow,
            "process_added_files",
            return_value=(True, {}),
        ) as process_added_files:
            result = workflow._process_commit_modified_file(
                "guide.md",
                {},
                pr_diff,
                {"mode": "commit", "source_repo": "acme/docs", "base_ref": "base", "head_ref": "head", "changed_files": []},
                object(),
                object(),
                {
                    "target_repo": "acme/docs",
                    "target_local_path": tmpdir,
                    "prefer_local_target_for_read": True,
                    "source_language": "English",
                    "target_language": "Chinese",
                    "ignore_files": [],
                },
                None,
            )

        self.assertEqual(result["status"], "success")
        get_source_file_content.assert_called_once()
        process_added_files.assert_called_once()
        self.assertEqual(
            process_added_files.call_args.args[0],
            {"guide.md": "# Guide\n\nNew content\n"},
        )

    def test_process_modified_file_reports_regular_file_failure_reason(self):
        pr_diff = "\n".join(
            [
                "File: guide.md",
                "@@ -1,1 +1,1 @@",
                "-Old",
                "+New",
                "-" * 80,
            ]
        )
        failure_reason = "Target file guide.md is missing or could not map 1 modified source section(s)"

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "guide.md").write_text("# Guide\n", encoding="utf-8")

            with mock.patch.object(
                workflow,
                "process_regular_modified_file",
                return_value=(False, failure_reason),
            ) as process_regular_modified_file:
                result = workflow._process_commit_modified_file(
                    "guide.md",
                    {},
                    pr_diff,
                    {"mode": "commit", "source_repo": "acme/docs", "base_ref": "base", "head_ref": "head", "changed_files": []},
                    object(),
                    object(),
                    {
                        "target_repo": "acme/docs",
                        "target_local_path": tmpdir,
                        "prefer_local_target_for_read": True,
                        "source_language": "English",
                        "target_language": "Chinese",
                        "ignore_files": [],
                    },
                    None,
                )

        self.assertEqual(result["status"], "failure")
        self.assertEqual(result["reason"], failure_reason)
        process_regular_modified_file.assert_called_once()

    def test_full_mode_fetches_source_files_even_when_not_in_compare_diff(self):
        fake_github_client = object()
        fake_ai_client = SimpleNamespace(model="fake-model")

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(workflow, "SOURCE_REPO", "acme/docs"))
            stack.enter_context(mock.patch.object(workflow, "TARGET_REPO", "acme/docs-cn"))
            stack.enter_context(mock.patch.object(workflow, "GITHUB_TOKEN", "token"))
            stack.enter_context(mock.patch.object(workflow, "TARGET_REPO_PATH", "/tmp/target"))
            stack.enter_context(mock.patch.object(workflow, "SOURCE_BASE_REF", ""))
            stack.enter_context(mock.patch.object(workflow, "SOURCE_HEAD_REF", "head"))
            stack.enter_context(mock.patch.object(workflow, "SOURCE_FILES", "guide.md"))
            stack.enter_context(mock.patch.object(workflow, "SOURCE_FOLDER", ""))
            stack.enter_context(mock.patch.object(workflow, "SOURCE_FILES_TRANSLATION_MODE", "full"))
            stack.enter_context(mock.patch.object(workflow, "SOURCE_REPO_PATH", ""))
            stack.enter_context(mock.patch.object(workflow, "TARGET_REF", ""))
            stack.enter_context(mock.patch.object(workflow, "clean_temp_output_dir"))
            stack.enter_context(mock.patch.object(workflow.Auth, "Token", return_value="token-auth"))
            stack.enter_context(mock.patch.object(workflow, "Github", return_value=fake_github_client))
            stack.enter_context(mock.patch.object(workflow, "UnifiedAIClient", return_value=fake_ai_client))
            stack.enter_context(mock.patch.object(workflow, "load_glossary", return_value=[]))
            stack.enter_context(mock.patch.object(workflow, "create_glossary_matcher", return_value=None))
            build_commit_diff_context = stack.enter_context(
                mock.patch.object(
                    workflow,
                    "build_commit_diff_context",
                )
            )
            build_local_commit_diff_context = stack.enter_context(
                mock.patch.object(workflow, "build_local_commit_diff_context")
            )
            analyze_source_changes = stack.enter_context(
                mock.patch.object(workflow, "analyze_source_changes")
            )
            stack.enter_context(mock.patch.object(workflow, "get_source_file_content", return_value="# Guide\n"))
            process_added_files = stack.enter_context(
                mock.patch.object(workflow, "process_added_files", return_value=(True, {}))
            )
            stack.enter_context(mock.patch.object(workflow, "git_add_successful_task_changes"))
            stack.enter_context(mock.patch.object(workflow.TranslationStats, "write_failure_report"))
            result = workflow.main()

        self.assertEqual(result, 0)
        build_commit_diff_context.assert_not_called()
        build_local_commit_diff_context.assert_not_called()
        analyze_source_changes.assert_not_called()
        process_added_files.assert_called_once()
        self.assertEqual(process_added_files.call_args.args[0], {"guide.md": "# Guide\n"})
        self.assertTrue(process_added_files.call_args.kwargs["overwrite_existing"])

    def test_manual_success_adds_corresponding_en_commit_marker(self):
        fake_github_client = object()
        fake_ai_client = SimpleNamespace(model="fake-model")
        zero_counts = {
            "added_files": 0,
            "deleted_files": 0,
            "toc_files": 0,
            "keyword_files": 0,
            "modified_sections": 0,
            "added_images": 0,
            "modified_images": 0,
            "deleted_images": 0,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "guide.md").write_text("# Guide\n\nBody\n", encoding="utf-8")
            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(workflow, "SOURCE_REPO", "acme/docs"))
                stack.enter_context(mock.patch.object(workflow, "TARGET_REPO", "acme/docs-cn"))
                stack.enter_context(mock.patch.object(workflow, "GITHUB_TOKEN", "token"))
                stack.enter_context(mock.patch.object(workflow, "TARGET_REPO_PATH", tmpdir))
                stack.enter_context(mock.patch.object(workflow, "SOURCE_BASE_REF", ""))
                stack.enter_context(mock.patch.object(workflow, "SOURCE_HEAD_REF", "abcdef1234567890"))
                stack.enter_context(mock.patch.object(workflow, "SOURCE_FILES", "guide.md"))
                stack.enter_context(mock.patch.object(workflow, "SOURCE_FOLDER", ""))
                stack.enter_context(mock.patch.object(workflow, "SOURCE_FILES_TRANSLATION_MODE", "full"))
                stack.enter_context(mock.patch.object(workflow, "COMMIT_SYNC_RUN_TYPE", "workflow_dispatch"))
                stack.enter_context(mock.patch.object(workflow, "SOURCE_REPO_PATH", ""))
                stack.enter_context(mock.patch.object(workflow, "TARGET_REF", ""))
                stack.enter_context(mock.patch.object(workflow, "clean_temp_output_dir"))
                stack.enter_context(mock.patch.object(workflow.Auth, "Token", return_value="token-auth"))
                stack.enter_context(mock.patch.object(workflow, "Github", return_value=fake_github_client))
                stack.enter_context(mock.patch.object(workflow, "UnifiedAIClient", return_value=fake_ai_client))
                stack.enter_context(mock.patch.object(workflow, "load_glossary", return_value=[]))
                stack.enter_context(mock.patch.object(workflow, "create_glossary_matcher", return_value=None))
                stack.enter_context(
                    mock.patch.object(
                        workflow,
                        "process_translation_group",
                        return_value={
                            "attempted": True,
                            "successful_file_paths": {"guide.md"},
                            "counts": zero_counts,
                        },
                    )
                )
                stack.enter_context(mock.patch.object(workflow, "git_add_changes"))
                stack.enter_context(mock.patch.object(workflow.TranslationStats, "write_failure_report"))

                result = workflow.main()

            self.assertEqual(result, 0)
            self.assertIn(
                "# Guide <!--Corresponding EN commit: abcdef1234567890-->",
                Path(tmpdir, "guide.md").read_text(encoding="utf-8"),
            )

    def test_manual_incremental_uses_existing_marker_cursor_and_updates_marker(self):
        fake_github_client = object()
        fake_ai_client = SimpleNamespace(model="fake-model")
        zero_counts = {
            "added_files": 0,
            "deleted_files": 0,
            "toc_files": 0,
            "keyword_files": 0,
            "modified_sections": 0,
            "added_images": 0,
            "modified_images": 0,
            "deleted_images": 0,
        }
        global_context = {
            "mode": "commit",
            "source_repo": "acme/docs",
            "target_repo": "acme/docs-cn",
            "base_ref": "aaaaaaa111111111",
            "head_ref": "ccccccc333333333",
            "changed_files": [
                SimpleNamespace(filename="guide.md", status="modified", patch="@@ -1 +1 @@\n-a\n+b\n"),
            ],
        }
        marker_context = {
            "mode": "commit",
            "source_repo": "acme/docs",
            "target_repo": "acme/docs-cn",
            "base_ref": "bbbbbbb222222222",
            "head_ref": "ccccccc333333333",
            "changed_files": [
                SimpleNamespace(filename="guide.md", status="modified", patch="@@ -1 +1 @@\n-b\n+c\n"),
            ],
        }
        group_paths = []

        def fake_process_translation_group(*args, **kwargs):
            paths = {file.filename for file in kwargs["filtered_changed_files"]}
            group_paths.append(paths)
            return {
                "attempted": True,
                "successful_file_paths": paths,
                "counts": zero_counts,
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "guide.md").write_text(
                "# Guide <!--Corresponding EN commit: bbbbbbb222222222-->\n\nBody\n",
                encoding="utf-8",
            )
            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(workflow, "SOURCE_REPO", "acme/docs"))
                stack.enter_context(mock.patch.object(workflow, "TARGET_REPO", "acme/docs-cn"))
                stack.enter_context(mock.patch.object(workflow, "GITHUB_TOKEN", "token"))
                stack.enter_context(mock.patch.object(workflow, "TARGET_REPO_PATH", tmpdir))
                stack.enter_context(mock.patch.object(workflow, "SOURCE_BASE_REF", "aaaaaaa111111111"))
                stack.enter_context(mock.patch.object(workflow, "SOURCE_HEAD_REF", "ccccccc333333333"))
                stack.enter_context(mock.patch.object(workflow, "SOURCE_FILES", "guide.md"))
                stack.enter_context(mock.patch.object(workflow, "SOURCE_FOLDER", ""))
                stack.enter_context(mock.patch.object(workflow, "SOURCE_FILES_TRANSLATION_MODE", "incremental"))
                stack.enter_context(mock.patch.object(workflow, "COMMIT_SYNC_RUN_TYPE", "workflow_dispatch"))
                stack.enter_context(mock.patch.object(workflow, "SOURCE_REPO_PATH", ""))
                stack.enter_context(mock.patch.object(workflow, "TARGET_REF", ""))
                stack.enter_context(mock.patch.object(workflow, "clean_temp_output_dir"))
                stack.enter_context(mock.patch.object(workflow.Auth, "Token", return_value="token-auth"))
                stack.enter_context(mock.patch.object(workflow, "Github", return_value=fake_github_client))
                stack.enter_context(mock.patch.object(workflow, "UnifiedAIClient", return_value=fake_ai_client))
                stack.enter_context(mock.patch.object(workflow, "load_glossary", return_value=[]))
                stack.enter_context(mock.patch.object(workflow, "create_glossary_matcher", return_value=None))
                build_context = stack.enter_context(
                    mock.patch.object(
                        workflow,
                        "build_incremental_diff_context",
                        side_effect=[global_context, marker_context],
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        workflow,
                        "process_translation_group",
                        side_effect=fake_process_translation_group,
                    )
                )
                update_markers = stack.enter_context(
                    mock.patch.object(
                        workflow,
                        "update_corresponding_en_commit_for_files",
                        wraps=workflow.update_corresponding_en_commit_for_files,
                    )
                )
                stack.enter_context(mock.patch.object(workflow, "git_add_changes"))
                stack.enter_context(mock.patch.object(workflow.TranslationStats, "write_failure_report"))

                result = workflow.main()

            self.assertEqual(result, 0)
            self.assertEqual(
                [call.args[:2] for call in build_context.call_args_list],
                [
                    ("aaaaaaa111111111", "ccccccc333333333"),
                    ("bbbbbbb222222222", "ccccccc333333333"),
                ],
            )
            self.assertEqual(group_paths, [{"guide.md"}])
            self.assertEqual(update_markers.call_args.args[0], {"guide.md"})
            content = Path(tmpdir, "guide.md").read_text(encoding="utf-8")
            self.assertIn("Corresponding EN commit: ccccccc333333333", content)
            self.assertNotIn("bbbbbbb222222222", content)

    def test_manual_incremental_updates_marker_when_marker_cursor_has_no_diff(self):
        fake_github_client = object()
        fake_ai_client = SimpleNamespace(model="fake-model")
        global_context = {
            "mode": "commit",
            "source_repo": "acme/docs",
            "target_repo": "acme/docs-cn",
            "base_ref": "aaaaaaa111111111",
            "head_ref": "ccccccc333333333",
            "changed_files": [
                SimpleNamespace(filename="guide.md", status="modified", patch="@@ -1 +1 @@\n-a\n+b\n"),
            ],
        }
        marker_context = {
            "mode": "commit",
            "source_repo": "acme/docs",
            "target_repo": "acme/docs-cn",
            "base_ref": "bbbbbbb222222222",
            "head_ref": "ccccccc333333333",
            "changed_files": [],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "guide.md").write_text(
                "# Guide <!--Corresponding EN commit: bbbbbbb222222222-->\n\nBody\n",
                encoding="utf-8",
            )
            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(workflow, "SOURCE_REPO", "acme/docs"))
                stack.enter_context(mock.patch.object(workflow, "TARGET_REPO", "acme/docs-cn"))
                stack.enter_context(mock.patch.object(workflow, "GITHUB_TOKEN", "token"))
                stack.enter_context(mock.patch.object(workflow, "TARGET_REPO_PATH", tmpdir))
                stack.enter_context(mock.patch.object(workflow, "SOURCE_BASE_REF", "aaaaaaa111111111"))
                stack.enter_context(mock.patch.object(workflow, "SOURCE_HEAD_REF", "ccccccc333333333"))
                stack.enter_context(mock.patch.object(workflow, "SOURCE_FILES", "guide.md"))
                stack.enter_context(mock.patch.object(workflow, "SOURCE_FOLDER", ""))
                stack.enter_context(mock.patch.object(workflow, "SOURCE_FILES_TRANSLATION_MODE", "incremental"))
                stack.enter_context(mock.patch.object(workflow, "COMMIT_SYNC_RUN_TYPE", "workflow_dispatch"))
                stack.enter_context(mock.patch.object(workflow, "SOURCE_REPO_PATH", ""))
                stack.enter_context(mock.patch.object(workflow, "TARGET_REF", ""))
                stack.enter_context(mock.patch.object(workflow, "clean_temp_output_dir"))
                stack.enter_context(mock.patch.object(workflow.Auth, "Token", return_value="token-auth"))
                stack.enter_context(mock.patch.object(workflow, "Github", return_value=fake_github_client))
                stack.enter_context(mock.patch.object(workflow, "UnifiedAIClient", return_value=fake_ai_client))
                stack.enter_context(mock.patch.object(workflow, "load_glossary", return_value=[]))
                stack.enter_context(mock.patch.object(workflow, "create_glossary_matcher", return_value=None))
                build_context = stack.enter_context(
                    mock.patch.object(
                        workflow,
                        "build_incremental_diff_context",
                        side_effect=[global_context, marker_context],
                    )
                )
                process_group = stack.enter_context(
                    mock.patch.object(workflow, "process_translation_group")
                )
                update_markers = stack.enter_context(
                    mock.patch.object(
                        workflow,
                        "update_corresponding_en_commit_for_files",
                        wraps=workflow.update_corresponding_en_commit_for_files,
                    )
                )
                stack.enter_context(mock.patch.object(workflow, "git_add_changes"))
                stack.enter_context(mock.patch.object(workflow.TranslationStats, "write_failure_report"))

                result = workflow.main()

            self.assertEqual(result, 0)
            self.assertEqual(build_context.call_count, 2)
            process_group.assert_not_called()
            self.assertEqual(update_markers.call_args.args[0], {"guide.md"})
            content = Path(tmpdir, "guide.md").read_text(encoding="utf-8")
            self.assertIn("Corresponding EN commit: ccccccc333333333", content)
            self.assertNotIn("bbbbbbb222222222", content)

    def test_scheduled_run_splits_marker_mismatches_and_removes_successful_marker(self):
        fake_github_client = object()
        fake_ai_client = SimpleNamespace(model="fake-model")
        zero_counts = {
            "added_files": 0,
            "deleted_files": 0,
            "toc_files": 0,
            "keyword_files": 0,
            "modified_sections": 0,
            "added_images": 0,
            "modified_images": 0,
            "deleted_images": 0,
        }
        global_context = {
            "mode": "commit",
            "source_repo": "acme/docs",
            "target_repo": "acme/docs-cn",
            "base_ref": "aaaaaaa111111111",
            "head_ref": "ccccccc333333333",
            "changed_files": [
                SimpleNamespace(filename="guide.md", status="modified", patch="@@ -1 +1 @@\n-a\n+b\n"),
                SimpleNamespace(filename="global.md", status="modified", patch="@@ -1 +1 @@\n-a\n+b\n"),
                SimpleNamespace(filename="other.md", status="modified", patch="@@ -1 +1 @@\n-a\n+b\n"),
            ],
        }
        marker_context = {
            "mode": "commit",
            "source_repo": "acme/docs",
            "target_repo": "acme/docs-cn",
            "base_ref": "bbbbbbb222222222",
            "head_ref": "ccccccc333333333",
            "changed_files": [
                SimpleNamespace(filename="guide.md", status="modified", patch="@@ -1 +1 @@\n-b\n+c\n"),
            ],
        }
        group_paths = []

        def fake_process_translation_group(*args, **kwargs):
            filtered_changed_files = kwargs["filtered_changed_files"]
            paths = {file.filename for file in filtered_changed_files}
            group_paths.append(paths)
            return {
                "attempted": True,
                "successful_file_paths": paths,
                "counts": zero_counts,
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "guide.md").write_text(
                "# Guide <!--Corresponding EN commit: bbbbbbb222222222-->\n\nBody\n",
                encoding="utf-8",
            )
            Path(tmpdir, "global.md").write_text(
                "# Global <!--Corresponding EN commit: aaaaaaa111111111-->\n\nBody\n",
                encoding="utf-8",
            )
            Path(tmpdir, "other.md").write_text("# Other\n\nBody\n", encoding="utf-8")
            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(workflow, "SOURCE_REPO", "acme/docs"))
                stack.enter_context(mock.patch.object(workflow, "TARGET_REPO", "acme/docs-cn"))
                stack.enter_context(mock.patch.object(workflow, "GITHUB_TOKEN", "token"))
                stack.enter_context(mock.patch.object(workflow, "TARGET_REPO_PATH", tmpdir))
                stack.enter_context(mock.patch.object(workflow, "SOURCE_BASE_REF", "aaaaaaa111111111"))
                stack.enter_context(mock.patch.object(workflow, "SOURCE_HEAD_REF", "ccccccc333333333"))
                stack.enter_context(mock.patch.object(workflow, "SOURCE_FILES", ""))
                stack.enter_context(mock.patch.object(workflow, "SOURCE_FOLDER", ""))
                stack.enter_context(mock.patch.object(workflow, "SOURCE_FILES_TRANSLATION_MODE", "incremental"))
                stack.enter_context(mock.patch.object(workflow, "COMMIT_SYNC_RUN_TYPE", "schedule"))
                stack.enter_context(mock.patch.object(workflow, "SOURCE_REPO_PATH", ""))
                stack.enter_context(mock.patch.object(workflow, "TARGET_REF", ""))
                stack.enter_context(mock.patch.object(workflow, "clean_temp_output_dir"))
                stack.enter_context(mock.patch.object(workflow.Auth, "Token", return_value="token-auth"))
                stack.enter_context(mock.patch.object(workflow, "Github", return_value=fake_github_client))
                stack.enter_context(mock.patch.object(workflow, "UnifiedAIClient", return_value=fake_ai_client))
                stack.enter_context(mock.patch.object(workflow, "load_glossary", return_value=[]))
                stack.enter_context(mock.patch.object(workflow, "create_glossary_matcher", return_value=None))
                stack.enter_context(
                    mock.patch.object(
                        workflow,
                        "build_incremental_diff_context",
                        side_effect=[global_context, marker_context],
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        workflow,
                        "process_translation_group",
                        side_effect=fake_process_translation_group,
                    )
                )
                update_markers = stack.enter_context(
                    mock.patch.object(
                        workflow,
                        "update_corresponding_en_commit_for_files",
                        wraps=workflow.update_corresponding_en_commit_for_files,
                    )
                )
                stack.enter_context(mock.patch.object(workflow, "git_add_changes"))
                stack.enter_context(mock.patch.object(workflow.TranslationStats, "write_failure_report"))

                result = workflow.main()

            self.assertEqual(result, 0)
            self.assertEqual(group_paths, [{"global.md", "other.md"}, {"guide.md"}])
            self.assertEqual(
                update_markers.call_args.args[0],
                {"global.md", "guide.md"},
            )
            self.assertNotIn(
                "Corresponding EN commit",
                Path(tmpdir, "guide.md").read_text(encoding="utf-8"),
            )
            self.assertNotIn(
                "Corresponding EN commit",
                Path(tmpdir, "global.md").read_text(encoding="utf-8"),
            )
            self.assertNotIn(
                "Corresponding EN commit",
                Path(tmpdir, "other.md").read_text(encoding="utf-8"),
            )

    def test_scheduled_run_removes_marker_when_per_file_cursor_matches_head(self):
        fake_github_client = object()
        fake_ai_client = SimpleNamespace(model="fake-model")
        zero_counts = {
            "added_files": 0,
            "deleted_files": 0,
            "toc_files": 0,
            "keyword_files": 0,
            "modified_sections": 0,
            "added_images": 0,
            "modified_images": 0,
            "deleted_images": 0,
        }
        global_context = {
            "mode": "commit",
            "source_repo": "acme/docs",
            "target_repo": "acme/docs-cn",
            "base_ref": "aaaaaaa111111111",
            "head_ref": "ccccccc333333333",
            "changed_files": [
                SimpleNamespace(filename="guide.md", status="modified", patch="@@ -1 +1 @@\n-a\n+b\n"),
                SimpleNamespace(filename="release-notes.md", status="modified", patch="@@ -1 +1 @@\n-a\n+b\n"),
            ],
        }

        def fake_process_translation_group(*args, **kwargs):
            paths = {file.filename for file in kwargs["filtered_changed_files"]}
            return {
                "attempted": True,
                "successful_file_paths": paths,
                "counts": zero_counts,
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "guide.md").write_text(
                "# Guide <!--Corresponding EN commit: ccccccc333333333-->\n\nBody\n",
                encoding="utf-8",
            )
            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(workflow, "SOURCE_REPO", "acme/docs"))
                stack.enter_context(mock.patch.object(workflow, "TARGET_REPO", "acme/docs-cn"))
                stack.enter_context(mock.patch.object(workflow, "GITHUB_TOKEN", "token"))
                stack.enter_context(mock.patch.object(workflow, "TARGET_REPO_PATH", tmpdir))
                stack.enter_context(mock.patch.object(workflow, "SOURCE_BASE_REF", "aaaaaaa111111111"))
                stack.enter_context(mock.patch.object(workflow, "SOURCE_HEAD_REF", "ccccccc333333333"))
                stack.enter_context(mock.patch.object(workflow, "SOURCE_FILES", ""))
                stack.enter_context(mock.patch.object(workflow, "SOURCE_FOLDER", ""))
                stack.enter_context(mock.patch.object(workflow, "SOURCE_FILES_TRANSLATION_MODE", "incremental"))
                stack.enter_context(mock.patch.object(workflow, "COMMIT_SYNC_RUN_TYPE", "schedule"))
                stack.enter_context(mock.patch.object(workflow, "SOURCE_REPO_PATH", ""))
                stack.enter_context(mock.patch.object(workflow, "TARGET_REF", ""))
                stack.enter_context(mock.patch.object(workflow, "clean_temp_output_dir"))
                stack.enter_context(mock.patch.object(workflow.Auth, "Token", return_value="token-auth"))
                stack.enter_context(mock.patch.object(workflow, "Github", return_value=fake_github_client))
                stack.enter_context(mock.patch.object(workflow, "UnifiedAIClient", return_value=fake_ai_client))
                stack.enter_context(mock.patch.object(workflow, "load_glossary", return_value=[]))
                stack.enter_context(mock.patch.object(workflow, "create_glossary_matcher", return_value=None))
                build_context = stack.enter_context(
                    mock.patch.object(
                        workflow,
                        "build_incremental_diff_context",
                        return_value=global_context,
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        workflow,
                        "process_translation_group",
                        side_effect=fake_process_translation_group,
                    )
                )
                update_markers = stack.enter_context(
                    mock.patch.object(
                        workflow,
                        "update_corresponding_en_commit_for_files",
                        wraps=workflow.update_corresponding_en_commit_for_files,
                    )
                )
                stack.enter_context(mock.patch.object(workflow, "git_add_changes"))
                stack.enter_context(mock.patch.object(workflow.TranslationStats, "write_failure_report"))

                result = workflow.main()

            self.assertEqual(result, 0)
            build_context.assert_called_once()
            self.assertEqual(update_markers.call_args.args[0], {"guide.md"})
            self.assertNotIn(
                "Corresponding EN commit",
                Path(tmpdir, "guide.md").read_text(encoding="utf-8"),
            )

    def test_scheduled_run_removes_marker_when_per_file_cursor_has_no_diff(self):
        fake_github_client = object()
        fake_ai_client = SimpleNamespace(model="fake-model")
        global_context = {
            "mode": "commit",
            "source_repo": "acme/docs",
            "target_repo": "acme/docs-cn",
            "base_ref": "aaaaaaa111111111",
            "head_ref": "ccccccc333333333",
            "changed_files": [
                SimpleNamespace(filename="guide.md", status="modified", patch="@@ -1 +1 @@\n-a\n+b\n"),
            ],
        }
        marker_context = {
            "mode": "commit",
            "source_repo": "acme/docs",
            "target_repo": "acme/docs-cn",
            "base_ref": "bbbbbbb222222222",
            "head_ref": "ccccccc333333333",
            "changed_files": [],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "guide.md").write_text(
                "# Guide <!--Corresponding EN commit: bbbbbbb222222222-->\n\nBody\n",
                encoding="utf-8",
            )
            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(workflow, "SOURCE_REPO", "acme/docs"))
                stack.enter_context(mock.patch.object(workflow, "TARGET_REPO", "acme/docs-cn"))
                stack.enter_context(mock.patch.object(workflow, "GITHUB_TOKEN", "token"))
                stack.enter_context(mock.patch.object(workflow, "TARGET_REPO_PATH", tmpdir))
                stack.enter_context(mock.patch.object(workflow, "SOURCE_BASE_REF", "aaaaaaa111111111"))
                stack.enter_context(mock.patch.object(workflow, "SOURCE_HEAD_REF", "ccccccc333333333"))
                stack.enter_context(mock.patch.object(workflow, "SOURCE_FILES", ""))
                stack.enter_context(mock.patch.object(workflow, "SOURCE_FOLDER", ""))
                stack.enter_context(mock.patch.object(workflow, "SOURCE_FILES_TRANSLATION_MODE", "incremental"))
                stack.enter_context(mock.patch.object(workflow, "COMMIT_SYNC_RUN_TYPE", "schedule"))
                stack.enter_context(mock.patch.object(workflow, "SOURCE_REPO_PATH", ""))
                stack.enter_context(mock.patch.object(workflow, "TARGET_REF", ""))
                stack.enter_context(mock.patch.object(workflow, "clean_temp_output_dir"))
                stack.enter_context(mock.patch.object(workflow.Auth, "Token", return_value="token-auth"))
                stack.enter_context(mock.patch.object(workflow, "Github", return_value=fake_github_client))
                stack.enter_context(mock.patch.object(workflow, "UnifiedAIClient", return_value=fake_ai_client))
                stack.enter_context(mock.patch.object(workflow, "load_glossary", return_value=[]))
                stack.enter_context(mock.patch.object(workflow, "create_glossary_matcher", return_value=None))
                build_context = stack.enter_context(
                    mock.patch.object(
                        workflow,
                        "build_incremental_diff_context",
                        side_effect=[global_context, marker_context],
                    )
                )
                process_group = stack.enter_context(
                    mock.patch.object(workflow, "process_translation_group")
                )
                update_markers = stack.enter_context(
                    mock.patch.object(
                        workflow,
                        "update_corresponding_en_commit_for_files",
                        wraps=workflow.update_corresponding_en_commit_for_files,
                    )
                )
                stack.enter_context(mock.patch.object(workflow, "git_add_changes"))
                stack.enter_context(mock.patch.object(workflow.TranslationStats, "write_failure_report"))

                result = workflow.main()

            self.assertEqual(result, 0)
            self.assertEqual(build_context.call_count, 2)
            process_group.assert_not_called()
            self.assertEqual(update_markers.call_args.args[0], {"guide.md"})
            self.assertNotIn(
                "Corresponding EN commit",
                Path(tmpdir, "guide.md").read_text(encoding="utf-8"),
            )

    def test_scheduled_run_removes_marker_for_unchanged_file_in_marker_group(self):
        fake_github_client = object()
        fake_ai_client = SimpleNamespace(model="fake-model")
        zero_counts = {
            "added_files": 0,
            "deleted_files": 0,
            "toc_files": 0,
            "keyword_files": 0,
            "modified_sections": 0,
            "added_images": 0,
            "modified_images": 0,
            "deleted_images": 0,
        }
        global_context = {
            "mode": "commit",
            "source_repo": "acme/docs",
            "target_repo": "acme/docs-cn",
            "base_ref": "aaaaaaa111111111",
            "head_ref": "ccccccc333333333",
            "changed_files": [
                SimpleNamespace(filename="changed.md", status="modified", patch="@@ -1 +1 @@\n-a\n+b\n"),
                SimpleNamespace(filename="unchanged.md", status="modified", patch="@@ -1 +1 @@\n-a\n+b\n"),
            ],
        }
        marker_context = {
            "mode": "commit",
            "source_repo": "acme/docs",
            "target_repo": "acme/docs-cn",
            "base_ref": "bbbbbbb222222222",
            "head_ref": "ccccccc333333333",
            "changed_files": [
                SimpleNamespace(filename="changed.md", status="modified", patch="@@ -1 +1 @@\n-b\n+c\n"),
            ],
        }

        def fake_process_translation_group(*args, **kwargs):
            paths = {file.filename for file in kwargs["filtered_changed_files"]}
            return {
                "attempted": True,
                "successful_file_paths": paths,
                "counts": zero_counts,
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "changed.md").write_text(
                "# Changed <!--Corresponding EN commit: bbbbbbb222222222-->\n\nBody\n",
                encoding="utf-8",
            )
            Path(tmpdir, "unchanged.md").write_text(
                "# Unchanged <!--Corresponding EN commit: bbbbbbb222222222-->\n\nBody\n",
                encoding="utf-8",
            )
            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(workflow, "SOURCE_REPO", "acme/docs"))
                stack.enter_context(mock.patch.object(workflow, "TARGET_REPO", "acme/docs-cn"))
                stack.enter_context(mock.patch.object(workflow, "GITHUB_TOKEN", "token"))
                stack.enter_context(mock.patch.object(workflow, "TARGET_REPO_PATH", tmpdir))
                stack.enter_context(mock.patch.object(workflow, "SOURCE_BASE_REF", "aaaaaaa111111111"))
                stack.enter_context(mock.patch.object(workflow, "SOURCE_HEAD_REF", "ccccccc333333333"))
                stack.enter_context(mock.patch.object(workflow, "SOURCE_FILES", ""))
                stack.enter_context(mock.patch.object(workflow, "SOURCE_FOLDER", ""))
                stack.enter_context(mock.patch.object(workflow, "SOURCE_FILES_TRANSLATION_MODE", "incremental"))
                stack.enter_context(mock.patch.object(workflow, "COMMIT_SYNC_RUN_TYPE", "schedule"))
                stack.enter_context(mock.patch.object(workflow, "SOURCE_REPO_PATH", ""))
                stack.enter_context(mock.patch.object(workflow, "TARGET_REF", ""))
                stack.enter_context(mock.patch.object(workflow, "clean_temp_output_dir"))
                stack.enter_context(mock.patch.object(workflow.Auth, "Token", return_value="token-auth"))
                stack.enter_context(mock.patch.object(workflow, "Github", return_value=fake_github_client))
                stack.enter_context(mock.patch.object(workflow, "UnifiedAIClient", return_value=fake_ai_client))
                stack.enter_context(mock.patch.object(workflow, "load_glossary", return_value=[]))
                stack.enter_context(mock.patch.object(workflow, "create_glossary_matcher", return_value=None))
                stack.enter_context(
                    mock.patch.object(
                        workflow,
                        "build_incremental_diff_context",
                        side_effect=[global_context, marker_context],
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        workflow,
                        "process_translation_group",
                        side_effect=fake_process_translation_group,
                    )
                )
                update_markers = stack.enter_context(
                    mock.patch.object(
                        workflow,
                        "update_corresponding_en_commit_for_files",
                        wraps=workflow.update_corresponding_en_commit_for_files,
                    )
                )
                stack.enter_context(mock.patch.object(workflow, "git_add_changes"))
                stack.enter_context(mock.patch.object(workflow.TranslationStats, "write_failure_report"))

                result = workflow.main()

            self.assertEqual(result, 0)
            self.assertEqual(
                update_markers.call_args.args[0],
                {"changed.md", "unchanged.md"},
            )
            self.assertNotIn(
                "Corresponding EN commit",
                Path(tmpdir, "changed.md").read_text(encoding="utf-8"),
            )
            self.assertNotIn(
                "Corresponding EN commit",
                Path(tmpdir, "unchanged.md").read_text(encoding="utf-8"),
            )

    def test_unspecified_run_type_keeps_legacy_global_diff_path(self):
        fake_github_client = object()
        fake_ai_client = SimpleNamespace(model="fake-model")
        zero_counts = {
            "added_files": 0,
            "deleted_files": 0,
            "toc_files": 0,
            "keyword_files": 0,
            "modified_sections": 0,
            "added_images": 0,
            "modified_images": 0,
            "deleted_images": 0,
        }
        global_context = {
            "mode": "commit",
            "source_repo": "acme/docs",
            "target_repo": "acme/docs-cn",
            "base_ref": "aaaaaaa111111111",
            "head_ref": "ccccccc333333333",
            "changed_files": [
                SimpleNamespace(filename="guide.md", status="modified", patch="@@ -1 +1 @@\n-a\n+b\n"),
                SimpleNamespace(filename="other.md", status="modified", patch="@@ -1 +1 @@\n-a\n+b\n"),
            ],
        }
        group_paths = []

        def fake_process_translation_group(*args, **kwargs):
            paths = {file.filename for file in kwargs["filtered_changed_files"]}
            group_paths.append(paths)
            return {
                "attempted": True,
                "successful_file_paths": paths,
                "counts": zero_counts,
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "guide.md").write_text(
                "# Guide <!--Corresponding EN commit: bbbbbbb222222222-->\n\nBody\n",
                encoding="utf-8",
            )
            Path(tmpdir, "other.md").write_text("# Other\n\nBody\n", encoding="utf-8")
            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(workflow, "SOURCE_REPO", "acme/docs"))
                stack.enter_context(mock.patch.object(workflow, "TARGET_REPO", "acme/docs-cn"))
                stack.enter_context(mock.patch.object(workflow, "GITHUB_TOKEN", "token"))
                stack.enter_context(mock.patch.object(workflow, "TARGET_REPO_PATH", tmpdir))
                stack.enter_context(mock.patch.object(workflow, "SOURCE_BASE_REF", "aaaaaaa111111111"))
                stack.enter_context(mock.patch.object(workflow, "SOURCE_HEAD_REF", "ccccccc333333333"))
                stack.enter_context(mock.patch.object(workflow, "SOURCE_FILES", ""))
                stack.enter_context(mock.patch.object(workflow, "SOURCE_FOLDER", ""))
                stack.enter_context(mock.patch.object(workflow, "SOURCE_FILES_TRANSLATION_MODE", "incremental"))
                stack.enter_context(mock.patch.object(workflow, "COMMIT_SYNC_RUN_TYPE", ""))
                stack.enter_context(mock.patch.object(workflow, "SOURCE_REPO_PATH", ""))
                stack.enter_context(mock.patch.object(workflow, "TARGET_REF", ""))
                stack.enter_context(mock.patch.object(workflow, "clean_temp_output_dir"))
                stack.enter_context(mock.patch.object(workflow.Auth, "Token", return_value="token-auth"))
                stack.enter_context(mock.patch.object(workflow, "Github", return_value=fake_github_client))
                stack.enter_context(mock.patch.object(workflow, "UnifiedAIClient", return_value=fake_ai_client))
                stack.enter_context(mock.patch.object(workflow, "load_glossary", return_value=[]))
                stack.enter_context(mock.patch.object(workflow, "create_glossary_matcher", return_value=None))
                build_context = stack.enter_context(
                    mock.patch.object(
                        workflow,
                        "build_incremental_diff_context",
                        return_value=global_context,
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        workflow,
                        "process_translation_group",
                        side_effect=fake_process_translation_group,
                    )
                )
                update_markers = stack.enter_context(
                    mock.patch.object(workflow, "update_corresponding_en_commit_for_files")
                )
                stack.enter_context(mock.patch.object(workflow.TranslationStats, "write_failure_report"))

                result = workflow.main()

            self.assertEqual(result, 0)
            build_context.assert_called_once()
            self.assertEqual(group_paths, [{"guide.md", "other.md"}])
            update_markers.assert_not_called()
            self.assertIn(
                "Corresponding EN commit",
                Path(tmpdir, "guide.md").read_text(encoding="utf-8"),
            )

    def test_scheduled_marker_group_diff_failure_continues_with_failure_report(self):
        fake_github_client = object()
        fake_ai_client = SimpleNamespace(model="fake-model")
        global_context = {
            "mode": "commit",
            "source_repo": "acme/docs",
            "target_repo": "acme/docs-cn",
            "base_ref": "aaaaaaa111111111",
            "head_ref": "ccccccc333333333",
            "changed_files": [
                SimpleNamespace(filename="guide.md", status="modified", patch="@@ -1 +1 @@\n-a\n+b\n"),
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "guide.md").write_text(
                "# Guide <!--Corresponding EN commit: bbbbbbb222222222-->\n\nBody\n",
                encoding="utf-8",
            )
            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(workflow, "SOURCE_REPO", "acme/docs"))
                stack.enter_context(mock.patch.object(workflow, "TARGET_REPO", "acme/docs-cn"))
                stack.enter_context(mock.patch.object(workflow, "GITHUB_TOKEN", "token"))
                stack.enter_context(mock.patch.object(workflow, "TARGET_REPO_PATH", tmpdir))
                stack.enter_context(mock.patch.object(workflow, "SOURCE_BASE_REF", "aaaaaaa111111111"))
                stack.enter_context(mock.patch.object(workflow, "SOURCE_HEAD_REF", "ccccccc333333333"))
                stack.enter_context(mock.patch.object(workflow, "SOURCE_FILES", ""))
                stack.enter_context(mock.patch.object(workflow, "SOURCE_FOLDER", ""))
                stack.enter_context(mock.patch.object(workflow, "SOURCE_FILES_TRANSLATION_MODE", "incremental"))
                stack.enter_context(mock.patch.object(workflow, "COMMIT_SYNC_RUN_TYPE", "schedule"))
                stack.enter_context(mock.patch.object(workflow, "SOURCE_REPO_PATH", ""))
                stack.enter_context(mock.patch.object(workflow, "TARGET_REF", ""))
                stack.enter_context(mock.patch.object(workflow, "clean_temp_output_dir"))
                stack.enter_context(mock.patch.object(workflow.Auth, "Token", return_value="token-auth"))
                stack.enter_context(mock.patch.object(workflow, "Github", return_value=fake_github_client))
                stack.enter_context(mock.patch.object(workflow, "UnifiedAIClient", return_value=fake_ai_client))
                stack.enter_context(mock.patch.object(workflow, "load_glossary", return_value=[]))
                stack.enter_context(mock.patch.object(workflow, "create_glossary_matcher", return_value=None))
                stack.enter_context(
                    mock.patch.object(
                        workflow,
                        "build_incremental_diff_context",
                        side_effect=[global_context, RuntimeError("compare failed")],
                    )
                )
                process_group = stack.enter_context(
                    mock.patch.object(workflow, "process_translation_group")
                )
                update_markers = stack.enter_context(
                    mock.patch.object(workflow, "update_corresponding_en_commit_for_files")
                )
                write_failure_report = stack.enter_context(
                    mock.patch.object(workflow.TranslationStats, "write_failure_report")
                )

                result = workflow.main()

            self.assertEqual(result, 0)
            process_group.assert_not_called()
            update_markers.assert_not_called()
            write_failure_report.assert_called_once()

    def test_apply_full_translation_mode_uses_head_file_content_and_removes_incremental_work(self):
        changed_files = [
            SimpleNamespace(filename="guide.md", status="modified"),
            SimpleNamespace(filename="TOC.md", status="modified"),
            SimpleNamespace(filename="deleted.md", status="removed"),
            SimpleNamespace(filename="renamed-new.md", previous_filename="renamed-old.md", status="renamed"),
            SimpleNamespace(filename="image.png", status="modified"),
        ]
        added_sections = {"guide.md": {"sections": {}}, "other.md": {"sections": {}}}
        modified_sections = {
            "guide.md": {"sections": {}},
            "TOC.md": {"sections": {}},
            "renamed-new.md": {"sections": {}},
        }
        deleted_sections = {"guide.md": {"1": "# Old"}, "deleted.md": {"1": "# Removed"}}
        added_files = {}
        toc_files = {"TOC.md": {"type": "toc"}}
        keyword_files = {}
        stats = workflow.TranslationStats()
        calls = []

        def fake_get_source_file_content(file_path, diff_context, github_client, ref_name="head_ref"):
            calls.append((file_path, ref_name, diff_context["head_ref"]))
            return f"HEAD content for {file_path}"

        with mock.patch.object(
            workflow,
            "get_source_file_content",
            side_effect=fake_get_source_file_content,
        ):
            full_paths = workflow.apply_source_files_full_translation_mode(
                "guide.md,TOC.md,renamed-old.md,unchanged.md",
                "",
                changed_files,
                {"head_ref": "latest-head"},
                object(),
                [],
                added_sections,
                modified_sections,
                deleted_sections,
                added_files,
                toc_files,
                keyword_files,
                stats,
            )

        self.assertEqual(full_paths, {"guide.md", "TOC.md", "renamed-new.md", "unchanged.md"})
        self.assertCountEqual(
            calls,
            [
                ("guide.md", "head_ref", "latest-head"),
                ("renamed-new.md", "head_ref", "latest-head"),
                ("TOC.md", "head_ref", "latest-head"),
                ("unchanged.md", "head_ref", "latest-head"),
            ],
        )
        self.assertEqual(
            added_files,
            {
                "guide.md": "HEAD content for guide.md",
                "renamed-new.md": "HEAD content for renamed-new.md",
                "TOC.md": "HEAD content for TOC.md",
                "unchanged.md": "HEAD content for unchanged.md",
            },
        )
        self.assertNotIn("guide.md", added_sections)
        self.assertIn("other.md", added_sections)
        self.assertNotIn("guide.md", modified_sections)
        self.assertNotIn("TOC.md", modified_sections)
        self.assertNotIn("renamed-new.md", modified_sections)
        self.assertNotIn("guide.md", deleted_sections)
        self.assertIn("deleted.md", deleted_sections)
        self.assertEqual(toc_files, {})
        self.assertEqual(stats.failed, [])

    def test_full_translation_collection_strips_related_resources_sections(self):
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

        with mock.patch.object(
            workflow,
            "get_full_translation_source_content",
            return_value=source_content,
        ):
            files, failures = workflow.collect_source_files_for_full_translation(
                {"guide.md"},
                [],
                {"head_ref": "head"},
                object(),
            )

        self.assertEqual(failures, {})
        self.assertEqual(set(files), {"guide.md"})
        self.assertIn("## Usage", files["guide.md"])
        self.assertNotIn("RelatedResources", files["guide.md"])
        self.assertNotIn("ResourceCard", files["guide.md"])

    def test_full_translation_collection_skips_related_resources_only_file(self):
        source_content = "\n".join(
            [
                "## Related resources",
                "",
                "<RelatedResources>",
                '  <ResourceCard title="Example" type="blog" link="https://example.com" />',
                "</RelatedResources>",
                "",
            ]
        )

        with mock.patch.object(
            workflow,
            "get_full_translation_source_content",
            return_value=source_content,
        ):
            files, failures = workflow.collect_source_files_for_full_translation(
                {"guide.md"},
                [],
                {"head_ref": "head"},
                object(),
            )

        self.assertEqual(files, {})
        self.assertEqual(failures, {})

    def test_commit_modified_file_filters_related_resources_from_ai_diff_context(self):
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
        pr_diff = "\n".join(
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
                "-" * 80,
            ]
        )

        def fake_get_source_file_content(file_path, diff_context, github_client, ref_name="head_ref"):
            return base_content if ref_name == "base_ref" else head_content

        with mock.patch.object(
            workflow,
            "get_source_file_content",
            side_effect=fake_get_source_file_content,
        ), mock.patch.object(
            workflow,
            "should_process_modified_file_as_added",
            return_value=False,
        ), mock.patch.object(
            workflow,
            "determine_file_processing_type",
            return_value="regular_modified",
        ), mock.patch.object(
            workflow,
            "process_regular_modified_file",
            return_value=(True, ""),
        ) as process_regular_modified_file:
            result = workflow._process_commit_modified_file(
                "guide.md",
                {"sections": {"3": "## Usage"}},
                pr_diff,
                {"base_ref": "base", "head_ref": "head"},
                object(),
                object(),
                {},
                None,
            )

        self.assertEqual(result["status"], "success")
        filtered_diff = process_regular_modified_file.call_args.args[2]
        self.assertIn("-Old usage.", filtered_diff)
        self.assertIn("+New usage.", filtered_diff)
        self.assertNotIn("RelatedResources", filtered_diff)
        self.assertNotIn("ResourceCard", filtered_diff)

    def test_commit_modified_file_keeps_related_resources_when_filter_disabled(self):
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
        pr_diff = "\n".join(
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
                "-" * 80,
            ]
        )

        def fake_get_source_file_content(file_path, diff_context, github_client, ref_name="head_ref"):
            return base_content if ref_name == "base_ref" else head_content

        with mock.patch.object(
            workflow,
            "get_source_file_content",
            side_effect=fake_get_source_file_content,
        ), mock.patch.object(
            workflow,
            "should_process_modified_file_as_added",
            return_value=False,
        ), mock.patch.object(
            workflow,
            "determine_file_processing_type",
            return_value="regular_modified",
        ), mock.patch.object(
            workflow,
            "process_regular_modified_file",
            return_value=(True, ""),
        ) as process_regular_modified_file:
            result = workflow._process_commit_modified_file(
                "guide.md",
                {"sections": {"3": "## Usage"}},
                pr_diff,
                {"base_ref": "base", "head_ref": "head"},
                object(),
                object(),
                {"ignore_resource_card_section": False},
                None,
            )

        self.assertEqual(result["status"], "success")
        kept_diff = process_regular_modified_file.call_args.args[2]
        self.assertIn("RelatedResources", kept_diff)
        self.assertIn("ResourceCard", kept_diff)

    def test_added_file_fallback_strips_related_resources_sections(self):
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
                '  <ResourceCard title="Example" />',
                "</RelatedResources>",
                "",
            ]
        )
        pr_diff = "\n".join(
            [
                "File: guide.md",
                "@@ -3,1 +3,1 @@",
                "-Old usage.",
                "+Use TiDB.",
            ]
        )

        def fake_get_source_file_content(file_path, diff_context, github_client, ref_name="head_ref"):
            return "# Guide\n\n## Usage\n\nOld usage.\n" if ref_name == "base_ref" else source_content

        with mock.patch.object(
            workflow,
            "get_source_file_content",
            side_effect=fake_get_source_file_content,
        ), mock.patch.object(
            workflow,
            "should_process_modified_file_as_added",
            return_value=True,
        ), mock.patch.object(
            workflow,
            "process_added_files",
            return_value=(True, {}),
        ) as process_added_files:
            result = workflow._process_commit_modified_file(
                "guide.md",
                {"sections": {"3": "## Usage"}},
                pr_diff,
                {"base_ref": "base", "head_ref": "head"},
                object(),
                object(),
                {},
                None,
            )

        self.assertEqual(result["status"], "success")
        added_content = process_added_files.call_args.args[0]["guide.md"]
        self.assertIn("## Usage", added_content)
        self.assertNotIn("RelatedResources", added_content)
        self.assertNotIn("ResourceCard", added_content)

    def test_collect_toc_scope_added_files_detects_new_cloud_links(self):
        toc_files = {
            "TOC-tidb-cloud.md": {
                "type": "toc",
                "source_base_content": "- [Old](/tidb-cloud/old.md)\n",
                "source_head_content": "\n".join(
                    [
                        "- [Old](/tidb-cloud/old.md)",
                        "- [New](/tidb-cloud/a-b-c.md#overview)",
                        "- [Shared](/shared/a-b-c.md)",
                    ]
                ),
            }
        }

        self.assertEqual(
            workflow.collect_toc_scope_added_files(toc_files),
            {"tidb-cloud/a-b-c.md", "shared/a-b-c.md"},
        )

    def test_collect_toc_scope_added_files_uses_aggregate_toc_scope(self):
        toc_files = {
            "TOC-tidb-cloud.md": {
                "type": "toc",
                "source_base_content": "- [Old](/tidb-cloud/old.md)\n",
                "source_head_content": "\n".join(
                    [
                        "- [Old](/tidb-cloud/old.md)",
                        "- [Shared](/shared/already-linked.md)",
                        "- [New](/shared/newly-linked.md)",
                    ]
                ),
            },
            "TOC-tidb-cloud-starter.md": {
                "type": "toc",
                "source_base_content": "- [Shared](/shared/already-linked.md)\n",
                "source_head_content": "- [Shared](/shared/already-linked.md)\n",
            },
        }

        self.assertEqual(
            workflow.collect_toc_scope_added_files(toc_files),
            {"shared/newly-linked.md"},
        )

    def test_collect_toc_scope_added_files_reads_all_configured_tocs(self):
        contents = {
            ("TOC-tidb-cloud.md", "base_ref"): "- [Old](/tidb-cloud/old.md)\n",
            ("TOC-tidb-cloud.md", "head_ref"): "\n".join(
                [
                    "- [Old](/tidb-cloud/old.md)",
                    "- [Shared](/shared/already-linked.md)",
                    "- [New](/shared/newly-linked.md)",
                ]
            ),
            ("TOC-tidb-cloud-starter.md", "base_ref"): "- [Shared](/shared/already-linked.md)\n",
            ("TOC-tidb-cloud-starter.md", "head_ref"): "- [Shared](/shared/already-linked.md)\n",
        }

        def fake_get_source_ref_content(file_path, diff_context, github_client, ref_name):
            return contents[(file_path, ref_name)]

        with mock.patch.object(
            workflow,
            "get_source_ref_content",
            side_effect=fake_get_source_ref_content,
        ):
            scope_added = workflow.collect_toc_scope_added_files(
                {},
                {"base_ref": "base", "head_ref": "head"},
                object(),
                ["TOC-tidb-cloud.md", "TOC-tidb-cloud-starter.md"],
            )

        self.assertEqual(scope_added, {"shared/newly-linked.md"})

    def test_apply_toc_scope_added_files_queues_full_translation_and_keeps_toc_work(self):
        toc_files = {
            "TOC-tidb-cloud.md": {
                "type": "toc",
                "source_base_content": "- [Old](/tidb-cloud/old.md)\n",
                "source_head_content": "\n".join(
                    [
                        "- [Old](/tidb-cloud/old.md)",
                        "- [New](/shared/a-b-c.md)",
                    ]
                ),
            }
        }
        added_sections = {"shared/a-b-c.md": {"sections": {}}, "other.md": {"sections": {}}}
        modified_sections = {"shared/a-b-c.md": {"sections": {}}}
        deleted_sections = {"shared/a-b-c.md": {"1": "# Old"}}
        added_files = {"shared/a-b-c.md": "diff-added content"}
        keyword_files = {"shared/a-b-c.md": {"type": "keyword"}}
        stats = workflow.TranslationStats()

        with mock.patch.object(
            workflow,
            "get_full_translation_source_content",
            return_value="# HEAD content\n",
        ) as get_full_translation_source_content:
            queued_paths = workflow.apply_toc_scope_added_files(
                toc_files,
                {"head_ref": "head"},
                object(),
                [],
                added_sections,
                modified_sections,
                deleted_sections,
                added_files,
                keyword_files,
                stats,
            )

        self.assertEqual(queued_paths, {"shared/a-b-c.md"})
        get_full_translation_source_content.assert_called_once()
        self.assertEqual(added_files, {"shared/a-b-c.md": "# HEAD content\n"})
        self.assertNotIn("shared/a-b-c.md", added_sections)
        self.assertIn("other.md", added_sections)
        self.assertNotIn("shared/a-b-c.md", modified_sections)
        self.assertNotIn("shared/a-b-c.md", deleted_sections)
        self.assertNotIn("shared/a-b-c.md", keyword_files)
        self.assertIn("TOC-tidb-cloud.md", toc_files)
        self.assertEqual(stats.failed, [])

    def test_process_translation_group_skips_toc_scope_when_source_files_explicit(self):
        changed_file = SimpleNamespace(
            filename="tidb-cloud/dedicated/_index.md",
            status="modified",
            patch="@@ -1 +1 @@\n-old\n+new\n",
        )
        zero_counts = {
            "added_files": 0,
            "deleted_files": 0,
            "toc_files": 0,
            "keyword_files": 0,
            "index_files": 1,
            "formatting_files": 0,
            "modified_sections": 0,
            "added_images": 0,
            "modified_images": 0,
            "deleted_images": 0,
        }

        with mock.patch.object(
            workflow,
            "analyze_source_changes",
            return_value=({}, {}, {}, {}, [], {}, {}, [], [], [], set(), {
                "tidb-cloud/dedicated/_index.md": {
                    "type": "index",
                    "source_base_content": "---\n---\n",
                    "source_head_content": "---\n---\n",
                }
            }, {}),
        ), mock.patch.object(
            workflow,
            "apply_toc_scope_added_files",
            side_effect=AssertionError("Cloud TOC scope should be skipped for explicit SOURCE_FILES"),
        ), mock.patch.object(
            workflow,
            "should_parallelize_file_processing",
            return_value=False,
        ), mock.patch.object(
            workflow,
            "run_file_tasks",
            return_value=[{
                "file_path": "tidb-cloud/dedicated/_index.md",
                "ok": True,
                "result": {"status": "success", "reason": ""},
                "error": None,
            }],
        ), mock.patch.object(
            workflow,
            "_record_translation_task_result",
            return_value=True,
        ), mock.patch.object(
            workflow,
            "git_add_successful_task_changes",
        ), mock.patch.object(
            workflow,
            "process_index_file",
            return_value=True,
        ):
            result = workflow.process_translation_group(
                "explicit SOURCE_FILES",
                source_files_translation_mode="incremental",
                source_files="tidb-cloud/dedicated/_index.md",
                source_folder="",
                diff_context={
                    "mode": "commit",
                    "source_repo": "pingcap/docs",
                    "target_repo": "pingcap/docs",
                    "base_ref": "base",
                    "head_ref": "head",
                },
                filtered_changed_files=[changed_file],
                pr_diff=changed_file.patch,
                github_client=object(),
                ai_client=object(),
                repo_config={
                    "source_language": "English",
                    "target_language": "Chinese",
                    "target_local_path": "/tmp/unused",
                },
                repo_configs={},
                glossary_matcher=None,
                commit_ignore_files=[],
                translation_stats=workflow.TranslationStats(),
            )

        self.assertTrue(result["attempted"])
        self.assertEqual(
            result["successful_file_paths"],
            {"tidb-cloud/dedicated/_index.md"},
        )
        self.assertEqual(result["counts"], zero_counts)

    def test_process_translation_group_bootstraps_full_files_without_incremental_diff(self):
        stats = workflow.TranslationStats()

        with mock.patch.object(
            workflow,
            "apply_source_files_full_translation_mode",
            return_value={"TOC-new.md", "new-guide.md"},
        ) as apply_full, mock.patch.object(
            workflow,
            "apply_toc_scope_added_files",
            side_effect=AssertionError("Explicit source scope should skip secondary expansion"),
        ), mock.patch.object(
            workflow,
            "should_parallelize_file_processing",
            return_value=False,
        ), mock.patch.object(
            workflow,
            "validate_successful_translation_structures",
        ):
            result = workflow.process_translation_group(
                "new TOC bootstrap",
                source_files_translation_mode="incremental",
                source_files="TOC-new.md,new-guide.md",
                source_folder="",
                diff_context={
                    "mode": "commit",
                    "source_repo": "pingcap/docs",
                    "target_repo": "pingcap/docs",
                    "base_ref": "base",
                    "head_ref": "head",
                },
                filtered_changed_files=[],
                pr_diff="",
                github_client=object(),
                ai_client=object(),
                repo_config={
                    "source_language": "English",
                    "target_language": "Chinese",
                    "target_local_path": "/tmp/unused",
                },
                repo_configs={},
                glossary_matcher=None,
                commit_ignore_files=[],
                translation_stats=stats,
                source_full_files="TOC-new.md,new-guide.md",
            )

        self.assertTrue(result["attempted"])
        apply_full.assert_called_once()
        self.assertEqual(
            apply_full.call_args.args[0],
            "TOC-new.md,new-guide.md",
        )

    def test_process_translation_group_applies_formatting_only_without_ai(self):
        file_path = "tidb-cloud/monitor-alert-slack.md"
        change_data = build_formatting_only_change(
            '# Slack\n\n<CustomContent plan="dedicated">\n    \n1. Unsubscribe.\n',
            '# Slack\n\n<CustomContent plan="dedicated">\n\n1. Unsubscribe.\n',
        )
        changed_file = SimpleNamespace(
            filename=file_path,
            status="modified",
            patch='@@ -3,3 +3,3 @@\n <CustomContent plan="dedicated">\n-    \n+\n 1. Unsubscribe.\n',
        )
        analysis_result = (
            {},
            {},
            {},
            {},
            [],
            {},
            {},
            [],
            [],
            [],
            set(),
            {},
            {file_path: change_data},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = Path(tmpdir, file_path)
            target_path.parent.mkdir(parents=True)
            target_path.write_text(
                '# Slack 中文\n\n<CustomContent plan="dedicated">\n    \n1. 取消订阅。\n',
                encoding="utf-8",
            )
            stats = workflow.TranslationStats()

            with mock.patch.object(
                workflow,
                "analyze_source_changes",
                return_value=analysis_result,
            ), mock.patch.object(
                workflow,
                "should_parallelize_file_processing",
                return_value=False,
            ), mock.patch.object(
                workflow,
                "git_add_successful_task_changes",
            ), mock.patch.object(
                workflow,
                "validate_successful_translation_structures",
            ), mock.patch.object(
                workflow,
                "_process_commit_modified_file",
                side_effect=AssertionError("regular translation path must not run"),
            ) as regular_processor:
                result = workflow.process_translation_group(
                    "formatting-only",
                    source_files_translation_mode="incremental",
                    source_files=file_path,
                    source_folder="",
                    diff_context={
                        "mode": "commit",
                        "source_repo": "acme/docs",
                        "target_repo": "acme/docs-cn",
                        "base_ref": "base",
                        "head_ref": "head",
                    },
                    filtered_changed_files=[changed_file],
                    pr_diff=changed_file.patch,
                    github_client=object(),
                    ai_client=object(),
                    repo_config={
                        "source_language": "English",
                        "target_language": "Chinese",
                        "target_local_path": tmpdir,
                    },
                    repo_configs={},
                    glossary_matcher=None,
                    commit_ignore_files=[],
                    translation_stats=stats,
                )

            regular_processor.assert_not_called()
            self.assertEqual({file_path}, result["successful_file_paths"])
            self.assertEqual(1, result["counts"]["formatting_files"])
            self.assertEqual([], stats.failed)
            self.assertIn(
                '<CustomContent plan="dedicated">\n\n1. 取消订阅。',
                target_path.read_text(encoding="utf-8"),
            )

    def test_process_translation_group_uses_ai_mapping_when_target_lines_shift(self):
        file_path = "tidb-cloud/monitor-alert-slack.md"
        change_data = build_formatting_only_change(
            '# Slack\n\n<CustomContent plan="dedicated">\n    \n1. Unsubscribe.\n',
            '# Slack\n\n<CustomContent plan="dedicated">\n\n1. Unsubscribe.\n',
        )
        changed_file = SimpleNamespace(
            filename=file_path,
            status="modified",
            patch='@@ -3,3 +3,3 @@\n <CustomContent plan="dedicated">\n-    \n+\n 1. Unsubscribe.\n',
        )
        change_data["source_patch"] = changed_file.patch
        analysis_result = (
            {},
            {},
            {},
            {},
            [],
            {},
            {},
            [],
            [],
            [],
            set(),
            {},
            {file_path: change_data},
        )

        class MappingAI:
            def __init__(self):
                self.calls = []

            def chat_completion(self, **kwargs):
                self.calls.append(kwargs)
                return '{"mappings":[{"source_line_number":4,"target_line_number":5}]}'

        ai_client = MappingAI()
        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = Path(tmpdir, file_path)
            target_path.parent.mkdir(parents=True)
            target_path.write_text(
                '# Slack 中文\n补充说明。\n\n<CustomContent plan="dedicated">\n'
                '    \n1. 取消订阅。\n<CustomContent plan="dedicated">\n'
                '    \n1. 保持订阅。\n',
                encoding="utf-8",
            )
            stats = workflow.TranslationStats()

            with mock.patch.object(
                workflow,
                "analyze_source_changes",
                return_value=analysis_result,
            ), mock.patch.object(
                workflow,
                "should_parallelize_file_processing",
                return_value=False,
            ), mock.patch.object(
                workflow,
                "git_add_successful_task_changes",
            ), mock.patch.object(
                workflow,
                "validate_successful_translation_structures",
            ), mock.patch.object(
                workflow,
                "_process_commit_modified_file",
                side_effect=AssertionError("regular translation path must not run"),
            ) as regular_processor:
                result = workflow.process_translation_group(
                    "formatting AI fallback",
                    source_files_translation_mode="incremental",
                    source_files=file_path,
                    source_folder="",
                    diff_context={
                        "mode": "commit",
                        "source_repo": "acme/docs",
                        "target_repo": "acme/docs-cn",
                        "base_ref": "base",
                        "head_ref": "head",
                    },
                    filtered_changed_files=[changed_file],
                    pr_diff=changed_file.patch,
                    github_client=object(),
                    ai_client=ai_client,
                    repo_config={
                        "source_language": "English",
                        "target_language": "Chinese",
                        "target_local_path": tmpdir,
                    },
                    repo_configs={},
                    glossary_matcher=None,
                    commit_ignore_files=[],
                    translation_stats=stats,
                )

            regular_processor.assert_not_called()
            self.assertEqual(1, len(ai_client.calls))
            self.assertEqual({file_path}, result["successful_file_paths"])
            self.assertEqual(1, result["counts"]["formatting_files"])
            self.assertEqual([], stats.failed)
            self.assertIn(
                '<CustomContent plan="dedicated">\n\n1. 取消订阅。',
                target_path.read_text(encoding="utf-8"),
            )
            self.assertIn(
                '<CustomContent plan="dedicated">\n    \n1. 保持订阅。',
                target_path.read_text(encoding="utf-8"),
            )

    def test_translation_stats_writes_failure_report(self):
        stats = workflow.TranslationStats()
        stats.mark_failure("cloud.md", "Matcher failed")

        with tempfile.TemporaryDirectory() as tmpdir:
            stats.write_failure_report(tmpdir)
            report = Path(tmpdir, "translation-failures.md").read_text(encoding="utf-8")

        self.assertIn("### Translation failures", report)
        self.assertIn("- `cloud.md`: Matcher failed", report)

    def test_translation_stats_writes_partial_output_to_failure_report(self):
        stats = workflow.TranslationStats()
        stats.mark_partial("partial.md", "AI response incomplete")

        with tempfile.TemporaryDirectory() as tmpdir:
            stats.write_failure_report(tmpdir)
            report = Path(tmpdir, "translation-failures.md").read_text(encoding="utf-8")
            records = json.loads(
                Path(tmpdir, "translation-failures.json").read_text(encoding="utf-8")
            )

        self.assertIn("### Partial or incomplete translations", report)
        self.assertEqual(records[0]["status"], "partial")

    def test_translation_stats_writes_structure_report_without_translation_failures(self):
        stats = workflow.TranslationStats()
        stats.mark_structure_error(
            StructureValidationIssue(
                file_path="cloud.md",
                reason="heading level sequence differs",
                source_compact="#x1 ##x1",
                target_compact="#x1 ###x1",
                first_difference="heading 2: source ##, target ###",
            )
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            stats.write_failure_report(tmpdir)
            report = Path(tmpdir, "translation-failures.md").read_text(encoding="utf-8")
            structure_json = json.loads(
                Path(tmpdir, "translation-structure-errors.json").read_text(encoding="utf-8")
            )

            self.assertFalse(Path(tmpdir, "translation-failures.json").exists())

        self.assertIn("### Docs with document structure mismatches after translation", report)
        self.assertIn("- `cloud.md`: heading level sequence differs", report)
        self.assertIn("Source: `#x1 ##x1`", report)
        self.assertEqual("cloud.md", structure_json[0]["file_path"])

    def test_translation_stats_writes_combined_failure_and_structure_report(self):
        stats = workflow.TranslationStats()
        stats.mark_failure("failed.md", "Matcher failed")
        stats.mark_structure_error(
            StructureValidationIssue(
                file_path="mismatch.md",
                reason="heading level sequence differs",
                source_compact="#x1 ##x1",
                target_compact="#x1 ##x2",
            )
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            stats.write_failure_report(tmpdir)
            report = Path(tmpdir, "translation-failures.md").read_text(encoding="utf-8")
            failure_json = json.loads(
                Path(tmpdir, "translation-failures.json").read_text(encoding="utf-8")
            )
            structure_json = json.loads(
                Path(tmpdir, "translation-structure-errors.json").read_text(encoding="utf-8")
            )

        failure_heading = "### Translation failures"
        structure_heading = "### Docs with document structure mismatches after translation"
        self.assertIn(failure_heading, report)
        self.assertIn(structure_heading, report)
        self.assertLess(report.index(failure_heading), report.index(structure_heading))
        self.assertIn("- `failed.md`: Matcher failed", report)
        self.assertIn("- `mismatch.md`: heading level sequence differs", report)
        self.assertEqual("failed.md", failure_json[0]["file_path"])
        self.assertEqual("mismatch.md", structure_json[0]["file_path"])

    def test_translation_stats_removes_stale_failure_report_when_no_failures(self):
        stats = workflow.TranslationStats()

        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir, "translation-failures.md")
            structure_json_path = Path(tmpdir, "translation-structure-errors.json")
            report_path.write_text("stale", encoding="utf-8")
            structure_json_path.write_text("stale", encoding="utf-8")
            stats.write_failure_report(tmpdir)

            self.assertFalse(report_path.exists())
            self.assertFalse(structure_json_path.exists())

    def test_validate_successful_translation_structures_records_mismatch_but_keeps_success(self):
        stats = workflow.TranslationStats()
        stats.mark_success("guide.md")

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "guide.md").write_text(
                "# 指南\n\n### Child\n",
                encoding="utf-8",
            )
            with mock.patch.object(workflow, "TARGET_REPO_PATH", tmpdir), mock.patch.object(
                workflow,
                "get_source_ref_content",
                return_value="# Guide\n\n## Child\n",
            ):
                issues = workflow.validate_successful_translation_structures(
                    {"guide.md", "image.png"},
                    {"head_ref": "head"},
                    object(),
                    stats,
                )

        self.assertEqual(["guide.md"], stats.succeeded)
        self.assertEqual(1, len(issues))
        self.assertEqual(1, len(stats.structure_errors))
        self.assertEqual("guide.md", stats.structure_errors[0].file_path)

    def test_structure_validation_ignores_filtered_related_resources_section(self):
        stats = workflow.TranslationStats()
        stats.mark_success("guide.md")
        source = (
            "# Guide\n\n## Content\n\nBody.\n\n"
            "## Related resources\n\n"
            "<RelatedResources>\n"
            '  <ResourceCard title="Video" link="https://example.com" />\n'
            "</RelatedResources>\n"
        )
        target = (
            "# 指南\n\n## 内容\n\n正文。\n\n"
            "## 相关资源\n\n"
            "<RelatedResources>\n"
            '  <ResourceCard title="本地化视频" link="https://example.com" />\n'
            "</RelatedResources>\n"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "guide.md").write_text(target, encoding="utf-8")
            with mock.patch.object(workflow, "TARGET_REPO_PATH", tmpdir), mock.patch.object(
                workflow,
                "get_source_ref_content",
                return_value=source,
            ):
                issues = workflow.validate_successful_translation_structures(
                    {"guide.md"},
                    {
                        "mode": "commit",
                        "head_ref": "head",
                        "repo_config": {"ignore_resource_card_section": True},
                    },
                    object(),
                    stats,
                )

        self.assertEqual([], issues)
        self.assertEqual([], stats.structure_errors)

    def test_structure_validation_reports_changed_heading_anchor_missing_from_target(self):
        stats = workflow.TranslationStats()
        stats.mark_success("guide.md")
        source = "# Guide\n\n## Audit filter events\n"
        target = "# ガイド\n\n## 監査フィルターイベント\n"
        pr_diff = "\n".join(
            [
                "File: guide.md",
                "@@ -3,1 +3,1 @@",
                "-## Auditing filter events",
                "+## Audit filter events",
                "-" * 80,
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "guide.md").write_text(target, encoding="utf-8")
            with mock.patch.object(
                workflow,
                "TARGET_REPO_PATH",
                tmpdir,
            ), mock.patch.object(
                workflow,
                "get_source_ref_content",
                return_value=source,
            ):
                issues = workflow.validate_successful_translation_structures(
                    {"guide.md"},
                    {
                        "mode": "commit",
                        "head_ref": "head",
                        "repo_config": {
                            "source_language": "English",
                            "target_language": "Japanese",
                        },
                    },
                    object(),
                    stats,
                    repo_config={
                        "source_language": "English",
                        "target_language": "Japanese",
                    },
                    pr_diff=pr_diff,
                )

        self.assertEqual(
            ["expected heading anchors are missing"],
            [issue.reason for issue in issues],
        )
        self.assertEqual(1, len(stats.structure_errors))


if __name__ == "__main__":
    unittest.main()

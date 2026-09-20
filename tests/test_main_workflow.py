import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import main_workflow


class MainWorkflowImageOnlyPrTest(unittest.TestCase):
    def test_product_name_is_resolved_from_environment_at_runtime(self):
        self.assertFalse(hasattr(main_workflow, "PRODUCT"))

        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(main_workflow.get_product_name(), "")

        with mock.patch.dict(os.environ, {"PRODUCT": "First Product"}, clear=False):
            self.assertEqual(main_workflow.get_product_name(), "First Product")

        with mock.patch.dict(os.environ, {"PRODUCT": "Second Product"}, clear=False):
            self.assertEqual(main_workflow.get_product_name(), "Second Product")

    def test_unknown_provider_does_not_break_module_import(self):
        env = os.environ.copy()
        env["AI_PROVIDER"] = "azureopenai"
        env["PYTHONPATH"] = str(SCRIPTS_DIR)

        result = subprocess.run(
            [sys.executable, "-c", "import main_workflow"],
            cwd=SCRIPTS_DIR,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_git_add_changes_can_be_disabled(self):
        with mock.patch.object(main_workflow, "SKIP_GIT_ADD", True), mock.patch.object(
            main_workflow.subprocess, "run"
        ) as subprocess_run:
            main_workflow.git_add_changes("/tmp/target")

        subprocess_run.assert_not_called()

    def test_main_continues_when_pr_has_only_image_changes(self):
        repo_config = {
            "source_repo": "acme/docs",
            "target_repo": "acme/docs-cn",
            "target_local_path": "/tmp/target",
            "prefer_local_target_for_read": False,
            "source_language": "English",
            "target_language": "Chinese",
        }
        source_context = {
            "mode": "pr",
            "source_repo": "acme/docs",
            "target_repo": "acme/docs-cn",
            "base_ref": "base123",
            "head_ref": "head123",
            "changed_files": [],
            "repo_config": repo_config,
            "source_description": "PR #1 commit range base123..head123: Update",
            "pr_number": 1,
            "title": "Update",
        }
        fake_github_client = object()

        with mock.patch.object(main_workflow, "SOURCE_PR_URL", "https://github.com/acme/docs/pull/1"), mock.patch.object(
            main_workflow, "TARGET_PR_URL", "https://github.com/acme/docs-cn/pull/2"
        ), mock.patch.object(main_workflow, "GITHUB_TOKEN", "token"), mock.patch.object(
            main_workflow, "TARGET_REPO_PATH", "/tmp/target"
        ), mock.patch.object(main_workflow, "clean_temp_output_dir"), mock.patch.object(
            main_workflow, "get_workflow_repo_configs", return_value={"acme/docs": repo_config}
        ), mock.patch.object(
            main_workflow, "get_workflow_repo_config", return_value=repo_config
        ), mock.patch.object(
            main_workflow.Auth, "Token", return_value="token-auth"
        ), mock.patch.object(
            main_workflow, "Github", return_value=fake_github_client
        ), mock.patch.object(
            main_workflow, "build_pr_diff_context", return_value=source_context
        ), mock.patch.object(
            main_workflow, "UnifiedAIClient", return_value=SimpleNamespace(model="fake-model")
        ), mock.patch.object(
            main_workflow, "load_glossary", return_value=[]
        ), mock.patch.object(
            main_workflow, "create_glossary_matcher", return_value=None
        ), mock.patch.object(
            main_workflow,
            "analyze_source_changes",
            return_value=({}, {}, {}, {}, [], {}, {}, ["docs/media/example.png"], [], [], set(), {}, {}),
        ) as analyze_source_changes, mock.patch.object(
            main_workflow, "process_all_images"
        ) as process_all_images, mock.patch.object(
            main_workflow, "git_add_changes"
        ):
            main_workflow.main()

        self.assertIs(analyze_source_changes.call_args.args[0], source_context)
        process_all_images.assert_called_once_with(
            ["docs/media/example.png"],
            [],
            [],
            source_context,
            fake_github_client,
            repo_config,
        )


class MainWorkflowRegressionTest(unittest.TestCase):
    def test_restructured_pr_file_uses_reconciler_instead_of_full_overwrite(self):
        added_files = {"guide.md": "# Guide\n\n## B\n\nNew\n"}
        restructured_files = {"guide.md"}
        run_report = main_workflow.RunReport()

        with tempfile.TemporaryDirectory() as tmpdir:
            target_file = Path(tmpdir) / "guide.md"
            target_file.write_text("# 指南\n\n## A\n\n旧译文\n", encoding="utf-8")

            def fake_source(file_path, _context, _github_client, ref_name="head_ref"):
                self.assertEqual(file_path, "guide.md")
                if ref_name == "head_ref":
                    return "# Guide\n\n## B\n\nNew\n"
                if ref_name == "base_ref":
                    return "# Guide\n\n## A\n\nOld\n"
                raise AssertionError(ref_name)

            with mock.patch.object(main_workflow, "TARGET_REPO_PATH", tmpdir), \
                mock.patch.object(main_workflow, "get_source_file_content", side_effect=fake_source), \
                mock.patch.object(main_workflow, "reconcile_restructured_file", return_value="# 指南\n\n## B\n\n新译文\n") as reconcile, \
                mock.patch.object(main_workflow, "git_add_changes") as git_add:
                reconciled = main_workflow.reconcile_restructured_files_for_pr(
                    restructured_files,
                    added_files,
                    {"mode": "pr"},
                    object(),
                    object(),
                    {"source_language": "Chinese", "target_language": "English"},
                    None,
                    run_report,
                )
            final_content = target_file.read_text(encoding="utf-8")

        self.assertEqual({"guide.md"}, reconciled)
        self.assertEqual({}, added_files)
        self.assertEqual(set(), restructured_files)
        self.assertEqual(["guide.md"], run_report.succeeded)
        self.assertEqual("# 指南\n\n## B\n\n新译文\n", final_content)
        reconcile.assert_called_once()
        git_add.assert_called_once_with(tmpdir)

    def test_restructured_pr_reconciliation_failure_does_not_fall_back_to_overwrite(self):
        added_files = {"guide.md": "# Guide\n\n## B\n\nNew\n"}
        restructured_files = {"guide.md"}
        run_report = main_workflow.RunReport()

        with tempfile.TemporaryDirectory() as tmpdir:
            target_file = Path(tmpdir) / "guide.md"
            original = "# 指南\n\n## A\n\n旧译文\n"
            target_file.write_text(original, encoding="utf-8")

            with mock.patch.object(main_workflow, "TARGET_REPO_PATH", tmpdir), \
                mock.patch.object(main_workflow, "get_source_file_content", return_value="# Guide\n"), \
                mock.patch.object(main_workflow, "reconcile_restructured_file", return_value=None), \
                mock.patch.object(main_workflow, "git_add_changes") as git_add:
                reconciled = main_workflow.reconcile_restructured_files_for_pr(
                    restructured_files,
                    added_files,
                    {"mode": "pr"},
                    object(),
                    object(),
                    {"source_language": "Chinese", "target_language": "English"},
                    None,
                    run_report,
                )
            final_content = target_file.read_text(encoding="utf-8")

        self.assertEqual(set(), reconciled)
        self.assertEqual({}, added_files)
        self.assertEqual(set(), restructured_files)
        self.assertEqual(original, final_content)
        self.assertEqual(1, len(run_report.failed))
        self.assertIn("declined", run_report.failed[0][1])
        git_add.assert_not_called()

    def test_restructured_pr_reports_plural_missing_source_snapshots(self):
        added_files = {"guide.md": "# Guide\n"}
        restructured_files = {"guide.md"}
        run_report = main_workflow.RunReport()

        with tempfile.TemporaryDirectory() as tmpdir:
            target_file = Path(tmpdir) / "guide.md"
            target_file.write_text("# 指南\n", encoding="utf-8")

            with mock.patch.object(main_workflow, "TARGET_REPO_PATH", tmpdir), \
                mock.patch.object(main_workflow, "get_source_file_content", return_value=None), \
                mock.patch.object(main_workflow, "reconcile_restructured_file") as reconcile:
                main_workflow.reconcile_restructured_files_for_pr(
                    restructured_files,
                    added_files,
                    {"mode": "pr"},
                    object(),
                    object(),
                    {"source_language": "Chinese", "target_language": "English"},
                    None,
                    run_report,
                )

        self.assertEqual(1, len(run_report.failed))
        self.assertIn(
            "HEAD and BASE source snapshots are unavailable",
            run_report.failed[0][1],
        )
        reconcile.assert_not_called()

    def test_workflow_repo_configs_accepts_pr_files_range_source_url(self):
        with mock.patch.object(
            main_workflow,
            "SOURCE_PR_URL",
            "https://github.com/acme/docs/pull/1/files/base123..head123?plain=1",
        ), mock.patch.object(
            main_workflow, "TARGET_PR_URL", "https://github.com/acme/docs-cn/pull/2"
        ), mock.patch.object(
            main_workflow, "TARGET_REPO_PATH", "/tmp/target"
        ), mock.patch.object(
            main_workflow, "PREFER_LOCAL_TARGET_FOR_READ", True
        ):
            configs = main_workflow.get_workflow_repo_configs()

        self.assertEqual({"acme/docs"}, set(configs.keys()))
        self.assertEqual(configs["acme/docs"]["target_repo"], "acme/docs-cn")
        self.assertTrue(configs["acme/docs"]["prefer_local_target_for_read"])

    def test_filter_docs_by_source_files_keeps_only_requested_paths(self):
        filtered = main_workflow.filter_docs_by_source_files(
            "guide.md, TOC-test.md",
            {"guide.md": {"sections": {}}},
            {"ignore.md": {"sections": {}}},
            {"guide.md": {"sections": {}}},
            {"add.md": "# Added"},
            ["ignore-delete.md", "guide.md"],
            {"TOC-test.md": {"type": "toc"}},
            {"keywords.md": {"type": "keyword"}},
            ["guide.png"],
            ["ignore-image.png"],
            ["TOC-test.md"],
        )

        self.assertEqual({"guide.md"}, set(filtered[0].keys()))
        self.assertEqual({}, filtered[1])
        self.assertEqual({"guide.md"}, set(filtered[2].keys()))
        self.assertEqual({}, filtered[3])
        self.assertEqual(["guide.md"], filtered[4])
        self.assertEqual({"TOC-test.md"}, set(filtered[5].keys()))
        self.assertEqual({}, filtered[6])
        self.assertEqual([], filtered[7])
        self.assertEqual([], filtered[8])
        self.assertEqual(["TOC-test.md"], filtered[9])

    def test_unmatched_modified_sections_are_formatted_for_failure_report(self):
        source_diff_dict = {
            "modified_1102": {
                "operation": "modified",
                "new_line_number": 1102,
                "new_content": "### tidb_analyze_column_options\n\nNew content\n",
            },
            "added_1200": {
                "operation": "added",
                "new_line_number": 1200,
                "new_content": "### new_section\n",
            },
        }

        missing = main_workflow.get_unmatched_modified_source_sections(
            source_diff_dict,
            {"added_1200": {}},
        )
        reason = main_workflow.format_unmatched_modified_sections_failure(
            "system-variables.md",
            missing,
        )

        self.assertEqual(len(missing), 1)
        self.assertEqual(missing[0]["key"], "modified_1102")
        self.assertEqual(missing[0]["title"], "tidb_analyze_column_options")
        self.assertIn("system-variables.md", reason)
        self.assertIn("modified_1102 (source line 1102): tidb_analyze_column_options", reason)
        self.assertIn("translation was skipped", reason)

    def test_regular_modified_file_skips_translation_when_modified_section_unmatched(self):
        source_diff_dict = {
            "modified_10": {
                "operation": "modified",
                "new_line_number": 10,
                "new_content": "### missing_source_section\n\nNew content\n",
                "old_content": "### missing_source_section\n\nOld content\n",
                "original_hierarchy": "## Variable reference > ### missing_source_section",
            },
            "modified_20": {
                "operation": "modified",
                "new_line_number": 20,
                "new_content": "### matched_source_section\n\nNew content\n",
                "old_content": "### matched_source_section\n\nOld content\n",
                "original_hierarchy": "## Variable reference > ### matched_source_section",
            },
        }
        matched_sections = {
            "modified_20": {
                "target_line": "20",
                "target_hierarchy": "## matched_source_section",
                "target_content": "### matched_source_section\n\n旧内容\n",
                "source_operation": "modified",
                "source_old_content": "### matched_source_section\n\nOld content\n",
                "source_new_content": "### matched_source_section\n\nNew content\n",
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_dir = Path(tmpdir)
            source_diff_file = temp_dir / "system-variables-source-diff-dict.json"
            source_diff_file.write_text(
                json.dumps(source_diff_dict),
                encoding="utf-8",
            )

            with mock.patch.object(main_workflow, "ensure_temp_output_dir", return_value=tmpdir), \
                mock.patch.object(main_workflow, "check_source_token_limit", return_value=(True, 10, 50000)), \
                mock.patch("diff_analyzer.get_target_hierarchy_and_content", return_value=({"20": "## matched_source_section"}, ["### matched_source_section"])), \
                mock.patch("section_matcher.match_source_diff_to_target", return_value=matched_sections), \
                mock.patch.object(main_workflow, "process_modified_sections") as process_modified_sections:
                success, reason = main_workflow.process_regular_modified_file(
                    "system-variables.md",
                    {"sections": {"10": "## missing_source_section", "20": "## matched_source_section"}},
                    "File: system-variables.md\n@@ -10,1 +10,1 @@\n-old\n+new",
                    {"mode": "commit"},
                    object(),
                    object(),
                    {
                        "target_repo": "pingcap/docs",
                        "target_local_path": tmpdir,
                        "prefer_local_target_for_read": True,
                        "source_language": "English",
                        "target_language": "Chinese",
                    },
                    120,
                    return_details=True,
                )

        self.assertFalse(success)
        self.assertIn("modified_10", reason)
        self.assertIn("missing_source_section", reason)
        process_modified_sections.assert_not_called()

    def test_regular_modified_file_applies_heading_level_only_without_ai_on_target_line_offset(self):
        source_diff_dict = {
            "modified_20": {
                "operation": "modified",
                "new_line_number": 20,
                "new_content": "### SQL statement\n\nNew content\n",
                "old_content": "## SQL statement\n\nNew content\n",
                "original_hierarchy": "## SQL statement",
                "heading_level_change_only": True,
                "old_heading_level": 2,
                "new_heading_level": 3,
            },
        }
        matched_sections = {
            "modified_20": {
                "target_line": "5",
                "target_hierarchy": "## Statements > ### SQL 语句",
                "target_content": "### SQL 语句\n\n旧内容\n",
                "source_operation": "modified",
                "source_old_content": "## SQL statement\n\nNew content\n",
                "source_new_content": "### SQL statement\n\nNew content\n",
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_dir = Path(tmpdir)
            source_diff_file = temp_dir / "example-source-diff-dict.json"
            source_diff_file.write_text(
                json.dumps(source_diff_dict),
                encoding="utf-8",
            )

            def assert_match_file(match_file, target_local_path, target_file_name=None):
                self.assertEqual(target_file_name, "example.md")
                match_data = json.loads(Path(match_file).read_text(encoding="utf-8"))
                self.assertEqual(
                    match_data["modified_20"]["target_new_content"],
                    "### SQL 语句\n\n旧内容\n",
                )
                self.assertEqual(match_data["modified_20"]["target_line"], "5")
                return True

            with mock.patch.object(main_workflow, "ensure_temp_output_dir", return_value=tmpdir), \
                mock.patch.object(main_workflow, "check_source_token_limit", return_value=(True, 10, 50000)), \
                mock.patch("diff_analyzer.get_target_hierarchy_and_content", return_value=({"5": "## Statements > ### SQL 语句"}, ["# SQL", "", "## Statements", "", "### SQL 语句"])), \
                mock.patch("section_matcher.match_source_diff_to_target", return_value=matched_sections), \
                mock.patch.object(main_workflow, "process_modified_sections") as process_modified_sections, \
                mock.patch("file_updater.update_target_document_from_match_data", side_effect=assert_match_file):
                success, reason = main_workflow.process_regular_modified_file(
                    "example.md",
                    {"sections": {"20": "### SQL statement"}},
                    "File: example.md\n@@ -20,1 +20,1 @@\n-## SQL statement\n+### SQL statement",
                    {"mode": "commit"},
                    object(),
                    object(),
                    {
                        "target_repo": "pingcap/docs",
                        "target_local_path": tmpdir,
                        "prefer_local_target_for_read": True,
                        "source_language": "English",
                        "target_language": "Chinese",
                    },
                    120,
                    return_details=True,
                )

        self.assertTrue(success)
        self.assertFalse(reason)
        process_modified_sections.assert_not_called()

    def test_regular_modified_file_applies_version_mark_only_without_ai(self):
        old_source = (
            "### `tidb_opt_partial_ordered_index_for_topn` "
            "<span class=\"version-mark\">New in v8.5.6</span>\n\n"
            "The body is unchanged.\n"
        )
        new_source = old_source.replace("v8.5.6", "v8.5.7")
        target_content = (
            "### `tidb_opt_partial_ordered_index_for_topn` "
            "<span class=\"version-mark\">从 v8.5.6 开始引入</span> "
            "{#stable-anchor}\n\n"
            "正文保持不变。\n"
        )
        source_diff_dict = {
            "modified_20": {
                "operation": "modified",
                "new_line_number": 20,
                "new_content": new_source,
                "old_content": old_source,
                "original_hierarchy": (
                    "## Variable reference > ### "
                    "`tidb_opt_partial_ordered_index_for_topn` "
                    "<span class=\"version-mark\">New in v8.5.6</span>"
                ),
            },
        }
        matched_sections = {
            "modified_20": {
                "target_line": "5",
                "target_hierarchy": "## 变量参考 > ### `tidb_opt_partial_ordered_index_for_topn`",
                "target_content": target_content,
                "source_operation": "modified",
                "source_old_content": old_source,
                "source_new_content": new_source,
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            source_diff_file = Path(tmpdir) / "system-variables-source-diff-dict.json"
            source_diff_file.write_text(
                json.dumps(source_diff_dict),
                encoding="utf-8",
            )

            def assert_match_file(match_file, target_local_path, target_file_name=None):
                match_data = json.loads(Path(match_file).read_text(encoding="utf-8"))
                updated = match_data["modified_20"]["target_new_content"]
                self.assertEqual(
                    updated,
                    target_content.replace("v8.5.6", "v8.5.7"),
                )
                self.assertIn("{#stable-anchor}", updated)
                self.assertIn("正文保持不变。", updated)
                return True

            with mock.patch.object(main_workflow, "ensure_temp_output_dir", return_value=tmpdir), \
                mock.patch.object(main_workflow, "check_source_token_limit", return_value=(True, 10, 50000)), \
                mock.patch("diff_analyzer.get_target_hierarchy_and_content", return_value=({"5": "## 变量参考 > ### variable"}, ["# 系统变量", "", "## 变量参考", "", "### variable"])), \
                mock.patch("section_matcher.match_source_diff_to_target", return_value=matched_sections), \
                mock.patch.object(main_workflow, "process_modified_sections") as process_modified_sections, \
                mock.patch("file_updater.update_target_document_from_match_data", side_effect=assert_match_file):
                success, reason = main_workflow.process_regular_modified_file(
                    "system-variables.md",
                    {"sections": {"20": "### variable"}},
                    (
                        "File: system-variables.md\n@@ -20,1 +20,1 @@\n"
                        "-<span>New in v8.5.6</span>\n"
                        "+<span>New in v8.5.7</span>"
                    ),
                    {"mode": "commit"},
                    object(),
                    object(),
                    {
                        "target_repo": "pingcap/docs",
                        "target_local_path": tmpdir,
                        "prefer_local_target_for_read": True,
                        "source_language": "English",
                        "target_language": "Chinese",
                    },
                    120,
                    return_details=True,
                )

        self.assertTrue(success)
        self.assertFalse(reason)
        process_modified_sections.assert_not_called()


if __name__ == "__main__":
    unittest.main()

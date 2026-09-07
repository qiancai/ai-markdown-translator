import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from resolve_cloud_source_files import (
    build_allowed_files,
    collect_new_toc_full_translation_files,
    collect_toc_scope_added_files,
    extract_markdown_doc_links,
    parse_git_name_status,
    resolve_source_files,
)


class ResolveCloudSourceFilesTest(unittest.TestCase):
    def test_new_tocs_deduplicate_links_and_only_bootstrap_new_scope(self):
        with tempfile.TemporaryDirectory() as source_tmpdir, tempfile.TemporaryDirectory() as target_tmpdir:
            source_root = Path(source_tmpdir)
            target_root = Path(target_tmpdir)
            subprocess.run(
                ["git", "init"],
                cwd=source_root,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=source_root, check=True)
            subprocess.run(["git", "config", "user.name", "Test User"], cwd=source_root, check=True)

            files = {
                "TOC-existing.md": "- [Shared](/shared.md)\n",
                "TOC-new-a.md": "- [Shared](/shared.md)\n- [A](/new-a.md)\n- [Common](/new-common.md)\n- [Head overlap](/head-overlap.md)\n",
                "TOC-new-b.md": "- [B](/new-b.md)\n- [Common](/new-common.md)\n",
                "shared.md": "# Shared\n",
                "new-a.md": "# A\n",
                "new-b.md": "# B\n",
                "new-common.md": "# Common\n",
                "head-overlap.md": "# Head overlap\n",
                "new/_index.md": "---\ntitle: New\n---\n",
            }
            for file_path, content in files.items():
                path = source_root / file_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
            subprocess.run(["git", "add", "."], cwd=source_root, check=True)
            subprocess.run(
                ["git", "commit", "-m", "source"],
                cwd=source_root,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            base_ref = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=source_root, text=True
            ).strip()
            (source_root / "TOC-existing.md").write_text(
                "- [Shared](/shared.md)\n- [Head overlap](/head-overlap.md)\n",
                encoding="utf-8",
            )
            subprocess.run(
                ["git", "commit", "-am", "existing TOC adds a link"],
                cwd=source_root,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            head_ref = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=source_root, text=True
            ).strip()

            (target_root / "TOC-existing.md").write_text(
                "- [共享](/shared.md)\n", encoding="utf-8"
            )

            full_files, new_tocs = collect_new_toc_full_translation_files(
                source_root,
                target_root,
                ["TOC-existing.md", "TOC-new-a.md", "TOC-new-b.md"],
                base_ref,
                head_ref,
                extra_files=["new/_index.md"],
            )

        self.assertEqual(new_tocs, {"TOC-new-a.md", "TOC-new-b.md"})
        self.assertEqual(
            full_files,
            {
                "TOC-new-a.md",
                "TOC-new-b.md",
                "new-a.md",
                "new-b.md",
                "new-common.md",
                "head-overlap.md",
                "new/_index.md",
            },
        )
        self.assertNotIn("shared.md", full_files)

    def test_existing_target_toc_does_not_trigger_bootstrap(self):
        with tempfile.TemporaryDirectory() as source_tmpdir, tempfile.TemporaryDirectory() as target_tmpdir:
            source_root = Path(source_tmpdir)
            target_root = Path(target_tmpdir)
            subprocess.run(
                ["git", "init"],
                cwd=source_root,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=source_root, check=True)
            subprocess.run(["git", "config", "user.name", "Test User"], cwd=source_root, check=True)
            (source_root / "TOC-cloud.md").write_text(
                "- [Guide](/guide.md)\n", encoding="utf-8"
            )
            (source_root / "guide.md").write_text("# Guide\n", encoding="utf-8")
            subprocess.run(["git", "add", "."], cwd=source_root, check=True)
            subprocess.run(
                ["git", "commit", "-m", "source"],
                cwd=source_root,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            source_ref = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=source_root, text=True
            ).strip()
            (target_root / "TOC-cloud.md").write_text(
                "- [指南](/guide.md)\n", encoding="utf-8"
            )

            full_files, new_tocs = collect_new_toc_full_translation_files(
                source_root,
                target_root,
                ["TOC-cloud.md"],
                source_ref,
                source_ref,
            )

        self.assertEqual(new_tocs, set())
        self.assertEqual(full_files, set())

    def test_build_allowed_files_keeps_links_with_anchors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "TOC-tidb-cloud.md").write_text(
                "- [Billing](/tidb-cloud/tidb-cloud-billing.md#invoices)\n",
                encoding="utf-8",
            )

            allowed = build_allowed_files(root, ["TOC-tidb-cloud.md"])

        self.assertIn("TOC-tidb-cloud.md", allowed)
        self.assertIn("tidb-cloud/tidb-cloud-billing.md", allowed)

    def test_build_allowed_files_includes_cloud_index_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "TOC-tidb-cloud.md").write_text(
                "- [Billing](/tidb-cloud/tidb-cloud-billing.md)\n",
                encoding="utf-8",
            )

            allowed = build_allowed_files(
                root,
                ["TOC-tidb-cloud.md"],
                extra_files=[
                    "tidb-cloud/essential/_index.md",
                    "/docs/tidb-cloud/starter/_index.md",
                ],
            )

        self.assertIn("tidb-cloud/essential/_index.md", allowed)
        self.assertIn("tidb-cloud/starter/_index.md", allowed)

    def test_extract_markdown_doc_links_uses_markdown_parser_edge_cases(self):
        links = extract_markdown_doc_links(
            """
- [![img](x.png)](/tidb-cloud/nested.md)
- [Reference][ref]

```md
[Fake](/tidb-cloud/fake.md)
```

[ref]: /tidb-cloud/reference.md#anchor
"""
        )

        self.assertEqual(
            links,
            [
                "tidb-cloud/nested.md",
                "tidb-cloud/reference.md",
            ],
        )

    def test_extract_markdown_doc_links_ignores_unsupported_mdx_files(self):
        links = extract_markdown_doc_links(
            "- [Markdown](/guide.md)\n- [MDX](/unsupported.mdx)\n"
        )

        self.assertEqual(links, ["guide.md"])

    def test_manual_file_names_must_be_in_cloud_scope(self):
        allowed = {"TOC-tidb-cloud.md", "tidb-cloud/in-scope.md"}

        with self.assertRaises(ValueError):
            resolve_source_files(allowed, input_file_names="not-cloud.md")

    def test_manual_basename_resolves_when_unique(self):
        allowed = {"TOC-tidb-cloud.md", "tidb-cloud/in-scope.md"}

        resolved = resolve_source_files(allowed, input_file_names="in-scope.md")

        self.assertEqual(resolved, ["tidb-cloud/in-scope.md"])

    def test_auto_mode_intersects_changed_files_with_allowed_files(self):
        allowed = {"TOC-tidb-cloud.md", "tidb-cloud/in-scope.md"}
        changed_rows = [
            {"filename": "tidb-cloud/in-scope.md", "previous_filename": ""},
            {"filename": "other.md", "previous_filename": ""},
        ]

        resolved = resolve_source_files(allowed, changed_rows=changed_rows)

        self.assertEqual(resolved, ["tidb-cloud/in-scope.md"])

    def test_auto_mode_includes_non_cloud_path_when_linked_from_cloud_toc(self):
        allowed = {"TOC-tidb-cloud.md", "shared/in-scope.md"}
        changed_rows = [
            {"filename": "shared/in-scope.md", "previous_filename": ""},
            {"filename": "shared/not-in-scope.md", "previous_filename": ""},
        ]

        resolved = resolve_source_files(allowed, changed_rows=changed_rows)

        self.assertEqual(resolved, ["shared/in-scope.md"])

    def test_auto_mode_includes_toc_scope_added_file_even_when_file_not_changed(self):
        allowed = {"TOC-tidb-cloud.md"}
        changed_rows = [
            {"filename": "TOC-tidb-cloud.md", "previous_filename": ""},
        ]

        resolved = resolve_source_files(
            allowed,
            changed_rows=changed_rows,
            always_include_files={"shared/newly-linked.md"},
        )

        self.assertEqual(resolved, ["TOC-tidb-cloud.md", "shared/newly-linked.md"])

    def test_collect_toc_scope_added_files_uses_base_and_head_tocs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            subprocess.run(
                ["git", "init"],
                cwd=root,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.name", "Test User"], cwd=root, check=True)
            (root / "TOC-tidb-cloud.md").write_text(
                "- [Old](/tidb-cloud/old.md)\n",
                encoding="utf-8",
            )
            (root / "TOC-tidb-cloud-starter.md").write_text(
                "- [Shared](/shared/already-linked.md)\n",
                encoding="utf-8",
            )
            subprocess.run(
                ["git", "add", "TOC-tidb-cloud.md", "TOC-tidb-cloud-starter.md"],
                cwd=root,
                check=True,
            )
            subprocess.run(
                ["git", "commit", "-m", "base"],
                cwd=root,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            base_ref = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
            (root / "TOC-tidb-cloud.md").write_text(
                "\n".join(
                    [
                        "- [Old](/tidb-cloud/old.md)",
                        "- [Shared](/shared/already-linked.md)",
                        "- [New](/shared/newly-linked.md)",
                    ]
                ),
                encoding="utf-8",
            )
            subprocess.run(
                ["git", "commit", "-am", "head"],
                cwd=root,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            head_ref = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()

            scope_added = collect_toc_scope_added_files(
                root,
                ["TOC-tidb-cloud.md", "TOC-tidb-cloud-starter.md"],
                base_ref,
                head_ref,
            )

        self.assertEqual(scope_added, {"shared/newly-linked.md"})

    def test_auto_mode_includes_changed_cloud_index_file_when_allowed(self):
        allowed = {"TOC-tidb-cloud.md", "tidb-cloud/essential/_index.md"}
        changed_rows = [
            {"filename": "tidb-cloud/essential/_index.md", "previous_filename": ""},
            {"filename": "tidb-cloud/not-index.md", "previous_filename": ""},
        ]

        resolved = resolve_source_files(allowed, changed_rows=changed_rows)

        self.assertEqual(resolved, ["tidb-cloud/essential/_index.md"])

    def test_manual_cloud_index_file_is_allowed_when_configured(self):
        allowed = {"TOC-tidb-cloud.md", "tidb-cloud/essential/_index.md"}

        resolved = resolve_source_files(
            allowed,
            input_file_names="tidb-cloud/essential/_index.md",
        )

        self.assertEqual(resolved, ["tidb-cloud/essential/_index.md"])

    def test_renamed_file_can_match_previous_allowed_path(self):
        allowed = {"tidb-cloud/old.md"}
        changed_rows = [
            {"filename": "tidb-cloud/new-location.md", "previous_filename": "tidb-cloud/old.md"},
        ]

        resolved = resolve_source_files(allowed, changed_rows=changed_rows)

        self.assertEqual(resolved, ["tidb-cloud/old.md"])

    def test_parse_git_name_status_handles_renames(self):
        rows = parse_git_name_status("R100\ttidb-cloud/old.md\ttidb-cloud/new.md\n")

        self.assertEqual(
            rows,
            [{"filename": "tidb-cloud/new.md", "previous_filename": "tidb-cloud/old.md"}],
        )


if __name__ == "__main__":
    unittest.main()

import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
REPO_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from log_sanitizer import safe_target_path
from special_file_utils import path_resource_key
from workflow_outcome import FileOutcomes
from file_io import atomic_write_bytes, atomic_write_text


class SafeTargetPathTest(unittest.TestCase):
    def test_rejects_absolute_traversal_and_git_metadata_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for unsafe in ("/tmp/escaped.md", "../escaped.md", ".git/config"):
                with self.subTest(unsafe=unsafe):
                    with self.assertRaises(ValueError):
                        safe_target_path(tmpdir, unsafe)

    def test_accepts_normal_repository_relative_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertEqual(
                safe_target_path(tmpdir, "docs/guide.md"),
                str(Path(tmpdir, "docs", "guide.md").resolve()),
            )


class PathResourceKeyTest(unittest.TestCase):
    def test_slash_and_literal_double_dash_paths_do_not_collide(self):
        nested = path_resource_key("a/b.md")
        literal = path_resource_key("a--b.md")

        self.assertNotEqual(nested, literal)
        self.assertEqual(nested, "a--b")
        self.assertEqual(literal, "a%2D%2Db")

    def test_percent_escaping_is_unambiguous(self):
        self.assertNotEqual(
            path_resource_key("a%2D%2Db.md"),
            path_resource_key("a--b.md"),
        )


class FileOutcomesTest(unittest.TestCase):
    def test_keeps_normal_mapping_truthiness_and_exposes_success_explicitly(self):
        outcomes = FileOutcomes()
        self.assertFalse(outcomes)
        self.assertTrue(outcomes.all_succeeded)

        outcomes.add("guide.md", "failed", "network error")
        self.assertTrue(outcomes)
        self.assertFalse(outcomes.all_succeeded)


class AtomicFileWriteTest(unittest.TestCase):
    def test_text_replace_preserves_existing_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir, "guide.md")
            target.write_text("old", encoding="utf-8")
            os.chmod(target, 0o640)

            atomic_write_text(str(target), "new\n")

            self.assertEqual(target.read_text(encoding="utf-8"), "new\n")
            self.assertEqual(target.stat().st_mode & 0o777, 0o640)
            self.assertEqual(list(Path(tmpdir).glob(".guide.md.*.tmp")), [])

    def test_binary_replace_writes_exact_bytes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir, "image.png")
            atomic_write_bytes(str(target), b"\x00\x01\xff")
            self.assertEqual(target.read_bytes(), b"\x00\x01\xff")

    def test_failed_replace_keeps_original_and_removes_temporary_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir, "guide.md")
            target.write_text("old", encoding="utf-8")

            with mock.patch("file_io.os.replace", side_effect=OSError("busy")):
                with self.assertRaisesRegex(OSError, "busy"):
                    atomic_write_text(str(target), "new")

            self.assertEqual(target.read_text(encoding="utf-8"), "old")
            self.assertEqual(list(Path(tmpdir).glob(".guide.md.*.tmp")), [])

    def test_fdopen_failure_closes_descriptor_and_removes_temporary_file(self):
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch(
            "file_io.os.fdopen", side_effect=OSError("fdopen failed")
        ), mock.patch("file_io.os.close", wraps=os.close) as close:
            target = Path(tmpdir, "guide.md")

            with self.assertRaisesRegex(OSError, "fdopen failed"):
                atomic_write_text(str(target), "new")

            close.assert_called_once()
            self.assertFalse(target.exists())
            self.assertEqual(list(Path(tmpdir).glob(".guide.md.*.tmp")), [])


class WorkflowActionPinTest(unittest.TestCase):
    def test_setup_python_v6_uses_verified_40_character_commit(self):
        expected = "ece7cb06caefa5fff74198d8649806c4678c61a1"
        for relative_path in (
            "sync-doc-pr-zh-to-en.yml",
            "sync-doc-updates-zh-to-en.yml",
        ):
            with self.subTest(path=relative_path):
                content = Path(REPO_ROOT, relative_path).read_text(encoding="utf-8")
                matches = re.findall(r"actions/setup-python@([0-9a-f]+)", content)
                self.assertTrue(matches)
                self.assertEqual(set(matches), {expected})
                self.assertEqual(len(expected), 40)


class WorkflowProviderSecretScopingTest(unittest.TestCase):
    def test_pr_workflow_only_populates_the_selected_provider_secrets(self):
        content = Path(
            REPO_ROOT, "sync-doc-pr-zh-to-en.yml"
        ).read_text(encoding="utf-8")

        expected_bindings = (
            "DEEPSEEK_API_TOKEN: ${{ github.event.inputs.ai_provider == "
            "'deepseek' && secrets.DEEPSEEK_API_TOKEN || '' }}",
            "GEMINI_API_TOKEN: ${{ github.event.inputs.ai_provider == "
            "'gemini' && secrets.GEMINI_API_TOKEN || '' }}",
            "AZURE_OPENAI_KEY: ${{ github.event.inputs.ai_provider == "
            "'azure' && secrets.AZURE_OPENAI_KEY || '' }}",
            "OPENAI_BASE_URL: ${{ github.event.inputs.ai_provider == "
            "'azure' && secrets.AZURE_OPENAI_BASE_URL || '' }}",
        )

        for binding in expected_bindings:
            with self.subTest(binding=binding):
                self.assertIn(binding, content)

        unconditional_bindings = (
            "DEEPSEEK_API_TOKEN: ${{ secrets.DEEPSEEK_API_TOKEN }}",
            "GEMINI_API_TOKEN: ${{ secrets.GEMINI_API_TOKEN }}",
            "AZURE_OPENAI_KEY: ${{ secrets.AZURE_OPENAI_KEY }}",
            "OPENAI_BASE_URL: ${{ secrets.AZURE_OPENAI_BASE_URL }}",
        )
        for binding in unconditional_bindings:
            with self.subTest(binding=binding):
                self.assertNotIn(binding, content)

    def test_fixed_azure_workflow_does_not_inject_other_provider_secrets(self):
        content = Path(
            REPO_ROOT, "sync-doc-updates-zh-to-en.yml"
        ).read_text(encoding="utf-8")

        self.assertIn(
            "AZURE_OPENAI_KEY: ${{ secrets.AZURE_OPENAI_KEY }}",
            content,
        )
        self.assertIn(
            "OPENAI_BASE_URL: ${{ secrets.AZURE_OPENAI_BASE_URL }}",
            content,
        )
        self.assertNotIn("DEEPSEEK_API_TOKEN:", content)
        self.assertNotIn("GEMINI_API_TOKEN:", content)


if __name__ == "__main__":
    unittest.main()

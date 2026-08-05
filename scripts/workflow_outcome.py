"""Shared file-level outcome and run-report primitives."""

import json
import os


VALID_OUTCOME_STATUSES = {"success", "partial", "failed", "skipped"}


def file_outcome(status, reason=""):
    if status not in VALID_OUTCOME_STATUSES:
        raise ValueError(f"Unsupported file outcome status: {status}")
    return {"status": status, "reason": reason or ""}


class FileOutcomes(dict):
    """Mapping of file paths to status/reason dictionaries."""

    def add(self, file_path, status, reason=""):
        self[file_path] = file_outcome(status, reason)

    @property
    def all_succeeded(self):
        return all(
            outcome.get("status") in {"success", "skipped"}
            for outcome in self.values()
        )

class RunReport:
    """Track complete, partial, failed, skipped, and structural outcomes."""

    def __init__(self):
        self.total = 0
        self.succeeded = []
        self.partial = []
        self.failed = []
        self.skipped = []
        self.structure_errors = []
        self.retry_refs = {}

    def mark_success(self, file_path):
        self.total += 1
        self.succeeded.append(file_path)

    def mark_partial(self, file_path, reason):
        self.total += 1
        self.partial.append((file_path, reason))

    def mark_failure(self, file_path, reason):
        self.total += 1
        self.failed.append((file_path, reason))

    def mark_skipped(self, file_path, reason):
        self.skipped.append((file_path, reason))

    def mark_structure_error(self, issue):
        self.structure_errors.append(issue)

    def set_retry_ref(self, file_path, source_ref):
        if file_path and source_ref:
            self.retry_refs[file_path] = source_ref

    def set_retry_refs_for_new_issues(self, partial_start, failed_start, source_ref):
        for file_path, _ in self.partial[partial_start:]:
            self.set_retry_ref(file_path, source_ref)
        for file_path, _ in self.failed[failed_start:]:
            self.set_retry_ref(file_path, source_ref)

    def record_outcomes(self, outcomes):
        for file_path, outcome in (outcomes or {}).items():
            status = outcome.get("status", "failed")
            reason = outcome.get("reason", "")
            if status == "success":
                self.mark_success(file_path)
            elif status == "partial":
                self.mark_partial(file_path, reason)
            elif status == "skipped":
                self.mark_skipped(file_path, reason)
            else:
                self.mark_failure(file_path, reason)

    def print_summary(self, printer=print):
        printer("\n📚 Translation attempt summary:")
        printer(f"   📄 Files attempted for translation: {self.total}")
        printer(f"   ✅ Successfully translated: {len(self.succeeded)}")
        printer(f"   ⚠️  Partially translated: {len(self.partial)}")
        printer(f"   ❌ Failed to translate: {len(self.failed)}")
        printer(f"   ⚠️  Document structure mismatches: {len(self.structure_errors)}")

        if self.partial:
            printer("   ⚠️  Partial files:")
            for file_path, reason in self.partial:
                printer(f"      - {file_path}: {reason}")
        if self.failed:
            printer("   ❌ Failed files:")
            for file_path, reason in self.failed:
                printer(f"      - {file_path}: {reason}")
        if self.skipped:
            printer(f"   ⏭️  Skipped files: {len(self.skipped)}")
            for file_path, reason in self.skipped:
                printer(f"      - {file_path}: {reason}")

    def write_failure_report(self, output_dir):
        """Write incomplete work so partial pushes retain a follow-up queue."""
        os.makedirs(output_dir, exist_ok=True)
        markdown_path = os.path.join(output_dir, "translation-failures.md")
        json_path = os.path.join(output_dir, "translation-failures.json")
        structure_json_path = os.path.join(output_dir, "translation-structure-errors.json")

        if not self.partial and not self.failed and not self.structure_errors:
            for path in (markdown_path, json_path, structure_json_path):
                if os.path.exists(path):
                    os.remove(path)
            return

        with open(markdown_path, "w", encoding="utf-8") as f:
            if self.partial:
                f.write("### Partial or incomplete translations\n\n")
                f.write(
                    "The following files contain useful generated output, but still require review or retry.\n\n"
                )
                for file_path, reason in self.partial:
                    f.write(f"- `{file_path}`: {reason}\n")
                if self.failed or self.structure_errors:
                    f.write("\n")

            if self.failed:
                f.write("### Translation failures\n\n")
                f.write(
                    "The following files were not translated automatically and must be handled manually before merging.\n\n"
                )
                for file_path, reason in self.failed:
                    f.write(f"- `{file_path}`: {reason}\n")
                if self.structure_errors:
                    f.write("\n")

            if self.structure_errors:
                f.write("### Docs with document structure mismatches after translation\n\n")
                f.write(
                    "The following files were translated, but their Markdown heading or CustomContent structure does not match the source HEAD file.\n\n"
                )
                for issue in self.structure_errors:
                    f.write(f"- `{issue.file_path}`: {issue.reason}\n")
                    if getattr(issue, "source_compact", ""):
                        f.write(f"  - Source: `{issue.source_compact}`\n")
                    if getattr(issue, "target_compact", ""):
                        f.write(f"  - Target: `{issue.target_compact}`\n")
                    if getattr(issue, "first_difference", ""):
                        f.write(f"  - First difference: {issue.first_difference}\n")

        incomplete = [
            {"file_path": path, "status": status, "reason": reason}
            for status, entries in (("partial", self.partial), ("failed", self.failed))
            for path, reason in entries
        ]
        if incomplete:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(incomplete, f, ensure_ascii=False, indent=2)
                f.write("\n")
        elif os.path.exists(json_path):
            os.remove(json_path)

        if self.structure_errors:
            with open(structure_json_path, "w", encoding="utf-8") as f:
                json.dump(
                    [issue.to_dict() for issue in self.structure_errors],
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
                f.write("\n")
        elif os.path.exists(structure_json_path):
            os.remove(structure_json_path)


def record_file_task_result(task_result, run_report):
    """Record one ``run_file_tasks`` result and return its status and reason."""
    file_path = task_result["file_path"]
    if not task_result.get("ok"):
        status = "failure"
        reason = task_result.get("error") or "File task raised an exception"
    else:
        result = task_result.get("result") or {}
        status = result.get("status", "failure")
        reason = result.get("reason", "")

    if status == "success":
        run_report.mark_success(file_path)
    elif status == "partial":
        run_report.mark_partial(file_path, reason)
    elif status == "skipped":
        run_report.mark_skipped(file_path, reason)
    else:
        status = "failure"
        run_report.mark_failure(file_path, reason)

    return status, reason

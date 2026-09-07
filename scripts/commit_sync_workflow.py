"""
Main entry point for commit-diff-based sync in GitHub Actions.

This workflow keeps the existing PR-based pipeline untouched and adds a
parallel entrypoint for translation based on an explicit commit compare range.
"""

import json
import os
import re
import tempfile
from collections import Counter

from github import Auth, Github

SOURCE_REPO = os.getenv("SOURCE_REPO")
TARGET_REPO = os.getenv("TARGET_REPO")
SOURCE_BRANCH = os.getenv("SOURCE_BRANCH", "master")
SOURCE_BASE_REF = os.getenv("SOURCE_BASE_REF", "")
SOURCE_HEAD_REF = os.getenv("SOURCE_HEAD_REF", "")
SOURCE_FOLDER = os.getenv("SOURCE_FOLDER", "")
SOURCE_FILES = os.getenv("SOURCE_FILES", "")
SOURCE_FULL_FILES = os.getenv("SOURCE_FULL_FILES", "")
SOURCE_FILES_TRANSLATION_MODE = os.getenv("SOURCE_FILES_TRANSLATION_MODE", "incremental")
SOURCE_LANGUAGE_OVERRIDE = os.getenv("SOURCE_LANGUAGE", "")
TARGET_LANGUAGE_OVERRIDE = os.getenv("TARGET_LANGUAGE", "")
COMMIT_IGNORE_FOLDERS_OVERRIDE = os.getenv("COMMIT_IGNORE_FOLDERS", "")
IGNORE_RESOURCE_CARD_SECTION = (
    os.getenv("IGNORE_RESOURCE_CARD_SECTION")
    or os.getenv("ignore-resource-card-section")
    or "Yes"
)
CLOUD_TOC_FILES = os.getenv("TOC_FILES") or os.getenv("CLOUD_TOC_FILES", "")
SOURCE_REPO_PATH = os.getenv("SOURCE_REPO_PATH", "")
TARGET_REF = os.getenv("TARGET_REF", "")
PREFER_LOCAL_TARGET_FOR_READ = os.getenv("PREFER_LOCAL_TARGET_FOR_READ", "false").lower() == "true"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
AI_PROVIDER = os.getenv("AI_PROVIDER", "deepseek")
TARGET_REPO_PATH = os.getenv("TARGET_REPO_PATH")
TERMS_PATH = os.getenv("TERMS_PATH", "")
FAIL_ON_TRANSLATION_ERROR = os.getenv("FAIL_ON_TRANSLATION_ERROR", "false").lower() == "true"
COMMIT_SYNC_RUN_TYPE = os.getenv("COMMIT_SYNC_RUN_TYPE", os.getenv("GITHUB_EVENT_NAME", ""))
TIDB_CLOUD_ABSOLUTE_LINK_PREFIX = os.getenv(
    "TIDB_CLOUD_ABSOLUTE_LINK_PREFIX",
    "https://docs.pingcap.com/tidbcloud/",
)
os.environ.setdefault(
    "TIDB_CLOUD_ABSOLUTE_LINK_PREFIX",
    TIDB_CLOUD_ABSOLUTE_LINK_PREFIX,
)

from ai_client import UnifiedAIClient, thread_safe_print
from diff_analyzer import (
    build_commit_diff_context,
    build_local_commit_diff_context,
    build_diff_text,
    get_source_file_content,
    infer_language_direction,
    analyze_source_changes,
    filter_related_resources_resource_card_diff,
    is_yes_option_enabled,
    preserve_base_related_resources_resource_card_sections,
    remove_related_resources_resource_card_sections,
)
from file_adder import process_added_files
from file_updater import (
    extract_added_explicit_heading_anchors,
    preprocess_diff_for_heading_anchor_stability,
)
from structural_reconciler import reconcile_restructured_file
from file_deleter import process_deleted_files
from file_io import atomic_write_text
from formatting_sync import apply_formatting_only_change_with_ai_fallback
from glossary import create_glossary_matcher, load_glossary
from image_processor import process_all_images
from keword_processor import process_keyword_file
from log_sanitizer import sanitize_exception_message, safe_target_path
from parallel_file_processor import (
    make_file_task,
    make_task_result,
    run_file_tasks,
    should_parallelize_file_processing,
)
from resolve_cloud_source_files import extract_markdown_doc_links
from main_workflow import (
    MAX_NON_SYSTEM_SECTIONS_FOR_AI,
    SPECIAL_FILES,
    clean_temp_output_dir,
    determine_file_processing_type,
    extract_file_diff_from_pr,
    git_add_changes,
    git_add_successful_task_changes,
    normalize_added_file_outcomes,
    process_regular_modified_file,
)
from index_file_processor import process_index_file
from toc_processor import process_toc_file
from translation_structure_validator import (
    extract_source_heading_identities,
    validate_markdown_heading_structures,
)
from workflow_ignore_config import load_workflow_ignore_config
from workflow_outcome import FileOutcomes, RunReport, record_file_task_result
from special_file_utils import path_resource_key

WORKFLOW_IGNORE_CONFIG = load_workflow_ignore_config()
COMMIT_BASED_MODE_IGNORE_FILES = WORKFLOW_IGNORE_CONFIG["COMMIT_BASED_MODE_IGNORE_FILES"]
COMMIT_BASED_MODE_IGNORE_FOLDERS = WORKFLOW_IGNORE_CONFIG["COMMIT_BASED_MODE_IGNORE_FOLDERS"]


def should_ignore_resource_card_section(repo_config):
    """Return whether commit mode should ignore RelatedResources ResourceCard sections."""
    return is_yes_option_enabled(
        (repo_config or {}).get("ignore_resource_card_section", True),
        default=True,
    )


SOURCE_FILES_TRANSLATION_MODE_ALIASES = {
    "incremental": "incremental",
    "diff": "incremental",
    "commit-diff": "incremental",
    "commit_diff": "incremental",
    "full": "full",
    "all": "full",
    "full-translation": "full",
    "full_translation": "full",
}
CORRESPONDING_EN_COMMIT_RE = re.compile(
    r"<!--\s*Corresponding EN commit:\s*([0-9a-fA-F]{7,64})\s*-->"
)
CORRESPONDING_EN_COMMIT_TEMPLATE = "<!--Corresponding EN commit: {sha}-->"
RUN_TYPE_MANUAL_ALIASES = {"manual", "workflow_dispatch"}
RUN_TYPE_SCHEDULED_ALIASES = {"schedule", "scheduled"}
PER_FILE_MARKER_GROUP_WARNING_THRESHOLD = 5
RETRY_LEDGER_FILE = "latest_translation_commit.json"


class TranslationStats(RunReport):
    """Backward-compatible name for the shared run report."""

    def print_summary(self):
        super().print_summary(printer=thread_safe_print)


def _record_translation_task_result(task_result, translation_stats):
    file_path = task_result["file_path"]
    status, reason = record_file_task_result(task_result, translation_stats)

    if status == "success":
        detail = f": {reason}" if reason else ""
        thread_safe_print(f"   ✅ Successfully processed {file_path}{detail}")
        return True
    if status == "partial":
        thread_safe_print(f"   ⚠️  Partially processed {file_path}: {reason}")
        return True
    if status == "skipped":
        thread_safe_print(f"   ⏭️  Skipped {file_path}: {reason}")
        return False

    thread_safe_print(f"   ❌ Failed to process {file_path}: {reason}")
    return False


def _process_commit_modified_file(
    source_file_path,
    file_sections,
    pr_diff,
    diff_context,
    github_client,
    ai_client,
    repo_config,
    glossary_matcher,
):
    thread_safe_print(f"\n📄 Processing modified file: {source_file_path}")

    file_specific_diff = extract_file_diff_from_pr(pr_diff, source_file_path) if pr_diff else ""
    if not file_specific_diff:
        if pr_diff:
            return make_task_result("failure", "No file-specific diff found")
        return make_task_result("failure", "No markdown patch text available")

    thread_safe_print(f"   📊 File-specific diff: {len(file_specific_diff)} chars")

    if should_process_modified_file_as_added(source_file_path, repo_config, github_client):
        thread_safe_print(
            f"   🆕 Target file is missing; processing {source_file_path} as a newly added file"
        )
        try:
            source_content = get_source_file_content(
                source_file_path,
                diff_context,
                github_client,
                ref_name="head_ref",
            )
        except Exception as e:
            return make_task_result(
                "failure",
                f"Could not get source file content for added-file fallback: {sanitize_exception_message(e)}",
            )
        if should_ignore_resource_card_section(repo_config):
            source_content, skipped_sections = remove_related_resources_resource_card_sections(source_content)
            if skipped_sections:
                thread_safe_print(
                    f"   🧹 Skipped {len(skipped_sections)} RelatedResources section(s) "
                    "from added-file fallback"
                )
                for section in skipped_sections:
                    thread_safe_print(f"      - {section['hierarchy']}")
            if not source_content.strip():
                return make_task_result(
                    "skipped",
                    "No translatable content remains after RelatedResources filtering",
                )

        outcomes = process_added_files(
            {source_file_path: source_content},
            diff_context,
            github_client,
            ai_client,
            repo_config,
            glossary_matcher=glossary_matcher,
            return_outcomes=True,
        )
        outcomes = normalize_added_file_outcomes(outcomes, source_file_path)
        outcome = outcomes.get(
            source_file_path,
            {"status": "failed", "reason": "Missing added-file fallback outcome"},
        )
        return make_task_result(outcome["status"], outcome.get("reason", ""))

    ignore_files = repo_config.get("ignore_files", COMMIT_BASED_MODE_IGNORE_FILES)
    file_type = determine_file_processing_type(source_file_path, file_sections, SPECIAL_FILES, ignore_files)
    thread_safe_print(f"   🔍 File processing type: {file_type}")

    if file_type == "special_file_toc":
        return make_task_result("skipped", "Already handled in TOC step")
    if file_type == "special_file_keyword":
        return make_task_result("skipped", "Already handled in keyword step")
    if file_type == "special_file_index":
        thread_safe_print(
            f"   📄 Normal _index.md file in modified_sections — processing as regular file"
        )
    elif file_type != "regular_modified":
        return make_task_result("failure", f"Unknown file processing type: {file_type}")

    if should_ignore_resource_card_section(repo_config):
        try:
            base_content_for_diff = get_source_file_content(
                source_file_path,
                diff_context,
                github_client,
                ref_name="base_ref",
            )
            head_content_for_diff = get_source_file_content(
                source_file_path,
                diff_context,
                github_client,
                ref_name="head_ref",
            )
            filtered_file_specific_diff = filter_related_resources_resource_card_diff(
                file_specific_diff,
                base_content_for_diff,
                head_content_for_diff,
            )
            if filtered_file_specific_diff != file_specific_diff:
                thread_safe_print(
                    "   🧹 Removed RelatedResources section lines from commit-mode AI diff context"
                )
                file_specific_diff = filtered_file_specific_diff
        except Exception as e:
            thread_safe_print(
                "   ⚠️  Could not filter RelatedResources from file diff context: "
                f"{sanitize_exception_message(e)}"
            )

    success, failure_reason = process_regular_modified_file(
        source_file_path,
        file_sections,
        file_specific_diff,
        diff_context,
        github_client,
        ai_client,
        repo_config,
        MAX_NON_SYSTEM_SECTIONS_FOR_AI,
        glossary_matcher=glossary_matcher,
        return_details=True,
    )
    if success:
        return make_task_result("partial" if failure_reason else "success", failure_reason)
    return make_task_result(
        "failure",
        failure_reason or "Regular modified file processor returned failure",
    )


def get_commit_repo_config():
    """Build a minimal repo config for commit-driven sync."""
    if SOURCE_LANGUAGE_OVERRIDE or TARGET_LANGUAGE_OVERRIDE:
        if not (SOURCE_LANGUAGE_OVERRIDE and TARGET_LANGUAGE_OVERRIDE):
            raise ValueError(
                "SOURCE_LANGUAGE and TARGET_LANGUAGE must both be set or both be empty. "
                f"Got SOURCE_LANGUAGE={SOURCE_LANGUAGE_OVERRIDE!r}, TARGET_LANGUAGE={TARGET_LANGUAGE_OVERRIDE!r}"
            )
        source_language = SOURCE_LANGUAGE_OVERRIDE
        target_language = TARGET_LANGUAGE_OVERRIDE
    else:
        source_language, target_language = infer_language_direction(SOURCE_REPO, TARGET_REPO)
    return {
        "source_repo": SOURCE_REPO,
        "target_repo": TARGET_REPO,
        "target_local_path": TARGET_REPO_PATH,
        "target_ref": TARGET_REF,
        "prefer_local_target_for_read": PREFER_LOCAL_TARGET_FOR_READ,
        "source_mode": "commit",
        "source_language": source_language,
        "target_language": target_language,
        "ignore_resource_card_section": is_yes_option_enabled(
            IGNORE_RESOURCE_CARD_SECTION,
            default=True,
        ),
    }


def get_commit_repo_configs():
    """Return repo configs keyed by source repo for analyzer compatibility."""
    repo_config = get_commit_repo_config()
    return {
        SOURCE_REPO: repo_config.copy(),
    }


def normalize_source_files(source_files, source_folder):
    """Normalize comma-separated source file filters relative to the optional folder.

    When SOURCE_FOLDER is set, bare file paths are treated as relative to that folder.
    """
    normalized = set()
    folder_prefix = source_folder.strip("/").strip()

    for item in source_files.split(","):
        rel = item.strip()
        if not rel:
            continue
        rel = rel.lstrip("/")
        if folder_prefix and not rel.startswith(folder_prefix + "/"):
            rel = f"{folder_prefix}/{rel}"
        normalized.add(rel)

    return normalized


def parse_comma_separated_list(value):
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def normalize_commit_sync_run_type(run_type):
    normalized = (run_type or "").strip().lower()
    if normalized in RUN_TYPE_MANUAL_ALIASES:
        return "manual"
    if normalized in RUN_TYPE_SCHEDULED_ALIASES:
        return "scheduled"
    return ""


def exclude_paths_from_marker_groups(marker_groups, excluded_paths):
    """Keep full-translation files out of per-file incremental cursor groups."""
    excluded_paths = set(excluded_paths or [])
    if not excluded_paths:
        return marker_groups

    return {
        source_ref: remaining_paths
        for source_ref, file_paths in marker_groups.items()
        if (remaining_paths := set(file_paths) - excluded_paths)
    }


def commits_match(left, right):
    """Return True when two commit identifiers refer to the same SHA prefix."""
    left = (left or "").strip().lower()
    right = (right or "").strip().lower()
    if not left or not right:
        return False
    if left == right:
        return True
    shorter, longer = sorted((left, right), key=len)
    return len(shorter) >= 7 and longer.startswith(shorter)


def get_safe_target_file_path(target_repo_path, source_file_path):
    if not target_repo_path or not source_file_path:
        return None

    try:
        target_file_path = safe_target_path(target_repo_path, source_file_path)
    except ValueError:
        return None
    return target_file_path


def read_target_file_content(target_repo_path, source_file_path):
    target_file_path = get_safe_target_file_path(target_repo_path, source_file_path)
    if not target_file_path or not os.path.exists(target_file_path):
        return None

    with open(target_file_path, "r", encoding="utf-8") as f:
        return f.read()


def load_retry_cursor_groups(target_repo_path):
    """Load persisted failed/partial file cursors grouped by source ref."""
    ledger_path = get_safe_target_file_path(target_repo_path, RETRY_LEDGER_FILE)
    if not ledger_path or not os.path.exists(ledger_path):
        return {}, set()
    try:
        with open(ledger_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        thread_safe_print(
            f"   ⚠️  Could not read retry cursor ledger: {sanitize_exception_message(e)}"
        )
        return {}, set()

    groups = {}
    paths = set()
    pending = data.get("pending") or {}
    if not isinstance(pending, dict):
        return {}, set()
    for file_path, source_ref in pending.items():
        if not isinstance(file_path, str) or not isinstance(source_ref, str):
            continue
        if not re.fullmatch(r"[0-9a-fA-F]{7,64}", source_ref.strip()):
            continue
        groups.setdefault(source_ref.strip(), set()).add(file_path)
        paths.add(file_path)
    return groups, paths


def update_retry_cursor_ledger(
    target_repo_path,
    translation_stats,
    completed_file_paths,
    source_branch=None,
    source_head_ref=None,
):
    """Atomically finalize the global cursor and any per-file retry cursors."""
    ledger_path = get_safe_target_file_path(target_repo_path, RETRY_LEDGER_FILE)
    if not ledger_path:
        translation_stats.mark_failure(
            RETRY_LEDGER_FILE,
            "Failed to update retry cursor ledger: unsafe target path",
        )
        return False
    temp_path = ""
    try:
        if os.path.exists(ledger_path):
            with open(ledger_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            if not source_branch or not source_head_ref:
                raise ValueError(
                    "cannot initialize a missing retry cursor ledger without "
                    "source_branch and source_head_ref"
                )
            thread_safe_print(
                f"   ℹ️  Initializing missing retry cursor ledger: {ledger_path}"
            )
            data = {}
        pending = data.get("pending")
        if not isinstance(pending, dict):
            pending = {}
        for file_path in completed_file_paths:
            pending.pop(file_path, None)
        pending.update(translation_stats.retry_refs)
        if source_branch is not None or source_head_ref is not None:
            if not source_branch or not source_head_ref:
                raise ValueError(
                    "source_branch and source_head_ref must be provided together"
                )
            data["target"] = source_branch
            data["sha"] = source_head_ref
        if pending:
            data["pending"] = dict(sorted(pending.items()))
        else:
            data.pop("pending", None)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=os.path.dirname(ledger_path),
            prefix=".latest_translation_commit.",
            suffix=".tmp",
            delete=False,
        ) as f:
            temp_path = f.name
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(temp_path, ledger_path)
        temp_path = ""
        return True
    except Exception as e:
        if temp_path:
            try:
                os.remove(temp_path)
            except OSError:
                pass
        translation_stats.mark_failure(
            RETRY_LEDGER_FILE,
            f"Failed to update retry cursor ledger: {sanitize_exception_message(e)}",
        )
        return False


def validate_successful_translation_structures(
    successful_file_paths,
    diff_context,
    github_client,
    translation_stats,
    repo_config=None,
    pr_diff="",
):
    """Validate document structure for successfully translated Markdown files."""
    markdown_file_paths = {
        file_path
        for file_path in successful_file_paths
        if file_path and file_path.lower().endswith(".md")
    }
    if not markdown_file_paths:
        return []

    thread_safe_print(
        f"\n🔎 Validating document structure for {len(markdown_file_paths)} translated Markdown file(s)..."
    )

    should_filter_related_resources = (
        isinstance(diff_context, dict)
        and diff_context.get("mode") == "commit"
        and should_ignore_resource_card_section(diff_context.get("repo_config"))
    )
    effective_repo_config = (
        repo_config
        or (
            diff_context.get("repo_config")
            if isinstance(diff_context, dict)
            else None
        )
        or {}
    )
    source_content_cache = {}

    def load_source_for_structure_validation(file_path):
        if file_path in source_content_cache:
            return source_content_cache[file_path]
        source_content = get_source_ref_content(
            file_path,
            diff_context,
            github_client,
            "head_ref",
        )
        if should_filter_related_resources and source_content:
            source_content, _ = remove_related_resources_resource_card_sections(
                source_content
            )
        source_content_cache[file_path] = source_content
        return source_content

    def load_target_for_structure_validation(file_path):
        target_content = read_target_file_content(TARGET_REPO_PATH, file_path)
        if should_filter_related_resources and target_content:
            target_content, _ = remove_related_resources_resource_card_sections(
                target_content
            )
        return target_content

    def load_expected_heading_anchors(file_path):
        file_diff = extract_file_diff_from_pr(pr_diff, file_path)
        if not file_diff:
            return []

        prompt_diff = preprocess_diff_for_heading_anchor_stability(
            file_diff,
            effective_repo_config.get("source_language", ""),
            effective_repo_config.get("target_language", ""),
            source_mode=(
                diff_context.get("mode", "")
                if isinstance(diff_context, dict)
                else ""
            ),
        )
        expected = extract_added_explicit_heading_anchors(prompt_diff)
        if not expected:
            return []

        source_content = load_source_for_structure_validation(file_path)
        source_counts = Counter(extract_source_heading_identities(source_content))
        canonical_expected = []
        for anchor in expected:
            if source_counts[anchor] > 0:
                canonical_expected.append(anchor)
                source_counts[anchor] -= 1
        return canonical_expected

    issues = validate_markdown_heading_structures(
        markdown_file_paths,
        # Validate the same canonical source view that commit-mode translation
        # uses. Otherwise an intentionally skipped RelatedResources section is
        # incorrectly reported as a missing target heading.
        source_content_loader=load_source_for_structure_validation,
        target_content_loader=load_target_for_structure_validation,
        expected_heading_anchors_loader=load_expected_heading_anchors,
    )

    if not issues:
        thread_safe_print("   ✅ Document structures match source HEAD")
        return []

    thread_safe_print(
        f"   ⚠️  Found {len(issues)} document structure mismatch(es)"
    )
    for issue in issues:
        thread_safe_print(f"      - {issue.file_path}: {issue.reason}")
        translation_stats.mark_structure_error(issue)
        translation_stats.set_retry_ref(
            issue.file_path,
            diff_context.get("base_ref", "") if isinstance(diff_context, dict) else "",
        )

    return issues


def get_corresponding_en_commit(content):
    match = CORRESPONDING_EN_COMMIT_RE.search(content or "")
    return match.group(1) if match else ""


def _line_ending(line):
    if line.endswith("\r\n"):
        return "\r\n"
    if line.endswith("\n"):
        return "\n"
    return ""


def upsert_corresponding_en_commit(content, commit_sha, add_if_missing):
    """Add or update the per-file source commit marker in translated Markdown."""
    marker = CORRESPONDING_EN_COMMIT_TEMPLATE.format(sha=commit_sha)
    if CORRESPONDING_EN_COMMIT_RE.search(content or ""):
        updated = CORRESPONDING_EN_COMMIT_RE.sub(marker, content, count=1)
        return updated, updated != content

    if not add_if_missing:
        return content, False

    lines = (content or "").splitlines(keepends=True)
    for index, line in enumerate(lines):
        if re.match(r"^#\s+\S", line):
            ending = _line_ending(line)
            body = line[: -len(ending)] if ending else line
            lines[index] = f"{body} {marker}{ending}"
            return "".join(lines), True

    ending = "\n"
    return f"{marker}{ending}{content or ''}", True


def remove_corresponding_en_commit(content):
    """Remove an existing per-file source commit marker from Markdown."""
    if not CORRESPONDING_EN_COMMIT_RE.search(content or ""):
        return content, False

    updated_lines = []
    changed = False
    for line in (content or "").splitlines(keepends=True):
        if not CORRESPONDING_EN_COMMIT_RE.search(line):
            updated_lines.append(line)
            continue

        changed = True
        updated_line = CORRESPONDING_EN_COMMIT_RE.sub("", line, count=1)
        if not updated_line.strip():
            continue

        ending = _line_ending(updated_line)
        body = updated_line[: -len(ending)] if ending else updated_line
        updated_lines.append(f"{body.rstrip()}{ending}")

    return "".join(updated_lines), changed


def update_corresponding_en_commit_for_files(
    file_paths,
    target_repo_path,
    commit_sha,
    add_if_missing,
    remove_if_present=False,
):
    """Add/update or remove Corresponding EN commit markers in translated files."""
    if add_if_missing and remove_if_present:
        raise ValueError(
            "add_if_missing and remove_if_present are mutually exclusive."
        )

    updated_files = []
    failures = {}

    for file_path in sorted(set(file_paths)):
        if not file_path.endswith(".md"):
            continue

        target_file_path = get_safe_target_file_path(target_repo_path, file_path)
        if not target_file_path:
            failures[file_path] = "Unsafe target file path"
            continue
        if not os.path.exists(target_file_path):
            continue

        try:
            with open(target_file_path, "r", encoding="utf-8") as f:
                content = f.read()
            if remove_if_present:
                updated_content, changed = remove_corresponding_en_commit(content)
            else:
                updated_content, changed = upsert_corresponding_en_commit(
                    content,
                    commit_sha,
                    add_if_missing=add_if_missing,
                )
            if not changed:
                continue
            atomic_write_text(target_file_path, updated_content)
            updated_files.append(file_path)
        except Exception as e:
            failures[file_path] = sanitize_exception_message(e)

    return updated_files, failures


def split_changed_files_by_corresponding_en_commit(
    changed_files,
    target_repo_path,
    global_base_ref,
):
    """Split scheduled files between the global cursor and per-file cursors."""
    global_changed_files = []
    marker_groups = {}
    marker_file_paths = set()

    for file in changed_files:
        file_path = getattr(file, "filename", "") or ""
        marker_commit = ""
        if file_path and file_path.endswith(".md"):
            try:
                target_content = read_target_file_content(target_repo_path, file_path)
            except Exception as e:
                thread_safe_print(
                    f"   ⚠️  Could not read target marker for {file_path}: "
                    f"{sanitize_exception_message(e)}"
                )
                target_content = None
            marker_commit = get_corresponding_en_commit(target_content or "")
            if marker_commit:
                marker_file_paths.add(file_path)

        if marker_commit and not commits_match(marker_commit, global_base_ref):
            marker_groups.setdefault(marker_commit, set()).add(file_path)
            continue

        global_changed_files.append(file)

    return global_changed_files, marker_groups, marker_file_paths


def changed_file_candidate_paths(file):
    paths = [getattr(file, "filename", "")]
    previous_filename = getattr(file, "previous_filename", None)
    if previous_filename:
        paths.append(previous_filename)
    return [path for path in paths if path]


def read_target_corresponding_en_commit(target_repo_path, file_path):
    try:
        target_content = read_target_file_content(target_repo_path, file_path)
    except Exception as e:
        thread_safe_print(
            f"   ⚠️  Could not read target marker for {file_path}: "
            f"{sanitize_exception_message(e)}"
        )
        target_content = None
    return get_corresponding_en_commit(target_content or "")


def split_manual_source_files_by_corresponding_en_commit(
    changed_files,
    source_files,
    source_folder,
    target_repo_path,
    global_base_ref,
):
    """Split manually requested files so per-file cursors override the global cursor."""
    requested_paths = normalize_source_files(source_files, source_folder)
    if not requested_paths:
        return changed_files, {}, set()

    changed_candidate_paths = {
        path
        for file in changed_files
        for path in changed_file_candidate_paths(file)
    }
    grouped_marker_paths = set()
    marker_groups = {}
    marker_file_paths = set()

    for file_path in sorted(requested_paths):
        if not file_path.endswith(".md"):
            continue

        marker_commit = read_target_corresponding_en_commit(
            target_repo_path,
            file_path,
        )
        if not marker_commit:
            continue

        marker_file_paths.add(file_path)
        if (
            not commits_match(marker_commit, global_base_ref)
            or file_path not in changed_candidate_paths
        ):
            marker_groups.setdefault(marker_commit, set()).add(file_path)
            grouped_marker_paths.add(file_path)

    if not grouped_marker_paths:
        return changed_files, marker_groups, marker_file_paths

    global_changed_files = [
        file
        for file in changed_files
        if not (set(changed_file_candidate_paths(file)) & grouped_marker_paths)
    ]
    return global_changed_files, marker_groups, marker_file_paths


def filter_changed_files(changed_files, source_folder="", source_files=""):
    """Filter changed files by folder and/or explicit file list.

    Renames are matched against both the current filename and previous_filename so
    moved files stay in scope for translation and cleanup.
    """
    folder_prefix = source_folder.strip("/").strip()
    folder_prefix = f"{folder_prefix}/" if folder_prefix else ""
    normalized_files = normalize_source_files(source_files, source_folder)

    def matches(file):
        paths = changed_file_candidate_paths(file)

        if folder_prefix and not any(
            path.startswith(folder_prefix) or path == folder_prefix.rstrip("/")
            for path in paths
        ):
            return False

        if normalized_files and not any(path in normalized_files for path in paths):
            return False

        return True

    return [file for file in changed_files if matches(file)]


def normalize_source_files_translation_mode(mode):
    """Normalize SOURCE_FILES_TRANSLATION_MODE into incremental or full."""
    normalized = (mode or "incremental").strip().lower()
    if not normalized:
        normalized = "incremental"

    if normalized not in SOURCE_FILES_TRANSLATION_MODE_ALIASES:
        valid_values = "incremental or full"
        raise ValueError(
            f"SOURCE_FILES_TRANSLATION_MODE must be {valid_values}; got {mode!r}."
        )

    return SOURCE_FILES_TRANSLATION_MODE_ALIASES[normalized]


def get_effective_source_files_translation_mode(mode, source_files):
    """Return the mode that actually applies to this run.

    Full-file translation is intentionally scoped to explicit SOURCE_FILES so a
    folder-only commit sync cannot accidentally rewrite every changed file.
    """
    normalized_mode = normalize_source_files_translation_mode(mode)
    if not source_files.strip():
        return "incremental"
    return normalized_mode


def resolve_full_translation_source_file_paths(source_files, source_folder, changed_files):
    """Resolve explicitly requested SOURCE_FILES to HEAD paths for full translation."""
    requested_paths = normalize_source_files(source_files, source_folder)
    resolved_paths = set()
    matched_requested_paths = set()

    for file in changed_files:
        current_path = getattr(file, "filename", "")
        previous_path = getattr(file, "previous_filename", None)
        candidate_paths = {path for path in (current_path, previous_path) if path}
        matched_paths = requested_paths & candidate_paths
        if not matched_paths:
            continue

        matched_requested_paths.update(matched_paths)
        if getattr(file, "status", "") != "removed" and current_path:
            resolved_paths.add(current_path)

    # Full mode must support explicit files that are not in the compare diff.
    resolved_paths.update(requested_paths - matched_requested_paths)
    return resolved_paths


def collect_source_files_for_full_translation(
    source_file_paths,
    changed_files,
    diff_context,
    github_client,
    ignore_files=None,
):
    """Fetch full HEAD content for selected markdown files."""
    ignore_files = set(ignore_files or [])
    full_translation_files = {}
    failures = {}

    removed_paths = {
        getattr(file, "filename", "")
        for file in changed_files
        if getattr(file, "status", "") == "removed"
    }

    for source_file_path in sorted(source_file_paths):
        if not source_file_path or not source_file_path.endswith(".md"):
            continue
        if source_file_path in ignore_files:
            thread_safe_print(f"   ⏭️  Skipping ignored file for full translation: {source_file_path}")
            continue
        if source_file_path in removed_paths:
            continue

        try:
            file_content = get_full_translation_source_content(
                source_file_path,
                diff_context,
                github_client,
            )
            if should_ignore_resource_card_section(diff_context.get("repo_config")):
                file_content, skipped_sections = remove_related_resources_resource_card_sections(file_content)
                if skipped_sections:
                    thread_safe_print(
                        f"   🧹 Skipped {len(skipped_sections)} RelatedResources section(s) "
                        f"from full commit-mode file translation: {source_file_path}"
                    )
                    for section in skipped_sections:
                        thread_safe_print(f"      - {section['hierarchy']}")
                if not file_content.strip():
                    thread_safe_print(
                        f"   ⏭️  Skipping {source_file_path}: no translatable content remains "
                        "after RelatedResources filtering"
                    )
                    continue
            full_translation_files[source_file_path] = file_content
        except Exception as e:
            failures[source_file_path] = (
                f"Could not get source file content for full translation: "
                f"{sanitize_exception_message(e)}"
            )

    return full_translation_files, failures


def get_local_git_file_content(source_repo_path, ref, source_file_path):
    import subprocess

    result = subprocess.run(
        ["git", "-C", source_repo_path, "show", f"{ref}:{source_file_path}"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        detail = sanitize_local_git_error(
            result.stderr,
            source_repo_path,
        )
        if detail:
            detail = f": {detail}"
        raise RuntimeError(
            f"git show failed for {source_file_path} at source ref {ref}{detail}"
        )
    if result.stderr.strip():
        thread_safe_print(
            f"   ⚠️  git show produced stderr for {source_file_path}; using stdout only."
        )
    return result.stdout


def get_source_ref_content(source_file_path, diff_context, github_client, ref_name):
    """Read source content from the requested source ref."""
    source_repo_path = diff_context.get("source_repo_path")
    if source_repo_path:
        return get_local_git_file_content(
            source_repo_path,
            diff_context[ref_name],
            source_file_path,
        )

    return get_source_file_content(
        source_file_path,
        diff_context,
        github_client,
        ref_name=ref_name,
    )


def get_full_translation_source_content(source_file_path, diff_context, github_client):
    """Read full-translation source content from local HEAD ref when available."""
    return get_source_ref_content(
        source_file_path,
        diff_context,
        github_client,
        "head_ref",
    )


def sanitize_local_git_error(stderr, source_repo_path):
    """Return a safe git error detail without the local checkout path."""
    detail = (stderr or "").strip()
    if source_repo_path:
        detail = detail.replace(source_repo_path, "[SOURCE_REPO_PATH]")
    return sanitize_exception_message(detail)


def remove_incremental_work_for_files(
    file_paths,
    added_sections,
    modified_sections,
    deleted_sections,
    toc_files,
    keyword_files,
    index_files=None,
    formatting_files=None,
):
    """Remove per-diff work queues for files that will be translated in full."""
    for file_path in file_paths:
        added_sections.pop(file_path, None)
        modified_sections.pop(file_path, None)
        deleted_sections.pop(file_path, None)
        toc_files.pop(file_path, None)
        keyword_files.pop(file_path, None)
        if index_files is not None:
            index_files.pop(file_path, None)
        if formatting_files is not None:
            formatting_files.pop(file_path, None)


def apply_source_files_full_translation_mode(
    source_files,
    source_folder,
    filtered_changed_files,
    diff_context,
    github_client,
    commit_ignore_files,
    added_sections,
    modified_sections,
    deleted_sections,
    added_files,
    toc_files,
    keyword_files,
    translation_stats,
    index_files=None,
    formatting_files=None,
):
    """Switch selected SOURCE_FILES from diff-based queues to full-file translation."""
    thread_safe_print(
        "\n📄 SOURCE_FILES_TRANSLATION_MODE=full: translating selected markdown files as complete files."
    )
    source_file_paths = resolve_full_translation_source_file_paths(
        source_files,
        source_folder,
        filtered_changed_files,
    )
    full_translation_files, full_translation_failures = collect_source_files_for_full_translation(
        source_file_paths,
        filtered_changed_files,
        diff_context,
        github_client,
        ignore_files=commit_ignore_files,
    )
    full_translation_file_paths = set(full_translation_files) | set(full_translation_failures)
    remove_incremental_work_for_files(
        full_translation_file_paths,
        added_sections,
        modified_sections,
        deleted_sections,
        toc_files,
        keyword_files,
        index_files=index_files,
        formatting_files=formatting_files,
    )
    added_files.update(full_translation_files)
    for file_path in full_translation_failures:
        added_files.pop(file_path, None)
    for file_path, reason in full_translation_failures.items():
        translation_stats.mark_failure(file_path, reason)
        thread_safe_print(f"   ❌ {file_path}: {reason}")

    return full_translation_file_paths


def collect_toc_scope_added_files_from_snapshots(toc_files):
    """Return links in aggregate head TOCs that were absent from aggregate base TOCs."""
    base_links = set()
    head_links = set()

    for toc_data in (toc_files or {}).values():
        source_base_content = toc_data.get("source_base_content")
        source_head_content = toc_data.get("source_head_content")
        if source_base_content is None or source_head_content is None:
            continue

        base_links.update(extract_markdown_doc_links(source_base_content))
        head_links.update(extract_markdown_doc_links(source_head_content))

    return head_links - base_links


def collect_toc_scope_added_files(toc_files, diff_context=None, github_client=None, toc_file_paths=None):
    """Return Markdown docs newly linked from the aggregate Cloud TOC scope."""
    if not toc_file_paths or diff_context is None:
        return collect_toc_scope_added_files_from_snapshots(toc_files)

    aggregate_toc_files = {}
    changed_toc_files = toc_files or {}

    for toc_file in toc_file_paths:
        if toc_file in changed_toc_files:
            aggregate_toc_files[toc_file] = changed_toc_files[toc_file]
            continue

        try:
            source_base_content = get_source_ref_content(
                toc_file,
                diff_context,
                github_client,
                "base_ref",
            )
        except Exception as e:
            thread_safe_print(
                f"   ⚠️  Could not get base Cloud TOC content for {toc_file}: "
                f"{sanitize_exception_message(e)}"
            )
            source_base_content = ""

        try:
            source_head_content = get_source_ref_content(
                toc_file,
                diff_context,
                github_client,
                "head_ref",
            )
        except Exception as e:
            thread_safe_print(
                f"   ⚠️  Could not get head Cloud TOC content for {toc_file}: "
                f"{sanitize_exception_message(e)}"
            )
            source_head_content = ""

        aggregate_toc_files[toc_file] = {
            "source_base_content": source_base_content,
            "source_head_content": source_head_content,
        }

    return collect_toc_scope_added_files_from_snapshots(aggregate_toc_files)


def get_cloud_toc_file_paths(toc_files):
    configured_toc_files = parse_comma_separated_list(CLOUD_TOC_FILES)
    if configured_toc_files:
        return configured_toc_files

    return sorted((toc_files or {}).keys())


def apply_toc_scope_added_files(
    toc_files,
    diff_context,
    github_client,
    commit_ignore_files,
    added_sections,
    modified_sections,
    deleted_sections,
    added_files,
    keyword_files,
    translation_stats,
    index_files=None,
    formatting_files=None,
):
    """Queue files newly linked from Cloud TOCs for full file-added translation."""
    scope_added_file_paths = collect_toc_scope_added_files(
        toc_files,
        diff_context,
        github_client,
        get_cloud_toc_file_paths(toc_files),
    )
    if not scope_added_file_paths:
        return set()

    thread_safe_print(
        "\n📄 Cloud TOC scope added files: "
        f"{', '.join(sorted(scope_added_file_paths))}"
    )
    scope_added_file_contents, scope_added_failures = collect_source_files_for_full_translation(
        scope_added_file_paths,
        [],
        diff_context,
        github_client,
        ignore_files=commit_ignore_files,
    )
    queued_file_paths = set(scope_added_file_contents) | set(scope_added_failures)

    remove_incremental_work_for_files(
        queued_file_paths,
        added_sections,
        modified_sections,
        deleted_sections,
        # Keep the changed TOC itself queued; only the newly linked doc should
        # switch from diff-based work to full file-added translation.
        {},
        keyword_files,
        index_files=index_files,
        formatting_files=formatting_files,
    )
    added_files.update(scope_added_file_contents)

    for file_path in scope_added_failures:
        added_files.pop(file_path, None)
    for file_path, reason in scope_added_failures.items():
        translation_stats.mark_failure(file_path, reason)
        thread_safe_print(f"   ❌ {file_path}: {reason}")

    return queued_file_paths


def source_filters_explicitly_include(folder_name):
    """Return True when commit-mode filters explicitly target a folder."""
    normalized_folder = folder_name.strip("/").strip()
    source_folder = SOURCE_FOLDER.strip("/").strip()

    if source_folder == normalized_folder:
        return True

    normalized_files = normalize_source_files(SOURCE_FILES, SOURCE_FOLDER)
    return any(
        path == normalized_folder or path.startswith(normalized_folder + "/")
        for path in normalized_files
    )


def get_commit_ignore_files():
    """Return commit-mode ignore files.

    PR-mode TOC ignores are intentionally not inherited here because commit sync
    can be scoped to exactly the files that should be mirrored.
    """
    return list(COMMIT_BASED_MODE_IGNORE_FILES)


def get_commit_ignore_folders():
    """Return commit-mode ignore folders from env override or config file."""
    if COMMIT_IGNORE_FOLDERS_OVERRIDE:
        return [f.strip().strip("/").strip() for f in COMMIT_IGNORE_FOLDERS_OVERRIDE.split(",") if f.strip()]
    return list(COMMIT_BASED_MODE_IGNORE_FOLDERS)


def should_process_modified_file_as_added(source_file_path, repo_config, github_client):
    """Return True when a modified source file has no writable target baseline."""
    target_local_path = repo_config.get("target_local_path")
    if target_local_path:
        target_file_path = get_safe_target_file_path(target_local_path, source_file_path)
        if not target_file_path:
            raise ValueError(f"Unsafe target file path: {source_file_path!r}")
        if os.path.exists(target_file_path):
            return False
        return True

    target_ref = repo_config.get("target_ref")
    if not target_ref:
        return False

    try:
        target_repo = github_client.get_repo(repo_config["target_repo"])
        target_repo.get_contents(source_file_path, ref=target_ref)
        return False
    except Exception:
        return True


def build_exclude_folders(repo_config):
    """Build the early-exclusion folder list for commit sync."""
    exclude_folders = []
    target_lang = (repo_config.get("target_language") or "").lower()
    source_lang = (repo_config.get("source_language") or "").lower()
    if not target_lang or target_lang == source_lang:
        return exclude_folders

    for folder_name in get_commit_ignore_folders():
        folder_name = folder_name.strip("/").strip()
        if not folder_name:
            continue
        if source_filters_explicitly_include(folder_name):
            thread_safe_print(f"ℹ️  Explicit source filters include '{folder_name}', skipping the default exclusion for commit sync.")
        else:
            exclude_folders.append(folder_name)
    return exclude_folders


def resolve_compare_range():
    """Resolve the explicit source compare range from environment variables."""
    base_ref = SOURCE_BASE_REF.strip()
    head_ref = SOURCE_HEAD_REF.strip()

    if not base_ref or not head_ref:
        raise ValueError(
            "SOURCE_BASE_REF and SOURCE_HEAD_REF must both be set for commit-based sync."
        )

    return base_ref, head_ref


def build_incremental_diff_context(
    base_ref,
    head_ref,
    github_client,
    repo_config,
    repo_configs,
):
    if SOURCE_REPO_PATH:
        diff_context = build_local_commit_diff_context(
            SOURCE_REPO,
            TARGET_REPO,
            base_ref,
            head_ref,
            SOURCE_REPO_PATH,
            repo_configs,
        )
    else:
        diff_context = build_commit_diff_context(
            SOURCE_REPO,
            TARGET_REPO,
            base_ref,
            head_ref,
            github_client,
            repo_configs,
        )
    diff_context["repo_config"] = repo_config
    return diff_context


def print_group_summary(group_label, counts):
    thread_safe_print("\n" + "-" * 80)
    thread_safe_print(f"📊 Group Summary: {group_label}")
    thread_safe_print("-" * 80)
    thread_safe_print(f"   📄 Added files: {counts['added_files']} processed")
    thread_safe_print(f"   🗑️  Deleted files: {counts['deleted_files']} processed")
    thread_safe_print(f"   📋 TOC files: {counts['toc_files']} processed")
    thread_safe_print(f"   📋 Keyword files: {counts['keyword_files']} processed")
    thread_safe_print(f"   📄 Index files: {counts.get('index_files', 0)} processed")
    thread_safe_print(f"   🧹 Formatting-only files: {counts.get('formatting_files', 0)} processed")
    thread_safe_print(f"   📝 Modified files: {counts['modified_sections']} processed")
    thread_safe_print(f"   🖼️  Added images: {counts['added_images']} processed")
    thread_safe_print(f"   🖼️  Modified images: {counts['modified_images']} processed")
    thread_safe_print(f"   🖼️  Deleted images: {counts['deleted_images']} processed")


def add_counts(total_counts, counts):
    for key, value in counts.items():
        total_counts[key] = total_counts.get(key, 0) + value


def reconcile_restructured_files(
    restructured_files,
    added_files,
    full_translation_file_paths,
    diff_context,
    github_client,
    ai_client,
    repo_config,
    glossary_matcher,
    translation_stats,
    successful_file_paths,
):
    """Reconcile restructured files in place, reusing existing translations.

    Each successfully reconciled file is written to the target and recorded as
    a success. If an existing translated file cannot be reconciled safely, keep
    it unchanged and record a failure; never turn a localized reconciliation
    failure into a destructive full-file overwrite. A genuinely missing target
    can still use the normal full-file creation path.
    """
    reconciled_paths = set()
    failed_paths = set()
    source_mode = (
        diff_context.get("mode", "commit") if isinstance(diff_context, dict) else "commit"
    )
    rr_ignore = source_mode == "commit" and should_ignore_resource_card_section(repo_config)

    for file_path in sorted(restructured_files):
        existing_target = read_target_file_content(TARGET_REPO_PATH, file_path)
        if not existing_target:
            thread_safe_print(
                f"   ℹ️  Reconciler: no existing target for {file_path}; "
                "using full translation"
            )
            continue

        def mark_reconciliation_failure(reason):
            thread_safe_print(
                f"   ❌ Reconciler: {reason}; preserving existing target file"
            )
            translation_stats.mark_failure(file_path, reason)
            failed_paths.add(file_path)

        try:
            head_source = get_source_ref_content(file_path, diff_context, github_client, "head_ref")
            base_source = get_source_ref_content(file_path, diff_context, github_client, "base_ref")
        except Exception as e:
            mark_reconciliation_failure(
                f"could not load source for {file_path}: "
                f"{sanitize_exception_message(e)}"
            )
            continue

        missing_snapshots = [
            label
            for label, content in (("HEAD", head_source), ("BASE", base_source))
            if content is None
        ]
        if missing_snapshots:
            if len(missing_snapshots) == 1:
                missing_detail = f"{missing_snapshots[0]} source snapshot is unavailable"
            else:
                missing_detail = (
                    f"{' and '.join(missing_snapshots)} source snapshots are unavailable"
                )
            mark_reconciliation_failure(
                f"{missing_detail} for {file_path}"
            )
            continue

        if rr_ignore:
            # A localized target can legitimately retain this ignored section.
            # Keep full documents aligned and make the section immutable by
            # restoring its BASE source form in the effective HEAD snapshot.
            head_source, preserved_sections = (
                preserve_base_related_resources_resource_card_sections(
                    base_source,
                    head_source,
                )
            )
            if head_source is None:
                mark_reconciliation_failure(
                    "RelatedResources section count changed between source BASE "
                    f"and HEAD for {file_path}"
                )
                continue
            if preserved_sections:
                thread_safe_print(
                    f"   🧹 Reconciler: ignored changes in "
                    f"{len(preserved_sections)} RelatedResources section(s)"
                )

        try:
            reconciled = reconcile_restructured_file(
                file_path,
                head_source,
                base_source,
                existing_target,
                ai_client,
                repo_config,
                glossary_matcher=glossary_matcher,
                source_mode=source_mode,
            )
        except Exception as e:
            mark_reconciliation_failure(
                f"reconciliation raised for {file_path}: "
                f"{sanitize_exception_message(e)}"
            )
            continue

        if reconciled is None:
            mark_reconciliation_failure(
                f"structure-preserving reconciliation declined {file_path}"
            )
            continue

        target_file_path = get_safe_target_file_path(TARGET_REPO_PATH, file_path)
        if not target_file_path:
            mark_reconciliation_failure(
                f"unsafe target path for {file_path}"
            )
            continue

        try:
            atomic_write_text(target_file_path, reconciled)
        except Exception as e:
            mark_reconciliation_failure(
                f"failed to write {target_file_path}: "
                f"{sanitize_exception_message(e)}"
            )
            continue

        thread_safe_print(f"   ✅ Reconciled (structure-preserving): {file_path}")
        translation_stats.mark_success(file_path)
        successful_file_paths.add(file_path)
        reconciled_paths.add(file_path)

    for file_path in reconciled_paths | failed_paths:
        added_files.pop(file_path, None)
        full_translation_file_paths.discard(file_path)

    if reconciled_paths:
        git_add_changes(TARGET_REPO_PATH)

    return reconciled_paths


def process_translation_group(
    group_label,
    *,
    source_files_translation_mode,
    source_files,
    source_folder,
    diff_context,
    filtered_changed_files,
    pr_diff,
    github_client,
    ai_client,
    repo_config,
    repo_configs,
    glossary_matcher,
    commit_ignore_files,
    translation_stats,
    source_full_files="",
):
    """Process one commit-sync translation group that shares the same base ref.

    Processing is intentionally best-effort across independent files. Successful
    files remain eligible for staging and push even when other files fail; every
    failed file must instead be recorded in TranslationStats so the generated
    failure report provides the follow-up queue.
    """
    thread_safe_print(f"\n🔁 Translation group: {group_label}")
    successful_file_paths = set()
    partial_start = len(translation_stats.partial)
    failed_start = len(translation_stats.failed)

    def finalize(result):
        translation_stats.set_retry_refs_for_new_issues(
            partial_start,
            failed_start,
            diff_context.get("base_ref", "") if isinstance(diff_context, dict) else "",
        )
        return result

    if source_files_translation_mode == "full":
        thread_safe_print("📊 Compare diff: skipped for full translation mode")
        thread_safe_print(
            "ℹ️  Full translation uses SOURCE_FILES from SOURCE_HEAD_REF "
            "instead of commit diff analysis."
        )
        added_sections = {}
        modified_sections = {}
        deleted_sections = {}
        added_files = {}
        deleted_files = []
        toc_files = {}
        keyword_files = {}
        index_files = {}
        formatting_files = {}
        added_images = []
        modified_images = []
        deleted_images = []
        restructured_files = set()
    else:
        if filtered_changed_files:
            thread_safe_print(f"📊 Filtered changed files: {len(filtered_changed_files)}")
            if pr_diff:
                thread_safe_print(f"✅ Built diff text: {len(pr_diff)} characters")
            else:
                thread_safe_print("ℹ️  No markdown patch text found; continuing in case there are file-level or image changes.")

            exclude_folders = build_exclude_folders(repo_config)

            try:
                added_sections, modified_sections, deleted_sections, added_files, deleted_files, toc_files, keyword_files, added_images, modified_images, deleted_images, restructured_files, index_files, formatting_files = analyze_source_changes(
                    diff_context,
                    github_client,
                    special_files=SPECIAL_FILES,
                    ignore_files=commit_ignore_files,
                    repo_configs=repo_configs,
                    max_non_system_sections=MAX_NON_SYSTEM_SECTIONS_FOR_AI,
                    pr_diff=pr_diff,
                    exclude_folders=exclude_folders,
                    include_formatting_files=True,
                )
            except Exception as e:
                translation_stats.mark_failure(
                    group_label,
                    f"Source diff analysis failed: {sanitize_exception_message(e)}",
                )
                return finalize({
                    "attempted": True,
                    "successful_file_paths": successful_file_paths,
                    "counts": {
                        "added_files": 0,
                        "deleted_files": 0,
                        "toc_files": 0,
                        "keyword_files": 0,
                        "index_files": 0,
                        "formatting_files": 0,
                        "modified_sections": 0,
                        "added_images": 0,
                        "modified_images": 0,
                        "deleted_images": 0,
                    },
                })
        else:
            thread_safe_print("ℹ️  No matching incremental source changes found for this translation group.")
            added_sections = {}
            modified_sections = {}
            deleted_sections = {}
            added_files = {}
            deleted_files = []
            toc_files = {}
            keyword_files = {}
            index_files = {}
            formatting_files = {}
            added_images = []
            modified_images = []
            deleted_images = []
            restructured_files = set()

    full_translation_file_paths = set()
    if source_files_translation_mode == "full":
        full_translation_file_paths = apply_source_files_full_translation_mode(
            source_files,
            source_folder,
            filtered_changed_files,
            diff_context,
            github_client,
            commit_ignore_files,
            added_sections,
            modified_sections,
            deleted_sections,
            added_files,
            toc_files,
            keyword_files,
            translation_stats,
            index_files=index_files,
            formatting_files=formatting_files,
        )
    else:
        if source_full_files.strip():
            thread_safe_print(
                "\n📄 Translating newly entered TOC scope files from complete HEAD content."
            )
            full_translation_file_paths.update(
                apply_source_files_full_translation_mode(
                    source_full_files,
                    source_folder,
                    filtered_changed_files,
                    diff_context,
                    github_client,
                    commit_ignore_files,
                    added_sections,
                    modified_sections,
                    deleted_sections,
                    added_files,
                    toc_files,
                    keyword_files,
                    translation_stats,
                    index_files=index_files,
                    formatting_files=formatting_files,
                )
            )
        explicit_source_files = normalize_source_files(source_files, source_folder)
        if explicit_source_files:
            thread_safe_print(
                "ℹ️  Explicit SOURCE_FILES provided; skipping Cloud TOC scope expansion."
            )
        else:
            full_translation_file_paths.update(
                apply_toc_scope_added_files(
                    toc_files,
                    diff_context,
                    github_client,
                    commit_ignore_files,
                    added_sections,
                    modified_sections,
                    deleted_sections,
                    added_files,
                    keyword_files,
                    translation_stats,
                    index_files=index_files,
                    formatting_files=formatting_files,
                )
            )

    # Restructured files were rerouted to added_files by
    # detect_restructured_file() in analyze_source_changes().  They need
    # overwrite_existing=True because the target file already exists.
    if restructured_files:
        thread_safe_print(
            f"\n🔄 Restructured files detected ({len(restructured_files)}), "
            "will use structure-preserving reconciliation:"
        )
        for path in sorted(restructured_files):
            thread_safe_print(f"   - {path}")
        full_translation_file_paths.update(restructured_files)

        # Try structure-preserving reconciliation first: rebuild each file in
        # HEAD order while reusing the existing translation for unchanged /
        # moved sections. Successfully reconciled files are removed from the
        # full-translation queue. Existing targets that cannot be reconciled
        # are preserved and reported as failures; only missing targets can use
        # full-file creation.
        reconcile_restructured_files(
            restructured_files,
            added_files,
            full_translation_file_paths,
            diff_context,
            github_client,
            ai_client,
            repo_config,
            glossary_matcher,
            translation_stats,
            successful_file_paths,
        )

    changed_file_paths = {
        file.filename for file in filtered_changed_files if getattr(file, "filename", None)
    }
    file_processing_count = (
        len(full_translation_file_paths)
        if source_files_translation_mode == "full"
        else len(changed_file_paths | full_translation_file_paths)
    )
    parallel_file_processing = should_parallelize_file_processing(file_processing_count)
    if parallel_file_processing:
        thread_safe_print(f"⚡ Translation has {file_processing_count} files; file-level translation will use parallel chunks.")

    if deleted_files:
        thread_safe_print(f"\n🗑️  Processing {len(deleted_files)} deleted files...")
        delete_outcomes = process_deleted_files(
            deleted_files, github_client, repo_config, return_outcomes=True
        )
        translation_stats.record_outcomes(delete_outcomes)
        if not delete_outcomes.all_succeeded:
            thread_safe_print(f"   ⚠️  Some file deletions failed")
        git_add_changes(TARGET_REPO_PATH)

    if added_files:
        thread_safe_print(f"\n📄 Processing {len(added_files)} added files...")
        added_tasks = []
        for file_path, file_content in added_files.items():
            def run_added_file(path=file_path, content=file_content):
                outcomes = process_added_files(
                    {path: content},
                    diff_context,
                    github_client,
                    ai_client,
                    repo_config,
                    glossary_matcher=glossary_matcher,
                    return_outcomes=True,
                    overwrite_existing=path in full_translation_file_paths,
                )
                outcomes = normalize_added_file_outcomes(outcomes, path)
                outcome = outcomes.get(
                    path,
                    {"status": "failed", "reason": "Missing added-file outcome"},
                )
                return make_task_result(outcome["status"], outcome.get("reason", ""))

            added_tasks.append(make_file_task(file_path, run_added_file))

        added_results = run_file_tasks(added_tasks, "added files", parallel_file_processing)
        for result in added_results:
            if _record_translation_task_result(result, translation_stats):
                successful_file_paths.add(result["file_path"])
        git_add_successful_task_changes(added_results, TARGET_REPO_PATH)

    if toc_files:
        thread_safe_print(f"\n📋 Processing {len(toc_files)} TOC files...")
        toc_tasks = []
        for file_path, toc_data in toc_files.items():
            def run_toc_file(path=file_path, data=toc_data):
                if data.get("type") != "toc":
                    return make_task_result("failure", f"Unknown TOC data type: {data.get('type')}")

                success = process_toc_file(
                    path,
                    data,
                    diff_context,
                    github_client,
                    ai_client,
                    repo_config,
                    glossary_matcher=glossary_matcher,
                )
                if success:
                    return make_task_result("success")
                return make_task_result("failure", "TOC processor returned failure")

            toc_tasks.append(make_file_task(file_path, run_toc_file))

        toc_results = run_file_tasks(toc_tasks, "TOC files", parallel_file_processing)
        for result in toc_results:
            if _record_translation_task_result(result, translation_stats):
                successful_file_paths.add(result["file_path"])
        git_add_successful_task_changes(toc_results, TARGET_REPO_PATH)

    if keyword_files:
        thread_safe_print(f"\n📋 Processing {len(keyword_files)} keyword files...")
        keyword_tasks = []
        for file_path, keyword_data in keyword_files.items():
            def run_keyword_file(path=file_path, data=keyword_data):
                if data.get("type") != "keyword":
                    return make_task_result("failure", f"Unknown keyword data type: {data.get('type')}")

                success = process_keyword_file(
                    path,
                    data,
                    diff_context,
                    github_client,
                    ai_client,
                    repo_config,
                )
                if success:
                    return make_task_result("success")
                return make_task_result("failure", "Keyword processor returned failure")

            keyword_tasks.append(make_file_task(file_path, run_keyword_file))

        keyword_results = run_file_tasks(keyword_tasks, "keyword files", parallel_file_processing)
        for result in keyword_results:
            if _record_translation_task_result(result, translation_stats):
                successful_file_paths.add(result["file_path"])
        git_add_successful_task_changes(keyword_results, TARGET_REPO_PATH)

    if index_files:
        thread_safe_print(f"\n📄 Processing {len(index_files)} _index.md files...")
        index_tasks = []
        for file_path, index_data in index_files.items():
            def run_index_file(path=file_path, data=index_data):
                if data.get("type") != "index":
                    return make_task_result("failure", f"Unknown index data type: {data.get('type')}")

                success = process_index_file(
                    path,
                    data,
                    diff_context,
                    github_client,
                    ai_client,
                    repo_config,
                    glossary_matcher=glossary_matcher,
                )
                if success:
                    return make_task_result("success")
                return make_task_result("failure", "_index.md processor returned failure")

            index_tasks.append(make_file_task(file_path, run_index_file))

        index_results = run_file_tasks(index_tasks, "_index.md files", parallel_file_processing)
        for result in index_results:
            if _record_translation_task_result(result, translation_stats):
                successful_file_paths.add(result["file_path"])
        git_add_successful_task_changes(index_results, TARGET_REPO_PATH)

    if formatting_files:
        thread_safe_print(
            f"\n🧹 Processing {len(formatting_files)} formatting-only files..."
        )
        formatting_tasks = []
        for file_path, formatting_data in formatting_files.items():
            def run_formatting_file(path=file_path, data=formatting_data):
                success, changed, reason, ai_attempted = (
                    apply_formatting_only_change_with_ai_fallback(
                        path,
                        data,
                        repo_config['target_local_path'],
                        ai_client,
                        repo_config['source_language'],
                        repo_config['target_language'],
                    )
                )
                if ai_attempted and success:
                    thread_safe_print(
                        f"   🤖 Used AI-assisted formatting line mapping for {path}"
                    )
                elif ai_attempted:
                    thread_safe_print(
                        f"   🤖 Attempted AI-assisted formatting line mapping for {path}"
                    )
                if not success:
                    return make_task_result("failure", reason)
                return make_task_result(
                    "success",
                    reason or ("Formatting already synchronized" if not changed else ""),
                )

            formatting_tasks.append(make_file_task(file_path, run_formatting_file))

        formatting_results = run_file_tasks(
            formatting_tasks,
            "formatting-only files",
            parallel_file_processing,
        )
        for result in formatting_results:
            if _record_translation_task_result(result, translation_stats):
                successful_file_paths.add(result["file_path"])
        git_add_successful_task_changes(formatting_results, TARGET_REPO_PATH)

    if modified_sections:
        thread_safe_print(f"\n📝 Processing {len(modified_sections)} modified files...")
        modified_tasks = []
        for source_file_path, file_sections in modified_sections.items():
            def run_modified_file(path=source_file_path, sections=file_sections):
                return _process_commit_modified_file(
                    path,
                    sections,
                    pr_diff,
                    diff_context,
                    github_client,
                    ai_client,
                    repo_config,
                    glossary_matcher,
                )

            modified_tasks.append(
                make_file_task(
                    source_file_path,
                    run_modified_file,
                    resource_key=path_resource_key(source_file_path),
                )
            )

        modified_results = run_file_tasks(modified_tasks, "modified files", parallel_file_processing)
        for result in modified_results:
            if _record_translation_task_result(result, translation_stats):
                successful_file_paths.add(result["file_path"])
        git_add_successful_task_changes(modified_results, TARGET_REPO_PATH)

    if added_images or modified_images or deleted_images:
        thread_safe_print("\n🖼️  Processing images...")
        image_outcomes = process_all_images(
            added_images,
            modified_images,
            deleted_images,
            diff_context,
            github_client,
            repo_config,
        )
        translation_stats.record_outcomes(image_outcomes)
        git_add_changes(TARGET_REPO_PATH)

    validate_successful_translation_structures(
        successful_file_paths,
        diff_context,
        github_client,
        translation_stats,
        repo_config=repo_config,
        pr_diff=pr_diff,
    )

    counts = {
        "added_files": len(added_files),
        "deleted_files": len(deleted_files),
        "toc_files": len(toc_files),
        "keyword_files": len(keyword_files),
        "index_files": len(index_files),
        "formatting_files": len(formatting_files),
        "modified_sections": len(modified_sections),
        "added_images": len(added_images),
        "modified_images": len(modified_images),
        "deleted_images": len(deleted_images),
    }
    print_group_summary(group_label, counts)
    return finalize({
        "attempted": True,
        "successful_file_paths": successful_file_paths,
        "counts": counts,
    })


def main():
    """Run the commit-based translation workflow."""
    try:
        source_files_translation_mode = get_effective_source_files_translation_mode(
            SOURCE_FILES_TRANSLATION_MODE,
            SOURCE_FILES,
        )
        configured_source_files_translation_mode = normalize_source_files_translation_mode(
            SOURCE_FILES_TRANSLATION_MODE
        )
    except Exception as e:
        thread_safe_print(f"❌ Invalid source files translation mode: {sanitize_exception_message(e)}")
        return 1

    required_env = {
        "SOURCE_REPO": SOURCE_REPO,
        "TARGET_REPO": TARGET_REPO,
        "GITHUB_TOKEN": GITHUB_TOKEN,
        "TARGET_REPO_PATH": TARGET_REPO_PATH,
        "SOURCE_HEAD_REF": SOURCE_HEAD_REF.strip(),
    }
    if source_files_translation_mode != "full":
        required_env["SOURCE_BASE_REF"] = SOURCE_BASE_REF.strip()

    if not all(required_env.values()):
        thread_safe_print("❌ Missing required environment variables:")
        thread_safe_print(f"   SOURCE_REPO: {SOURCE_REPO}")
        thread_safe_print(f"   TARGET_REPO: {TARGET_REPO}")
        thread_safe_print(f"   GITHUB_TOKEN: {'Set' if GITHUB_TOKEN else 'Not set'}")
        thread_safe_print(f"   TARGET_REPO_PATH: {TARGET_REPO_PATH}")
        if source_files_translation_mode != "full":
            thread_safe_print(f"   SOURCE_BASE_REF: {SOURCE_BASE_REF or '(empty)'}")
        thread_safe_print(f"   SOURCE_HEAD_REF: {SOURCE_HEAD_REF or '(empty)'}")
        return 1

    thread_safe_print("🔧 Auto Commit Sync Tool")
    thread_safe_print(f"📍 Source Repo: {SOURCE_REPO}")
    thread_safe_print(f"📍 Target Repo: {TARGET_REPO}")
    thread_safe_print(f"🌿 Source Branch: {SOURCE_BRANCH}")
    if source_files_translation_mode == "full":
        thread_safe_print("🧭 Compare Range: (not used in full mode)")
        thread_safe_print(f"🧭 Source Head Ref: {SOURCE_HEAD_REF}")
    else:
        thread_safe_print(f"🧭 Compare Range: {SOURCE_BASE_REF}...{SOURCE_HEAD_REF}")
    thread_safe_print(f"📁 Source Folder Filter: {SOURCE_FOLDER or '(none)'}")
    thread_safe_print(f"📄 Source File Filter: {SOURCE_FILES or '(none)'}")
    thread_safe_print(f"📄 Full Source File Bootstrap: {SOURCE_FULL_FILES or '(none)'}")
    if configured_source_files_translation_mode != source_files_translation_mode:
        thread_safe_print(
            "📄 Source Files Translation Mode: "
            f"{source_files_translation_mode} "
            f"(configured {configured_source_files_translation_mode}, ignored because SOURCE_FILES is empty)"
        )
    else:
        thread_safe_print(f"📄 Source Files Translation Mode: {source_files_translation_mode}")
    thread_safe_print(f"📦 Source Repo Path: {SOURCE_REPO_PATH or '(remote compare API)'}")
    thread_safe_print(f"🎯 Target Ref: {TARGET_REF or '(default branch)'}")
    thread_safe_print(f"📖 Prefer Local Target Read: {PREFER_LOCAL_TARGET_FOR_READ}")
    thread_safe_print(f"📁 Target Repo Path: {TARGET_REPO_PATH}")
    thread_safe_print(f"🤖 AI Provider: {AI_PROVIDER}")
    thread_safe_print(f"🚦 Fail on Translation Error: {FAIL_ON_TRANSLATION_ERROR}")
    run_type = normalize_commit_sync_run_type(COMMIT_SYNC_RUN_TYPE)
    thread_safe_print(f"🧭 Commit Sync Run Type: {run_type or '(unspecified)'}")

    clean_temp_output_dir()

    auth = Auth.Token(GITHUB_TOKEN)
    github_client = Github(auth=auth)
    repo_config = get_commit_repo_config()
    repo_configs = get_commit_repo_configs()

    try:
        ai_client = UnifiedAIClient(provider=AI_PROVIDER)
        thread_safe_print(f"🤖 AI Provider: {AI_PROVIDER.upper()} ({ai_client.model})")
    except Exception as e:
        thread_safe_print(f"❌ Failed to initialize AI client: {sanitize_exception_message(e)}")
        return 1

    commit_ignore_files = get_commit_ignore_files()
    repo_config["ignore_files"] = commit_ignore_files
    repo_configs[SOURCE_REPO]["ignore_files"] = commit_ignore_files

    terms_path = TERMS_PATH
    if not terms_path and TARGET_REPO_PATH:
        candidate = os.path.join(TARGET_REPO_PATH, "resources", "terms.md")
        if os.path.exists(candidate):
            terms_path = candidate
    thread_safe_print(f"\n📚 Loading glossary from: {terms_path or '(not configured)'}")
    glossary = load_glossary(terms_path) if terms_path else []
    glossary_matcher = create_glossary_matcher(glossary)
    translation_stats = TranslationStats()
    all_successful_file_paths = set()
    marker_aligned_file_paths = set()
    marker_file_paths = set()
    total_counts = {
        "added_files": 0,
        "deleted_files": 0,
        "toc_files": 0,
        "keyword_files": 0,
        "index_files": 0,
        "formatting_files": 0,
        "modified_sections": 0,
        "added_images": 0,
        "modified_images": 0,
        "deleted_images": 0,
    }

    if source_files_translation_mode == "full":
        base_ref = SOURCE_BASE_REF.strip()
        head_ref = SOURCE_HEAD_REF.strip()
        thread_safe_print(f"\n🚀 Starting commit-based full sync for SOURCE_FILES at: {head_ref}")
        diff_context = {
            "mode": "commit",
            "source_repo": SOURCE_REPO,
            "target_repo": TARGET_REPO,
            # Full mode does not compare refs; base_ref is kept only for shared
            # source-context helpers that expect the key to exist.
            "base_ref": base_ref or head_ref,
            "head_ref": head_ref,
            "changed_files": [],
            "repo_config": repo_config,
            "source_repo_path": SOURCE_REPO_PATH,
            "source_url": f"https://github.com/{SOURCE_REPO}/tree/{head_ref}",
            "source_description": f"full SOURCE_FILES at {head_ref}",
        }
        group_result = process_translation_group(
            "full SOURCE_FILES",
            source_files_translation_mode="full",
            source_files=SOURCE_FILES,
            source_folder=SOURCE_FOLDER,
            diff_context=diff_context,
            filtered_changed_files=[],
            pr_diff="",
            github_client=github_client,
            ai_client=ai_client,
            repo_config=repo_config,
            repo_configs=repo_configs,
            glossary_matcher=glossary_matcher,
            commit_ignore_files=commit_ignore_files,
            translation_stats=translation_stats,
        )
        all_successful_file_paths.update(group_result["successful_file_paths"])
        add_counts(total_counts, group_result["counts"])
    else:
        try:
            base_ref, head_ref = resolve_compare_range()
        except Exception as e:
            thread_safe_print(f"❌ Failed to resolve compare range: {sanitize_exception_message(e)}")
            return 1

        thread_safe_print(f"\n🚀 Starting commit-based sync for: {base_ref}...{head_ref}")

        try:
            diff_context = build_incremental_diff_context(
                base_ref,
                head_ref,
                github_client,
                repo_config,
                repo_configs,
            )
        except Exception as e:
            thread_safe_print(f"❌ Failed to build commit diff context: {sanitize_exception_message(e)}")
            return 1

        filtered_changed_files = filter_changed_files(
            diff_context["changed_files"],
            source_folder=SOURCE_FOLDER,
            source_files=SOURCE_FILES,
        )
        marker_groups = {}
        marker_file_paths = set()
        if run_type == "scheduled":
            filtered_changed_files, marker_groups, marker_file_paths = split_changed_files_by_corresponding_en_commit(
                filtered_changed_files,
                TARGET_REPO_PATH,
                base_ref,
            )
            retry_groups, retry_file_paths = load_retry_cursor_groups(TARGET_REPO_PATH)
            if retry_file_paths:
                filtered_changed_files = [
                    file
                    for file in filtered_changed_files
                    if not retry_file_paths.intersection(changed_file_candidate_paths(file))
                ]
                for retry_ref, file_paths in retry_groups.items():
                    marker_groups.setdefault(retry_ref, set()).update(file_paths)
                thread_safe_print(
                    "\n♻️  Retrying files from persisted per-file cursors: "
                    + ", ".join(sorted(retry_file_paths))
                )
        elif run_type == "manual" and SOURCE_FILES.strip():
            filtered_changed_files, marker_groups, marker_file_paths = split_manual_source_files_by_corresponding_en_commit(
                filtered_changed_files,
                SOURCE_FILES,
                SOURCE_FOLDER,
                TARGET_REPO_PATH,
                base_ref,
            )

        full_source_file_paths = normalize_source_files(
            SOURCE_FULL_FILES,
            SOURCE_FOLDER,
        )
        if full_source_file_paths:
            # A complete HEAD bootstrap supersedes both target-file markers and
            # persisted retry cursors. Keeping these paths in marker_groups
            # would process the same file again through an incremental range.
            marker_groups = exclude_paths_from_marker_groups(
                marker_groups,
                full_source_file_paths,
            )
            marker_file_paths -= full_source_file_paths

        if marker_groups:
            thread_safe_print("\n📌 Files with per-file Corresponding EN commit cursors:")
            for marker_ref, file_paths in sorted(marker_groups.items()):
                thread_safe_print(
                    f"   {marker_ref}: {', '.join(sorted(file_paths))}"
                )
        if len(marker_groups) > PER_FILE_MARKER_GROUP_WARNING_THRESHOLD:
            thread_safe_print(
                "   ⚠️  Many distinct per-file Corresponding EN commit cursors "
                f"detected ({len(marker_groups)} groups). This may increase compare API calls."
            )

        if filtered_changed_files or SOURCE_FULL_FILES.strip():
            diff_context["changed_files"] = filtered_changed_files
            pr_diff = build_diff_text(filtered_changed_files)
            group_result = process_translation_group(
                f"global cursor {base_ref}...{head_ref}",
                source_files_translation_mode="incremental",
                source_files=SOURCE_FILES,
                source_folder=SOURCE_FOLDER,
                diff_context=diff_context,
                filtered_changed_files=filtered_changed_files,
                pr_diff=pr_diff,
                github_client=github_client,
                ai_client=ai_client,
                repo_config=repo_config,
                repo_configs=repo_configs,
                glossary_matcher=glossary_matcher,
                commit_ignore_files=commit_ignore_files,
                translation_stats=translation_stats,
                source_full_files=SOURCE_FULL_FILES,
            )
            all_successful_file_paths.update(group_result["successful_file_paths"])
            add_counts(total_counts, group_result["counts"])
        elif not marker_groups:
            thread_safe_print("ℹ️  No matching source changes found after folder/file filtering.")

        for marker_ref, file_paths in sorted(marker_groups.items()):
            if commits_match(marker_ref, head_ref):
                thread_safe_print(
                    f"ℹ️  Skipping per-file cursor {marker_ref}; it already matches SOURCE_HEAD_REF."
                )
                marker_aligned_file_paths.update(file_paths)
                continue

            try:
                marker_diff_context = build_incremental_diff_context(
                    marker_ref,
                    head_ref,
                    github_client,
                    repo_config,
                    repo_configs,
                )
            except Exception as e:
                reason = f"Failed to build per-file cursor diff context: {sanitize_exception_message(e)}"
                for file_path in sorted(file_paths):
                    translation_stats.mark_failure(file_path, reason)
                    translation_stats.set_retry_ref(file_path, marker_ref)
                continue

            marker_source_files = ",".join(sorted(file_paths))
            marker_filtered_changed_files = filter_changed_files(
                marker_diff_context["changed_files"],
                source_folder="",
                source_files=marker_source_files,
            )
            marker_changed_paths = {
                path
                for file in marker_filtered_changed_files
                for path in (
                    getattr(file, "filename", None),
                    getattr(file, "previous_filename", None),
                )
                if path
            }
            marker_aligned_file_paths.update(set(file_paths) - marker_changed_paths)
            marker_diff_context["changed_files"] = marker_filtered_changed_files
            if not marker_filtered_changed_files:
                thread_safe_print(
                    "ℹ️  No source changes found since per-file cursor "
                    f"{marker_ref} for: {marker_source_files}"
                )
                continue

            marker_pr_diff = build_diff_text(marker_filtered_changed_files)
            group_result = process_translation_group(
                f"per-file cursor {marker_ref}...{head_ref}",
                source_files_translation_mode="incremental",
                source_files=marker_source_files,
                source_folder="",
                diff_context=marker_diff_context,
                filtered_changed_files=marker_filtered_changed_files,
                pr_diff=marker_pr_diff,
                github_client=github_client,
                ai_client=ai_client,
                repo_config=repo_config,
                repo_configs=repo_configs,
                glossary_matcher=glossary_matcher,
                commit_ignore_files=commit_ignore_files,
                translation_stats=translation_stats,
            )
            all_successful_file_paths.update(group_result["successful_file_paths"])
            add_counts(total_counts, group_result["counts"])

    if run_type in {"manual", "scheduled"}:
        add_if_missing = run_type == "manual"
        remove_if_present = run_type == "scheduled"
        marker_update_paths = (
            all_successful_file_paths | marker_aligned_file_paths
            if add_if_missing
            else (all_successful_file_paths | marker_aligned_file_paths) & marker_file_paths
        )
    else:
        marker_update_paths = set()

    # A partial translation is useful enough to stage and push, but it is not a
    # completed cursor checkpoint.  Keeping its old marker prevents a future
    # run from silently treating an incomplete response as fully synchronized.
    incomplete_translation_paths = {
        path for path, _ in translation_stats.partial + translation_stats.failed
    }
    incomplete_translation_paths.update(
        issue.file_path for issue in translation_stats.structure_errors
    )
    marker_update_paths -= incomplete_translation_paths

    if marker_update_paths:
        updated_marker_files, marker_failures = update_corresponding_en_commit_for_files(
            marker_update_paths,
            TARGET_REPO_PATH,
            SOURCE_HEAD_REF.strip(),
            add_if_missing=add_if_missing,
            remove_if_present=remove_if_present,
        )
        if updated_marker_files:
            if add_if_missing:
                thread_safe_print(
                    "\n📌 Added or updated Corresponding EN commit markers: "
                    f"{', '.join(updated_marker_files)}"
                )
            else:
                thread_safe_print(
                    "\n📌 Removed Corresponding EN commit markers from scheduled successes: "
                    f"{', '.join(updated_marker_files)}"
                )
            git_add_changes(TARGET_REPO_PATH)
        for file_path, reason in marker_failures.items():
            translation_stats.mark_failure(
                file_path,
                f"Failed to update Corresponding EN commit marker: {reason}",
            )

    if run_type == "scheduled":
        completed_paths = (
            set(translation_stats.succeeded) | marker_aligned_file_paths
        ) - incomplete_translation_paths
        if update_retry_cursor_ledger(
            TARGET_REPO_PATH,
            translation_stats,
            completed_paths,
            source_branch=SOURCE_BRANCH,
            source_head_ref=SOURCE_HEAD_REF.strip(),
        ):
            git_add_changes(TARGET_REPO_PATH)

    thread_safe_print("\n" + "=" * 80)
    thread_safe_print("📊 Final Summary:")
    thread_safe_print("=" * 80)
    thread_safe_print(f"   📄 Added files: {total_counts['added_files']} processed")
    thread_safe_print(f"   🗑️  Deleted files: {total_counts['deleted_files']} processed")
    thread_safe_print(f"   📋 TOC files: {total_counts['toc_files']} processed")
    thread_safe_print(f"   📋 Keyword files: {total_counts['keyword_files']} processed")
    thread_safe_print(f"   📄 Index files: {total_counts.get('index_files', 0)} processed")
    thread_safe_print(f"   🧹 Formatting-only files: {total_counts.get('formatting_files', 0)} processed")
    thread_safe_print(f"   📝 Modified files: {total_counts['modified_sections']} processed")
    thread_safe_print(f"   🖼️  Added images: {total_counts['added_images']} processed")
    thread_safe_print(f"   🖼️  Modified images: {total_counts['modified_images']} processed")
    thread_safe_print(f"   🖼️  Deleted images: {total_counts['deleted_images']} processed")
    translation_stats.print_summary()
    translation_stats.write_failure_report(os.path.join(os.path.dirname(__file__), "temp_output"))
    thread_safe_print("=" * 80)
    if translation_stats.partial or translation_stats.failed:
        thread_safe_print(
            "⚠️  The commit-based sync workflow preserved useful per-file output "
            "and recorded incomplete work in the failure report."
        )
    if translation_stats.structure_errors:
        thread_safe_print("⚠️  The commit-based sync workflow completed with document structure mismatches.")
    if translation_stats.failed and FAIL_ON_TRANSLATION_ERROR:
        return 1
    if (
        not translation_stats.partial
        and not translation_stats.failed
        and not translation_stats.structure_errors
    ):
        thread_safe_print("🎉 The commit-based sync workflow completed successfully!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

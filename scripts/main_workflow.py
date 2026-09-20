"""
Main Entry Point for GitHub Workflow
Orchestrates the entire auto-sync workflow in GitHub Actions environment
"""

import sys
import os
import json
import re
import subprocess
import tiktoken
from github import Github, Auth
from product_specific_handler import get_product_name

# Configuration from environment variables
SOURCE_PR_URL = os.getenv("SOURCE_PR_URL")
TARGET_PR_URL = os.getenv("TARGET_PR_URL")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
AI_PROVIDER = os.getenv("AI_PROVIDER", "deepseek")
TARGET_REPO_PATH = os.getenv("TARGET_REPO_PATH")
SOURCE_FILES = os.getenv("SOURCE_FILES", "")
SKIP_GIT_ADD = os.getenv("SKIP_GIT_ADD", "false").lower() == "true"
PREFER_LOCAL_TARGET_FOR_READ = os.getenv("PREFER_LOCAL_TARGET_FOR_READ", "true").lower() == "true"
TIDB_CLOUD_ABSOLUTE_LINK_PREFIX = os.getenv(
    "TIDB_CLOUD_ABSOLUTE_LINK_PREFIX",
    "https://docs.pingcap.com/tidbcloud/",
)
os.environ.setdefault(
    "TIDB_CLOUD_ABSOLUTE_LINK_PREFIX",
    TIDB_CLOUD_ABSOLUTE_LINK_PREFIX,
)

# Import all modules
from ai_client import UnifiedAIClient, get_provider_max_tokens, thread_safe_print, print_lock
from diff_analyzer import (
    analyze_source_changes,
    build_diff_text,
    build_pr_diff_context,
    get_source_file_content,
    get_repo_config,
    get_target_hierarchy_and_content,
    get_source_file_hierarchy,
    parse_pr_url,
)
from image_processor import process_all_images
from file_adder import process_added_files
from file_deleter import process_deleted_files
from file_io import atomic_write_text, read_safe_target_text
from file_updater import process_files_in_batches, process_added_sections, process_modified_sections
from formatting_sync import apply_formatting_only_change_with_ai_fallback
from toc_processor import process_toc_file
from keword_processor import process_keyword_file
from section_matcher import match_source_diff_to_target
from glossary import load_glossary, create_glossary_matcher
from log_sanitizer import sanitize_exception_message, safe_target_path
from parallel_file_processor import (
    count_unique_file_paths,
    make_file_task,
    make_task_result,
    run_file_tasks,
    should_parallelize_file_processing,
)
from special_file_utils import is_index_file_name, is_toc_file_name, path_resource_key
from structural_reconciler import reconcile_restructured_file
from workflow_ignore_config import load_workflow_ignore_config
from workflow_outcome import RunReport, record_file_task_result

# Glossary terms path (optional, defaults to resources/terms.md in the docs repo)
TERMS_PATH = os.getenv("TERMS_PATH", "")

# Processing limit configuration
def positive_int_env(name, default):
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


MAX_NON_SYSTEM_SECTIONS_FOR_AI = positive_int_env(
    "MAX_NON_SYSTEM_SECTIONS_FOR_AI", 120
)
SOURCE_TOKEN_LIMIT = positive_int_env(
    "SOURCE_TOKEN_LIMIT", 50000
)  # Applies to regular modified-file source sections.

# Special file configuration
SPECIAL_FILES = ["TOC.md", "keywords.md"]
WORKFLOW_IGNORE_CONFIG = load_workflow_ignore_config()
PR_MODE_IGNORE_FILES = WORKFLOW_IGNORE_CONFIG["PR_MODE_IGNORE_FILES"]
PR_MODE_IGNORE_FOLDERS = WORKFLOW_IGNORE_CONFIG["PR_MODE_IGNORE_FOLDERS"]


def normalize_added_file_outcomes(result, file_path):
    """Adapt legacy bool/tuple mocks and callers to the per-file outcome contract."""
    if isinstance(result, dict):
        return result
    if isinstance(result, tuple):
        success = bool(result[0])
        failures = result[1] if len(result) > 1 and isinstance(result[1], dict) else {}
        return {
            file_path: {
                "status": "success" if success else "failed",
                "reason": failures.get(file_path, "") if not success else "",
            }
        }
    return {
        file_path: {
            "status": "success" if result else "failed",
            "reason": "" if result else "Added file processor returned failure",
        }
    }

# Repository configuration for workflow
def get_workflow_repo_configs():
    """Get repository configuration based on environment variables"""
    if not SOURCE_PR_URL or not TARGET_PR_URL:
        raise ValueError("SOURCE_PR_URL and TARGET_PR_URL must be set")
    
    # Parse source and target repo info
    source_owner, source_repo, _ = parse_pr_url(SOURCE_PR_URL)
    target_owner, target_repo, _ = parse_pr_url(TARGET_PR_URL)
    
    source_repo_key = f"{source_owner}/{source_repo}"
    target_repo_key = f"{target_owner}/{target_repo}"
    
    # Determine language direction based on repo names
    if source_repo.endswith('-cn') and not target_repo.endswith('-cn'):
        # Chinese to English
        source_language = "Chinese"
        target_language = "English"
    elif not source_repo.endswith('-cn') and target_repo.endswith('-cn'):
        # English to Chinese
        source_language = "English"
        target_language = "Chinese"
    else:
        # Default fallback
        source_language = "English"
        target_language = "Chinese"
    
    return {
        source_repo_key: {
            "target_repo": target_repo_key,
            "target_local_path": TARGET_REPO_PATH,
            "prefer_local_target_for_read": PREFER_LOCAL_TARGET_FOR_READ,
            "source_mode": "pr",
            "source_language": source_language,
            "target_language": target_language
        }
    }

def ensure_temp_output_dir():
    """Ensure the temp_output directory exists"""
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    temp_dir = os.path.join(script_dir, "temp_output")
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir

def clean_temp_output_dir():
    """Clean the temp_output directory at the start of execution"""
    import shutil
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    temp_dir = os.path.join(script_dir, "temp_output")
    if os.path.exists(temp_dir):
        if os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir)
            thread_safe_print(f"🧹 Cleaned existing temp_output directory")
        else:
            # Remove file if it exists
            os.remove(temp_dir)
            thread_safe_print(f"🧹 Removed existing temp_output file")
    os.makedirs(temp_dir, exist_ok=True)
    thread_safe_print(f"📁 Created temp_output directory: {temp_dir}")
    return temp_dir

def git_add_changes(target_repo_path):
    """Stage all current changes in the target repo so they survive a later failure."""
    if SKIP_GIT_ADD:
        thread_safe_print("   ⏭️  SKIP_GIT_ADD=true; skipping git add .")
        return

    result = subprocess.run(
        ["git", "add", "."],
        cwd=target_repo_path,
        capture_output=True, text=True
    )
    if result.returncode == 0:
        thread_safe_print(f"   ✅ git add . completed in {target_repo_path}")
    else:
        thread_safe_print(f"   ❌ git add . failed: {result.stderr}")

def git_add_successful_task_changes(task_results, target_repo_path):
    """Stage successful task writes after a parallel batch has finished."""
    has_success = any(
        result["ok"]
        and result["result"]
        and result["result"].get("status") in {"success", "partial"}
        for result in task_results
    )
    if not has_success:
        return

    # Keep staging out of worker threads so git add never races with active file writes.
    git_add_changes(target_repo_path)


def reconcile_restructured_files_for_pr(
    restructured_files,
    added_files,
    source_context,
    github_client,
    ai_client,
    repo_config,
    glossary_matcher,
    run_report,
):
    """Handle PR-mode restructured files without full-file overwrite.

    Source section moves need the structural reconciler, which reuses existing
    target-language sections for unchanged content. If reconciliation fails,
    preserve the existing target file and mark the file as failed instead of
    falling back to destructive full-file translation.
    """
    reconciled_paths = set()
    failed_paths = set()
    source_mode = (
        source_context.get("mode", "pr")
        if isinstance(source_context, dict)
        else "pr"
    )

    for file_path in sorted(restructured_files):
        existing_target = read_safe_target_text(TARGET_REPO_PATH, file_path)
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
            run_report.mark_failure(file_path, reason)
            failed_paths.add(file_path)

        try:
            head_source = get_source_file_content(
                file_path,
                source_context,
                github_client,
                ref_name="head_ref",
            )
            base_source = get_source_file_content(
                file_path,
                source_context,
                github_client,
                ref_name="base_ref",
            )
        except Exception as error:
            mark_reconciliation_failure(
                f"could not load source for {file_path}: "
                f"{sanitize_exception_message(error)}"
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
        except Exception as error:
            mark_reconciliation_failure(
                f"reconciliation raised for {file_path}: "
                f"{sanitize_exception_message(error)}"
            )
            continue

        if reconciled is None:
            mark_reconciliation_failure(
                f"structure-preserving reconciliation declined {file_path}"
            )
            continue

        try:
            target_file_path = safe_target_path(TARGET_REPO_PATH, file_path)
        except ValueError as error:
            mark_reconciliation_failure(sanitize_exception_message(error))
            continue

        try:
            atomic_write_text(target_file_path, reconciled)
        except Exception as error:
            mark_reconciliation_failure(
                f"failed to write {target_file_path}: "
                f"{sanitize_exception_message(error)}"
            )
            continue

        thread_safe_print(f"   ✅ Reconciled (structure-preserving): {file_path}")
        run_report.mark_success(file_path)
        reconciled_paths.add(file_path)

    for file_path in reconciled_paths | failed_paths:
        added_files.pop(file_path, None)
        restructured_files.discard(file_path)

    if reconciled_paths:
        git_add_changes(TARGET_REPO_PATH)

    return reconciled_paths


def estimate_tokens(text):
    """Calculate accurate token count using tiktoken (GPT-4/3.5 encoding)"""
    if not text:
        return 0
    try:
        enc = tiktoken.get_encoding("cl100k_base")  # GPT-4/3.5 encoding
        tokens = enc.encode(text)
        return len(tokens)
    except Exception as e:
        # Fallback to character approximation if tiktoken fails
        thread_safe_print(
            f"   ⚠️  Tiktoken encoding failed: {sanitize_exception_message(e)}, using character approximation"
        )
        return len(text) // 4

def print_token_estimation(prompt_text, context="AI translation"):
    """Print accurate token consumption for a request"""
    actual_tokens = estimate_tokens(prompt_text)
    char_count = len(prompt_text)
    thread_safe_print(f"   💰 {context}")
    thread_safe_print(f"      📝 Input: {char_count:,} characters")
    thread_safe_print(f"      🔢 Actual tokens: {actual_tokens:,} (using tiktoken cl100k_base)")
    return actual_tokens

def check_source_token_limit(source_diff_dict_file, token_limit=SOURCE_TOKEN_LIMIT):
    """Check if the total tokens of all new_content in source-diff-dict exceeds the limit"""
    try:
        with open(source_diff_dict_file, 'r', encoding='utf-8') as f:
            source_diff_dict = json.load(f)
        
        total_new_content = ""
        section_count = 0
        
        for key, section_data in source_diff_dict.items():
            if isinstance(section_data, dict):
                new_content = section_data.get('new_content', '')
                if new_content:
                    total_new_content += new_content + "\n"
                    section_count += 1
        
        if not total_new_content.strip():
            thread_safe_print(f"   ⚠️  No new_content found in {source_diff_dict_file}")
            return True, 0, 0  # Allow processing if no content to check
        
        total_tokens = estimate_tokens(total_new_content)
        char_count = len(total_new_content)
        
        thread_safe_print(f"   📊 Source token limit check:")
        thread_safe_print(f"      📝 Total new_content: {char_count:,} characters from {section_count} sections")
        thread_safe_print(f"      🔢 Total tokens: {total_tokens:,}")
        thread_safe_print(f"      🚧 Token limit: {token_limit:,}")
        
        if total_tokens > token_limit:
            thread_safe_print(f"      ❌ Token limit exceeded! ({total_tokens:,} > {token_limit:,})")
            return False, total_tokens, token_limit
        else:
            thread_safe_print(f"      ✅ Within token limit ({total_tokens:,} ≤ {token_limit:,})")
            return True, total_tokens, token_limit
            
    except Exception as e:
        thread_safe_print(
            f"   ❌ Error checking token limit for {source_diff_dict_file}: {sanitize_exception_message(e)}"
        )
        return True, 0, 0  # Allow processing on error to avoid blocking


def extract_source_diff_section_title(section_data):
    """Return a compact source section title for failure reports."""
    for field in ("new_content", "old_content"):
        content = section_data.get(field) or ""
        for line in str(content).splitlines():
            stripped = line.strip()
            if re.match(r'^#{1,10}\s+\S', stripped):
                return re.sub(r'^#{1,10}\s+', '', stripped).strip()

    hierarchy = section_data.get("original_hierarchy", "")
    if hierarchy:
        leaf = hierarchy.split(" > ")[-1]
        return re.sub(r'^#{1,10}\s+', '', leaf.strip()).strip()

    return "(unknown section)"


def get_unmatched_modified_source_sections(source_diff_dict, matched_sections):
    """Find modified source sections that could not be mapped to target sections."""
    matched_keys = set(matched_sections or {})
    missing = []

    for key, section_data in source_diff_dict.items():
        if section_data.get("operation") != "modified":
            continue
        if key in matched_keys:
            continue

        missing.append(
            {
                "key": key,
                "line": section_data.get("new_line_number", "?"),
                "title": extract_source_diff_section_title(section_data),
            }
        )

    return missing


def format_unmatched_modified_sections_failure(file_path, missing_sections):
    """Build a one-line reason suitable for translation failure reports."""
    details = "; ".join(
        f"{item['key']} (source line {item['line']}): {item['title']}"
        for item in missing_sections
    )
    return (
        f"Target file {file_path} is missing or could not map "
        f"{len(missing_sections)} modified source section(s), so translation was skipped. "
        f"Missing sections: {details}. "
        "Please add or sync these sections in the target branch first, then rerun translation for this file."
    )


def get_pr_diff(pr_url, github_client):
    """Get the diff content from a GitHub PR (from auto-sync-pr-changes.py)"""
    try:
        from diff_analyzer import get_pr_diff as get_diff
        return get_diff(pr_url, github_client)
    except Exception as e:
        thread_safe_print(f"   ❌ Error getting PR diff: {sanitize_exception_message(e)}")
        return None

def filter_diff_by_operation_type(pr_diff, operation_type, target_sections=None):
    """Filter PR diff to only include changes relevant to specific operation type"""
    
    if not pr_diff:
        return ""
    
    if operation_type == "modified":
        # For modified sections, we want the full diff but focus on changed content
        return pr_diff
    elif operation_type == "added":
        # For added sections, we want to show what was added
        filtered_lines = []
        for line in pr_diff.split('\n'):
            if line.startswith('+') and not line.startswith('+++'):
                filtered_lines.append(line)
            elif line.startswith('@@') or line.startswith('File:'):
                filtered_lines.append(line)
        return '\n'.join(filtered_lines)
    elif operation_type == "deleted":
        # For deleted sections, we want to show what was removed
        filtered_lines = []
        for line in pr_diff.split('\n'):
            if line.startswith('-') and not line.startswith('---'):
                filtered_lines.append(line)
            elif line.startswith('@@') or line.startswith('File:'):
                filtered_lines.append(line)
        return '\n'.join(filtered_lines)
    
    return pr_diff

def filter_diff_for_target_file(pr_diff, target_file, source_diff_dict):
    """Extract file-specific diff from the complete PR diff based on source files that map to the target file"""
    if not pr_diff or not source_diff_dict:
        return pr_diff
    
    # Extract source files that contribute to this target file
    source_files = set()
    for key, section_data in source_diff_dict.items():
        if isinstance(section_data, dict):
            source_file = section_data.get('source_file', '')
            if source_file:
                source_files.add(source_file)
    
    if not source_files:
        thread_safe_print(f"   ⚠️  No source files found in source_diff_dict, using complete PR diff")
        return pr_diff
    
    thread_safe_print(f"   📄 Source files contributing to {target_file}: {list(source_files)}")
    
    # Filter PR diff to only include changes from these source files
    filtered_lines = []
    current_file = None
    include_section = False
    
    for line in pr_diff.split('\n'):
        if line.startswith('File: '):
            current_file = line.replace('File: ', '').strip()
            include_section = current_file in source_files
            if include_section:
                filtered_lines.append(line)
        elif line.startswith('-' * 80):
            if include_section:
                filtered_lines.append(line)
        elif include_section:
            filtered_lines.append(line)
    
    file_specific_diff = '\n'.join(filtered_lines)
    thread_safe_print(f"   📊 Filtered diff: {len(file_specific_diff)} chars (from {len(pr_diff)} chars)")
    
    return file_specific_diff if file_specific_diff.strip() else pr_diff

def extract_file_diff_from_pr(pr_diff, source_file_path):
    """Extract diff content for a specific source file from the complete PR diff"""
    if not pr_diff:
        return ""
    
    filtered_lines = []
    current_file = None
    include_section = False
    
    for line in pr_diff.split('\n'):
        if line.startswith('File: '):
            current_file = line.replace('File: ', '').strip()
            include_section = (current_file == source_file_path)
            if include_section:
                filtered_lines.append(line)
        elif line.startswith('-' * 80):
            if include_section:
                filtered_lines.append(line)
                include_section = False  # End of this file's section
        elif include_section:
            filtered_lines.append(line)
    
    return '\n'.join(filtered_lines)

def determine_file_processing_type(source_file_path, file_sections, special_files=None, ignore_files=None):
    """Determine how to process a file based on operation type and file characteristics"""
    basename = os.path.basename(source_file_path)
    ignore_files = PR_MODE_IGNORE_FILES if ignore_files is None else ignore_files
    
    # Check if this is a special file (like TOC.md or keywords.md)
    if is_toc_file_name(source_file_path, ignore_files):
        return "special_file_toc"

    if is_index_file_name(source_file_path, ignore_files):
        return "special_file_index"

    if special_files and basename in special_files:
        if basename == "keywords.md":
            if isinstance(file_sections, dict) and file_sections.get("keyword_regular_only"):
                return "regular_modified"
            return "special_file_keyword"
    
    # For all other modified files, use regular processing
    return "regular_modified"

def process_regular_modified_file(source_file_path, file_sections, file_diff, source_context_or_pr_url, github_client, ai_client, repo_config, max_sections, glossary_matcher=None, return_details=False):
    """Process a regular markdown file that has been modified"""
    def finish(success, reason=""):
        if return_details:
            return success, reason
        return success

    try:
        thread_safe_print(f"   📝 Processing as regular modified file: {source_file_path}")
        
        # Extract the actual sections from the file_sections structure
        # file_sections contains: {'sections': {...}, 'original_hierarchy': {...}, 'current_hierarchy': {...}}
        if isinstance(file_sections, dict) and 'sections' in file_sections:
            actual_sections = file_sections['sections']
        else:
            # Fallback: assume file_sections is already the sections dict
            actual_sections = file_sections
        
        thread_safe_print(f"   📊 Extracted sections: {len(actual_sections)} sections")
        
        # CRITICAL: Load the source-diff-dict.json and perform matching
        import json
        import os
        from section_matcher import match_source_diff_to_target
        from diff_analyzer import get_target_hierarchy_and_content
        
        # Load source-diff-dict.json with file prefix
        temp_dir = ensure_temp_output_dir()
        file_prefix = path_resource_key(source_file_path)
        source_diff_dict_file = os.path.join(temp_dir, f"{file_prefix}-source-diff-dict.json")
        if os.path.exists(source_diff_dict_file):
            with open(source_diff_dict_file, 'r', encoding='utf-8') as f:
                source_diff_dict = json.load(f)
            thread_safe_print(f"   📂 Loaded source diff dict with {len(source_diff_dict)} sections from {source_diff_dict_file}")
            
            # Check source token limit before proceeding with processing
            thread_safe_print(f"   🔍 Checking source token limit...")
            within_limit, total_tokens, token_limit = check_source_token_limit(source_diff_dict_file)
            if not within_limit:
                thread_safe_print(f"   🚫 Skipping file processing: source content exceeds token limit")
                thread_safe_print(f"      📊 Total tokens: {total_tokens:,} > Limit: {token_limit:,}")
                thread_safe_print(f"      ⏭️  File {source_file_path} will not be processed")
                return finish(
                    False,
                    f"Source content exceeds token limit ({total_tokens:,} > {token_limit:,})",
                )
                
        else:
            thread_safe_print(f"   ❌ {source_diff_dict_file} not found")
            return finish(False, f"{source_diff_dict_file} not found")
        
        # Get target file hierarchy and content
        target_repo = repo_config['target_repo']
        target_hierarchy, target_lines = get_target_hierarchy_and_content(
            source_file_path,
            github_client,
            target_repo,
            repo_config.get('target_local_path'),
            repo_config.get('prefer_local_target_for_read', False),
            repo_config.get('target_ref'),
        )
        
        if not target_hierarchy or not target_lines:
            thread_safe_print(f"   ❌ Could not get target file content for {source_file_path}")
            return finish(False, f"Could not get target file content for {source_file_path}")
        
        thread_safe_print(f"   📖 Target file: {len(target_hierarchy)} sections, {len(target_lines)} lines")
        
        source_mode = source_context_or_pr_url.get("mode", "pr") if isinstance(source_context_or_pr_url, dict) else "pr"

        def load_source_hierarchy_context():
            source_base_hierarchy = get_source_file_hierarchy(
                source_file_path,
                source_context_or_pr_url,
                github_client,
                get_base_version=True,
            )
            source_head_hierarchy = get_source_file_hierarchy(
                source_file_path,
                source_context_or_pr_url,
                github_client,
                get_base_version=False,
            )
            thread_safe_print(
                f"   🧭 Source hierarchy context for retry: "
                f"{len(source_base_hierarchy)} base sections, {len(source_head_hierarchy)} head sections"
            )
            return source_base_hierarchy, source_head_hierarchy

        # Perform source diff to target matching
        thread_safe_print(f"   🔗 Matching source diff to target...")
        enhanced_sections = match_source_diff_to_target(
            source_diff_dict, 
            target_hierarchy, 
            target_lines, 
            ai_client, 
            repo_config, 
            max_sections,
            get_provider_max_tokens(AI_PROVIDER),
            source_mode=source_mode,
            source_hierarchy_provider=load_source_hierarchy_context,
        )
        
        if not enhanced_sections:
            thread_safe_print(f"   ❌ No sections matched")
            return finish(False, "No sections matched")
        
        thread_safe_print(f"   ✅ Matched {len(enhanced_sections)} sections")
        
        # Save the match result for reference
        match_file = os.path.join(temp_dir, f"{path_resource_key(source_file_path)}-match_source_diff_to_target.json")
        with open(match_file, 'w', encoding='utf-8') as f:
            json.dump(enhanced_sections, f, ensure_ascii=False, indent=2)
        thread_safe_print(f"   💾 Saved match result to: {match_file}")

        missing_modified_sections = get_unmatched_modified_source_sections(
            source_diff_dict,
            enhanced_sections,
        )
        if missing_modified_sections:
            failure_reason = format_unmatched_modified_sections_failure(
                source_file_path,
                missing_modified_sections,
            )
            thread_safe_print(
                f"   ❌ Skipping {source_file_path}: "
                f"{len(missing_modified_sections)} modified source section(s) were not found in target"
            )
            for item in missing_modified_sections:
                thread_safe_print(
                    f"      - {item['key']} (source line {item['line']}): {item['title']}"
                )
            return finish(False, failure_reason)
        
        # Separate deterministic/minimal heading changes from sections needing AI translation.
        from file_updater import apply_heading_level_change_to_target
        from structural_reconciler import reconcile_version_mark_only_change
        heading_level_only_results = {}
        version_mark_only_results = {}
        sections_for_ai = {}

        for key, section_info in enhanced_sections.items():
            source_entry = source_diff_dict.get(key, {})
            version_mark_result = None
            if source_entry.get('operation') == 'modified':
                version_mark_result = reconcile_version_mark_only_change(
                    source_entry.get('new_content')
                    or section_info.get('source_new_content', ''),
                    source_entry.get('old_content')
                    or section_info.get('source_old_content', ''),
                    section_info.get('target_content', ''),
                    ai_client,
                    repo_config['source_language'],
                    repo_config['target_language'],
                    glossary_matcher=glossary_matcher,
                    source_mode=source_mode,
                )

            if version_mark_result is not None:
                version_mark_only_results[key] = version_mark_result
                thread_safe_print(
                    f"   ♻️  Version-mark-only: {key} — updated only the span "
                    "inner text and reused the rest of the target section"
                )
            elif source_entry.get('heading_level_change_only'):
                target_content = section_info.get('target_content', '')
                old_level = source_entry['old_heading_level']
                new_level = source_entry['new_heading_level']
                new_target = apply_heading_level_change_to_target(
                    target_content, old_level, new_level
                )
                heading_level_only_results[key] = new_target
                thread_safe_print(
                    f"   📐 Heading-level-only: {key} "
                    f"({'#' * old_level} → {'#' * new_level}) — applied directly to target"
                )
            else:
                sections_for_ai[key] = section_info

        if heading_level_only_results or version_mark_only_results:
            thread_safe_print(
                f"   ♻️  Applied {len(version_mark_only_results)} version-mark-only and "
                f"{len(heading_level_only_results)} heading-level-only update(s); "
                f"{len(sections_for_ai)} section(s) still need normal AI translation"
            )

        # Step 2: Get AI translation for the remaining matched sections
        ai_updated_sections = {}
        ai_success = True

        if sections_for_ai:
            thread_safe_print(f"   🤖 Getting AI translation for {len(sections_for_ai)} matched sections...")

            file_data = {
                source_file_path: {
                    'type': 'enhanced_sections',
                    'sections': sections_for_ai
                }
            }

            results = process_modified_sections(file_data, file_diff, source_context_or_pr_url, github_client, ai_client, repo_config, max_sections, glossary_matcher=glossary_matcher)

            if results and len(results) > 0:
                file_path, ai_success, ai_updated_sections = results[0]
                if not ai_success or not isinstance(ai_updated_sections, dict):
                    failure_reason = "AI translation failed or returned invalid results"
                    if hasattr(ai_updated_sections, "failures") and ai_updated_sections.failures:
                        failure_reason = "; ".join(ai_updated_sections.failures)
                    thread_safe_print(f"   ⚠️  AI translation failed or returned invalid results")
                    return finish(False, failure_reason)
            else:
                if not heading_level_only_results and not version_mark_only_results:
                    thread_safe_print(f"   ⚠️  No results from process_modified_sections")
                    return finish(False, "No results from process_modified_sections")
                ai_updated_sections = {}
        else:
            thread_safe_print(
                "   ♻️  All matched sections used deterministic/minimal heading "
                "updates — skipping normal section translation"
            )

        # Step 3: Update match_source_diff_to_target.json with results
        thread_safe_print(f"   📝 Step 3: Updating {match_file} with results...")

        with open(match_file, 'r', encoding='utf-8') as f:
            match_data = json.load(f)

        updated_count = 0
        for key, section_data in match_data.items():
            operation = section_data.get('source_operation', '')

            if operation == 'deleted':
                section_data['target_new_content'] = None
            elif key in version_mark_only_results:
                section_data['target_new_content'] = version_mark_only_results[key]
                updated_count += 1
            elif key in heading_level_only_results:
                section_data['target_new_content'] = heading_level_only_results[key]
                updated_count += 1
            elif key in ai_updated_sections:
                section_data['target_new_content'] = ai_updated_sections[key]
                updated_count += 1
            else:
                section_data['target_new_content'] = section_data.get('target_content', '')

        with open(match_file, 'w', encoding='utf-8') as f:
            json.dump(match_data, f, ensure_ascii=False, indent=2)

        thread_safe_print(f"   ✅ Updated {updated_count} sections in {match_file}")

        # Abort before writing if any chunk translations failed
        chunk_failures = getattr(ai_updated_sections, "failures", [])
        if chunk_failures:
            thread_safe_print(f"   ⚠️  Skipping file update due to chunk translation failures")
            return finish(False, "; ".join(chunk_failures))

        # Step 4: Apply updates to target document
        thread_safe_print(f"   📝 Step 4: Applying updates to target document...")
        from file_updater import update_target_document_from_match_data

        success = update_target_document_from_match_data(match_file, repo_config['target_local_path'], source_file_path)
        if success:
            thread_safe_print(f"   🎉 Target document successfully updated!")
            partial_reasons = getattr(ai_updated_sections, "partial_reasons", [])
            return finish(True, "; ".join(partial_reasons))
        else:
            thread_safe_print(f"   ❌ Failed to update target document")
            return finish(False, f"Failed to update target document for {source_file_path}")
        
    except Exception as e:
        thread_safe_print(
            f"   ❌ Error processing regular modified file {source_file_path}: {sanitize_exception_message(e)}"
        )
        return finish(False, f"Error processing regular modified file {source_file_path}: {sanitize_exception_message(e)}")


def _process_single_modified_file_for_pr(
    source_file_path,
    file_sections,
    pr_diff,
    source_context_or_pr_url,
    github_client,
    ai_client,
    repo_config,
    glossary_matcher,
):
    thread_safe_print(f"\n📄 Processing modified file: {source_file_path}")

    thread_safe_print(f"   🔍 Extracting file-specific diff for: {source_file_path}")
    file_specific_diff = extract_file_diff_from_pr(pr_diff, source_file_path) if pr_diff else ""

    if not file_specific_diff:
        if pr_diff:
            return make_task_result("skipped", "No file-specific diff found")
        return make_task_result("skipped", "No markdown patch text available for section-level translation")

    thread_safe_print(f"   📊 File-specific diff: {len(file_specific_diff)} chars")

    file_type = determine_file_processing_type(source_file_path, file_sections, SPECIAL_FILES, PR_MODE_IGNORE_FILES)
    thread_safe_print(f"   🔍 File processing type: {file_type}")

    if file_type == "special_file_toc":
        return make_task_result("skipped", "Special file already processed in Step 3.3")
    if file_type == "special_file_keyword":
        return make_task_result("skipped", "Keyword file already processed in Step 3.3b")
    if file_type not in ("regular_modified", "special_file_index"):
        return make_task_result("failure", f"Unknown file processing type: {file_type}")

    success, failure_reason = process_regular_modified_file(
        source_file_path,
        file_sections,
        file_specific_diff,
        source_context_or_pr_url,
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


def is_under_exclude_folder(file_path, folder_name):
    """Check if a file path is under the given folder"""
    return file_path.startswith(folder_name + "/") or file_path == folder_name

def filter_docs_by_folder(folder_name, added_sections, modified_sections, deleted_sections,
                          added_files, deleted_files, toc_files, keyword_files,
                          added_images, modified_images, deleted_images,
                          label=None):
    """Remove all entries under the given folder from every result category.
    Returns the filtered versions of all inputs."""
    if label is None:
        label = folder_name

    def filter_dict(d):
        return {k: v for k, v in d.items() if not is_under_exclude_folder(k, folder_name)}

    def filter_list(lst):
        return [item for item in lst if not is_under_exclude_folder(item, folder_name)]

    skipped = []
    for d in (added_sections, modified_sections, deleted_sections, added_files, toc_files, keyword_files):
        for k in list(d.keys()):
            if is_under_exclude_folder(k, folder_name):
                skipped.append(k)
    for lst in (deleted_files, added_images, modified_images, deleted_images):
        for item in lst:
            if is_under_exclude_folder(item, folder_name):
                skipped.append(item)

    if skipped:
        thread_safe_print(f"\n🚫 Skipping {len(skipped)} {label} doc entries under '{folder_name}/':")
        for s in skipped:
            thread_safe_print(f"   ⏭️  {s}")

    return (
        filter_dict(added_sections),
        filter_dict(modified_sections),
        filter_dict(deleted_sections),
        filter_dict(added_files),
        filter_list(deleted_files),
        filter_dict(toc_files),
        filter_dict(keyword_files),
        filter_list(added_images),
        filter_list(modified_images),
        filter_list(deleted_images),
    )


def normalize_source_files(source_files):
    """Normalize a comma-separated source file filter list."""
    normalized = []
    seen = set()
    for item in source_files.split(","):
        path = item.strip()
        if path.startswith("./"):
            path = path[2:]
        if not path or path in seen:
            continue
        seen.add(path)
        normalized.append(path)
    return normalized


def filter_docs_by_source_files(source_files, added_sections, modified_sections, deleted_sections,
                                added_files, deleted_files, toc_files, keyword_files,
                                added_images, modified_images, deleted_images):
    """Keep only entries whose repo-relative path is explicitly listed."""
    normalized_files = set(normalize_source_files(source_files))
    if not normalized_files:
        return (
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
        )

    def keep(path):
        return path in normalized_files

    def filter_dict(d):
        return {k: v for k, v in d.items() if keep(k)}

    def filter_list(lst):
        return [item for item in lst if keep(item)]

    discovered_paths = set()
    for d in (added_sections, modified_sections, deleted_sections, added_files, toc_files, keyword_files):
        discovered_paths.update(d.keys())
    for lst in (deleted_files, added_images, modified_images, deleted_images):
        discovered_paths.update(lst)

    missing_paths = sorted(normalized_files - discovered_paths)
    thread_safe_print(f"\n📄 Applying SOURCE_FILES filter: {', '.join(sorted(normalized_files))}")
    if missing_paths:
        thread_safe_print("   ⚠️  Requested source files not found in this PR diff:")
        for path in missing_paths:
            thread_safe_print(f"      - {path}")

    return (
        filter_dict(added_sections),
        filter_dict(modified_sections),
        filter_dict(deleted_sections),
        filter_dict(added_files),
        filter_list(deleted_files),
        filter_dict(toc_files),
        filter_dict(keyword_files),
        filter_list(added_images),
        filter_list(modified_images),
        filter_list(deleted_images),
    )

def get_workflow_repo_config(pr_url, repo_configs):
    """Get repository configuration for workflow environment"""
    from diff_analyzer import parse_pr_url
    
    owner, repo, pr_number = parse_pr_url(pr_url)
    source_repo = f"{owner}/{repo}"
    
    if source_repo not in repo_configs:
        raise ValueError(f"Unsupported source repository: {source_repo}. Supported: {list(repo_configs.keys())}")
    
    config = repo_configs[source_repo].copy()
    config['source_repo'] = source_repo
    config['pr_number'] = pr_number
    
    return config

def main():
    """Main function - orchestrates the entire workflow for GitHub Actions"""
    
    # Validate environment variables
    if not all([SOURCE_PR_URL, TARGET_PR_URL, GITHUB_TOKEN, TARGET_REPO_PATH]):
        thread_safe_print("❌ Missing required environment variables:")
        thread_safe_print(f"   SOURCE_PR_URL: {SOURCE_PR_URL}")
        thread_safe_print(f"   TARGET_PR_URL: {TARGET_PR_URL}")
        thread_safe_print(f"   GITHUB_TOKEN: {'Set' if GITHUB_TOKEN else 'Not set'}")
        thread_safe_print(f"   TARGET_REPO_PATH: {TARGET_REPO_PATH}")
        sys.exit(1)
    
    thread_safe_print(f"🔧 Auto PR Sync Tool (GitHub Workflow Version)")
    thread_safe_print(f"📍 Source PR URL: {SOURCE_PR_URL}")
    thread_safe_print(f"📍 Target PR URL: {TARGET_PR_URL}")
    thread_safe_print(f"🤖 AI Provider: {AI_PROVIDER}")
    thread_safe_print(f"🏷️  Product: {get_product_name()}")
    thread_safe_print(f"📁 Target Repo Path: {TARGET_REPO_PATH}")
    thread_safe_print(f"📖 Prefer Local Target Read: {PREFER_LOCAL_TARGET_FOR_READ}")
    
    # Clean and prepare temp_output directory
    clean_temp_output_dir()
    temp_output_dir = ensure_temp_output_dir()
    run_report = RunReport()
    
    # Get repository configuration using workflow config
    try:
        repo_configs = get_workflow_repo_configs()
        workflow_repo_config = get_workflow_repo_config(SOURCE_PR_URL, repo_configs)
        thread_safe_print(
            f"📁 Source Repo: {workflow_repo_config['source_repo']} "
            f"({workflow_repo_config['source_language']})"
        )
        thread_safe_print(
            f"📁 Target Repo: {workflow_repo_config['target_repo']} "
            f"({workflow_repo_config['target_language']})"
        )
        thread_safe_print(f"📁 Target Path: {workflow_repo_config['target_local_path']}")
    except ValueError as e:
        thread_safe_print(f"❌ {sanitize_exception_message(e)}")
        sys.exit(1)
    
    # Initialize clients
    auth = Auth.Token(GITHUB_TOKEN)
    github_client = Github(auth=auth)
    
    # Initialize unified AI client
    try:
        ai_client = UnifiedAIClient(provider=AI_PROVIDER)
        thread_safe_print(f"🤖 AI Provider: {AI_PROVIDER.upper()} ({ai_client.model})")
    except Exception as e:
        thread_safe_print(f"❌ Failed to initialize AI client: {sanitize_exception_message(e)}")
        sys.exit(1)
    
    # Load glossary and create matcher for term-aware translation
    terms_path = TERMS_PATH
    if not terms_path and TARGET_REPO_PATH:
        candidate = os.path.join(TARGET_REPO_PATH, "resources", "terms.md")
        if os.path.exists(candidate):
            terms_path = candidate
    thread_safe_print(f"\n📚 Loading glossary from: {terms_path or '(not configured)'}")
    glossary = load_glossary(terms_path) if terms_path else []
    glossary_matcher = create_glossary_matcher(glossary)
    
    thread_safe_print(f"\n🚀 Starting auto-sync for PR: {SOURCE_PR_URL}")
    
    # Step 1: Get PR diff
    thread_safe_print(f"\n📋 Step 1: Getting PR diff...")
    try:
        source_context = build_pr_diff_context(SOURCE_PR_URL, github_client, repo_configs)
    except Exception as e:
        thread_safe_print(f"❌ Could not build source diff context: {sanitize_exception_message(e)}")
        sys.exit(1)
    repo_config = source_context["repo_config"]
    pr_diff = build_diff_text(source_context["changed_files"])
    thread_safe_print(f"📋 Source diff: {source_context['source_description']}")
    if pr_diff:
        thread_safe_print(f"✅ Got PR diff: {len(pr_diff)} characters")
    else:
        thread_safe_print("⚠️  No markdown patch text found in this PR, continuing with file/image processing only")
    
    # Build list of folders to exclude early (before expensive per-file processing)
    exclude_folders = []
    if repo_config.get('target_language') == 'Chinese':
        exclude_folders = PR_MODE_IGNORE_FOLDERS
    
    # Step 2: Analyze source changes with operation categorization
    thread_safe_print(f"\n📊 Step 2: Analyzing source changes...")
    added_sections, modified_sections, deleted_sections, added_files, deleted_files, toc_files, keyword_files, added_images, modified_images, deleted_images, restructured_files, index_files, formatting_files = analyze_source_changes(
        source_context, github_client,
        special_files=SPECIAL_FILES, 
        ignore_files=PR_MODE_IGNORE_FILES,
        repo_configs=repo_configs,
        max_non_system_sections=MAX_NON_SYSTEM_SECTIONS_FOR_AI,
        pr_diff=pr_diff,
        exclude_folders=exclude_folders,
        source_files=[path.strip() for path in SOURCE_FILES.split(",") if path.strip()] if SOURCE_FILES else None,
        include_formatting_files=True,
    )
    if SOURCE_FILES:
        added_sections, modified_sections, deleted_sections, added_files, deleted_files, toc_files, keyword_files, added_images, modified_images, deleted_images = filter_docs_by_source_files(
            SOURCE_FILES,
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
        )
        requested_source_files = set(normalize_source_files(SOURCE_FILES))
        formatting_files = {
            path: data
            for path, data in formatting_files.items()
            if path in requested_source_files
        }
        restructured_files = {
            path for path in restructured_files if path in requested_source_files
        }
    # Restructured files are initially queued in added_files by
    # analyze_source_changes(). For existing target files, reconcile the
    # structure while reusing unchanged translations before the added-file path
    # can overwrite the whole document.
    if restructured_files:
        thread_safe_print(
            f"\n🔄 Restructured files detected ({len(restructured_files)}), "
            "will use structure-preserving reconciliation when possible:"
        )
        for path in sorted(restructured_files):
            thread_safe_print(f"   - {path}")
        reconcile_restructured_files_for_pr(
            restructured_files,
            added_files,
            source_context,
            github_client,
            ai_client,
            repo_config,
            glossary_matcher,
            run_report,
        )
    diff_file_count = count_unique_file_paths(
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
        formatting_files,
    )
    parallel_file_processing = should_parallelize_file_processing(diff_file_count)
    if parallel_file_processing:
        thread_safe_print(f"⚡ Diff has {diff_file_count} files; file-level translation will use parallel chunks.")
    
    # Step 3: Process different types of files based on operation type.
    #
    # Best-effort behavior is intentional here: file-level failures must not
    # discard successful translations from the same run. For example, if 95 of
    # 100 independent files succeed, those 95 files should remain eligible for
    # staging and push while the failed files are logged for follow-up.
    thread_safe_print(f"\n📋 Step 3: Processing files based on operation type...")
    
    # Step 3.1: Process deleted files (file-level deletions)
    if deleted_files:
        thread_safe_print(f"\n🗑️  Step 3.1: Processing {len(deleted_files)} deleted files...")
        delete_outcomes = process_deleted_files(
            deleted_files, github_client, repo_config, return_outcomes=True
        )
        run_report.record_outcomes(delete_outcomes)
        if not delete_outcomes.all_succeeded:
            thread_safe_print(f"   ⚠️  Some file deletions failed")
        thread_safe_print(f"   ✅ Deleted files processed")
        git_add_changes(TARGET_REPO_PATH)
    
    # Step 3.2: Process added files (file-level additions) one by one
    if added_files:
        thread_safe_print(f"\n📄 Step 3.2: Processing {len(added_files)} added files...")

        added_tasks = []
        for file_path, file_content in added_files.items():
            def run_added_file(path=file_path, content=file_content):
                outcomes = process_added_files(
                    {path: content},
                    source_context,
                    github_client,
                    ai_client,
                    repo_config,
                    glossary_matcher=glossary_matcher,
                    return_outcomes=True,
                    overwrite_existing=path in restructured_files,
                )
                outcomes = normalize_added_file_outcomes(outcomes, path)
                outcome = outcomes.get(path, {"status": "failed", "reason": "Missing added-file outcome"})
                return make_task_result(outcome["status"], outcome.get("reason", ""))

            added_tasks.append(make_file_task(file_path, run_added_file))

        added_results = run_file_tasks(added_tasks, "added files", parallel_file_processing)
        for result in added_results:
            file_path = result["file_path"]
            task_result = result["result"] or {}
            if result["ok"] and task_result.get("status") == "success":
                thread_safe_print(f"   ✅ Successfully processed added file {file_path}")
                run_report.mark_success(file_path)
            elif result["ok"] and task_result.get("status") == "partial":
                reason = task_result.get("reason", "Incomplete added-file translation")
                thread_safe_print(f"   ⚠️  Partially processed added file {file_path}: {reason}")
                run_report.mark_partial(file_path, reason)
            else:
                reason = result["error"] or task_result.get("reason") or "Added file processor returned failure"
                thread_safe_print(f"   ❌ Failed to process added file {file_path}: {reason}")
                run_report.mark_failure(file_path, reason)
        git_add_successful_task_changes(added_results, TARGET_REPO_PATH)
        thread_safe_print(f"   ✅ Added files processed")
    
    # Step 3.3: Process special files (TOC.md and similar)
    if toc_files:
        thread_safe_print(f"\n📋 Step 3.3: Processing {len(toc_files)} special files (TOC)...")

        toc_tasks = []
        for file_path, toc_data in toc_files.items():
            def run_toc_file(path=file_path, data=toc_data):
                if data.get("type") != "toc":
                    return make_task_result("failure", f"Unknown TOC data type: {data.get('type')}")
                success = process_toc_file(
                    path,
                    data,
                    source_context,
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
            file_path = result["file_path"]
            task_result = result["result"] or {}
            if result["ok"] and task_result.get("status") == "success":
                thread_safe_print(f"   ✅ Successfully processed TOC file {file_path}")
                run_report.mark_success(file_path)
            else:
                reason = result["error"] or task_result.get("reason") or "TOC processor returned failure"
                thread_safe_print(f"   ❌ Failed to process TOC file {file_path}: {reason}")
                run_report.mark_failure(file_path, reason)
        thread_safe_print(f"   ✅ Special files processed")
        git_add_successful_task_changes(toc_results, TARGET_REPO_PATH)
    
    # Step 3.3b: Process keyword files (keywords.md)
    if keyword_files:
        thread_safe_print(f"\n📋 Step 3.3b: Processing {len(keyword_files)} keyword files...")
        keyword_tasks = []
        for file_path, keyword_data in keyword_files.items():
            def run_keyword_file(path=file_path, data=keyword_data):
                if data.get("type") != "keyword":
                    return make_task_result("failure", f"Unknown keyword data type: {data.get('type')}")
                success = process_keyword_file(path, data, source_context, github_client, ai_client, repo_config)
                if success:
                    return make_task_result("success")
                return make_task_result("failure", "Keyword processor returned failure")

            keyword_tasks.append(make_file_task(file_path, run_keyword_file))

        keyword_results = run_file_tasks(keyword_tasks, "keyword files", parallel_file_processing)
        keyword_success = all(
            result["ok"] and result["result"] and result["result"].get("status") == "success"
            for result in keyword_results
        )
        if not keyword_success:
            for result in keyword_results:
                if result["ok"] and result["result"] and result["result"].get("status") == "success":
                    run_report.mark_success(result["file_path"])
                    continue
                reason = result["error"] or (result["result"] or {}).get("reason") or "Keyword processor returned failure"
                thread_safe_print(f"   ❌ Failed to process keyword file {result['file_path']}: {reason}")
                run_report.mark_failure(result["file_path"], reason)
            thread_safe_print("   ⚠️  Some keyword files failed; preserving successful file results")
        else:
            for result in keyword_results:
                run_report.mark_success(result["file_path"])
        thread_safe_print(f"   ✅ Keyword files processed")
        git_add_successful_task_changes(keyword_results, TARGET_REPO_PATH)
    
    # Step 3.4: Process formatting-only files
    if formatting_files:
        thread_safe_print(
            f"\n🧹 Step 3.4: Processing {len(formatting_files)} formatting-only files..."
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
            file_path = result["file_path"]
            status, reason = record_file_task_result(result, run_report)
            if status == "success":
                detail = f": {reason}" if reason else ""
                thread_safe_print(
                    f"   ✅ Synchronized formatting for {file_path}{detail}"
                )
            elif status == "partial":
                thread_safe_print(f"   ⚠️  Partially synchronized {file_path}: {reason}")
            elif status == "skipped":
                thread_safe_print(f"   ⏭️  Skipped formatting sync for {file_path}: {reason}")
            else:
                thread_safe_print(f"   ❌ Failed to synchronize {file_path}: {reason}")
        git_add_successful_task_changes(formatting_results, TARGET_REPO_PATH)

    # Step 3.5: Process modified files (section-level modifications)
    if modified_sections:
        thread_safe_print(f"\n📝 Step 3.5: Processing {len(modified_sections)} modified files...")

        modified_tasks = []
        for source_file_path, file_sections in modified_sections.items():
            def run_modified_file(path=source_file_path, sections=file_sections):
                return _process_single_modified_file_for_pr(
                    path,
                    sections,
                    pr_diff,
                    source_context,
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
            source_file_path = result["file_path"]
            if not result["ok"]:
                thread_safe_print(f"   ❌ Failed to process {source_file_path}: {result['error']}")
                run_report.mark_failure(source_file_path, result["error"])
                continue

            task_result = result["result"] or {}
            status = task_result.get("status", "failure")
            reason = task_result.get("reason", "")
            if status == "success":
                thread_safe_print(f"   ✅ Successfully processed {source_file_path}")
                run_report.mark_success(source_file_path)
            elif status == "partial":
                thread_safe_print(f"   ⚠️  Partially processed {source_file_path}: {reason}")
                run_report.mark_partial(source_file_path, reason)
            elif status == "skipped":
                thread_safe_print(f"   ⏭️  Skipped {source_file_path}: {reason}")
                run_report.mark_skipped(source_file_path, reason)
            else:
                thread_safe_print(f"   ❌ Failed to process {source_file_path}: {reason}")
                run_report.mark_failure(source_file_path, reason)

        git_add_successful_task_changes(modified_results, TARGET_REPO_PATH)
    
    # Step 3.6: Process images (added, modified, deleted)
    if added_images or modified_images or deleted_images:
        thread_safe_print(f"\n🖼️  Step 3.6: Processing images...")
        image_outcomes = process_all_images(
            added_images,
            modified_images,
            deleted_images,
            source_context,
            github_client,
            repo_config,
        )
        run_report.record_outcomes(image_outcomes)
        thread_safe_print(f"   ✅ Images processed")
        git_add_changes(TARGET_REPO_PATH)
    
    # Final summary
    thread_safe_print(f"\n" + "="*80)
    thread_safe_print(f"📊 Final Summary:")
    thread_safe_print(f"="*80)
    thread_safe_print(f"   📄 Added files: {len(added_files)} processed")
    thread_safe_print(f"   🗑️  Deleted files: {len(deleted_files)} processed")
    thread_safe_print(f"   📋 TOC files: {len(toc_files)} processed")
    thread_safe_print(f"   📋 Keyword files: {len(keyword_files)} processed")
    thread_safe_print(f"   🧹 Formatting-only files: {len(formatting_files)} processed")
    thread_safe_print(f"   📝 Modified files: {len(modified_sections)} processed")
    thread_safe_print(f"   🖼️  Added images: {len(added_images)} processed")
    thread_safe_print(f"   🖼️  Modified images: {len(modified_images)} processed")
    thread_safe_print(f"   🖼️  Deleted images: {len(deleted_images)} processed")
    run_report.print_summary(printer=thread_safe_print)
    run_report.write_failure_report(temp_output_dir)
    thread_safe_print(f"="*80)
    if run_report.partial or run_report.failed or run_report.structure_errors:
        thread_safe_print(
            "⚠️  Workflow preserved useful translated output and recorded "
            "incomplete work in translation-failures.md."
        )
    else:
        thread_safe_print("🎉 Workflow completed successfully!")

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        thread_safe_print(f"❌ Unhandled exception: {sanitize_exception_message(e)}")
        sys.exit(1)

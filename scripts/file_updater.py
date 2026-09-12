"""
File Updater Module
Handles processing and translation of updated files and sections
"""

import os
import re
import json
import ast
import difflib
import threading
from concurrent.futures import ThreadPoolExecutor
from github import Github
from openai import OpenAI
from file_io import atomic_write_text
from heading_anchor_utils import (
    EXPLICIT_HEADING_ANCHOR_RE,
    build_heading_anchor_slug,
)
from log_sanitizer import sanitize_exception_message, safe_target_path
from product_specific_handler import get_product_name, rewrite_tidb_version_anchors_in_sections
from special_file_utils import path_resource_key, source_scope_includes_folder
from svg_preprocessor import (
    strip_svgs,
    restore_svgs,
    strip_svgs_from_sections_and_diff,
    restore_svgs_in_dict,
)

# Thread-safe printing
print_lock = threading.Lock()

def thread_safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)

def verbose_logging_enabled():
    return os.getenv("VERBOSE_WORKFLOW_LOGS", "true").lower() in ("1", "true", "yes", "on")

def verbose_thread_safe_print(*args, **kwargs):
    if verbose_logging_enabled():
        thread_safe_print(*args, **kwargs)

def read_text_lines_preserve_newlines(file_path):
    """Read text lines without normalizing existing line endings."""
    with open(file_path, 'r', encoding='utf-8', newline='') as f:
        return f.readlines()

def write_text_lines_preserve_newlines(file_path, lines):
    """Write text lines while preserving line endings and the original EOF newline state."""
    lines = normalize_trailing_blank_lines(
        lines,
        preserve_final_newline=file_ends_with_newline(file_path),
    )
    atomic_write_text(file_path, "".join(lines), newline="")

def file_ends_with_newline(file_path):
    """Return whether the existing file ends with a newline byte."""
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return False

    with open(file_path, 'rb') as f:
        f.seek(-1, os.SEEK_END)
        return f.read(1) in (b'\n', b'\r')

def normalize_trailing_blank_lines(lines, preserve_final_newline=True):
    """Remove blank-only lines at EOF while preserving the original final newline state."""
    normalized = list(lines)
    preferred_newline = "\n"
    for line in reversed(normalized):
        if line.endswith("\r\n"):
            preferred_newline = "\r\n"
            break
        if line.endswith("\n"):
            preferred_newline = "\n"
            break

    while normalized and not normalized[-1].strip():
        normalized.pop()

    if normalized:
        if preserve_final_newline:
            if not normalized[-1].endswith(("\n", "\r")):
                normalized[-1] += preferred_newline
        else:
            normalized[-1] = normalized[-1].rstrip("\r\n")

    return normalized

def is_markdown_heading(line):
    """Return True only for real markdown headings at column 0."""
    if not line or not isinstance(line, str):
        return False
    if line != line.lstrip():
        return False
    return re.match(r'^#{1,10}\s+\S', line) is not None


NON_TOP_LEVEL_HEADING_RE = re.compile(r'^(#{2,10})\s+(.+?)\s*$')
DOC_VARIABLE_EXAMPLE = "{{{ .starter }}}"
ALIASES_LINE_RE = re.compile(r'^(?P<prefix>\+?)(?P<indent>\s*)aliases:(?P<spacing>\s*)(?P<value>.+?)\s*$')
TIDB_CLOUD_LINK_RE = re.compile(r'\[([^\]]+)\]\((/tidb-cloud/[^)]+)\)')
TIDB_CLOUD_ABSOLUTE_LINK_PREFIX = os.getenv(
    "TIDB_CLOUD_ABSOLUTE_LINK_PREFIX",
    "https://docs.pingcap.com/tidbcloud/",
)
TRANSLATION_CHUNK_SECTION_THRESHOLD = int(os.getenv("TRANSLATION_CHUNK_SECTION_THRESHOLD", "10"))
TRANSLATION_CHUNK_TOKEN_THRESHOLD = int(os.getenv("TRANSLATION_CHUNK_TOKEN_THRESHOLD", "12000"))
SYSTEM_TRANSLATION_CHUNK_SIZE = int(os.getenv("SYSTEM_TRANSLATION_CHUNK_SIZE", "20"))
REGULAR_TRANSLATION_CHUNK_SIZE = int(os.getenv("REGULAR_TRANSLATION_CHUNK_SIZE", "8"))
TRANSLATION_CHUNK_MAX_SECTIONS_ENV = os.getenv("TRANSLATION_CHUNK_MAX_SECTIONS")
TRANSLATION_CHUNK_MAX_SECTIONS = int(
    TRANSLATION_CHUNK_MAX_SECTIONS_ENV or str(REGULAR_TRANSLATION_CHUNK_SIZE)
)
TRANSLATION_CHUNK_CHAR_LIMIT = int(os.getenv("TRANSLATION_CHUNK_CHAR_LIMIT", "40000"))


class TranslationResult(dict):
    """Translated sections with fatal and non-fatal completeness metadata."""

    def __init__(self, initial=None, failures=None, partial_reasons=None):
        super().__init__(initial or {})
        self.failures = list(failures or [])
        self.partial_reasons = list(partial_reasons or [])


def has_explicit_heading_anchor(heading_line):
    """Return True when a markdown heading already carries an explicit anchor."""
    if not heading_line:
        return False
    return EXPLICIT_HEADING_ANCHOR_RE.search(heading_line.strip()) is not None


def add_heading_anchor_if_needed(heading_line):
    """Append an explicit anchor to a non-top-level heading when safe to do so."""
    stripped = heading_line.rstrip()
    if has_explicit_heading_anchor(stripped):
        return heading_line

    match = NON_TOP_LEVEL_HEADING_RE.match(stripped)
    if not match:
        return heading_line

    heading_text = match.group(2)
    slug = build_heading_anchor_slug(heading_text)
    if not slug:
        return heading_line

    return f"{stripped} {{#{slug}}}"


def extract_heading_anchor_slug(heading_line):
    """Return the anchor slug implied by a heading line, if any."""
    if not heading_line:
        return ""

    stripped = heading_line.rstrip()
    explicit_match = EXPLICIT_HEADING_ANCHOR_RE.search(stripped)
    if explicit_match:
        return explicit_match.group(1).strip()

    match = NON_TOP_LEVEL_HEADING_RE.match(stripped)
    if not match:
        return ""

    return build_heading_anchor_slug(match.group(2))


def get_source_mode(source_context_or_pr_url):
    """Return the high-level source mode for the current workflow input."""
    if isinstance(source_context_or_pr_url, dict):
        return source_context_or_pr_url.get("mode", "")
    return "pr"


def should_apply_tidb_cloud_link_rewrite(source_language, target_language, source_mode=""):
    """Return True when /tidb-cloud/ markdown links should become absolute URLs."""
    if (source_language or "").lower() != "english" or (target_language or "").lower() != "chinese":
        return False

    normalized_mode = (source_mode or "").lower()
    if normalized_mode == "pr":
        return True
    if normalized_mode != "commit":
        return False

    return source_scope_includes_folder(
        "ai",
        source_folder=os.getenv("SOURCE_FOLDER", ""),
        source_files=os.getenv("SOURCE_FILES", ""),
    )


def get_tidb_cloud_absolute_link_prefix():
    """Return the configured absolute link prefix for /tidb-cloud/ rewrites."""
    return os.getenv("TIDB_CLOUD_ABSOLUTE_LINK_PREFIX", TIDB_CLOUD_ABSOLUTE_LINK_PREFIX)


LANGUAGE_ALIAS_PREFIX = {
    "chinese": "/zh",
    "japanese": "/ja",
}


def get_language_alias_prefix(target_language):
    """Return the URL alias prefix for a target language, e.g. '/zh' or '/ja'."""
    return LANGUAGE_ALIAS_PREFIX.get((target_language or "").lower(), "")


def normalize_aliases_value(value, lang_prefix):
    """Add a language prefix (e.g. /zh, /ja) to alias paths that do not already carry it."""
    if not lang_prefix:
        return value

    try:
        aliases = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value

    if not isinstance(aliases, list):
        return value

    normalized = []
    changed = False
    for alias in aliases:
        if not isinstance(alias, str):
            return value

        updated = alias
        if alias.startswith("/") and alias != lang_prefix and not alias.startswith(lang_prefix + "/"):
            updated = f"{lang_prefix}{alias}"
        normalized.append(updated)
        if updated != alias:
            changed = True

    if not changed:
        return value

    return "[" + ",".join(repr(alias) for alias in normalized) + "]"


def preprocess_aliases_line(line, lang_prefix, diff_added_only=False):
    """Normalize aliases lines by adding the target-language prefix."""
    if diff_added_only and not line.startswith("+"):
        return line

    if not lang_prefix:
        return line

    match = ALIASES_LINE_RE.match(line)
    if not match:
        return line

    normalized_value = normalize_aliases_value(match.group("value"), lang_prefix)
    if normalized_value == match.group("value"):
        return line

    return (
        f"{match.group('prefix')}{match.group('indent')}aliases:"
        f"{match.group('spacing')}{normalized_value}"
    )


def preprocess_aliases_line_for_zh(line, diff_added_only=False):
    """Backward-compatible wrapper: normalize aliases for /zh prefix."""
    return preprocess_aliases_line(line, "/zh", diff_added_only=diff_added_only)


def build_tidb_cloud_absolute_url(relative_url):
    """Convert /tidb-cloud/... markdown paths to docs.pingcap.com/tidbcloud absolute URLs."""
    anchor = ""
    path = relative_url
    if "#" in relative_url:
        path, anchor = relative_url.split("#", 1)

    filename = os.path.basename(path.rstrip("/"))
    if filename.endswith(".md"):
        filename = filename[:-3]

    if not filename:
        return relative_url

    prefix = get_tidb_cloud_absolute_link_prefix()
    absolute = f"{prefix.rstrip('/')}/{filename}"
    if anchor:
        absolute = f"{absolute}#{anchor}"
    return absolute


def preprocess_tidb_cloud_links_in_line(line, diff_added_only=False):
    """Rewrite /tidb-cloud/ markdown links to absolute URLs."""
    if diff_added_only and not line.startswith("+"):
        return line

    def repl(match):
        label = match.group(1)
        relative_url = match.group(2)
        absolute_url = build_tidb_cloud_absolute_url(relative_url)
        return f"[{label}]({absolute_url})"

    return TIDB_CLOUD_LINK_RE.sub(repl, line)


def _extract_source_section_heading_lines(source_sections):
    """Return exact real heading lines found outside source-section fences."""
    heading_lines = set()
    for content in (source_sections or {}).values():
        in_code_block = False
        code_block_delimiter = None
        for line in str(content or "").splitlines():
            fence_marker = _get_fence_marker(line)
            if fence_marker:
                if not in_code_block:
                    in_code_block = True
                    code_block_delimiter = fence_marker
                elif line.strip().startswith(code_block_delimiter):
                    in_code_block = False
                    code_block_delimiter = None
                continue
            if not in_code_block and is_markdown_heading(line):
                heading_lines.add(line.rstrip())
    return heading_lines


def preprocess_diff_for_heading_anchor_stability(
    pr_diff,
    source_language,
    target_language,
    source_mode="",
    source_sections=None,
):
    """Add prompt-only stability tweaks for commit-based English -> non-English translation."""
    if not pr_diff:
        return pr_diff

    if (source_language or "").lower() != "english":
        return pr_diff

    normalized_target = (target_language or "").lower()
    if normalized_target == "english":
        return pr_diff

    enable_commit_only_preprocessing = (source_mode or "").lower() == "commit"
    enable_tidb_cloud_link_rewrite = should_apply_tidb_cloud_link_rewrite(
        source_language,
        target_language,
        source_mode=source_mode,
    )

    if not enable_commit_only_preprocessing and not enable_tidb_cloud_link_rewrite:
        return pr_diff

    lang_prefix = get_language_alias_prefix(target_language)
    source_heading_lines = (
        _extract_source_section_heading_lines(source_sections)
        if source_sections is not None
        else None
    )

    def source_confirms_heading(heading_line):
        return (
            source_heading_lines is None
            or heading_line.rstrip() in source_heading_lines
        )

    lines = pr_diff.splitlines()
    processed_lines = []
    in_code_block = False
    code_block_delimiter = None
    i = 0
    while i < len(lines):
        line = lines[i]

        # Track fenced code block state (strip diff +/- prefix for detection).
        content_for_fence = line
        if line.startswith('+') or line.startswith('-'):
            if not line.startswith('+++') and not line.startswith('---'):
                content_for_fence = line[1:]
        fence_marker = _get_fence_marker(content_for_fence)
        if fence_marker:
            if not in_code_block:
                in_code_block = True
                code_block_delimiter = fence_marker
            elif content_for_fence.strip().startswith(code_block_delimiter):
                in_code_block = False
                code_block_delimiter = None

        if enable_commit_only_preprocessing and not in_code_block and line.startswith('-') and not line.startswith('---'):
            removed_heading = line[1:]
            removed_slug = extract_heading_anchor_slug(removed_heading)
            buffered = [line]
            j = i + 1
            consumed_replacement = False
            buf_in_code_block = False
            buf_code_block_delimiter = None

            while j < len(lines) and lines[j].startswith('+') and not lines[j].startswith('+++'):
                added_line = lines[j]
                added_heading = added_line[1:]

                # Track fence state within the buffered + lines.
                buf_fence = _get_fence_marker(added_heading)
                if buf_fence:
                    if not buf_in_code_block:
                        buf_in_code_block = True
                        buf_code_block_delimiter = buf_fence
                    elif added_heading.strip().startswith(buf_code_block_delimiter):
                        buf_in_code_block = False
                        buf_code_block_delimiter = None

                if not buf_in_code_block:
                    added_slug = extract_heading_anchor_slug(added_heading)

                    if (
                        removed_slug
                        and added_slug
                        and removed_slug != added_slug
                        and source_confirms_heading(added_heading)
                    ):
                        added_line = f"+{add_heading_anchor_if_needed(added_heading)}"
                        consumed_replacement = True
                    elif (
                        added_slug
                        and not removed_slug
                        and source_confirms_heading(added_heading)
                    ):
                        added_line = f"+{add_heading_anchor_if_needed(added_heading)}"

                # Apply aliases/links preprocessing to buffered + lines.
                buffered_content = added_line[1:]
                if lang_prefix:
                    added_line = "+" + preprocess_aliases_line(buffered_content, lang_prefix, diff_added_only=False)
                if enable_tidb_cloud_link_rewrite:
                    added_line = "+" + preprocess_tidb_cloud_links_in_line(added_line[1:], diff_added_only=False)

                buffered.append(added_line)
                j += 1

            # Propagate fence state from buffered lines to the outer tracker.
            if buf_in_code_block:
                in_code_block = True
                code_block_delimiter = buf_code_block_delimiter

            if consumed_replacement or len(buffered) > 1:
                processed_lines.extend(buffered)
                i = j
                continue

        if enable_commit_only_preprocessing:
            if not in_code_block and line.startswith('+') and not line.startswith('+++'):
                content = line[1:]
                processed_content = (
                    add_heading_anchor_if_needed(content)
                    if source_confirms_heading(content)
                    else content
                )
                if processed_content != content:
                    line = f"+{processed_content}"
            line = preprocess_aliases_line(line, lang_prefix, diff_added_only=True)
        if enable_tidb_cloud_link_rewrite:
            line = preprocess_tidb_cloud_links_in_line(line, diff_added_only=True)
        processed_lines.append(line)
        i += 1

    return '\n'.join(processed_lines)


def _get_fence_marker(line):
    """Return a markdown fence marker (``` or ~~~) if the line opens/closes a code block."""
    match = re.match(r'^(`{3,}|~{3,})', (line or "").strip())
    return match.group(1) if match else None


def _iter_changed_diff_lines_outside_fences(pr_diff, change_prefix):
    """Yield one diff side's changed lines while tracking that side's fences."""
    opposite_prefix = "-" if change_prefix == "+" else "+"
    in_code_block = False
    code_block_delimiter = None

    for line in (pr_diff or "").splitlines():
        if line.startswith("@@"):
            # A hunk does not expose Markdown parser state before its first
            # context line. Reset here so an unclosed fence in one hunk cannot
            # incorrectly affect a later, unrelated hunk.
            in_code_block = False
            code_block_delimiter = None
            continue
        if line.startswith(("+++", "---")):
            continue
        if line.startswith(opposite_prefix):
            continue

        is_changed_line = line.startswith(change_prefix)
        if is_changed_line:
            content = line[1:]
        elif line.startswith(" "):
            content = line[1:]
        else:
            continue

        fence_marker = _get_fence_marker(content)
        if fence_marker:
            if not in_code_block:
                in_code_block = True
                code_block_delimiter = fence_marker
            elif content.strip().startswith(code_block_delimiter):
                in_code_block = False
                code_block_delimiter = None
            continue

        if is_changed_line and not in_code_block:
            yield content


def extract_added_explicit_heading_anchors(pr_diff):
    """Return explicit anchors from added non-H1 headings outside code fences."""
    anchors = []
    for content in _iter_changed_diff_lines_outside_fences(pr_diff, "+"):
        heading_match = NON_TOP_LEVEL_HEADING_RE.match(content.rstrip())
        if not heading_match:
            continue
        anchor_match = EXPLICIT_HEADING_ANCHOR_RE.search(content.rstrip())
        if anchor_match:
            anchors.append(anchor_match.group(1).strip())

    return anchors


def _prompt_heading_anchor_replacements(prompt_pr_diff):
    """Map source heading lines to anchors added to the prompt diff."""
    replacements = {}
    for content in _iter_changed_diff_lines_outside_fences(
        prompt_pr_diff,
        "+",
    ):
        anchor_match = EXPLICIT_HEADING_ANCHOR_RE.search(content.rstrip())
        if not anchor_match:
            continue
        heading_without_anchor = EXPLICIT_HEADING_ANCHOR_RE.sub(
            "",
            content.rstrip(),
        ).rstrip()
        if NON_TOP_LEVEL_HEADING_RE.match(heading_without_anchor):
            replacements[heading_without_anchor] = anchor_match.group(1).strip()

    return replacements


def preprocess_source_sections_for_heading_anchor_stability(
    source_sections,
    prompt_pr_diff,
):
    """Apply the prompt diff's synthetic heading anchors to source sections."""
    replacements = _prompt_heading_anchor_replacements(prompt_pr_diff)
    if not replacements:
        return dict(source_sections or {})

    processed_sections = {}
    for key, content in (source_sections or {}).items():
        processed_lines = []
        in_code_block = False
        code_block_delimiter = None

        for line in str(content or "").splitlines(keepends=True):
            line_ending = ""
            body = line
            if body.endswith("\r\n"):
                body, line_ending = body[:-2], "\r\n"
            elif body.endswith("\n"):
                body, line_ending = body[:-1], "\n"

            fence_marker = _get_fence_marker(body)
            if fence_marker:
                if not in_code_block:
                    in_code_block = True
                    code_block_delimiter = fence_marker
                elif body.strip().startswith(code_block_delimiter):
                    in_code_block = False
                    code_block_delimiter = None
                processed_lines.append(line)
                continue

            if not in_code_block and not has_explicit_heading_anchor(body):
                anchor = replacements.get(body.rstrip())
                if anchor:
                    body = f"{body.rstrip()} {{#{anchor}}}"

            processed_lines.append(body + line_ending)

        processed_sections[key] = "".join(processed_lines)

    return processed_sections


def _extract_section_headings(content):
    """Return heading records outside code fences for one section."""
    records = []
    lines = str(content or "").splitlines(keepends=True)
    in_code_block = False
    code_block_delimiter = None

    for line_index, line in enumerate(lines):
        body = line.rstrip("\r\n")
        fence_marker = _get_fence_marker(body)
        if fence_marker:
            if not in_code_block:
                in_code_block = True
                code_block_delimiter = fence_marker
            elif body.strip().startswith(code_block_delimiter):
                in_code_block = False
                code_block_delimiter = None
            continue

        if in_code_block or not is_markdown_heading(body):
            continue

        heading_match = re.match(r"^(#{1,10})\s+(.+?)\s*$", body)
        if not heading_match:
            continue
        anchor_match = EXPLICIT_HEADING_ANCHOR_RE.search(body)
        records.append(
            {
                "line_index": line_index,
                "level": len(heading_match.group(1)),
                "title": EXPLICIT_HEADING_ANCHOR_RE.sub(
                    "",
                    heading_match.group(2).strip(),
                ).strip(),
                "anchor": anchor_match.group(1).strip() if anchor_match else "",
            }
        )

    return lines, records


def _parse_heading_line(heading_line):
    """Parse one Markdown heading line without accepting surrounding content."""
    if not isinstance(heading_line, str):
        return None
    stripped = heading_line.strip()
    if not stripped or len(stripped.splitlines()) != 1:
        return None

    heading_match = re.match(r"^(#{1,10})\s+(.+?)\s*$", stripped)
    if not heading_match:
        return None
    anchor_match = EXPLICIT_HEADING_ANCHOR_RE.search(stripped)
    title = EXPLICIT_HEADING_ANCHOR_RE.sub(
        "",
        heading_match.group(2).strip(),
    ).strip()
    if not title:
        return None

    return {
        "level": len(heading_match.group(1)),
        "title": title,
        "anchor": anchor_match.group(1).strip() if anchor_match else "",
    }


def _replace_heading_anchor(line, anchor):
    """Set one heading's explicit anchor while preserving its line ending."""
    line_ending = ""
    body = line
    if body.endswith("\r\n"):
        body, line_ending = body[:-2], "\r\n"
    elif body.endswith("\n"):
        body, line_ending = body[:-1], "\n"
    body = EXPLICIT_HEADING_ANCHOR_RE.sub("", body.rstrip()).rstrip()
    return f"{body} {{#{anchor}}}{line_ending}"


def _replace_heading_title(line, level, title, anchor):
    """Replace a heading title while preserving its line ending and anchor."""
    line_ending = ""
    if line.endswith("\r\n"):
        line_ending = "\r\n"
    elif line.endswith("\n"):
        line_ending = "\n"

    anchor_suffix = f" {{#{anchor}}}" if anchor else ""
    return f"{'#' * level} {title}{anchor_suffix}{line_ending}"


def _heading_titles_equal(left, right):
    """Compare heading titles while ignoring insignificant whitespace and case."""
    def normalize(value):
        return re.sub(
            r"\s+",
            " ",
            str(value or "").strip(),
        ).casefold()

    return normalize(left) == normalize(right)


def _extract_changed_heading_specs(pr_diff):
    """Return added heading records and their paired removed headings."""
    _, hunks = _parse_diff_hunks(pr_diff)
    specs = []

    for hunk in hunks:
        header_match = _HUNK_HEADER_RE.match(hunk[0])
        if not header_match:
            continue
        new_line_number = int(header_match.group(3))
        pending_removed = []
        old_fence = None
        new_fence = None

        def update_fence(current_fence, content):
            marker = _get_fence_marker(content)
            if not marker:
                return current_fence
            if current_fence is None:
                return marker
            if content.strip().startswith(current_fence):
                return None
            return current_fence

        for raw_line in hunk[1:]:
            if raw_line.startswith("\\"):
                continue
            prefix = raw_line[:1]
            content = raw_line[1:].rstrip("\r\n")

            if prefix == " ":
                pending_removed = []
                old_fence = update_fence(old_fence, content)
                new_fence = update_fence(new_fence, content)
                new_line_number += 1
                continue

            if prefix == "-":
                fence_before = old_fence
                old_fence = update_fence(old_fence, content)
                if fence_before is None and _get_fence_marker(content) is None:
                    heading = _parse_heading_line(content)
                    if heading:
                        pending_removed.append(heading)
                continue

            if prefix == "+":
                fence_before = new_fence
                new_fence = update_fence(new_fence, content)
                if fence_before is None and _get_fence_marker(content) is None:
                    heading = _parse_heading_line(content)
                    if heading:
                        old_heading = None
                        for index, candidate in enumerate(pending_removed):
                            if candidate["level"] == heading["level"]:
                                old_heading = pending_removed.pop(index)
                                break
                        specs.append(
                            {
                                "old": old_heading,
                                "new": heading,
                                "new_line_number": new_line_number,
                            }
                        )
                new_line_number += 1

    return specs


def _build_heading_translation_tasks(
    source_sections,
    target_sections,
    updated_sections,
    prompt_pr_diff,
    source_language,
    target_language,
):
    """Return only changed headings whose main translation needs repair."""
    specs = [
        spec
        for spec in _extract_changed_heading_specs(prompt_pr_diff)
        if not spec["old"]
        or spec["old"]["title"] != spec["new"]["title"]
    ]
    if not specs:
        return [], [], []

    prompt_source_sections = preprocess_source_sections_for_heading_anchor_stability(
        source_sections,
        prompt_pr_diff,
    )
    locations = []
    partial_reasons = []
    updated_section_keys = set(updated_sections or {})
    for key, source_content in prompt_source_sections.items():
        # A heading retry may repair content inside a returned section, but it
        # must never fabricate a section that the main response omitted.
        if key not in updated_section_keys:
            continue

        source_lines, source_headings = _extract_section_headings(source_content)
        target_content = (target_sections or {}).get(key, "")
        output_content = (updated_sections or {}).get(key, "")
        _, target_headings = _extract_section_headings(target_content)
        _, output_headings = _extract_section_headings(output_content)

        source_levels = [heading["level"] for heading in source_headings]
        target_levels = [heading["level"] for heading in target_headings]
        output_levels = [heading["level"] for heading in output_headings]
        if len(source_headings) > 1 and (
            source_levels != output_levels
            or (target_content and source_levels != target_levels)
        ):
            partial_reasons.append(
                "changed heading repair is ambiguous for multi-heading "
                f"section {key}; preserving the main translation output"
            )
            continue

        source_heading_line_indexes = {
            heading["line_index"] for heading in source_headings
        }
        source_has_non_heading_content = any(
            line.strip()
            for line_index, line in enumerate(source_lines)
            if line_index not in source_heading_line_indexes
        )
        key_line_number = _extract_line_number_from_key(key)
        for ordinal, source_heading in enumerate(source_headings):
            locations.append(
                {
                    "key": key,
                    "ordinal": ordinal,
                    "key_line_number": key_line_number,
                    "source": source_heading,
                    "source_has_nonblank_prefix": any(
                        line.strip()
                        for line in source_lines[:source_heading["line_index"]]
                    ),
                    "source_has_non_heading_content": (
                        source_has_non_heading_content
                    ),
                    "target": (
                        target_headings[ordinal]
                        if ordinal < len(target_headings)
                        else None
                    ),
                    "output": (
                        output_headings[ordinal]
                        if ordinal < len(output_headings)
                        else None
                    ),
                }
            )

    mapped_headings = []
    tasks = []
    used_locations = set()
    for spec in specs:
        source_contains_heading = any(
            location["source"]["level"] == spec["new"]["level"]
            and location["source"]["title"] == spec["new"]["title"]
            for location in locations
        )
        if not source_contains_heading:
            # Chunk filtering works at hunk granularity, so one chunk can see
            # changed headings that belong to source sections in another chunk.
            continue

        matching_locations = []
        for location_index, location in enumerate(locations):
            if location_index in used_locations:
                continue
            source_heading = location["source"]
            if (
                source_heading["level"] != spec["new"]["level"]
                or source_heading["title"] != spec["new"]["title"]
            ):
                continue

            exact_line_match = (
                location["key_line_number"] == spec["new_line_number"]
            )
            intro_h1_match = (
                location["key"] == "intro_section"
                and spec["new"]["level"] == 1
                and location["ordinal"] == 0
            )
            distance = (
                abs(location["key_line_number"] - spec["new_line_number"])
                if location["key_line_number"] is not None
                else 999999
            )
            matching_locations.append(
                (
                    0 if exact_line_match else 1,
                    0 if intro_h1_match else 1,
                    0 if location["ordinal"] == 0 else 1,
                    distance,
                    location_index,
                    location,
                )
            )

        if not matching_locations:
            partial_reasons.append(
                "changed heading could not be mapped to a source section: "
                f"{'#' * spec['new']['level']} {spec['new']['title']}"
            )
            continue

        *_, location_index, location = min(matching_locations)
        used_locations.add(location_index)
        output_heading = location["output"]
        target_heading = location["target"]
        expected_anchor = (
            spec["new"]["anchor"]
            or location["source"]["anchor"]
            or (target_heading["anchor"] if target_heading else "")
            or (output_heading["anchor"] if output_heading else "")
        )
        record = {
            "key": location["key"],
            "ordinal": location["ordinal"],
            "level": spec["new"]["level"],
            "old_source_title": (
                spec["old"]["title"] if spec["old"] else ""
            ),
            "new_source_title": spec["new"]["title"],
            "old_target_title": (
                target_heading["title"] if target_heading else ""
            ),
            "main_output_title": (
                output_heading["title"] if output_heading else ""
            ),
            "main_output_level": (
                output_heading["level"] if output_heading else None
            ),
            "source_has_nonblank_prefix": (
                location["source_has_nonblank_prefix"]
            ),
            "source_has_non_heading_content": (
                location["source_has_non_heading_content"]
            ),
            "expected_anchor": expected_anchor,
        }
        mapped_headings.append(record)

        repair_reason = ""
        if not output_heading:
            repair_reason = "missing"
        elif output_heading["level"] != spec["new"]["level"]:
            repair_reason = "wrong_level"
        elif (
            record["old_target_title"]
            and _heading_titles_equal(
                record["main_output_title"],
                record["old_target_title"],
            )
        ):
            repair_reason = "unchanged"
        elif (
            source_language.casefold() != target_language.casefold()
            and any(
                _heading_titles_equal(
                    record["main_output_title"],
                    source_title,
                )
                for source_title in (
                    record["old_source_title"],
                    record["new_source_title"],
                )
                if source_title
            )
        ):
            repair_reason = "source_language"

        if repair_reason:
            record = dict(record)
            record["id"] = f"heading_{len(tasks) + 1:03d}"
            record["repair_reason"] = repair_reason
            tasks.append(record)

    return tasks, mapped_headings, partial_reasons


def _build_heading_translation_prompt(
    tasks,
    source_language,
    target_language,
    glossary_matcher=None,
):
    """Build a small prompt containing only changed heading translations."""
    payload = {}
    for task in tasks:
        level_prefix = "#" * task["level"]
        payload[task["id"]] = {
            "old_source_heading": (
                f"{level_prefix} {task['old_source_title']}"
                if task["old_source_title"]
                else None
            ),
            "new_source_heading": (
                f"{level_prefix} {task['new_source_title']}"
            ),
            "old_target_heading": (
                f"{level_prefix} {task['old_target_title']}"
                if task["old_target_title"]
                else None
            ),
            "main_output_heading": (
                f"{'#' * task['main_output_level']} "
                f"{task['main_output_title']}"
                if task["main_output_title"]
                and task["main_output_level"]
                else None
            ),
            "repair_reason": task["repair_reason"],
        }

    glossary_text = ""
    if glossary_matcher:
        from glossary import filter_terms_for_content, format_terms_for_prompt
        matched_terms = filter_terms_for_content(
            glossary_matcher,
            json.dumps(payload, ensure_ascii=False),
            source_language=source_language,
        )
        if matched_terms:
            glossary_text = (
                "\n"
                + format_terms_for_prompt(
                    matched_terms,
                    source_language=source_language,
                    target_language=target_language,
                )
                + "\n"
            )

    prompt = f"""Translate only the changed Markdown headings from {source_language} to {target_language} that the main translation left suspicious.

For a renamed heading, use the old source and target headings to preserve the established target-language style while applying every semantic change in the new source heading.
For a newly added heading, translate the new source heading naturally into {target_language}.

Rules:
- Return a JSON object with exactly the same keys as the input.
- Each value must be an object with exactly two fields:
  - `status`: `updated` or `already_equivalent`
  - `heading`: exactly one Markdown heading line with the same number of `#` characters as the new source heading
- Use `updated` when the heading needs a new target-language translation.
- Use `already_equivalent` only when `main_output_heading` is already correct and should remain unchanged. This includes an established target translation that still fully expresses the renamed source heading, and headings made entirely of product names, API names, code identifiers, configuration keys, or other technical text that must remain identical to the source language.
- Return no body text, explanation, code fence, or additional line.
- Do not add or change explicit heading anchors. The program restores anchors separately.
- Preserve inline Markdown, HTML/MDX tags, and placeholders exactly.
- Preserve technical product names and follow the glossary when present.

Input:
{json.dumps(payload, ensure_ascii=False, indent=2)}
{glossary_text}
Return only the JSON object."""
    return prompt


def _apply_targeted_heading(repaired, task, translated_title):
    """Apply one validated heading title, inserting a missing first heading."""
    key = task["key"]
    content = repaired.get(key, "")
    output_lines, output_headings = _extract_section_headings(content)

    if task["repair_reason"] == "missing":
        if (
            task["ordinal"] != 0
            or task["source_has_nonblank_prefix"]
            or (
                not content.strip()
                and task["source_has_non_heading_content"]
            )
        ):
            return False

        line_ending = "\r\n" if "\r\n" in content else "\n"
        anchor_suffix = (
            f" {{#{task['expected_anchor']}}}"
            if task["expected_anchor"]
            else ""
        )
        heading_line = (
            f"{'#' * task['level']} {translated_title}"
            f"{anchor_suffix}{line_ending}"
        )
        if not content:
            repaired[key] = heading_line
        elif content.startswith(("\n", "\r\n")):
            repaired[key] = heading_line + content
        else:
            repaired[key] = heading_line + line_ending + content
        return True

    if task["ordinal"] < len(output_headings):
        output_heading = output_headings[task["ordinal"]]
        output_lines[output_heading["line_index"]] = _replace_heading_title(
            output_lines[output_heading["line_index"]],
            task["level"],
            translated_title,
            task["expected_anchor"],
        )
        repaired[key] = "".join(output_lines)
        return True

    return False


def translate_changed_headings_with_ai(
    source_sections,
    target_sections,
    updated_sections,
    prompt_pr_diff,
    ai_client,
    source_language,
    target_language,
    target_file_prefix,
    prompt_suffix,
    glossary_matcher=None,
):
    """Retry only suspicious changed headings and splice validated titles back."""
    tasks, mapped_headings, partial_reasons = _build_heading_translation_tasks(
        source_sections,
        target_sections,
        updated_sections,
        prompt_pr_diff,
        source_language,
        target_language,
    )
    repaired = dict(updated_sections or {})
    already_equivalent_keys = set()
    unresolved_missing_keys = {
        task["key"]
        for task in tasks
        if task["repair_reason"] == "missing"
    }

    def preserve_unresolved_missing_sections():
        """Avoid writing heading-less or heading-only content after a failed repair."""
        for key in unresolved_missing_keys:
            original_target = (target_sections or {}).get(key, "")
            if original_target:
                repaired[key] = original_target
            else:
                repaired.pop(key, None)

    # Anchor ownership remains programmatic even when the targeted AI request
    # is unnecessary, fails, or returns an unusable title.
    for record in mapped_headings:
        output_lines, output_headings = _extract_section_headings(
            repaired.get(record["key"], "")
        )
        if record["ordinal"] >= len(output_headings):
            continue
        output_heading = output_headings[record["ordinal"]]
        output_lines[output_heading["line_index"]] = _replace_heading_title(
            output_lines[output_heading["line_index"]],
            output_heading["level"],
            output_heading["title"],
            record["expected_anchor"],
        )
        repaired[record["key"]] = "".join(output_lines)

    if not tasks:
        return repaired, partial_reasons, already_equivalent_keys

    prompt = _build_heading_translation_prompt(
        tasks,
        source_language,
        target_language,
        glossary_matcher=glossary_matcher,
    )
    temp_dir = get_temp_output_dir()
    prompt_file = os.path.join(
        temp_dir,
        f"{target_file_prefix}_prompt-for-ai-heading-translation"
        f"{prompt_suffix}.txt",
    )
    with open(prompt_file, "w", encoding="utf-8") as file:
        file.write(prompt)
    thread_safe_print(
        f"   🔤 Translating {len(tasks)} changed heading(s) separately..."
    )

    try:
        ai_response = ai_client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        response = parse_updated_sections(ai_response)
    except Exception as error:
        reason = sanitize_exception_message(error)
        partial_reasons.append(
            f"targeted heading translation failed: {reason}"
        )
        preserve_unresolved_missing_sections()
        return repaired, partial_reasons, already_equivalent_keys

    if not isinstance(response, dict):
        partial_reasons.append(
            "targeted heading translation did not return a JSON object"
        )
        preserve_unresolved_missing_sections()
        return repaired, partial_reasons, already_equivalent_keys

    expected_task_ids = {task["id"] for task in tasks}
    unexpected_task_ids = sorted(set(response) - expected_task_ids)
    if unexpected_task_ids:
        partial_reasons.append(
            "targeted heading translation returned unexpected keys: "
            + ", ".join(unexpected_task_ids)
        )

    for task in tasks:
        task_id = task["id"]
        if task_id not in response:
            partial_reasons.append(
                f"targeted heading translation missing result for "
                f"{task['key']} ({task_id})"
            )
            continue

        task_result = response[task_id]
        if not isinstance(task_result, dict):
            partial_reasons.append(
                f"targeted heading translation returned invalid result for "
                f"{task['key']} ({task_id})"
            )
            continue
        if set(task_result) != {"status", "heading"}:
            partial_reasons.append(
                f"targeted heading translation returned invalid fields for "
                f"{task['key']} ({task_id})"
            )
            continue

        status = task_result.get("status")
        if status not in {"updated", "already_equivalent"}:
            partial_reasons.append(
                f"targeted heading translation returned invalid status for "
                f"{task['key']} ({task_id})"
            )
            continue

        translated_heading = _parse_heading_line(task_result.get("heading"))
        if not translated_heading:
            partial_reasons.append(
                f"targeted heading translation returned invalid Markdown for "
                f"{task['key']} ({task_id})"
            )
            continue
        if translated_heading["level"] != task["level"]:
            partial_reasons.append(
                f"targeted heading translation changed heading level for "
                f"{task['key']} ({task_id})"
            )
            continue

        if status == "already_equivalent":
            if (
                not task["main_output_title"]
                or not _heading_titles_equal(
                    translated_heading["title"],
                    task["main_output_title"],
                )
            ):
                partial_reasons.append(
                    f"targeted heading translation made an invalid "
                    f"already_equivalent claim for {task['key']} ({task_id})"
                )
                continue
        else:
            if (
                task["old_target_title"]
                and _heading_titles_equal(
                    translated_heading["title"],
                    task["old_target_title"],
                )
            ):
                partial_reasons.append(
                    f"targeted heading translation stayed unchanged for "
                    f"{task['key']} ({task_id})"
                )
                continue
            if (
                source_language.casefold() != target_language.casefold()
                and any(
                    _heading_titles_equal(
                        translated_heading["title"],
                        source_title,
                    )
                    for source_title in (
                        task["old_source_title"],
                        task["new_source_title"],
                    )
                    if source_title
                )
            ):
                partial_reasons.append(
                    f"targeted translation left heading in the source language "
                    f"for {task['key']} ({task_id})"
                )
                continue

        if not _apply_targeted_heading(
            repaired,
            task,
            translated_heading["title"],
        ):
            partial_reasons.append(
                f"targeted heading translation could not be applied for "
                f"{task['key']} ({task_id})"
            )
            continue

        unresolved_missing_keys.discard(task["key"])

        # This exemption is consumed only by the modified-H1 completeness
        # check, so a non-H1 result must never suppress an H1 failure.
        if status == "already_equivalent" and task["level"] == 1:
            already_equivalent_keys.add(task["key"])

    preserve_unresolved_missing_sections()

    if getattr(ai_response, "completion_status", "complete") == "incomplete":
        partial_reasons.append(
            "targeted heading translation response incomplete: "
            + (
                getattr(ai_response, "completion_reason", "")
                or "unknown reason"
            )
        )

    results_file = os.path.join(
        temp_dir,
        f"{target_file_prefix}_updated_headings_from_ai{prompt_suffix}.json",
    )
    with open(results_file, "w", encoding="utf-8") as file:
        json.dump(response, file, ensure_ascii=False, indent=2)

    return repaired, partial_reasons, already_equivalent_keys


def restore_expected_heading_anchors(
    source_sections,
    updated_sections,
    prompt_pr_diff,
):
    """Restore changed-heading anchors deterministically after AI translation."""
    prompt_source_sections = preprocess_source_sections_for_heading_anchor_stability(
        source_sections,
        prompt_pr_diff,
    )
    expected_replacements = _prompt_heading_anchor_replacements(prompt_pr_diff)
    if not expected_replacements:
        return dict(updated_sections or {}), []

    restored = dict(updated_sections or {})
    partial_reasons = []
    for key, source_content in prompt_source_sections.items():
        if key not in restored:
            continue

        _, source_headings = _extract_section_headings(source_content)
        expected_headings = []
        for ordinal, heading in enumerate(source_headings):
            source_heading_without_anchor = (
                f"{'#' * heading['level']} {heading['title']}"
            )
            expected_anchor = expected_replacements.get(
                source_heading_without_anchor
            )
            if expected_anchor:
                expected_headings.append(
                    (ordinal, heading["level"], expected_anchor)
                )

        if not expected_headings:
            continue

        output_lines, output_headings = _extract_section_headings(restored[key])
        for ordinal, expected_level, expected_anchor in expected_headings:
            if ordinal >= len(output_headings):
                partial_reasons.append(
                    f"changed heading missing from AI output for {key}: "
                    f"expected anchor {{#{expected_anchor}}}"
                )
                continue

            output_heading = output_headings[ordinal]
            if output_heading["level"] != expected_level:
                partial_reasons.append(
                    f"changed heading level differs for {key}: "
                    f"expected {'#' * expected_level} with anchor "
                    f"{{#{expected_anchor}}}"
                )
                continue

            output_lines[output_heading["line_index"]] = _replace_heading_anchor(
                output_lines[output_heading["line_index"]],
                expected_anchor,
            )

        restored[key] = "".join(output_lines)

    return restored, partial_reasons


def _modified_h1_titles_from_diff(pr_diff):
    """Return H1 titles whose source title text actually changed."""
    return [
        spec["new"]["title"]
        for spec in _extract_changed_heading_specs(pr_diff)
        if (
            spec["old"]
            and spec["old"]["level"] == 1
            and spec["new"]["level"] == 1
            and spec["old"]["title"] != spec["new"]["title"]
        )
    ]


def find_unapplied_modified_h1_sections(
    source_sections,
    target_sections,
    updated_sections,
    prompt_pr_diff,
    already_equivalent_keys=None,
):
    """Report modified H1 sections whose translated heading stayed unchanged."""
    modified_h1_titles = set(_modified_h1_titles_from_diff(prompt_pr_diff))
    if not modified_h1_titles:
        return []

    equivalent_keys = set(already_equivalent_keys or ())
    partial_reasons = []
    for key, source_content in (source_sections or {}).items():
        if key in equivalent_keys:
            continue
        _, source_headings = _extract_section_headings(source_content)
        if not source_headings or source_headings[0]["level"] != 1:
            continue
        if source_headings[0]["title"] not in modified_h1_titles:
            continue

        _, target_headings = _extract_section_headings(
            (target_sections or {}).get(key, "")
        )
        _, updated_headings = _extract_section_headings(
            (updated_sections or {}).get(key, "")
        )
        old_h1 = (
            target_headings[0]["title"]
            if target_headings and target_headings[0]["level"] == 1
            else ""
        )
        new_h1 = (
            updated_headings[0]["title"]
            if updated_headings and updated_headings[0]["level"] == 1
            else ""
        )
        if old_h1 and (not new_h1 or new_h1 == old_h1):
            partial_reasons.append(
                f"changed source H1 produced an unchanged target H1 for {key}"
            )

    return partial_reasons

def _heading_match_candidates(target_hierarchy):
    """Return full-path and leaf heading candidates for section-start matching."""
    if not target_hierarchy:
        return []

    candidates = []
    for candidate in (
        target_hierarchy.strip(),
        target_hierarchy.split(' > ')[-1].strip(),
    ):
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    return candidates

def _line_matches_heading_candidates(raw_line, heading_candidates):
    """Return True when a file line matches one of the expected heading texts."""
    if not is_markdown_heading(raw_line):
        return False
    if not heading_candidates:
        return True
    return raw_line.strip() in heading_candidates

def resolve_section_start_line(target_lines, target_line_num, target_hierarchy):
    """Resolve a robust 0-based section start line for replace/delete."""
    if target_line_num <= 0:
        return 0

    candidate = target_line_num - 1
    heading_candidates = _heading_match_candidates(target_hierarchy)
    heading_text = heading_candidates[-1] if heading_candidates else (target_hierarchy or "").strip()

    # 1) Exact candidate
    if 0 <= candidate < len(target_lines) and _line_matches_heading_candidates(
        target_lines[candidate], heading_candidates
    ):
        return candidate

    # 2) Off-by-one previous line
    if candidate - 1 >= 0 and _line_matches_heading_candidates(
        target_lines[candidate - 1], heading_candidates
    ):
        thread_safe_print(f"   🔧 Adjusted start line from {target_line_num} to {target_line_num - 1} (off-by-one heading)")
        return candidate - 1

    # 3) Search exact heading text in file
    if heading_candidates:
        for idx, raw_line in enumerate(target_lines):
            if _line_matches_heading_candidates(raw_line, heading_candidates):
                thread_safe_print(f"   🔧 Resolved start line by heading text at line {idx + 1}")
                return idx

    # 4) Search nearby matching heading around candidate
    for delta in range(1, 6):
        for idx in (candidate - delta, candidate + delta):
            if 0 <= idx < len(target_lines) and _line_matches_heading_candidates(
                target_lines[idx], heading_candidates
            ):
                thread_safe_print(f"   🔧 Resolved start line by nearby matching heading at line {idx + 1}")
                return idx

    # 5) Last resort: nearest heading around candidate
    for delta in range(1, 6):
        for idx in (candidate - delta, candidate + delta):
            if 0 <= idx < len(target_lines) and is_markdown_heading(target_lines[idx]):
                thread_safe_print(f"   🔧 Resolved start line by nearby heading at line {idx + 1}")
                return idx

    return max(0, min(candidate, len(target_lines) - 1))

def count_changed_lines(old_text, new_text):
    """Count changed lines between two text blocks."""
    old_lines = old_text.splitlines()
    new_lines = new_text.splitlines()
    matcher = difflib.SequenceMatcher(None, old_lines, new_lines)
    changed = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != 'equal':
            changed += max(i2 - i1, j2 - j1)
    return changed

def count_changed_diff_lines(pr_diff):
    """Count changed non-header diff lines in unified diff text."""
    if not pr_diff:
        return 0
    count = 0
    for line in pr_diff.splitlines():
        if line.startswith('+++') or line.startswith('---') or line.startswith('@@') or line.startswith('File:'):
            continue
        if line.startswith('+') or line.startswith('-'):
            count += 1
    return count

def extract_literal_replacements_from_pr_diff(pr_diff):
    """Extract deterministic literal replacements from adjacent -/+ diff lines."""
    replacements = []
    if not pr_diff:
        return replacements

    lines = pr_diff.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('-') and not line.startswith('---'):
            old_line = line[1:]
            j = i + 1
            while j < len(lines) and lines[j].startswith('+') and not lines[j].startswith('+++'):
                new_line = lines[j][1:]
                if old_line != new_line:
                    old_links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', old_line)
                    new_links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', new_line)

                    # Prefer markdown-link text updates when URL stays the same.
                    for old_text, old_url in old_links:
                        for new_text, new_url in new_links:
                            if old_url == new_url and old_text != new_text:
                                replacements.append((f'[{old_text}]({old_url})', f'[{new_text}]({new_url})'))
                                replacements.append((old_text, new_text))

                    # Fallback full-line replacement for exact match scenarios.
                    replacements.append((old_line, new_line))
                j += 1
            i = j
            continue
        i += 1

    # Keep order but deduplicate
    deduped = []
    seen = set()
    for old, new in replacements:
        if old and new and old != new and (old, new) not in seen:
            deduped.append((old, new))
            seen.add((old, new))
    return deduped

def apply_literal_replacements(text, replacements):
    """Apply deterministic literal replacements in order."""
    updated = text
    for old, new in replacements:
        if old in updated:
            updated = updated.replace(old, new)
    return updated

def enforce_minimal_target_updates(target_sections, updated_sections, pr_diff):
    """Guardrail: prevent large style rewrites for small source diffs."""
    if not updated_sections:
        return updated_sections

    diff_changed_lines = count_changed_diff_lines(pr_diff)
    # Small source diff should produce small target edits.
    # Keep a little buffer for language expansion.
    max_allowed = max(2, diff_changed_lines * 3)
    replacements = extract_literal_replacements_from_pr_diff(pr_diff)

    guarded = {}
    for key, updated_text in updated_sections.items():
        original_text = target_sections.get(key, "")
        if not original_text:
            guarded[key] = updated_text
            continue

        changed = count_changed_lines(original_text, updated_text)
        if changed <= max_allowed:
            guarded[key] = updated_text
            continue

        # AI changed too much; apply deterministic replacements instead.
        deterministic = apply_literal_replacements(original_text, replacements)
        if deterministic != original_text:
            thread_safe_print(f"   🛡️  Minimal-change guard applied for {key}: {changed} changed lines -> deterministic patch")
            guarded[key] = deterministic
        else:
            thread_safe_print(f"   🛡️  Minimal-change guard kept original for {key}: {changed} changed lines exceeds limit {max_allowed}")
            guarded[key] = original_text

    return guarded


_tiktoken_encoding = None
_tiktoken_unavailable = False


def estimate_translation_tokens(text):
    """Estimate token count for chunk-mode routing."""
    global _tiktoken_encoding, _tiktoken_unavailable
    if not text:
        return 0
    if _tiktoken_unavailable:
        return max(1, len(str(text)) // 4)
    if _tiktoken_encoding is None:
        try:
            import tiktoken
            _tiktoken_encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _tiktoken_unavailable = True
            return max(1, len(str(text)) // 4)
    return len(_tiktoken_encoding.encode(str(text)))


def get_target_file_prefix_for_debug(target_file_name, target_sections):
    """Build the temp_output file prefix used by prompt/result debug files."""
    if target_file_name:
        return path_resource_key(target_file_name)

    if target_sections:
        first_key = next(iter(target_sections.keys()), "")
        if "_" in first_key:
            parts = first_key.split("_")
            if len(parts) > 1:
                return parts[0]

    return "unknown"


def get_temp_output_dir():
    """Return the scripts/temp_output directory and create it if needed."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    temp_dir = os.path.join(script_dir, "temp_output")
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir


def extract_first_heading_title(content):
    """Return the first markdown heading title from a section."""
    for line in str(content or "").splitlines():
        if is_markdown_heading(line):
            return re.sub(r'^#{1,10}\s+', '', line.strip()).strip()
    return ""


def estimate_chunk_section_chars(key, source_sections, target_sections=None):
    """Estimate how much prompt text a section contributes before diff/glossary."""
    target_sections = target_sections or {}
    return len(str(source_sections.get(key, ""))) + len(str(target_sections.get(key, "")))


def get_translation_chunk_max_sections(target_file_name=None):
    """Return per-file chunk size, preserving larger chunks for dense reference docs."""
    if TRANSLATION_CHUNK_MAX_SECTIONS_ENV:
        return TRANSLATION_CHUNK_MAX_SECTIONS

    basename = os.path.basename(target_file_name or "")
    if basename in {
        "system-variables.md",
        "configuration-file.md",
        "tidb-configuration-file.md",
        "tikv-configuration-file.md",
        "pd-configuration-file.md",
        "tiflash-configuration.md",
    }:
        return SYSTEM_TRANSLATION_CHUNK_SIZE

    return REGULAR_TRANSLATION_CHUNK_SIZE


def build_translation_chunks(source_sections, target_sections=None, max_sections=None):
    """Split ordered section keys by section count and approximate character budget."""
    chunks = []
    current_keys = []
    current_chars = 0
    max_sections = max_sections or TRANSLATION_CHUNK_MAX_SECTIONS

    for key in source_sections:
        section_chars = estimate_chunk_section_chars(key, source_sections, target_sections)

        if current_keys and (
            len(current_keys) >= max_sections
            or current_chars + section_chars > TRANSLATION_CHUNK_CHAR_LIMIT
        ):
            chunks.append(
                {
                    "type": "balanced",
                    "keys": current_keys,
                    "limit": max_sections,
                    "chars": current_chars,
                }
            )
            current_keys = []
            current_chars = 0

        current_keys.append(key)
        current_chars += section_chars

    if current_keys:
        chunks.append(
            {
                "type": "balanced",
                "keys": current_keys,
                "limit": max_sections,
                "chars": current_chars,
            }
        )

    return chunks


def summarize_chunk_sections(chunk_keys, source_sections, max_items=6):
    """Build a compact section list for failure reports."""
    names = []
    for key in chunk_keys:
        title = extract_first_heading_title(source_sections.get(key, ""))
        cleaned = title.replace("`", "").strip() if title else key
        names.append(cleaned or key)

    if len(names) <= max_items:
        return ", ".join(names)

    shown = ", ".join(names[:max_items])
    return f"{shown}, ... (+{len(names) - max_items} more)"


def should_use_translation_chunk_mode(source_sections):
    """Decide whether a section update should be translated in smaller chunks."""
    total_tokens = sum(
        estimate_translation_tokens(content)
        for content in source_sections.values()
        if content
    )
    section_count = len(source_sections)
    use_chunk_mode = (
        section_count > TRANSLATION_CHUNK_SECTION_THRESHOLD
        or total_tokens > TRANSLATION_CHUNK_TOKEN_THRESHOLD
    )

    return use_chunk_mode, total_tokens


def _extract_line_number_from_key(key):
    """Extract the source line number from a section key like 'modified_390'."""
    parts = key.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return int(parts[1])
    return None


def _parse_diff_hunks(pr_diff):
    """Split a unified diff into its header and individual hunks."""
    if not pr_diff:
        return "", []
    lines = pr_diff.splitlines(True)
    header_lines = []
    hunks = []
    current_hunk = []
    for line in lines:
        if line.startswith("@@"):
            if current_hunk:
                hunks.append(current_hunk)
            current_hunk = [line]
        elif current_hunk:
            current_hunk.append(line)
        else:
            header_lines.append(line)
    if current_hunk:
        hunks.append(current_hunk)
    return "".join(header_lines), hunks


_HUNK_HEADER_RE = re.compile(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@')


def _hunk_new_line_range(hunk_header):
    """Return (start, end) line range for the new-file side of a hunk."""
    m = _HUNK_HEADER_RE.match(hunk_header)
    if not m:
        return None, None
    start = int(m.group(3))
    count = int(m.group(4)) if m.group(4) is not None else 1
    return start, start + count - 1


def filter_diff_for_chunk_sections(pr_diff, chunk_keys, all_section_keys):
    """Return only the diff hunks that overlap with the given chunk's sections."""
    if not pr_diff or not chunk_keys:
        return pr_diff or ""

    prefix_section_keys = {"frontmatter", "intro_section"}
    chunk_keys_by_line = {}
    for key in chunk_keys:
        line_number = _extract_line_number_from_key(key)
        if line_number is not None:
            chunk_keys_by_line.setdefault(line_number, []).append(key)
    chunk_line_numbers = sorted(chunk_keys_by_line)
    chunk_prefix_keys = [
        key for key in chunk_keys if key in prefix_section_keys
    ]
    if not chunk_line_numbers and not chunk_prefix_keys:
        return pr_diff

    all_line_numbers = sorted({
        n for n in (_extract_line_number_from_key(k) for k in all_section_keys) if n is not None
    })

    section_ranges = {}
    if chunk_prefix_keys:
        prefix_end = (
            min(all_line_numbers) - 1
            if all_line_numbers
            else 999999
        )
        for key in chunk_prefix_keys:
            section_ranges[key] = (1, max(1, prefix_end))

    all_line_indexes = {
        line_number: index
        for index, line_number in enumerate(all_line_numbers)
    }
    for line_number, keys in chunk_keys_by_line.items():
        index = all_line_indexes[line_number]
        end = (
            all_line_numbers[index + 1] - 1
            if index + 1 < len(all_line_numbers)
            else 999999
        )
        for key in keys:
            section_ranges[key] = (line_number, end)

    header, hunks = _parse_diff_hunks(pr_diff)
    relevant = []
    covered_section_keys = set()
    for hunk in hunks:
        h_start, h_end = _hunk_new_line_range(hunk[0])
        if h_start is None:
            relevant.append(hunk)
            continue
        hunk_is_relevant = False
        for key, (r_start, r_end) in section_ranges.items():
            if h_start <= r_end and h_end >= r_start:
                covered_section_keys.add(key)
                hunk_is_relevant = True
        if hunk_is_relevant:
            relevant.append(hunk)

    required_modified_keys = {
        key
        for key in chunk_keys
        if key in prefix_section_keys or key.startswith("modified_")
    }
    uncovered_modified_keys = sorted(
        required_modified_keys - covered_section_keys
    )
    if uncovered_modified_keys:
        thread_safe_print(
            "   ⚠️  Chunk diff coverage is incomplete for modified section(s): "
            f"{', '.join(uncovered_modified_keys)}; using the full file diff"
        )
        return pr_diff

    if not relevant:
        return header.rstrip("\n")
    return header + "".join("".join(h) for h in relevant)


def _prepare_translation_prompt(
    pr_diff, source_sections, target_sections,
    source_language, target_language, source_mode,
    glossary_matcher=None, chunk_label=None, post_change_context_keys=None,  # post_change_context_keys kept for backward compat, unused
):
    """Build the AI translation prompt string.

    Returns (prompt, prompt_pr_diff) so callers can reuse the preprocessed diff
    for enforce_minimal_target_updates.
    """
    thread_safe_print(f"   📊 Source sections: {len(source_sections)} sections")
    thread_safe_print(f"   📊 Target sections: {len(target_sections)} sections")

    total_source_chars = sum(len(str(c)) for c in source_sections.values())
    total_target_chars = sum(len(str(c)) for c in target_sections.values())
    thread_safe_print(f"   📏 Content size: Source={total_source_chars:,} chars, Target={total_target_chars:,} chars")

    thread_safe_print(f"   🤖 Getting AI translation for {len(source_sections)} sections...")

    prompt_pr_diff = preprocess_diff_for_heading_anchor_stability(
        pr_diff,
        source_language,
        target_language,
        source_mode=source_mode,
        source_sections=source_sections,
    )
    prompt_source_sections = preprocess_source_sections_for_heading_anchor_stability(
        source_sections,
        prompt_pr_diff,
    )
    formatted_source_sections = json.dumps(
        prompt_source_sections,
        ensure_ascii=False,
        indent=2,
    )
    formatted_target_sections = json.dumps(
        target_sections,
        ensure_ascii=False,
        indent=2,
    )
    source_sections_heading = (
        f"1. Source sections in {source_language} (post-change, i.e. after applying the diff):"
    )

    glossary_prompt_section = ""
    glossary_instruction = ""
    # A chunk with no overlapping hunk contains only a "File:" header.  Do
    # not invoke the glossary matcher for that empty slice; doing so adds cost
    # and can accidentally select terms unrelated to this chunk.
    has_changed_diff_line = any(
        (line.startswith("+") and not line.startswith("+++"))
        or (line.startswith("-") and not line.startswith("---"))
        for line in (prompt_pr_diff or "").splitlines()
    )
    if glossary_matcher and has_changed_diff_line:
        from glossary import filter_terms_for_content, format_terms_for_prompt
        matched_terms = filter_terms_for_content(
            glossary_matcher,
            prompt_pr_diff or '',
            source_language=source_language,
        )
        if matched_terms:
            glossary_text = format_terms_for_prompt(
                matched_terms,
                source_language=source_language,
                target_language=target_language,
            )
            glossary_prompt_section = f"\n4. {glossary_text}\n"
            glossary_instruction = "\nWhen translating the target content, use the translations of the terms provided below to maintain consistency."
            thread_safe_print(f"   📚 Matched {len(matched_terms)} glossary terms for prompt")

    product_name = get_product_name()
    documentation_subject = f"{product_name} user documentation" if product_name else "User documentation"

    prompt = f"""````markdown
You are an expert technical writer in the database domain, proficient in writing clear, concise, and easy-to-understand user documentation.

{documentation_subject} is maintained in both {source_language} and {target_language}. Some content in the {source_language} documentation has been updated through a Git diff, and the corresponding content in {target_language} needs to be updated accordingly.

Your task is to update the target sections in {target_language} according to the Git diff in {source_language}. I will provide:
- The latest source sections in {source_language}
- The Git diff in {source_language}
- The current target sections in {target_language}
- The glossary for terms in both languages

CORE PRINCIPLE:
First determine the operation type of each section:
- If a section key begins with "added_", classify it as an added section regardless of whether its current target section is empty. The key prefix always takes precedence over target content.
- Otherwise, classify it as a modified section. Its current target section normally contains existing content.

Then apply the corresponding rule:
- Added section: Translate the complete latest source section into {target_language}, including natural-language content in unchanged context lines that do not begin with "+" in the Git diff.
- Modified section: Treat the Git diff as the ONLY source of truth for determining what needs to be modified in the target content. Continue to follow the strict diff-only rules below and preserve target content that is not changed by the diff byte-for-byte.

Instructions:

1. Analyze the diff precisely
   - The source sections already represent the FINAL state after the diff is applied.
   - The Git diff contains:
     - Added or updated source lines (beginning with "+")
     - Removed or replaced source lines (beginning with "-")
     - Unchanged context lines (Lines without diff markers)
   - In modified sections, only lines that are changed in the diff are eligible for modification in the target language.
   - In modified sections, unchanged context lines are NOT eligible for any modification in the target language.

2. Apply STRICT minimal edits in modified sections in {target_language}
   - Update ONLY the target-language content corresponding to actual changed source-language content.
   - Modify only the minimum necessary text required by the diff.
   - Do NOT rewrite entire paragraphs, lists, or sections if only a small fragment changed.
   - If a paragraph contains both changed and unchanged sentences:
     - Modify only the sentence fragments required by the diff
     - Preserve all other text exactly as-is

3. In modified sections, for lines not included in the diff or the unchanged context lines in the diff, do not modify them in {target_language}, which means:
     - Do not retranslate them.
     - Do not normalize whitespace or punctuation.
     - Do not adjust terminology.
     - Even if the translation of an unchanged line in {source_language} is not included in the target section, do not add that line to the target section unless the corresponding source line appears as an added line (lines beginning with "+") in the Git diff.
     - Even if the current translation of an unchanged line in {source_language} does not perfectly match the latest source content, leave it unchanged in the target section unless the corresponding source line appears as an added line (lines beginning with "+") in the Git diff.

4. For changed lines in {source_language}, follow the translation rules below:

    - Preserve ALL Markdown formatting (headers, links, code blocks, tables, etc.)
    - If the changed lines include table headers or table rows, translate the natural-language content in these headers or rows into the {target_language}.
    - If the changed part of a changed line contains a link, translate only the natural-language part of the link text into the {target_language}, and keep the linked file path in {source_language}.
    - Do NOT translate:
        - Code examples, SQL queries, configuration values, doc variables/placeholders such as {DOC_VARIABLE_EXAMPLE}, and Mermaid diagram code blocks (```mermaid ... ```). Preserve doc variables exactly as they appear, including triple braces and when they appear inside HTML attributes or tab labels.
        - Explicit heading anchors such as {{#example-test}} in the section titles.
        - Preserve HTML/MDX component tags exactly, including tag names, attributes, and closing tags, such as <CustomContent plan="premium"> and </CustomContent>.
        - File paths, URLs, and command line examples
        - Variable names and system configuration parameters
        - Some text wrapped in ** (such as **Create Resource** on the **Project** page) are UI button or label names, keep them in English if the context of that paragraph indicates that it is UI text.
    - Maintain the exact structure and indentation
    - Keep all special characters and formatting intact
    - {glossary_instruction}

5. Keep the JSON structure unchanged, only modify section content where required by the diff.

6. Conflict-resolution priority

    When rules conflict, follow this priority order:
    1. Translate added sections completely according to the added-section rule in the CORE PRINCIPLE
    2. Preserve unchanged target content in modified sections exactly
    3. Apply minimal edits in {target_language} according to specifically changed lines in {source_language}
    4. Maintain valid syntax/formatting

Input:

{source_sections_heading}
{formatted_source_sections}

Note: These sections are provided only to understand the final source context. Do not use them to introduce any target-language changes unless the exact source lines are changed in the Git diff.

2. GitHub PR changes (Git Diff):
{prompt_pr_diff}

3. Current target sections in {target_language}:
{formatted_target_sections}

4. Glossary for terms in {source_language} and {target_language}:
{glossary_prompt_section}

Please return the complete updated JSON in the same format as target sections, without any additional explanatory text.
````"""

    return prompt, prompt_pr_diff


def _execute_ai_translation(
    prompt, ai_client, target_sections, source_sections, prompt_pr_diff,
    target_file_prefix, prompt_suffix,
    source_language, target_language, source_mode="",
    glossary_matcher=None,
):
    """Send prompt to AI, parse, enforce minimal updates, save, and return result dict."""
    formatted_source_preview = ""
    formatted_target_preview = ""
    try:
        source_block = prompt.split("1. Source sections in", 1)[1].split(
            "\n2. GitHub PR changes (Git Diff):", 1
        )[0]
        src_json = json.loads(source_block.split("\n", 1)[1].strip())
        formatted_source_preview = json.dumps(src_json, ensure_ascii=False, indent=2)[:500]
    except Exception:
        formatted_source_preview = "(preview unavailable)"
    try:
        tgt_json = json.loads(prompt.split(f"3. Current target sections in")[1].split("\n4. Glossary")[0].strip().rsplit("\n", 1)[0])
        formatted_target_preview = json.dumps(tgt_json, ensure_ascii=False, indent=2)[:500]
    except Exception:
        formatted_target_preview = "(preview unavailable)"

    target_section_count = len(target_sections) if hasattr(target_sections, "__len__") else 0
    thread_safe_print(
        f"\n   📤 AI update request ({source_language} → {target_language}): "
        f"{target_section_count} target section(s), {len(prompt):,} prompt chars"
    )
    verbose_thread_safe_print(f"   " + "="*80)
    verbose_thread_safe_print(f"   Source Sections: {formatted_source_preview}...")
    verbose_thread_safe_print(
        "   PR Diff (first 500 chars): "
        f"{prompt_pr_diff[:500] if prompt_pr_diff else '(none)'}..."
    )
    verbose_thread_safe_print(f"   Target Sections: {formatted_target_preview}...")
    verbose_thread_safe_print(f"   " + "="*80)

    try:
        from main import print_token_estimation
        print_token_estimation(prompt, f"Document translation ({source_language} → {target_language})")
    except ImportError:
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            actual_tokens = len(enc.encode(prompt))
            char_count = len(prompt)
            thread_safe_print(f"   💰 Document translation ({source_language} → {target_language})")
            thread_safe_print(f"      📝 Input: {char_count:,} characters")
            thread_safe_print(f"      🔢 Actual tokens: {actual_tokens:,} (using tiktoken cl100k_base)")
        except Exception:
            estimated_tokens = len(prompt) // 4
            char_count = len(prompt)
            thread_safe_print(f"   💰 Document translation ({source_language} → {target_language})")
            thread_safe_print(f"      📝 Input: {char_count:,} characters")
            thread_safe_print(f"      🔢 Estimated tokens: ~{estimated_tokens:,} (fallback: 4 chars/token approximation)")

    temp_dir = get_temp_output_dir()
    try:
        ai_response = ai_client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        thread_safe_print(f"   📝 AI translation response received ({len(ai_response or ''):,} chars)")
        verbose_thread_safe_print(f"   📋 AI response (first 500 chars): {(ai_response or '')[:500]}...")

        result = parse_updated_sections(ai_response)
        if not isinstance(result, dict):
            return TranslationResult(failures=["AI response did not contain a valid JSON object"])
        expected_keys = set(target_sections)
        returned_keys = set(result)
        extra_keys = sorted(returned_keys - expected_keys)
        partial_reasons = []
        if extra_keys:
            result = {
                key: value for key, value in result.items() if key in expected_keys
            }
            if not result:
                return TranslationResult(
                    failures=[
                        "AI response contained no expected section keys; unexpected keys: "
                        + ", ".join(extra_keys)
                    ]
                )
            partial_reasons.append(
                "AI response ignored unexpected section keys: " + ", ".join(extra_keys)
            )
        main_result_keys = set(result)
        result = rewrite_tidb_version_anchors_in_sections(
            result,
            source_language,
            target_language,
            source_mode=source_mode,
        )
        result = enforce_minimal_target_updates(
            target_sections,
            result,
            prompt_pr_diff,
        )
        (
            result,
            heading_partial_reasons,
            already_equivalent_heading_keys,
        ) = translate_changed_headings_with_ai(
            source_sections,
            target_sections,
            result,
            prompt_pr_diff,
            ai_client,
            source_language,
            target_language,
            target_file_prefix,
            prompt_suffix,
            glossary_matcher=glossary_matcher,
        )
        result, anchor_partial_reasons = restore_expected_heading_anchors(
            source_sections,
            result,
            prompt_pr_diff,
        )
        h1_partial_reasons = find_unapplied_modified_h1_sections(
            source_sections,
            target_sections,
            result,
            prompt_pr_diff,
            already_equivalent_keys=already_equivalent_heading_keys,
        )
        partial_reasons.extend(anchor_partial_reasons)
        partial_reasons.extend(heading_partial_reasons)
        partial_reasons.extend(h1_partial_reasons)
        if getattr(ai_response, "completion_status", "complete") == "incomplete":
            partial_reasons.append(
                "AI response incomplete: "
                + (getattr(ai_response, "completion_reason", "") or "unknown reason")
            )
        # Heading repair must not hide a section omitted by the main response.
        missing_keys = sorted(expected_keys - main_result_keys)
        if missing_keys:
            partial_reasons.append(
                "AI response missing section keys: " + ", ".join(missing_keys)
            )
        result = TranslationResult(
            result,
            partial_reasons=list(dict.fromkeys(partial_reasons)),
        )
        thread_safe_print(f"   📊 Parsed {len(result)} sections from AI response")

        ai_results_file = os.path.join(
            temp_dir,
            f"{target_file_prefix}_updated_sections_from_ai{prompt_suffix}.json",
        )
        with open(ai_results_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        thread_safe_print(f"   💾 AI results saved to {ai_results_file}")
        return result

    except Exception as e:
        thread_safe_print(f"   ❌ AI translation failed: {sanitize_exception_message(e)}")
        return TranslationResult(
            failures=[f"AI translation failed: {sanitize_exception_message(e)}"]
        )


def get_updated_sections_from_ai_chunked(
    pr_diff,
    target_sections,
    source_sections,
    ai_client,
    source_language,
    target_language,
    target_file_name=None,
    glossary_matcher=None,
    dry_run=False,
    source_mode="",
    chunk_routing_sections=None,
    post_change_context_keys=None,
):
    """Translate large section sets chunk by chunk and merge successful chunks.

    Phase 1 builds and saves all prompts (with per-chunk filtered diffs).
    Phase 2 sends each prompt to AI and collects results.
    """
    target_file_prefix = get_target_file_prefix_for_debug(target_file_name, target_sections)
    routing_sections = chunk_routing_sections or source_sections
    # The diff header is the authoritative source path.  Debug/test prefixes
    # can differ from it, and using those prefixes would select the wrong
    # per-document chunk policy.
    diff_file_match = re.search(r"^File:\s*(.+?)\s*$", pr_diff or "", re.MULTILINE)
    chunk_policy_file = (
        diff_file_match.group(1) if diff_file_match else target_file_name
    )
    chunk_max_sections = get_translation_chunk_max_sections(chunk_policy_file)
    chunks = build_translation_chunks(
        routing_sections,
        target_sections,
        max_sections=chunk_max_sections,
    )
    total_chunks = len(chunks)
    all_section_keys = list(source_sections.keys())
    post_change_context_key_set = set(post_change_context_keys or [])

    thread_safe_print(
        f"   🧩 Chunk mode enabled: {len(source_sections)} sections split into {total_chunks} chunks"
    )

    # ------------------------------------------------------------------
    # Phase 1: Build and save all prompts
    # ------------------------------------------------------------------
    prepared = []
    temp_dir = get_temp_output_dir()
    for chunk_index, chunk in enumerate(chunks, 1):
        chunk_keys = chunk["keys"]
        chunk_source = {key: source_sections[key] for key in chunk_keys}
        chunk_target = {
            key: target_sections[key]
            for key in chunk_keys
            if key in target_sections
        }
        chunk_diff = filter_diff_for_chunk_sections(pr_diff, chunk_keys, all_section_keys)
        section_summary = summarize_chunk_sections(chunk_keys, routing_sections)
        chunk_label = f"part-{chunk_index:03d}"

        thread_safe_print(
            f"   🧩 Building prompt for chunk {chunk_index}/{total_chunks}: "
            f"{len(chunk_keys)} {chunk['type']} section(s) ({section_summary})"
        )

        prompt, prompt_pr_diff = _prepare_translation_prompt(
            chunk_diff, chunk_source, chunk_target,
            source_language, target_language, source_mode,
            glossary_matcher=glossary_matcher,
            chunk_label=chunk_label,
            post_change_context_keys=[
                key for key in chunk_keys if key in post_change_context_key_set
            ],
        )

        prompt_file = os.path.join(
            temp_dir,
            f"{target_file_prefix}_prompt-for-ai-translation.{chunk_label}.txt",
        )
        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(prompt)
        thread_safe_print(f"   💾 Prompt saved to {prompt_file} ({len(prompt):,} chars)")

        prepared.append({
            "index": chunk_index,
            "keys": chunk_keys,
            "type": chunk["type"],
            "prompt": prompt,
            "prompt_pr_diff": prompt_pr_diff,
            "chunk_source": chunk_source,
            "chunk_target": chunk_target,
            "section_summary": section_summary,
            "chunk_label": chunk_label,
        })

    thread_safe_print(
        f"\n   📝 All {total_chunks} chunk prompts generated. "
        + ("Dry-run: skipping AI calls." if dry_run else "Starting AI translation...")
    )

    if dry_run:
        return TranslationResult()

    # ------------------------------------------------------------------
    # Phase 2: Send each prompt to AI and collect results
    # ------------------------------------------------------------------
    merged_result = TranslationResult()
    failures = []
    partial_reasons = []

    for cp in prepared:
        chunk_index = cp["index"]
        thread_safe_print(
            f"\n   🤖 Sending chunk {chunk_index}/{total_chunks} to AI "
            f"({cp['section_summary']})"
        )

        chunk_result = _execute_ai_translation(
            cp["prompt"],
            ai_client,
            cp["chunk_target"],
            cp["chunk_source"],
            cp["prompt_pr_diff"],
            target_file_prefix,
            f".{cp['chunk_label']}",
            source_language,
            target_language,
            source_mode=source_mode,
            glossary_matcher=glossary_matcher,
        )

        if not chunk_result:
            failures.extend(getattr(chunk_result, "failures", []))
            failures.append(
                f"failed to translate chunk {chunk_index}/{total_chunks} "
                f"(sections: {cp['section_summary']})"
            )
            continue

        merged_result.update(chunk_result)
        partial_reasons.extend(getattr(chunk_result, "partial_reasons", []))
        missing_keys = [key for key in cp["keys"] if key not in chunk_result]
        if missing_keys:
            missing_summary = summarize_chunk_sections(missing_keys, routing_sections)
            partial_reasons.append(
                f"failed to translate chunk {chunk_index}/{total_chunks} "
                f"(missing sections: {missing_summary})"
            )

    if merged_result:
        partial_reasons.extend(failures)
        merged_result.partial_reasons = list(dict.fromkeys(partial_reasons))
    else:
        merged_result.failures = list(dict.fromkeys(failures))

    ai_results_file = os.path.join(temp_dir, f"{target_file_prefix}_updated_sections_from_ai.json")
    with open(ai_results_file, 'w', encoding='utf-8') as f:
        json.dump(merged_result, f, ensure_ascii=False, indent=2)
    thread_safe_print(f"   💾 Merged AI chunk results saved to {ai_results_file}")

    reported_reasons = merged_result.partial_reasons or merged_result.failures
    if reported_reasons:
        thread_safe_print(f"   ⚠️  Chunk translation completed with {len(reported_reasons)} issue(s)")
        for reason in reported_reasons:
            thread_safe_print(f"      ❌ {reason}")

    return merged_result


def get_updated_sections_from_ai(pr_diff, target_sections, source_old_content_dict, ai_client, source_language, target_language, target_file_name=None, glossary_matcher=None, dry_run=False, source_mode="", _disable_chunking=False, _chunk_label=None, _chunk_routing_content_dict=None, _post_change_context_keys=None):
    """Use AI to update target sections based on source content (post-change), PR diff, and target sections."""
    if not source_old_content_dict or not target_sections:
        return {}
    
    # Filter out deleted sections and prepare source sections from old content
    source_sections = {}
    for key, old_content in source_old_content_dict.items():
        # Skip deleted sections
        if 'deleted' in key:
            continue
        
        # Handle null values by using empty string
        content = old_content if old_content is not None else ""
        source_sections[key] = content

    # Strip SVG tags from all content before sending to AI to save tokens
    # and avoid truncation.  The svg_map is used to restore originals after
    # the AI responds.
    source_sections, target_sections, pr_diff, svg_map = \
        strip_svgs_from_sections_and_diff(source_sections, target_sections, pr_diff)
    if svg_map:
        thread_safe_print(f"   🖼️  Replaced {len(svg_map)} SVG(s) with placeholders for AI translation")

    if _chunk_routing_content_dict:
        routing_sections = {
            key: _chunk_routing_content_dict.get(key, source_sections[key])
            for key in source_sections
        }
    else:
        routing_sections = source_sections

    def _restore_result(result):
        """Restore SVG placeholders in the AI result dict."""
        if not svg_map or not result:
            return result
        restored = restore_svgs_in_dict(result, svg_map)
        if hasattr(result, "failures"):
            restored_obj = TranslationResult(
                restored,
                failures=result.failures,
                partial_reasons=getattr(result, "partial_reasons", []),
            )
            return restored_obj
        return restored

    if not _disable_chunking:
        use_chunk_mode, total_source_tokens = should_use_translation_chunk_mode(routing_sections)
        thread_safe_print(
            f"   🔢 Source new_content tokens for translation routing: ~{total_source_tokens:,}"
        )
        if use_chunk_mode:
            result = get_updated_sections_from_ai_chunked(
                pr_diff,
                target_sections,
                source_sections,
                ai_client,
                source_language,
                target_language,
                target_file_name,
                glossary_matcher=glossary_matcher,
                dry_run=dry_run,
                source_mode=source_mode,
                chunk_routing_sections=routing_sections,
                post_change_context_keys=_post_change_context_keys,
            )
            return _restore_result(result)

    prompt, prompt_pr_diff = _prepare_translation_prompt(
        pr_diff, source_sections, target_sections,
        source_language, target_language, source_mode,
        glossary_matcher=glossary_matcher,
        chunk_label=_chunk_label,
        post_change_context_keys=_post_change_context_keys,
    )

    # Save prompt to file for reference
    target_file_prefix = get_target_file_prefix_for_debug(target_file_name, target_sections)
    temp_dir = get_temp_output_dir()
    
    prompt_suffix = f".{_chunk_label}" if _chunk_label else ""
    prompt_file = os.path.join(temp_dir, f"{target_file_prefix}_prompt-for-ai-translation{prompt_suffix}.txt")
    with open(prompt_file, 'w', encoding='utf-8') as f:
        f.write(prompt)
    
    thread_safe_print(f"\n💾 Prompt saved to {prompt_file}")
    thread_safe_print(f"📝 Prompt length: {len(prompt)} characters")
    thread_safe_print(f"📊 Source sections: {len(source_sections)}")
    thread_safe_print(f"📊 Target sections: {len(target_sections)}")

    if dry_run:
        thread_safe_print(f"⏸️  Dry-run mode: prompt saved, skipping AI call")
        return {}

    thread_safe_print(f"🤖 Sending prompt to AI...")

    result = _execute_ai_translation(
        prompt, ai_client, target_sections, source_sections, prompt_pr_diff,
        target_file_prefix, prompt_suffix,
        source_language, target_language,
        source_mode=source_mode,
        glossary_matcher=glossary_matcher,
    )
    return _restore_result(result)

def parse_updated_sections(ai_response):
    """Parse AI response and extract JSON (from get-updated-target-sections.py)"""
    # Ensure temp_output directory exists for debug files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    temp_dir = os.path.join(script_dir, "temp_output")
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        print(f"\n   🔧 Parsing AI response...")
        print(f"   Raw response length: {len(ai_response)} characters")
        
        # Try to extract JSON from AI response
        cleaned_response = ai_response.strip()
        
        # Remove markdown code blocks if present
        if cleaned_response.startswith('```json'):
            cleaned_response = cleaned_response[7:]
            print(f"   📝 Removed '```json' prefix")
        elif cleaned_response.startswith('```'):
            cleaned_response = cleaned_response[3:]
            print(f"   📝 Removed '```' prefix")
        
        if cleaned_response.endswith('```'):
            cleaned_response = cleaned_response[:-3]
            print(f"   📝 Removed '```' suffix")
        
        cleaned_response = cleaned_response.strip()
        
        print(f"   📝 Cleaned response length: {len(cleaned_response)} characters")
        verbose_thread_safe_print(f"   📝 First 200 chars: {cleaned_response[:200]}...")
        verbose_thread_safe_print(f"   📝 Last 200 chars: ...{cleaned_response[-200:]}")
        
        # Try to find JSON content between curly braces
        start_idx = cleaned_response.find('{')
        end_idx = cleaned_response.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_content = cleaned_response[start_idx:end_idx+1]
            print(f"   📝 Extracted JSON content length: {len(json_content)} characters")
            
            try:
                # Parse JSON
                updated_sections = json.loads(json_content)
                print(f"   ✅ Successfully parsed JSON with {len(updated_sections)} sections")
                return updated_sections
            except json.JSONDecodeError as e:
                print(f"   ⚠️  JSON seems incomplete, trying to fix...")
                
                # Try to fix incomplete JSON by finding the last complete entry
                lines = json_content.split('\n')
                fixed_lines = []
                in_value = False
                quote_count = 0
                
                for line in lines:
                    if '"' in line:
                        quote_count += line.count('"')
                    
                    fixed_lines.append(line)
                    
                    # If we have an even number of quotes, we might have a complete entry
                    if quote_count % 2 == 0 and (line.strip().endswith(',') or line.strip().endswith('"')):
                        # Try to parse up to this point
                        potential_json = '\n'.join(fixed_lines)
                        if not potential_json.rstrip().endswith('}'):
                            # Remove trailing comma and add closing brace
                            if potential_json.rstrip().endswith(','):
                                potential_json = potential_json.rstrip()[:-1] + '\n}'
                            else:
                                potential_json += '\n}'
                        
                        try:
                            partial_sections = json.loads(potential_json)
                            print(f"   🔧 Fixed JSON with {len(partial_sections)} sections")
                            return partial_sections
                        except:
                            continue
                
                # If all else fails, return the original error
                raise e
        else:
            print(f"   ❌ Could not find valid JSON structure in response")
            return None
        
    except json.JSONDecodeError as e:
        print(f"   ❌ Error parsing AI response as JSON: {sanitize_exception_message(e)}")
        print(f"   📝 Error at position: {e.pos if hasattr(e, 'pos') else 'unknown'}")
        
        # Save debug info
        debug_file = os.path.join(temp_dir, f"ai_response_debug_{os.getpid()}.txt")
        with open(debug_file, 'w', encoding='utf-8') as f:
            f.write("Original AI Response:\n")
            f.write("="*80 + "\n")
            f.write(ai_response)
            f.write("\n" + "="*80 + "\n")
            f.write("Cleaned Response:\n")
            f.write("-"*80 + "\n")
            f.write(cleaned_response if 'cleaned_response' in locals() else "Not available")
        
        print(f"   📁 Debug info saved to: {debug_file}")
        return None
    except Exception as e:
        print(f"   ❌ Unexpected error parsing AI response: {sanitize_exception_message(e)}")
        return None


def replace_frontmatter_content(lines, new_content):
    """Replace content from beginning of file to first top-level header"""
    # Find the first top-level header
    first_header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith('# '):
            first_header_idx = i
            break
    
    if first_header_idx is None:
        # No top-level header found, replace entire content
        return new_content.split('\n')
    
    # Replace content from start to before first header
    new_lines = new_content.split('\n')
    return new_lines + lines[first_header_idx:]


def replace_toplevel_section_content(lines, target_line_num, new_content):
    """Replace content from top-level header to first next-level header"""
    start_idx = target_line_num - 1  # Convert to 0-based index
    
    # Find the end of top-level section (before first ## header)
    end_idx = len(lines)
    for i in range(start_idx + 1, len(lines)):
        line = lines[i].strip()
        if line.startswith('##'):  # Found first next-level header
            end_idx = i
            break
    
    # Replace the top-level section content (from start_idx to end_idx)
    new_lines = new_content.split('\n')
    return lines[:start_idx] + new_lines + lines[end_idx:]


def update_local_document(file_path, updated_sections, hierarchy_dict, target_local_path):
    """Update local document using hierarchy-based section identification (from update-target-doc-v2.py)"""
    local_path = safe_target_path(target_local_path, file_path)
    
    if not os.path.exists(local_path):
        print(f"   ❌ Local file not found: {local_path}")
        return False
    
    try:
        # Read document content
        with open(local_path, 'r', encoding='utf-8') as f:
            document_content = f.read()
        
        lines = document_content.split('\n')
        
        replacements_made = []
        
        # Use a unified approach: build a complete replacement plan first, then execute it
        # This avoids line number shifts during the replacement process
        
        # Find section boundaries for ALL sections
        section_boundaries = find_section_boundaries(lines, hierarchy_dict)
        
        # Create a comprehensive replacement plan
        replacement_plan = []
        
        for line_num, new_content in updated_sections.items():
            if line_num == "0":
                # Special handling for frontmatter
                first_header_idx = None
                for i, line in enumerate(lines):
                    if line.strip().startswith('# '):
                        first_header_idx = i
                        break
                
                replacement_plan.append({
                    'type': 'frontmatter',
                    'start': 0,
                    'end': first_header_idx if first_header_idx is not None else len(lines),
                    'new_content': new_content,
                    'line_num': line_num
                })
                
            elif line_num in hierarchy_dict:
                hierarchy = hierarchy_dict[line_num]
                if ' > ' not in hierarchy:  # Top-level section
                    # Special handling for top-level sections
                    start_idx = int(line_num) - 1
                    end_idx = len(lines)
                    for i in range(start_idx + 1, len(lines)):
                        line = lines[i].strip()
                        if line.startswith('##'):
                            end_idx = i
                            break
                    
                    replacement_plan.append({
                        'type': 'toplevel',
                        'start': start_idx,
                        'end': end_idx,
                        'new_content': new_content,
                        'line_num': line_num
                    })
                else:
                    # Regular section
                    if line_num in section_boundaries:
                        boundary = section_boundaries[line_num]
                        replacement_plan.append({
                            'type': 'regular',
                            'start': boundary['start'],
                            'end': boundary['end'],
                            'new_content': new_content,
                            'line_num': line_num,
                            'hierarchy': boundary['hierarchy']
                        })
                    else:
                        print(f"      ⚠️  Section at line {line_num} not found in hierarchy")
        
        # Sort replacement plan: process from bottom to top of the document to avoid line shifts
        # Sort by start line in reverse order (highest line number first)
        replacement_plan.sort(key=lambda x: -x['start'])
        
        # Execute replacements in the planned order (from bottom to top)
        print(f"      📋 Executing {len(replacement_plan)} replacements from bottom to top:")
        for i, replacement in enumerate(replacement_plan):
            print(f"      {i+1}. {replacement['type']} (line {replacement.get('line_num', '0')}, start: {replacement['start']})")
        
        for replacement in replacement_plan:
            start = replacement['start']
            end = replacement['end']
            new_content = replacement['new_content']
            new_lines = new_content.split('\n')
            
            # Replace the content
            lines = lines[:start] + new_lines + lines[end:]
            
            # Record the replacement
            original_line_count = end - start
            line_diff = len(new_lines) - original_line_count
            
            replacements_made.append({
                'type': replacement['type'],
                'line_num': replacement.get('line_num', 'N/A'),
                'hierarchy': replacement.get('hierarchy', 'N/A'),
                'start': start,
                'end': end,
                'original_lines': original_line_count,
                'new_lines': len(new_lines),
                'line_diff': line_diff
            })
            
            print(f"      ✅ Updated {replacement['type']} section: {replacement.get('line_num', 'frontmatter')}")
        
        # Save updated document
        atomic_write_text(local_path, '\n'.join(lines))
        
        print(f"   ✅ Updated {len(replacements_made)} sections")
        for replacement in replacements_made:
            print(f"      📝 Line {replacement['line_num']}: {replacement['hierarchy']}")
        
        return True
        
    except Exception as e:
        thread_safe_print(f"   ❌ Error updating file: {sanitize_exception_message(e)}")
        return False

def find_section_boundaries(lines, hierarchy_dict):
    """Find the start and end line for each section based on hierarchy (from update-target-doc-v2.py)"""
    section_boundaries = {}
    
    # Sort sections by line number
    sorted_sections = sorted(hierarchy_dict.items(), key=lambda x: int(x[0]))
    
    for i, (line_num, hierarchy) in enumerate(sorted_sections):
        start_line = int(line_num) - 1  # Convert to 0-based index
        
        # Find end line (start of next section at same or higher level)
        end_line = len(lines)  # Default to end of document
        
        if start_line >= len(lines):
            continue
            
        # Get current section level
        raw_current_line = lines[start_line]
        current_line = raw_current_line.strip()
        if not is_markdown_heading(raw_current_line):
            continue
            
        current_level = len(current_line.split()[0])  # Count # characters
        
        # Look for next section at same or higher level.
        # Ignore markdown-like headers inside fenced code blocks.
        in_code_block = False
        code_block_delimiter = None
        for j in range(start_line + 1, len(lines)):
            raw_line = lines[j]
            line = raw_line.strip()

            fence_match = re.match(r'^(`{3,}|~{3,})', line)
            if fence_match:
                if not in_code_block:
                    in_code_block = True
                    code_block_delimiter = fence_match.group(1)
                elif line.startswith(code_block_delimiter):
                    in_code_block = False
                    code_block_delimiter = None
                continue

            if not in_code_block and is_markdown_heading(raw_line):
                line_level = len(line.split()[0]) if line.split() else 0
                if line_level <= current_level:
                    end_line = j
                    break
        
        section_boundaries[line_num] = {
            'start': start_line,
            'end': end_line,
            'hierarchy': hierarchy,
            'level': current_level
        }
    
    return section_boundaries

def insert_sections_into_document(file_path, translated_sections, target_insertion_points, target_local_path):
    """Insert translated sections into the target document at specified points"""
    
    if not translated_sections or not target_insertion_points:
        thread_safe_print(f"   ⚠️  No sections or insertion points provided")
        return False
    
    local_path = safe_target_path(target_local_path, file_path)
    
    if not os.path.exists(local_path):
        thread_safe_print(f"   ❌ Local file not found: {local_path}")
        return False
    
    try:
        # Read document content
        with open(local_path, 'r', encoding='utf-8') as f:
            document_content = f.read()
        
        lines = document_content.split('\n')
        thread_safe_print(f"   📄 Document has {len(lines)} lines")
        
        # Sort insertion points by line number in descending order to avoid position shifts
        sorted_insertions = sorted(
            target_insertion_points.items(), 
            key=lambda x: x[1]['insertion_after_line'], 
            reverse=True
        )
        
        insertions_made = []
        
        for group_id, point_data in sorted_insertions:
            insertion_after_line = point_data['insertion_after_line']
            new_sections = point_data['new_sections']
            insertion_type = point_data['insertion_type']
            
            thread_safe_print(f"     📌 Inserting {len(new_sections)} sections after line {insertion_after_line}")
            
            # Convert 1-based line number to 0-based index for insertion point
            # insertion_after_line is 1-based, so insertion_index should be insertion_after_line - 1
            insertion_index = insertion_after_line - 1
            
            # Prepare new content to insert
            new_content_lines = []
            
            # Add an empty line before the new sections if not already present
            if insertion_index < len(lines) and lines[insertion_index].strip():
                new_content_lines.append("")
            
            # Add each translated section
            for section_line_num in new_sections:
                # Find the corresponding translated content
                section_hierarchy = None
                section_content = None
                
                # Search for the section in translated_sections by line number or hierarchy
                for hierarchy, content in translated_sections.items():
                    # Try to match by hierarchy or find the content
                    if str(section_line_num) in hierarchy or content:  # This is a simplified matching
                        section_hierarchy = hierarchy
                        section_content = content
                        break
                
                if section_content:
                    # Split content into lines and add to insertion
                    content_lines = section_content.split('\n')
                    new_content_lines.extend(content_lines)
                    
                    # Add spacing between sections
                    if section_line_num != new_sections[-1]:  # Not the last section
                        new_content_lines.append("")
                    
                    thread_safe_print(f"       ✅ Added section: {section_hierarchy}")
                else:
                    thread_safe_print(f"       ⚠️  Could not find translated content for section at line {section_line_num}")
            
            # Add an empty line after the new sections if not already present
            # Check if the new content already ends with an empty line
            if new_content_lines and not new_content_lines[-1].strip():
                # Content already ends with empty line, don't add another
                pass
            elif insertion_index + 1 < len(lines) and lines[insertion_index + 1].strip():
                # Next line has content and our content doesn't end with empty line, add one
                new_content_lines.append("")
            
            # Insert the new content (insert after insertion_index line, before the next line)
            # If insertion_after_line is 251, we want to insert at position 252 (0-based index 251)
            lines = lines[:insertion_index + 1] + new_content_lines + lines[insertion_index + 1:]
            
            insertions_made.append({
                'group_id': group_id,
                'insertion_after_line': insertion_after_line,
                'sections_count': len(new_sections),
                'lines_added': len(new_content_lines),
                'insertion_type': insertion_type
            })
        
        # Save updated document
        atomic_write_text(local_path, '\n'.join(lines))
        
        thread_safe_print(f"   ✅ Successfully inserted {len(insertions_made)} section groups")
        for insertion in insertions_made:
            thread_safe_print(f"      📝 {insertion['group_id']}: {insertion['sections_count']} sections, {insertion['lines_added']} lines after line {insertion['insertion_after_line']}")
        
        return True
        
    except Exception as e:
        thread_safe_print(f"   ❌ Error inserting sections: {sanitize_exception_message(e)}")
        return False

def process_modified_sections(modified_sections, pr_diff, source_context_or_pr_url, github_client, ai_client, repo_config, max_non_system_sections=120, glossary_matcher=None, dry_run=False):
    """Process modified sections with full data structure support"""
    results = []
    
    for file_path, file_data in modified_sections.items():
        thread_safe_print(f"\n📄 Processing {file_path}")
        
        try:
            # Call process_single_file with the complete data structure
            success, message = process_single_file(
                file_path, 
                file_data,  # Pass the complete data structure (includes 'sections', 'original_hierarchy', etc.)
                pr_diff, 
                source_context_or_pr_url, 
                github_client, 
                ai_client, 
                repo_config, 
                max_non_system_sections,
                glossary_matcher=glossary_matcher,
                dry_run=dry_run
            )
            
            if success:
                thread_safe_print(f"   ✅ Successfully processed {file_path}")
                results.append((file_path, True, message))
            else:
                thread_safe_print(f"   ❌ Failed to process {file_path}: {message}")
                results.append((file_path, False, message))
                
        except Exception as e:
            sanitized = sanitize_exception_message(e)
            thread_safe_print(f"   ❌ Error processing {file_path}: {sanitized}")
            results.append((file_path, False, f"Error processing {file_path}: {sanitized}"))
    
    return results

def process_single_file_deletion(file_path, source_sections, source_context_or_pr_url, github_client, ai_client, repo_config, max_non_system_sections=120):
    """Process deletion of sections in a single file"""
    
    # Import needed functions
    from diff_analyzer import get_target_hierarchy_and_content
    from section_matcher import (
        find_direct_matches_for_special_files, 
        filter_non_system_sections, 
        get_corresponding_sections,
        is_system_variable_or_config,
        clean_title_for_matching,
        parse_ai_response,
        find_matching_line_numbers,
    )
    
    # Get target file hierarchy and content
    target_hierarchy, target_lines = get_target_hierarchy_and_content(
        file_path,
        github_client,
        repo_config['target_repo'],
        repo_config.get('target_local_path'),
        repo_config.get('prefer_local_target_for_read', False),
        repo_config.get('target_ref'),
    )
    
    if not target_hierarchy:
        return False, f"Could not get target hierarchy for {file_path}"
    
    # Separate system variables from regular sections for hybrid mapping
    system_sections = {}
    regular_sections = {}
    
    for line_num, hierarchy in source_sections.items():
        # Extract title for checking
        if ' > ' in hierarchy:
            title = hierarchy.split(' > ')[-1]
        else:
            title = hierarchy
        
        cleaned_title = clean_title_for_matching(title)
        if is_system_variable_or_config(cleaned_title):
            system_sections[line_num] = hierarchy
        else:
            regular_sections[line_num] = hierarchy
    
    sections_to_delete = []
    
    # Process system variables with direct matching
    if system_sections:
        thread_safe_print(f"   🎯 Direct matching for {len(system_sections)} system sections...")
        matched_dict, failed_matches, skipped_sections = find_direct_matches_for_special_files(
            system_sections, target_hierarchy, target_lines
        )
        
        for target_line_num, hierarchy_string in matched_dict.items():
            sections_to_delete.append(int(target_line_num))
            thread_safe_print(f"      ✅ Marked system section for deletion: line {target_line_num}")
        
        if failed_matches:
            thread_safe_print(f"      ❌ Failed to match {len(failed_matches)} system sections")
            for failed_line in failed_matches:
                thread_safe_print(f"         - Line {failed_line}: {system_sections[failed_line]}")
    
    # Process regular sections with AI matching
    if regular_sections:
        thread_safe_print(f"   🤖 AI matching for {len(regular_sections)} regular sections...")
        
        # Filter target hierarchy for AI
        filtered_target_hierarchy = filter_non_system_sections(target_hierarchy)
        
        # Check if filtered hierarchy is reasonable for AI
        if len(filtered_target_hierarchy) > max_non_system_sections:
            thread_safe_print(f"      ❌ Target hierarchy too large for AI: {len(filtered_target_hierarchy)} > {max_non_system_sections}")
        else:
            # Get AI mapping (convert dict values to lists as expected by the function)
            source_list = list(regular_sections.values())
            target_list = list(filtered_target_hierarchy.values())
            
            ai_mapping = get_corresponding_sections(
                source_list, 
                target_list, 
                ai_client,
                repo_config['source_language'], 
                repo_config['target_language'],
                max_tokens=20000
            )
            
            if ai_mapping:
                # Parse AI response and find matching line numbers
                ai_sections = parse_ai_response(ai_mapping)
                ai_matched = find_matching_line_numbers(ai_sections, target_hierarchy)
                
                for source_line, target_line in ai_matched.items():
                    try:
                        sections_to_delete.append(int(target_line))
                        thread_safe_print(f"      ✅ Marked regular section for deletion: line {target_line}")
                    except ValueError as e:
                        thread_safe_print(
                            f"      ❌ Error converting target_line to int: {target_line}, error: {sanitize_exception_message(e)}"
                        )
                        # If target_line is not a number, try to find it in target_hierarchy
                        for line_num, hierarchy in target_hierarchy.items():
                            if target_line in hierarchy or hierarchy in target_line:
                                sections_to_delete.append(int(line_num))
                                thread_safe_print(f"      ✅ Found matching section at line {line_num}: {hierarchy}")
                                break
    
    # Delete the sections from local document
    if sections_to_delete:
        success = delete_sections_from_document(file_path, sections_to_delete, repo_config['target_local_path'])
        if success:
            return True, f"Successfully deleted {len(sections_to_delete)} sections from {file_path}"
        else:
            return False, f"Failed to delete sections from {file_path}"
    else:
        return False, f"No sections to delete in {file_path}"

def delete_sections_from_document(file_path, sections_to_delete, target_local_path):
    """Delete specified sections from the local document"""
    target_file_path = safe_target_path(target_local_path, file_path)
    
    if not os.path.exists(target_file_path):
        thread_safe_print(f"   ❌ Target file not found: {target_file_path}")
        return False
    
    try:
        # Read current file content without normalizing unrelated line endings.
        lines = read_text_lines_preserve_newlines(target_file_path)
        content = ''.join(lines)
        
        # Import needed function
        from diff_analyzer import build_hierarchy_dict
        
        # Build hierarchy to understand section boundaries
        target_hierarchy = build_hierarchy_dict(content)
        
        # Sort sections to delete in reverse order to maintain line numbers
        sections_to_delete.sort(reverse=True)
        
        thread_safe_print(f"   🗑️  Deleting {len(sections_to_delete)} sections from {file_path}")
        
        for section_line in sections_to_delete:
            section_start = section_line - 1  # Convert to 0-based index
            
            if section_start < 0 or section_start >= len(lines):
                thread_safe_print(f"      ❌ Invalid section line: {section_line}")
                continue
            
            # Find section end
            section_end = len(lines) - 1  # Default to end of file
            
            # Look for next header at same or higher level
            raw_current_line = lines[section_start]
            current_line = raw_current_line.strip()
            if is_markdown_heading(raw_current_line):
                current_level = len(current_line.split('#')[1:])  # Count # characters
                
                for i in range(section_start + 1, len(lines)):
                    raw_line = lines[i]
                    line = raw_line.strip()
                    if is_markdown_heading(raw_line):
                        line_level = len(line.split('#')[1:])
                        if line_level <= current_level:
                            section_end = i - 1
                            break
            
            # Delete section (from section_start to section_end inclusive)
            thread_safe_print(f"      🗑️  Deleting lines {section_start + 1} to {section_end + 1}")
            del lines[section_start:section_end + 1]
        
        # Write updated content back to file while preserving untouched line endings.
        write_text_lines_preserve_newlines(target_file_path, lines)
        
        thread_safe_print(f"   ✅ Updated file: {target_file_path}")
        return True
        
    except Exception as e:
        thread_safe_print(
            f"   ❌ Error deleting sections from {target_file_path}: {sanitize_exception_message(e)}"
        )
        return False

def process_single_file(file_path, source_sections, pr_diff, source_context_or_pr_url, github_client, ai_client, repo_config, max_non_system_sections=120, glossary_matcher=None, dry_run=False):
    """Process a single file - thread-safe function for parallel processing"""
    thread_id = threading.current_thread().name
    source_mode = get_source_mode(source_context_or_pr_url)
    thread_safe_print(f"\n📄 [{thread_id}] Processing {file_path}")
    
    try:
        # Check if this is a TOC file with special operations
        if isinstance(source_sections, dict) and 'type' in source_sections and source_sections['type'] == 'toc':
            from toc_processor import process_toc_file
            return process_toc_file(
                file_path,
                source_sections,
                source_context_or_pr_url,
                github_client,
                ai_client,
                repo_config,
                glossary_matcher=glossary_matcher,
            )
        
        # Check if this is enhanced sections
        if isinstance(source_sections, dict) and 'sections' in source_sections:
            if source_sections.get('type') == 'enhanced_sections':
                # Skip all the matching logic and directly extract data
                thread_safe_print(f"   [{thread_id}] 🚀 Using enhanced sections data, skipping matching logic")
                enhanced_sections = source_sections['sections']
                
                # Extract target sections and source old content from enhanced sections
                # Maintain the exact order from match_source_diff_to_target.json
                from collections import OrderedDict
                target_sections = OrderedDict()
                source_old_content_dict = OrderedDict()
                source_routing_content_dict = OrderedDict()
                
                # Process in the exact order they appear in enhanced_sections (which comes from match_source_diff_to_target.json)
                for key, section_info in enhanced_sections.items():
                    if isinstance(section_info, dict):
                        operation = section_info.get('source_operation', '')
                        
                        # Skip deleted sections - they shouldn't be in the enhanced_sections anyway
                        if operation == 'deleted':
                            continue
                        
                        # Always use source_new_content (post-change) so the AI
                        # sees the final source state alongside the diff and
                        # current target.  This avoids the AI missing newly added
                        # paragraphs that only appeared in the diff.
                        source_content = section_info.get('source_new_content', '')
                        if not source_content:
                            source_content = section_info.get('source_old_content', '')
                        routing_content = source_content
                        
                        # For target sections: use target_content for modified, empty string for added
                        if operation == 'added':
                            target_content = ""  # Added sections have no existing target content
                        else:  # modified
                            target_content = section_info.get('target_content', '')
                        
                        # Add to both dictionaries using the same key from match_source_diff_to_target.json
                        if source_content is not None:
                            source_old_content_dict[key] = source_content
                        source_routing_content_dict[key] = routing_content if routing_content is not None else source_content
                        target_sections[key] = target_content
                
                thread_safe_print(f"   [{thread_id}] 📊 Extracted: {len(target_sections)} target sections, {len(source_old_content_dict)} source old content entries")
                
                # Update sections with AI (get-updated-target-sections.py logic)
                thread_safe_print(f"   [{thread_id}] 🤖 Getting updated sections from AI...")
                updated_sections = get_updated_sections_from_ai(
                    pr_diff,
                    target_sections,
                    source_old_content_dict,
                    ai_client,
                    repo_config['source_language'],
                    repo_config['target_language'],
                    file_path,
                    glossary_matcher=glossary_matcher,
                    dry_run=dry_run,
                    source_mode=source_mode,
                    _chunk_routing_content_dict=source_routing_content_dict,
                )
                if dry_run:
                    thread_safe_print(f"   [{thread_id}] ⏸️  Dry-run: prompt saved for {file_path}")
                    return True, {}
                if not updated_sections:
                    thread_safe_print(f"   [{thread_id}] ⚠️  Could not get AI update")
                    chunk_failures = getattr(updated_sections, "failures", [])
                    if chunk_failures:
                        return False, "; ".join(chunk_failures)
                    return False, f"Could not get AI update for {file_path}"

                chunk_failures = getattr(updated_sections, "failures", [])
                if chunk_failures:
                    thread_safe_print(f"   [{thread_id}] ⚠️  Skipping file update due to chunk translation failures")
                    return False, "; ".join(chunk_failures)

                # Return the AI results for further processing
                thread_safe_print(f"   [{thread_id}] ✅ Successfully got AI translation results for {file_path}")
                return True, updated_sections  # Return the actual AI results
                    
            else:
                # New format: complete data structure
                actual_sections = source_sections['sections']
        
        # Regular file processing continues here for old format
        # Get target hierarchy and content (get-target-affected-hierarchy.py logic)
        from diff_analyzer import get_target_hierarchy_and_content
        target_hierarchy, target_lines = get_target_hierarchy_and_content(
            file_path,
            github_client,
            repo_config['target_repo'],
            repo_config.get('target_local_path'),
            repo_config.get('prefer_local_target_for_read', False),
            repo_config.get('target_ref'),
        )
        if not target_hierarchy:
            thread_safe_print(f"   [{thread_id}] ⚠️  Could not get target content")
            return False, f"Could not get target content for {file_path}"
        else:
            # Old format: direct dict
            actual_sections = source_sections
            
        # Only do mapping if we don't have enhanced sections
        if 'enhanced_sections' not in locals() or not enhanced_sections:
            # Separate different types of sections
            from section_matcher import is_system_variable_or_config
            system_var_sections = {}
            toplevel_sections = {}
            frontmatter_sections = {}
            regular_sections = {}
            
            for line_num, hierarchy in actual_sections.items():
                if line_num == "0" and hierarchy == "frontmatter":
                    # Special handling for frontmatter
                    frontmatter_sections[line_num] = hierarchy
                else:
                    # Extract the leaf title from hierarchy
                    leaf_title = hierarchy.split(' > ')[-1] if ' > ' in hierarchy else hierarchy
                    
                    if is_system_variable_or_config(leaf_title):
                        system_var_sections[line_num] = hierarchy
                    elif leaf_title.startswith('# '):
                        # Top-level titles need special handling
                        toplevel_sections[line_num] = hierarchy
                    else:
                        regular_sections[line_num] = hierarchy
        
        thread_safe_print(f"   [{thread_id}] 📊 Found {len(system_var_sections)} system variable/config, {len(toplevel_sections)} top-level, {len(frontmatter_sections)} frontmatter, and {len(regular_sections)} regular sections")
        
        target_affected = {}
        
        # Process frontmatter sections with special handling
        if frontmatter_sections:
            thread_safe_print(f"   [{thread_id}] 📄 Processing frontmatter section...")
            # For frontmatter, we simply map it to line 0 in target
            for line_num, hierarchy in frontmatter_sections.items():
                target_affected[line_num] = hierarchy
            thread_safe_print(f"   [{thread_id}] ✅ Mapped {len(frontmatter_sections)} frontmatter section")
        
        # Process top-level titles with special matching
        if toplevel_sections:
            thread_safe_print(f"   [{thread_id}] 🔝 Top-level title matching for {len(toplevel_sections)} sections...")
            from section_matcher import find_toplevel_title_matches
            toplevel_matched, toplevel_failed, toplevel_skipped = find_toplevel_title_matches(toplevel_sections, target_lines)
            
            if toplevel_matched:
                target_affected.update(toplevel_matched)
                thread_safe_print(f"   [{thread_id}] ✅ Top-level matched {len(toplevel_matched)} sections")
            
            if toplevel_failed:
                thread_safe_print(f"   [{thread_id}] ⚠️  {len(toplevel_failed)} top-level sections failed matching")
                for failed in toplevel_failed:
                    thread_safe_print(f"       ❌ {failed['hierarchy']}: {failed['reason']}")
        
        # Process system variables/config sections with direct matching
        if system_var_sections:
            thread_safe_print(f"   [{thread_id}] 🎯 Direct matching {len(system_var_sections)} system variable/config sections...")
            from section_matcher import find_direct_matches_for_special_files
            direct_matched, failed_matches, skipped_sections = find_direct_matches_for_special_files(system_var_sections, target_hierarchy, target_lines)
            
            if direct_matched:
                target_affected.update(direct_matched)
                thread_safe_print(f"   [{thread_id}] ✅ Direct matched {len(direct_matched)} system variable/config sections")
            
            if failed_matches:
                thread_safe_print(f"   [{thread_id}] ⚠️  {len(failed_matches)} system variable/config sections failed direct matching")
                for failed in failed_matches:
                    thread_safe_print(f"       ❌ {failed['hierarchy']}: {failed['reason']}")
        
        # Process regular sections with AI mapping using filtered target hierarchy
        if regular_sections:
            thread_safe_print(f"   [{thread_id}] 🤖 AI mapping {len(regular_sections)} regular sections...")
            
            # Filter target hierarchy to only include non-system sections for AI mapping
            from section_matcher import filter_non_system_sections
            filtered_target_hierarchy = filter_non_system_sections(target_hierarchy)
            
            # Check if filtered target hierarchy exceeds the maximum allowed for AI mapping
            if len(filtered_target_hierarchy) > max_non_system_sections:
                thread_safe_print(f"   [{thread_id}] ❌ Too many non-system sections ({len(filtered_target_hierarchy)} > {max_non_system_sections})")
                thread_safe_print(f"   [{thread_id}] ⚠️  Skipping AI mapping for regular sections to avoid complexity")
                
                # If no system sections were matched either, return error
                if not target_affected:
                    error_message = f"File {file_path} has too many non-system sections ({len(filtered_target_hierarchy)} > {max_non_system_sections}) and no system variable sections were matched"
                    return False, error_message
                
                # Continue with only system variable matches if available
                thread_safe_print(f"   [{thread_id}] ✅ Proceeding with {len(target_affected)} system variable/config sections only")
            else:
                # Proceed with AI mapping using filtered hierarchy
                from section_matcher import get_corresponding_sections
                source_list = list(regular_sections.values())
                target_list = list(filtered_target_hierarchy.values())
                
                ai_response = get_corresponding_sections(source_list, target_list, ai_client, repo_config['source_language'], repo_config['target_language'], max_tokens=20000)
                if ai_response:
                    # Parse AI response and find matching line numbers in the original (unfiltered) hierarchy
                    from section_matcher import parse_ai_response, find_matching_line_numbers
                    ai_sections = parse_ai_response(ai_response)
                    ai_matched = find_matching_line_numbers(ai_sections, target_hierarchy)  # Use original hierarchy for line number lookup
                    
                    if ai_matched:
                        target_affected.update(ai_matched)
                        thread_safe_print(f"   [{thread_id}] ✅ AI mapped {len(ai_matched)} regular sections")
                    else:
                        thread_safe_print(f"   [{thread_id}] ⚠️  AI mapping failed for regular sections")
                else:
                    thread_safe_print(f"   [{thread_id}] ⚠️  Could not get AI response for regular sections")
        
        # Summary of mapping results
        thread_safe_print(f"   [{thread_id}] 📊 Total mapped: {len(target_affected)} out of {len(actual_sections)} sections")
        
        if not target_affected:
            thread_safe_print(f"   [{thread_id}] ⚠️  Could not map sections")
            return False, f"Could not map sections for {file_path}"
        
        thread_safe_print(f"   [{thread_id}] ✅ Mapped {len(target_affected)} sections")
        
        # Extract target sections (get-target-affected-sections.py logic)
        thread_safe_print(f"   [{thread_id}] 📝 Extracting target sections...")
        from diff_analyzer import extract_affected_sections
        target_sections = extract_affected_sections(target_affected, target_lines)
        
        # Extract source old content from the enhanced data structure
        thread_safe_print(f"   [{thread_id}] 📖 Extracting source old content...")
        source_old_content_dict = {}
        source_routing_content_dict = {}
        
        # Handle different data structures for source_sections
        if isinstance(source_sections, dict) and 'sections' in source_sections:
            # New format: complete data structure with enhanced matching info
            # Always prefer source_new_content (post-change) so the AI sees
            # the final source state alongside the diff.
            for key, section_info in source_sections.items():
                if isinstance(section_info, dict) and ('source_new_content' in section_info or 'source_old_content' in section_info):
                    source_content = section_info.get('source_new_content') or section_info.get('source_old_content', '')
                    source_old_content_dict[key] = source_content
                    source_routing_content_dict[key] = source_content
        else:
            # Fallback: if we don't have the enhanced structure, we need to get it differently
            thread_safe_print(f"   [{thread_id}] ⚠️  Source sections missing enhanced structure, using fallback")
            # For now, create empty dict to avoid errors - this should be addressed in the calling code
            source_old_content_dict = {}
        
        # Update sections with AI (get-updated-target-sections.py logic)
        thread_safe_print(f"   [{thread_id}] 🤖 Getting updated sections from AI...")
        updated_sections = get_updated_sections_from_ai(
            pr_diff, target_sections, source_old_content_dict, ai_client,
            repo_config['source_language'], repo_config['target_language'], file_path,
            glossary_matcher=glossary_matcher, dry_run=dry_run, source_mode=source_mode,
            _chunk_routing_content_dict=source_routing_content_dict or None,
        )
        if dry_run:
            thread_safe_print(f"   [{thread_id}] ⏸️  Dry-run: prompt saved for {file_path}")
            return True, {}
        if not updated_sections:
            thread_safe_print(f"   [{thread_id}] ⚠️  Could not get AI update")
            chunk_failures = getattr(updated_sections, "failures", [])
            if chunk_failures:
                return False, "; ".join(chunk_failures)
            return False, f"Could not get AI update for {file_path}"

        chunk_failures = getattr(updated_sections, "failures", [])
        if chunk_failures:
            thread_safe_print(f"   [{thread_id}] ⚠️  Skipping file update due to chunk translation failures")
            return False, "; ".join(chunk_failures)
        
        # Update local document (update-target-doc-v2.py logic)
        thread_safe_print(f"   [{thread_id}] 💾 Updating local document...")
        success = update_local_document(file_path, updated_sections, target_affected, repo_config['target_local_path'])
        
        if success:
            thread_safe_print(f"   [{thread_id}] 🎉 Successfully updated {file_path}")
            return True, f"Successfully updated {file_path}"
        else:
            thread_safe_print(f"   [{thread_id}] ❌ Failed to update {file_path}")
            return False, f"Failed to update {file_path}"
            
    except Exception as e:
        sanitized = sanitize_exception_message(e)
        thread_safe_print(f"   [{thread_id}] ❌ Error processing {file_path}: {sanitized}")
        return False, f"Error processing {file_path}: {sanitized}"

def process_added_sections(added_sections, pr_diff, source_context_or_pr_url, github_client, ai_client, repo_config, max_non_system_sections=120, glossary_matcher=None):
    """Process added sections by translating and inserting them"""
    if not added_sections:
        thread_safe_print("\n➕ No added sections to process")
        return

    source_mode = get_source_mode(source_context_or_pr_url)
    
    thread_safe_print(f"\n➕ Processing added sections from {len(added_sections)} files...")
    
    # Import needed functions
    from section_matcher import map_insertion_points_to_target
    from diff_analyzer import get_target_hierarchy_and_content
    
    for file_path, section_data in added_sections.items():
        thread_safe_print(f"\n➕ Processing added sections in {file_path}")
        
        source_sections = section_data['sections']
        insertion_points = section_data['insertion_points']
        
        # Get target file hierarchy and content
        target_hierarchy, target_lines = get_target_hierarchy_and_content(
            file_path,
            github_client,
            repo_config['target_repo'],
            repo_config.get('target_local_path'),
            repo_config.get('prefer_local_target_for_read', False),
            repo_config.get('target_ref'),
        )
        
        if not target_hierarchy:
            thread_safe_print(f"   ❌ Could not get target hierarchy for {file_path}")
            continue
        
        # Map insertion points to target language
        target_insertion_points = map_insertion_points_to_target(
            insertion_points, target_hierarchy, target_lines, file_path, source_context_or_pr_url, github_client, ai_client, repo_config, max_non_system_sections
        )
        
        if not target_insertion_points:
            thread_safe_print(f"   ❌ No insertion points mapped for {file_path}")
            continue
        
        # Use AI to translate/update new sections (similar to modified sections)
        # Since we're now using source_old_content, we need to extract it from the added sections
        source_old_content_dict = {}
        for key, content in source_sections.items():
            # For added sections, source_old_content is typically None or empty
            # We use the new content (from the source file) as the content to translate
            source_old_content_dict[key] = content if content is not None else ""
        
        # Get target sections (empty for new sections, but we need the structure)
        target_sections = {}  # New sections don't have existing target content
        
        # Use the same AI function to translate the new sections
        translated_sections = get_updated_sections_from_ai(
            pr_diff, 
            target_sections, 
            source_old_content_dict, 
            ai_client,
            repo_config['source_language'], 
            repo_config['target_language'],
            file_path,
            glossary_matcher=glossary_matcher,
            source_mode=source_mode
        )
        
        if translated_sections:
            # Insert translated sections into document
            insert_sections_into_document(file_path, translated_sections, target_insertion_points, repo_config['target_local_path'])
            thread_safe_print(f"   ✅ Successfully inserted {len(translated_sections)} sections in {file_path}")
        else:
            thread_safe_print(f"   ⚠️  No sections were translated for {file_path}")

def process_files_in_batches(source_changes, pr_diff, source_context_or_pr_url, github_client, ai_client, repo_config, operation_type="modified", batch_size=5, max_non_system_sections=120, glossary_matcher=None):
    """Process files in parallel batches"""
    # Handle different data formats
    if isinstance(source_changes, dict):
        files = []
        for path, data in source_changes.items():
            if isinstance(data, dict):
                if 'type' in data and data['type'] == 'toc':
                    # TOC file with special operations
                    files.append((path, data))
                elif 'sections' in data:
                    # New format: extract sections for processing
                    files.append((path, data['sections']))
                else:
                    # Old format: direct dict
                    files.append((path, data))
            else:
                # Old format: direct dict
                files.append((path, data))
    else:
        files = list(source_changes.items())
    
    total_files = len(files)
    
    if total_files == 0:
        return []
    
    thread_safe_print(f"\n🔄 Processing {total_files} files in batches of {batch_size}")
    
    results = []
    
    # Process files in batches
    for i in range(0, total_files, batch_size):
        batch = files[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total_files + batch_size - 1) // batch_size
        
        thread_safe_print(f"\n📦 Batch {batch_num}/{total_batches}: Processing {len(batch)} files")
        
        # Process current batch in parallel
        with ThreadPoolExecutor(max_workers=len(batch), thread_name_prefix=f"Batch{batch_num}") as executor:
            # Submit all files in current batch
            future_to_file = {}
            for file_path, source_sections in batch:
                future = executor.submit(
                    process_single_file, 
                    file_path, 
                    source_sections, 
                    pr_diff, 
                    source_context_or_pr_url, 
                    github_client, 
                    ai_client,
                    repo_config,
                    max_non_system_sections,
                    glossary_matcher=glossary_matcher
                )
                future_to_file[future] = file_path
            
            # Collect results as they complete
            from concurrent.futures import as_completed
            batch_results = []
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    success, message = future.result()
                    batch_results.append((file_path, success, message))
                except Exception as e:
                    batch_results.append(
                        (file_path, False, f"Exception in thread: {sanitize_exception_message(e)}")
                    )
            
            results.extend(batch_results)
        
        # Brief pause between batches to avoid overwhelming the APIs
        if i + batch_size < total_files:
            thread_safe_print(f"   ⏸️  Waiting 2 seconds before next batch...")
            import time
            time.sleep(2)
    
    return results

def _detect_and_fix_cross_parent_moves(sections_with_line):
    """Detect sections that moved to a different parent and split them into delete + insert.

    When a source file restructure moves a section from one parent heading to
    another (e.g. ``### Modify project roles`` under ``## Manage project
    access`` becomes ``### Remove instance access`` under a new ``## Manage
    instance access``), the matcher maps the change to the *old* target
    position.  An in-place replace would leave the translated content under the
    wrong parent.

    This function detects such cases and converts the single "modified/replace"
    entry into two entries — a *delete* at the old position and an *insert*
    next to the newly added parent — so the content lands in the right place.
    """
    from section_matcher import extract_first_heading_from_content, clean_title_for_matching

    def _source_line(key):
        parts = key.split('_')
        if len(parts) >= 2 and parts[-1].isdigit():
            return int(parts[-1])
        return 0

    def _heading_level(heading_line):
        if not heading_line:
            return 0
        stripped = heading_line.lstrip()
        level = 0
        for ch in stripped:
            if ch == '#':
                level += 1
            else:
                break
        return level

    added_parents = []
    for key, section_data, line_num in sections_with_line:
        if section_data.get('source_operation') != 'added':
            continue
        new_content = section_data.get('source_new_content', '')
        heading = extract_first_heading_from_content(new_content)
        if heading and _heading_level(heading) == 2:
            added_parents.append((_source_line(key), key, section_data, line_num))
    added_parents.sort(key=lambda x: x[0])

    if not added_parents:
        return sections_with_line

    result = []
    for key, section_data, line_num in sections_with_line:
        if section_data.get('source_operation') != 'modified':
            result.append((key, section_data, line_num))
            continue

        old_content = section_data.get('source_old_content', '') or ''
        new_content = section_data.get('source_new_content', '') or ''

        if old_content.strip():
            result.append((key, section_data, line_num))
            continue

        old_hierarchy = section_data.get('source_original_hierarchy', '')
        old_leaf = old_hierarchy.rsplit(' > ', 1)[-1] if old_hierarchy else ''
        old_title = clean_title_for_matching(old_leaf)

        new_heading = extract_first_heading_from_content(new_content)
        new_title = clean_title_for_matching(new_heading) if new_heading else ''

        if not old_title or not new_title or old_title == new_title:
            result.append((key, section_data, line_num))
            continue

        src_line = _source_line(key)
        parent_match = None
        for p_src_line, p_key, p_data, p_line_num in reversed(added_parents):
            if p_src_line < src_line:
                parent_match = (p_key, p_data, p_line_num)
                break

        if parent_match is None:
            result.append((key, section_data, line_num))
            continue

        p_key, p_data, p_target_line = parent_match

        thread_safe_print(
            f"   🔀 Cross-parent move detected: {key}"
            f"\n      Old position: {old_leaf} (target line {line_num})"
            f"\n      New parent: {p_key} (target line {p_target_line})"
            f"\n      → Splitting into DELETE at line {line_num} + INSERT before line {p_target_line}"
        )

        delete_data = dict(section_data)
        delete_data['source_operation'] = 'deleted'
        delete_data['target_new_content'] = None
        result.append((key + '_delete', delete_data, line_num))

        insert_data = dict(section_data)
        insert_data['insertion_type'] = 'before_reference'
        result.append((key + '_insert', insert_data, p_target_line))

    return result


def apply_heading_level_change_to_target(target_content, old_level, new_level):
    """Apply a heading level change to target content.

    Sets the target heading directly to new_level. This is correct because
    source and target documents should have matching heading levels, so the
    target heading should always end up at the same level as the source.

    Only the first heading line is changed; body content is preserved as-is.
    """
    if not target_content:
        return target_content

    lines = target_content.split('\n')
    first_line = lines[0]
    match = re.match(r'^(#{1,6})\s+(.*)', first_line)
    if not match:
        return target_content

    title_text = match.group(2)

    target_new_level = max(1, min(6, new_level))

    lines[0] = '#' * target_new_level + ' ' + title_text
    return '\n'.join(lines)


SUSPICIOUS_DUPLICATE_BODY_MIN_CHARS = 300
SUSPICIOUS_DUPLICATE_BODY_MIN_LINES = 3
SUSPICIOUS_DUPLICATE_SOURCE_SIMILARITY = 0.50


def _normalized_section_body(content):
    """Normalize a translated/source section body without its first heading."""
    if not isinstance(content, str) or not content.strip():
        return "", 0

    body_lines = []
    removed_heading = False
    for raw_line in content.splitlines():
        if not removed_heading and is_markdown_heading(raw_line):
            removed_heading = True
            continue
        line = re.sub(r"\s+", " ", raw_line.strip())
        if line:
            body_lines.append(line)

    return "\n".join(body_lines), len(body_lines)


def _find_suspicious_duplicate_changed_bodies(match_data):
    """Find high-confidence duplicated translations before writing a file.

    Limit this guard to an added/modified pair with long, exactly duplicated
    target bodies whose source bodies are materially different. Short shared
    notes and intentionally repeated source instructions remain allowed.
    """
    candidates = []
    for key, section_data in (match_data or {}).items():
        operation = section_data.get("source_operation", "")
        if operation not in ("added", "modified"):
            continue

        source_body, _ = _normalized_section_body(
            section_data.get("source_new_content", "")
        )
        target_body, target_line_count = _normalized_section_body(
            section_data.get("target_new_content", "")
        )
        if not source_body or not target_body:
            continue
        if (
            len(target_body) < SUSPICIOUS_DUPLICATE_BODY_MIN_CHARS
            or target_line_count < SUSPICIOUS_DUPLICATE_BODY_MIN_LINES
        ):
            continue

        candidates.append(
            {
                "key": key,
                "operation": operation,
                "source_body": source_body,
                "target_body": target_body,
            }
        )

    issues = []
    for index, left in enumerate(candidates):
        for right in candidates[index + 1:]:
            if {left["operation"], right["operation"]} != {"added", "modified"}:
                continue
            if left["target_body"] != right["target_body"]:
                continue

            source_similarity = difflib.SequenceMatcher(
                None,
                left["source_body"],
                right["source_body"],
            ).ratio()
            if source_similarity >= SUSPICIOUS_DUPLICATE_SOURCE_SIMILARITY:
                continue

            issues.append(
                f"{left['key']} and {right['key']} have identical long target "
                "bodies but materially different source sections "
                f"(source similarity {source_similarity:.0%})"
            )

    return issues


def update_target_document_from_match_data(match_file_path, target_local_path, target_file_name=None):
    """
    Update target document using data from match_source_diff_to_target.json
    This integrates the logic from test_target_update.py
    
    Args:
        match_file_path: Path to the match_source_diff_to_target.json file
        target_local_path: Local path to the target repository 
        target_file_name: Optional target file name (if not provided, will be extracted from match_file_path)
    """
    import json
    import os
    from pathlib import Path
    
    # Load match data
    if not os.path.exists(match_file_path):
        thread_safe_print(f"❌ {match_file_path} file does not exist")
        return False
    
    with open(match_file_path, 'r', encoding='utf-8') as f:
        match_data = json.load(f)
    
    thread_safe_print(f"✅ Loaded {len(match_data)} section matching data from {match_file_path}")
    thread_safe_print(f"   Reading translation results directly from target_new_content field")
    
    if not match_data:
        thread_safe_print("❌ No matching data found")
        return False

    duplicate_body_issues = _find_suspicious_duplicate_changed_bodies(match_data)
    if duplicate_body_issues:
        thread_safe_print(
            "❌ Suspicious duplicate translated section bodies detected; "
            "preserving the existing target file"
        )
        for issue in duplicate_body_issues:
            thread_safe_print(f"   - {issue}")
        return False
    
    # Sort sections by target_line from large to small (modify from back to front)
    sections_with_line = []
    
    for key, section_data in match_data.items():
        operation = section_data.get('source_operation', '')
        target_new_content = section_data.get('target_new_content')
        
        # For deleted sections, target_new_content should be null
        if operation == 'deleted':
            if target_new_content is not None:
                thread_safe_print(f"   ⚠️  Deleted section {key} has non-null target_new_content, should be fixed")
            thread_safe_print(f"   🗑️  Including deleted section: {key}")
        elif not target_new_content:
            thread_safe_print(f"   ⚠️  Skipping section without target_new_content: {key}")
            continue
        
        target_line = section_data.get('target_line')
        if target_line and target_line != 'unknown':
            try:
                # Handle special case for bottom sections
                if target_line == "-1":
                    line_num = -1  # Special marker for bottom sections
                else:
                    line_num = int(target_line)
                sections_with_line.append((key, section_data, line_num))
            except ValueError:
                thread_safe_print(f"⚠️  Skipping invalid target_line: {target_line} for {key}")
    
    # ------------------------------------------------------------------
    # Detect cross-parent section moves: when a "modified" section's
    # content was completely replaced with content that belongs under a
    # newly added parent heading, split it into delete + insert so the
    # new content lands under the correct parent.
    # ------------------------------------------------------------------
    sections_with_line = _detect_and_fix_cross_parent_moves(sections_with_line)

    # Separate sections into different processing groups
    bottom_modified_sections = []  # Process first: modify existing content at document end
    regular_sections = []          # Process second: normal operations from back to front
    bottom_added_sections = []     # Process last: append new content to document end
    
    for key, section_data, line_num in sections_with_line:
        target_hierarchy = section_data.get('target_hierarchy', '')
        
        if target_hierarchy.startswith('bottom-modified-'):
            bottom_modified_sections.append((key, section_data, line_num))
        elif target_hierarchy.startswith('bottom-added-'):
            bottom_added_sections.append((key, section_data, line_num))
        else:
            regular_sections.append((key, section_data, line_num))
    
    # Sort each group appropriately
    def get_source_line_num(item):
        key, section_data, line_num = item
        if '_' in key and key.split('_')[1].isdigit():
            return int(key.split('_')[1])
        return 0
    
    # Bottom modified: sort by source line number (large to small)
    bottom_modified_sections.sort(key=lambda x: -get_source_line_num(x))
    
    # Regular sections: sort by target_line (large to small), then by source line number
    regular_sections.sort(key=lambda x: (-x[2], -get_source_line_num(x)))
    
    # Bottom added: sort by source line number (small to large) for proper document order
    bottom_added_sections.sort(key=lambda x: get_source_line_num(x))
    
    # Combine all sections in processing order
    all_sections = bottom_modified_sections + regular_sections + bottom_added_sections
    
    thread_safe_print(f"\n📊 Processing order: bottom-modified -> regular -> bottom-added")
    thread_safe_print(f"   📋 Bottom modified sections: {len(bottom_modified_sections)}")
    thread_safe_print(f"   📋 Regular sections: {len(regular_sections)}")  
    thread_safe_print(f"   📋 Bottom added sections: {len(bottom_added_sections)}")
    
    if not all_sections:
        thread_safe_print("❌ No valid sections found for update")
        return False
    
    if verbose_logging_enabled():
        thread_safe_print(f"\n📊 Detailed processing order:")
        for i, (key, section_data, line_num) in enumerate(all_sections, 1):
            operation = section_data.get('source_operation', '')
            hierarchy = section_data.get('target_hierarchy', '')
            insertion_type = section_data.get('insertion_type', '')

            # Extract source line number for display
            source_line_num = int(key.split('_')[1]) if '_' in key and key.split('_')[1].isdigit() else 'N/A'

            # Display target_line with special handling for bottom sections
            target_display = "END" if line_num == -1 else str(line_num)

            # Determine section group
            if hierarchy.startswith('bottom-modified-'):
                group = "BotMod"
            elif hierarchy.startswith('bottom-added-'):
                group = "BotAdd"
            else:
                group = "Regular"

            if operation == 'deleted':
                action = "delete"
            elif insertion_type == "before_reference":
                action = "insert"
            elif line_num == -1:
                action = "append"
            else:
                action = "replace"

            thread_safe_print(f"   {i:2}. [{group:7}] Target:{target_display:>3} Src:{source_line_num:3} | {key:15} ({operation:8}) | {action:7} | {hierarchy}")
    
    # Determine target file name
    if target_file_name is None:
        # Extract target file name from match file path
        # e.g., "tikv-configuration-file-match_source_diff_to_target.json" -> "tikv-configuration-file.md"
        match_filename = os.path.basename(match_file_path)
        if match_filename.endswith('-match_source_diff_to_target.json'):
            extracted_name = match_filename[:-len('-match_source_diff_to_target.json')] + '.md'
            target_file_name = extracted_name
            thread_safe_print(f"   📂 Extracted target file name from match file: {target_file_name}")
        else:
            # Fallback: try to determine from source hierarchy
            first_entry = next(iter(match_data.values()))
            source_hierarchy = first_entry.get('source_original_hierarchy', '')
            
            if 'TiFlash' in source_hierarchy or 'tiflash' in source_hierarchy.lower():
                target_file_name = "tiflash/tiflash-configuration.md"
            else:
                # Default to command-line flags for other cases
                target_file_name = "command-line-flags-for-tidb-configuration.md"
            thread_safe_print(f"   📂 Determined target file name from hierarchy: {target_file_name}")
    else:
        thread_safe_print(f"   📂 Using provided target file name: {target_file_name}")
    
    target_file_path = safe_target_path(target_local_path, target_file_name)
    thread_safe_print(f"\n📄 Target file path: {target_file_path}")
    
    # Update target document
    thread_safe_print(f"\n🚀 Starting target document update, will modify {len(all_sections)} sections...")
    success = update_target_document_sections(all_sections, target_file_path)
    
    return success

def _count_markdown_headings(content):
    """Count markdown headings in content, skipping fenced code blocks."""
    count = 0
    in_code_block = False
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith('```') or stripped.startswith('~~~'):
            in_code_block = not in_code_block
            continue
        if not in_code_block and is_markdown_heading(line):
            count += 1
    return count


def _get_first_heading_level(content):
    """Return the heading level (1-6) of the first markdown heading, or 0."""
    in_code_block = False
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith('```') or stripped.startswith('~~~'):
            in_code_block = not in_code_block
            continue
        if not in_code_block and is_markdown_heading(line):
            return len(line) - len(line.lstrip('#'))
    return 0


def _has_added_child_entries(all_sections, modified_heading_level):
    """Check whether all_sections contains added entries whose heading level
    is deeper than modified_heading_level.  When such entries exist the extra
    sub-headings an AI might produce for the parent modified section are
    already handled, so truncation is safe."""
    for _, s_data, _ in all_sections:
        if s_data.get('source_operation') != 'added':
            continue
        added_src = s_data.get('source_new_content', '')
        if added_src:
            added_level = _get_first_heading_level(added_src)
            if added_level > modified_heading_level:
                return True
    return False


def _truncate_overexpanded_translation(target_new_content, source_new_content):
    """Truncate AI translation that overexpands beyond the source content scope.

    When a parent section gets restructured (e.g., new ##### sub-headings added
    under ####), the AI may include those sub-headings in the modified section's
    translation even though they are handled as separate 'added' entries. This
    causes duplication.  Truncate at the first heading that exceeds the heading
    count present in source_new_content.
    """
    if not source_new_content or not target_new_content:
        return target_new_content

    source_heading_count = _count_markdown_headings(source_new_content)
    if source_heading_count == 0:
        return target_new_content

    target_heading_count = 0
    in_code_block = False
    target_lines = target_new_content.splitlines(keepends=True)

    for i, line in enumerate(target_lines):
        stripped = line.strip()
        if stripped.startswith('```') or stripped.startswith('~~~'):
            in_code_block = not in_code_block
            continue
        if not in_code_block and is_markdown_heading(line):
            target_heading_count += 1
            if target_heading_count > source_heading_count:
                truncated = ''.join(target_lines[:i]).rstrip('\n') + '\n'
                total_target = _count_markdown_headings(target_new_content)
                thread_safe_print(
                    f"   ⚠️  Truncated AI overexpansion: {total_target} headings "
                    f"in AI response vs {source_heading_count} in source; "
                    f"trimmed at line {i + 1}"
                )
                return truncated

    return target_new_content


def update_target_document_sections(all_sections, target_file_path):
    """
    Update target document sections - integrated from test_target_update.py
    """
    thread_safe_print(f"\n🚀 Starting target document update: {target_file_path}")
    
    # Read target document
    if not os.path.exists(target_file_path):
        thread_safe_print(f"❌ Target file does not exist: {target_file_path}")
        return False
    
    target_lines = read_text_lines_preserve_newlines(target_file_path)
    
    thread_safe_print(f"📄 Target document total lines: {len(target_lines)}")
    
    # Process modifications in order (bottom-modified -> regular -> bottom-added)
    for i, (key, section_data, target_line_num) in enumerate(all_sections, 1):
        operation = section_data.get('source_operation', '')
        insertion_type = section_data.get('insertion_type', '')
        target_hierarchy = section_data.get('target_hierarchy', '')
        target_new_content = section_data.get('target_new_content')
        target_end_marker = section_data.get('target_end_marker')
        
        thread_safe_print(f"\n📝 {i}/{len(all_sections)} Processing {key} (Line {target_line_num})")
        thread_safe_print(f"   Operation type: {operation}")
        thread_safe_print(f"   Target section: {target_hierarchy}")
        
        if operation == 'deleted':
            # Delete logic: remove the specified section
            if target_line_num == -1:
                thread_safe_print(f"   ❌ Invalid delete operation for bottom section")
                continue
                
            thread_safe_print(f"   🗑️  Delete mode: removing section starting at line {target_line_num}")
            
            # Find section end position
            start_line = resolve_section_start_line(target_lines, target_line_num, target_hierarchy)
            
            if start_line >= len(target_lines):
                thread_safe_print(f"   ❌ Line number out of range: {target_line_num} > {len(target_lines)}")
                continue
            
            # Find section end position
            end_line = find_section_end_for_update(
                target_lines,
                start_line,
                target_hierarchy,
                end_marker=target_end_marker,
            )
            
            thread_safe_print(f"   📍 Delete range: line {start_line + 1} to {end_line}")
            thread_safe_print(f"   📄 Delete content: {target_lines[start_line].strip()[:50]}...")
            
            # Delete content
            deleted_lines = target_lines[start_line:end_line]
            target_lines[start_line:end_line] = []
            
            thread_safe_print(f"   ✅ Deleted {len(deleted_lines)} lines of content")
            
        elif target_new_content is None:
            thread_safe_print(f"   ⚠️  Skipping: target_new_content is null")
            continue
            
        elif not target_new_content:
            thread_safe_print(f"   ⚠️  Skipping: target_new_content is empty")
            continue
            
        else:
            # Handle content format
            verbose_thread_safe_print(f"   📄 Content preview: {repr(target_new_content[:80])}...")
            
            if target_hierarchy.startswith('bottom-'):
                # Bottom section special handling
                if target_hierarchy.startswith('bottom-modified-'):
                    # Bottom modified: find and replace existing content at document end
                    thread_safe_print(f"   🔄 Bottom modified section: replacing existing content at document end")
                    
                    old_content = (section_data.get('source_old_content') or '').strip()
                    
                    if old_content:
                        # Search backwards from end to find the matching section
                        found_line = None
                        for idx in range(len(target_lines) - 1, -1, -1):
                            line_content = target_lines[idx].strip()
                            if line_content == old_content:
                                found_line = idx
                                thread_safe_print(f"   📍 Found target section at line {found_line + 1}: {line_content[:50]}...")
                                break
                        
                        if found_line is not None:
                            # Find section end
                            end_line = find_section_end_for_update(target_lines, found_line, target_hierarchy)
                            
                            # Ensure content format is correct
                            if not target_new_content.endswith('\n'):
                                target_new_content += '\n'
                            
                            # Split content by lines
                            new_lines = target_new_content.splitlines(keepends=True)
                            
                            # Replace content
                            target_lines[found_line:end_line] = new_lines
                            
                            thread_safe_print(f"   ✅ Replaced {end_line - found_line} lines with {len(new_lines)} lines")
                        else:
                            thread_safe_print(f"   ⚠️  Could not find target section, appending to end instead")
                            # Fallback: append to end
                            if not target_new_content.endswith('\n'):
                                target_new_content += '\n'
                            if target_lines and target_lines[-1].strip():
                                target_new_content = '\n' + target_new_content
                            new_lines = target_new_content.splitlines(keepends=True)
                            target_lines.extend(new_lines)
                            thread_safe_print(f"   ✅ Appended {len(new_lines)} lines to end of document")
                    else:
                        thread_safe_print(f"   ⚠️  No old_content found, appending to end instead")
                        # Fallback: append to end
                        if not target_new_content.endswith('\n'):
                            target_new_content += '\n'
                        if target_lines and target_lines[-1].strip():
                            target_new_content = '\n' + target_new_content
                        new_lines = target_new_content.splitlines(keepends=True)
                        target_lines.extend(new_lines)
                        thread_safe_print(f"   ✅ Appended {len(new_lines)} lines to end of document")
                        
                elif target_hierarchy.startswith('bottom-added-'):
                    # Bottom added: append new content to end of document
                    thread_safe_print(f"   🔚 Bottom added section: appending new content to end")
                    
                    # Ensure content format is correct
                    if not target_new_content.endswith('\n'):
                        target_new_content += '\n'
                    
                    # Add spacing before new section if needed
                    if target_lines and target_lines[-1].strip():
                        target_new_content = '\n' + target_new_content
                    
                    # Split content by lines
                    new_lines = target_new_content.splitlines(keepends=True)
                    
                    # Append to end of document
                    target_lines.extend(new_lines)
                    
                    thread_safe_print(f"   ✅ Appended {len(new_lines)} lines to end of document")
                else:
                    # Other bottom sections: append to end
                    thread_safe_print(f"   🔚 Other bottom section: appending to end of document")
                    
                    # Ensure content format is correct
                    if not target_new_content.endswith('\n'):
                        target_new_content += '\n'
                    
                    # Add spacing before new section if needed
                    if target_lines and target_lines[-1].strip():
                        target_new_content = '\n' + target_new_content
                    
                    # Split content by lines
                    new_lines = target_new_content.splitlines(keepends=True)
                    
                    # Append to end of document
                    target_lines.extend(new_lines)
                    
                    thread_safe_print(f"   ✅ Appended {len(new_lines)} lines to end of document")
                
            elif target_hierarchy == "intro_section":
                # Intro section: from first # heading to first ## heading
                thread_safe_print(f"   📄 Intro section mode: replacing from first # to first ##")
                
                # Find first # heading in current buffer
                first_heading_line = None
                for i, line in enumerate(target_lines):
                    if line.strip().startswith('# '):
                        first_heading_line = i
                        break
                if first_heading_line is None:
                    thread_safe_print(f"   ⚠️  No # heading found in target, skipping intro_section update")
                    continue
                
                # Find first ## heading in current buffer
                first_level2_line = None
                for i, line in enumerate(target_lines):
                    if is_markdown_heading(line) and line.strip().startswith('## '):
                        first_level2_line = i
                        break
                if first_level2_line is None:
                    first_level2_line = len(target_lines)
                
                thread_safe_print(f"   📍 Intro section range: line {first_heading_line + 1} to {first_level2_line}")
                
                # Split new content by lines, preserving original structure
                new_lines = target_new_content.splitlines(keepends=True)
                
                # Ensure content ends with proper newline
                if target_new_content.endswith('\n') and not new_lines[-1].endswith('\n\n'):
                    new_lines.append('\n')
                elif target_new_content and not target_new_content.endswith('\n'):
                    if new_lines and not new_lines[-1].endswith('\n'):
                        new_lines[-1] += '\n'
                
                # Replace from # heading to ## heading (leaves frontmatter untouched)
                target_lines[first_heading_line:first_level2_line] = new_lines
                
                thread_safe_print(f"   ✅ Replaced {first_level2_line - first_heading_line} lines of intro section with {len(new_lines)} lines")
                
            elif target_hierarchy == "frontmatter":
                # Frontmatter special handling: directly replace front lines
                thread_safe_print(f"   📄 Frontmatter mode: directly replacing document beginning")
                
                # Find the first top-level heading position
                first_header_line = 0
                for i, line in enumerate(target_lines):
                    if line.strip().startswith('# '):
                        first_header_line = i
                        break
                
                thread_safe_print(f"   📍 Frontmatter range: line 1 to {first_header_line}")
                
                # Split new content by lines, preserving original structure including trailing empty lines
                new_lines = target_new_content.splitlines(keepends=True)
                
                # If the original content ends with \n, it means there should be an empty line after the last content line
                # splitlines() doesn't create this empty line, so we need to add it manually
                if target_new_content.endswith('\n'):
                    new_lines.append('\n')
                elif target_new_content:
                    # If content doesn't end with newline, ensure the last line has one
                    if not new_lines[-1].endswith('\n'):
                        new_lines[-1] += '\n'
                
                # Replace frontmatter
                target_lines[0:first_header_line] = new_lines
                
                thread_safe_print(f"   ✅ Replaced {first_header_line} lines of frontmatter with {len(new_lines)} lines")
                
            elif insertion_type == "before_reference":
                # Insert logic: insert before specified line
                if target_line_num == -1:
                    thread_safe_print(f"   ❌ Invalid insert operation for bottom section")
                    continue
                    
                thread_safe_print(f"   📍 Insert mode: inserting before line {target_line_num}")
                
                # Ensure content format is correct
                if not target_new_content.endswith('\n'):
                    target_new_content += '\n'
                
                # Ensure spacing between sections
                if not target_new_content.endswith('\n\n'):
                    target_new_content += '\n'
                
                # Split content by lines
                new_lines = target_new_content.splitlines(keepends=True)
                
                # Insert at specified position
                insert_position = target_line_num - 1  # Convert to 0-based index
                if insert_position < 0:
                    insert_position = 0
                elif insert_position > len(target_lines):
                    insert_position = len(target_lines)
                
                # Execute insertion
                for j, line in enumerate(new_lines):
                    target_lines.insert(insert_position + j, line)
                
                thread_safe_print(f"   ✅ Inserted {len(new_lines)} lines of content")
                
            else:
                # Replace logic: find target section and replace
                if target_line_num == -1:
                    thread_safe_print(f"   ❌ Invalid replace operation for bottom section")
                    continue
                    
                thread_safe_print(f"   🔄 Replace mode: replacing section starting at line {target_line_num}")
                
                # Guard against AI overexpansion for modified sections:
                # the AI may include sub-headings that are handled by
                # separate 'added' entries, causing duplication.
                # Only truncate when we can confirm added child entries
                # exist that would handle the extra headings.
                if operation == 'modified':
                    source_new_content = section_data.get('source_new_content', '')
                    if source_new_content:
                        mod_level = _get_first_heading_level(source_new_content)
                        if mod_level > 0 and _has_added_child_entries(all_sections, mod_level):
                            target_new_content = _truncate_overexpanded_translation(
                                target_new_content, source_new_content
                            )
                
                # Ensure content format is correct
                if not target_new_content.endswith('\n'):
                    target_new_content += '\n'
                
                # Ensure spacing between sections
                if not target_new_content.endswith('\n\n'):
                    target_new_content += '\n'
                
                # Find section end position
                start_line = resolve_section_start_line(target_lines, target_line_num, target_hierarchy)
                
                if start_line >= len(target_lines):
                    thread_safe_print(f"   ❌ Line number out of range: {target_line_num} > {len(target_lines)}")
                    continue
                
                # Find section end position
                end_line = find_section_end_for_update(
                    target_lines,
                    start_line,
                    target_hierarchy,
                    end_marker=target_end_marker,
                )
                
                thread_safe_print(f"   📍 Replace range: line {start_line + 1} to {end_line}")
                
                # Split new content by lines
                new_lines = target_new_content.splitlines(keepends=True)
                
                # Replace content
                target_lines[start_line:end_line] = new_lines
                
                thread_safe_print(f"   ✅ Replaced {end_line - start_line} lines with {len(new_lines)} lines")
    
    
    write_text_lines_preserve_newlines(target_file_path, target_lines)
    
    thread_safe_print(f"\n✅ Target document update completed!")
    thread_safe_print(f"📄 Updated file: {target_file_path}")
    
    return True

def find_section_end_for_update(lines, start_line, target_hierarchy, end_marker=None):
    """Find section end position - based on test_target_update.py logic"""
    current_line = lines[start_line].strip()

    if end_marker:
        for i in range(start_line + 1, len(lines)):
            if end_marker in lines[i]:
                thread_safe_print(f"     📍 End marker '{end_marker}' found at line {i + 1}")
                return i
    
    if target_hierarchy == "frontmatter":
        # Frontmatter special handling: from --- to second ---, then to first top-level heading
        if start_line == 0 and current_line.startswith('---'):
            # Find second ---
            for i in range(start_line + 1, len(lines)):
                if lines[i].strip() == '---':
                    # Found frontmatter end, but need to include up to next content start
                    # Look for first non-empty line or first heading
                    for j in range(i + 1, len(lines)):
                        line = lines[j].strip()
                        if line and line.startswith('# '):
                            thread_safe_print(f"     📍 Frontmatter ends at line {j} (before first top-level heading)")
                            return j
                        elif line and not line.startswith('#'):
                            # If there's other content, end there
                            thread_safe_print(f"     📍 Frontmatter ends at line {j} (before other content)")
                            return j
                    # If no other content found, end after second ---
                    thread_safe_print(f"     📍 Frontmatter ends at line {i+1} (after second ---)")
                    return i + 1
        # If not standard frontmatter format, find first top-level heading
        for i in range(start_line + 1, len(lines)):
            if is_markdown_heading(lines[i]) and lines[i].startswith('# '):
                thread_safe_print(f"     📍 Frontmatter ends at line {i} (before first top-level heading)")
                return i
        # If no top-level heading found, process entire file
        return len(lines)
    
    if is_markdown_heading(lines[start_line]):
        # Use file_updater.py method to calculate heading level
        current_level = len(current_line.split()[0]) if current_line.split() else 0
        thread_safe_print(f"     🔍 Current heading level: {current_level} (heading: {current_line[:50]}...)")
        
        # Special handling for top-level headings: only process until first second-level heading
        in_code_block = False
        code_block_delimiter = None
        if current_level == 1:
            for i in range(start_line + 1, len(lines)):
                raw_line = lines[i]
                line = raw_line.strip()

                fm = re.match(r'^(`{3,}|~{3,})', line)
                if fm:
                    if not in_code_block:
                        in_code_block = True
                        code_block_delimiter = fm.group(1)
                    elif line.startswith(code_block_delimiter):
                        in_code_block = False
                        code_block_delimiter = None
                    continue

                if not in_code_block and is_markdown_heading(raw_line) and line.startswith('##'):  # Find first second-level heading
                    thread_safe_print(f"     📍 Top-level heading ends at line {i} (before first second-level heading)")
                    return i
            # If no second-level heading found, look for next top-level heading
            for i in range(start_line + 1, len(lines)):
                raw_line = lines[i]
                line = raw_line.strip()

                fm = re.match(r'^(`{3,}|~{3,})', line)
                if fm:
                    if not in_code_block:
                        in_code_block = True
                        code_block_delimiter = fm.group(1)
                    elif line.startswith(code_block_delimiter):
                        in_code_block = False
                        code_block_delimiter = None
                    continue

                if not in_code_block and is_markdown_heading(raw_line) and line.startswith('#') and not line.startswith('##'):
                    thread_safe_print(f"     📍 Top-level heading ends at line {i} (before next top-level heading)")
                    return i
        else:
            # For other level headings, stop at ANY header to get only direct content
            # This prevents including sub-sections in the update range
            for i in range(start_line + 1, len(lines)):
                raw_line = lines[i]
                line = raw_line.strip()

                fm = re.match(r'^(`{3,}|~{3,})', line)
                if fm:
                    if not in_code_block:
                        in_code_block = True
                        code_block_delimiter = fm.group(1)
                    elif line.startswith(code_block_delimiter):
                        in_code_block = False
                        code_block_delimiter = None
                    continue

                if not in_code_block and is_markdown_heading(raw_line):
                    # Stop at ANY header to get only direct content
                    thread_safe_print(f"     📍 Found header at line {i}: {line[:30]}... (stopping for direct content only)")
                    return i
        
        # If not found, return file end
        thread_safe_print(f"     📍 No end position found, using file end")
        return len(lines)
    
    # Non-heading line, only replace current line
    return start_line + 1

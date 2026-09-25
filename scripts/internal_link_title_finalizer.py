"""Finalize translated internal link titles after all files are written."""

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
import difflib
import html
import os
import posixpath
import re
from pathlib import PurePosixPath
from urllib.parse import unquote, urlsplit

from file_io import atomic_write_text, read_safe_target_text
from log_sanitizer import safe_target_path


FENCE_RE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")
HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})[ \t]+(.+?)\s*$")
EXPLICIT_ANCHOR_RE = re.compile(r"\s+\{#([^{}\s]+)\}\s*$")
CLOSING_HASHES_RE = re.compile(r"\s+#+\s*$")
HTML_TAG_RE = re.compile(r"<[^>]+>")
INLINE_LINK_RE = re.compile(r"!?\[([^\]]+)\]\([^)]+\)")
WHITESPACE_RE = re.compile(r"\s+")
ESCAPED_DESTINATION_CHARACTER_RE = re.compile(r"\\([\\()<> ])")


@dataclass(frozen=True)
class Heading:
    level: int
    title: str
    explicit_anchor: str
    generated_anchor: str


@dataclass(frozen=True)
class HeadingIndex:
    h1_headings: tuple
    headings_by_anchor: dict

    def resolve(self, fragment):
        if not fragment:
            if len(self.h1_headings) == 1:
                return self.h1_headings[0]
            return None

        matches = self.headings_by_anchor.get(fragment, ())
        if len(matches) == 1:
            return matches[0]
        return None


@dataclass(frozen=True)
class LinkOccurrence:
    label_start: int
    label_end: int
    label: str
    destination: str


def normalize_comparison_text(text):
    return WHITESPACE_RE.sub(" ", html.unescape(text or "")).strip()


def is_escaped(text, index):
    backslashes = 0
    index -= 1
    while index >= 0 and text[index] == "\\":
        backslashes += 1
        index -= 1
    return backslashes % 2 == 1


def find_balanced_closer(text, start, opener, closer):
    depth = 1
    index = start
    while index < len(text):
        character = text[index]
        if character == "\\" and index + 1 < len(text):
            index += 2
            continue
        if character == opener:
            depth += 1
        elif character == closer:
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return -1


def extract_link_destination(parenthesized_content):
    content = parenthesized_content.strip()
    if not content:
        return ""
    if content.startswith("<"):
        closing = content.find(">", 1)
        if closing == -1:
            return ""
        return content[1:closing]

    depth = 0
    index = 0
    while index < len(content):
        character = content[index]
        if character == "\\" and index + 1 < len(content):
            index += 2
            continue
        if character == "(":
            depth += 1
        elif character == ")" and depth:
            depth -= 1
        elif character.isspace() and depth == 0:
            break
        index += 1
    return content[:index]


def iter_links_in_segment(segment, segment_offset=0):
    index = 0
    active_code_ticks = 0
    while index < len(segment):
        if segment[index] == "`":
            tick_end = index + 1
            while tick_end < len(segment) and segment[tick_end] == "`":
                tick_end += 1
            tick_count = tick_end - index
            if active_code_ticks == 0:
                active_code_ticks = tick_count
            elif tick_count == active_code_ticks:
                active_code_ticks = 0
            index = tick_end
            continue

        if active_code_ticks:
            index += 1
            continue

        if segment[index] != "[" or is_escaped(segment, index):
            index += 1
            continue
        label_end = find_balanced_closer(segment, index + 1, "[", "]")
        if label_end == -1 or label_end + 1 >= len(segment):
            index += 1
            continue
        if index > 0 and segment[index - 1] == "!" and not is_escaped(
            segment,
            index - 1,
        ):
            index = label_end + 1
            continue
        if segment[label_end + 1] != "(":
            index = label_end + 1
            continue

        destination_end = find_balanced_closer(
            segment,
            label_end + 2,
            "(",
            ")",
        )
        if destination_end == -1:
            index = label_end + 1
            continue

        destination = extract_link_destination(
            segment[label_end + 2 : destination_end]
        )
        if destination:
            yield LinkOccurrence(
                label_start=segment_offset + index + 1,
                label_end=segment_offset + label_end,
                label=segment[index + 1 : label_end],
                destination=destination,
            )
        index = destination_end + 1


def plain_heading_text(title):
    text = INLINE_LINK_RE.sub(lambda match: match.group(1), title)
    text = HTML_TAG_RE.sub("", text)
    text = text.replace("`", "")
    text = text.replace("**", "").replace("__", "")
    text = text.replace("*", "").replace("~", "")
    return normalize_comparison_text(text)


def slugify_heading(title):
    plain = plain_heading_text(title).lower()
    output = []
    previous_was_separator = False
    for character in plain:
        if character.isalnum() or character in {"_", "-"}:
            output.append(character)
            previous_was_separator = character == "-"
        elif character.isspace() and output and not previous_was_separator:
            output.append("-")
            previous_was_separator = True
    return "".join(output).strip("-")


def extract_heading_index(content):
    headings = []
    slug_counts = defaultdict(int)
    fence_character = None
    fence_length = 0

    for line in str(content or "").splitlines():
        fence_match = FENCE_RE.match(line)
        if fence_match:
            marker = fence_match.group(1)
            if fence_character is None:
                fence_character = marker[0]
                fence_length = len(marker)
            elif marker[0] == fence_character and len(marker) >= fence_length:
                fence_character = None
                fence_length = 0
            continue
        if fence_character is not None:
            continue

        heading_match = HEADING_RE.match(line)
        if not heading_match:
            continue
        level = len(heading_match.group(1))
        title = CLOSING_HASHES_RE.sub("", heading_match.group(2)).strip()
        anchor_match = EXPLICIT_ANCHOR_RE.search(title)
        explicit_anchor = ""
        if anchor_match:
            explicit_anchor = anchor_match.group(1)
            title = title[: anchor_match.start()].rstrip()

        base_slug = slugify_heading(title)
        generated_anchor = base_slug
        if base_slug:
            duplicate_number = slug_counts[base_slug]
            if duplicate_number:
                generated_anchor = f"{base_slug}-{duplicate_number}"
            slug_counts[base_slug] += 1
        headings.append(
            Heading(
                level=level,
                title=title,
                explicit_anchor=explicit_anchor,
                generated_anchor=generated_anchor,
            )
        )

    anchors = defaultdict(list)
    for heading in headings:
        for anchor in {heading.explicit_anchor, heading.generated_anchor} - {""}:
            anchors[anchor].append(heading)
    return HeadingIndex(
        h1_headings=tuple(heading for heading in headings if heading.level == 1),
        headings_by_anchor={
            anchor: tuple(matches) for anchor, matches in anchors.items()
        },
    )


def resolve_internal_markdown_target(current_file, destination):
    destination = ESCAPED_DESTINATION_CHARACTER_RE.sub(r"\1", destination)
    parsed = urlsplit(destination)
    if parsed.scheme or parsed.netloc or destination.startswith("//"):
        return None, ""

    decoded_path = unquote(parsed.path)
    if not decoded_path:
        target = current_file
    elif decoded_path.startswith("/"):
        target = PurePosixPath(decoded_path.lstrip("/"))
    else:
        normalized = posixpath.normpath(str(current_file.parent / decoded_path))
        if normalized == ".." or normalized.startswith("../"):
            return None, ""
        target = PurePosixPath(normalized)

    if target.suffix.lower() != ".md":
        return None, ""
    return target, unquote(parsed.fragment)


def non_fenced_line_numbers(content):
    outside = set()
    fence_character = None
    fence_length = 0
    for line_number, line in enumerate(content.splitlines(), 1):
        fence_match = FENCE_RE.match(line)
        if fence_match:
            marker = fence_match.group(1)
            if fence_character is None:
                fence_character = marker[0]
                fence_length = len(marker)
            elif marker[0] == fence_character and len(marker) >= fence_length:
                fence_character = None
                fence_length = 0
            continue
        if fence_character is None:
            outside.add(line_number)
    return outside


def iter_changed_added_lines_by_file(changed_files, source_loader):
    for file in changed_files or []:
        file_path = getattr(file, "filename", "")
        patch = getattr(file, "patch", None)
        if not file_path or not file_path.endswith(".md") or not patch:
            continue

        if "](" not in patch:
            continue
        source_content = source_loader(file_path)
        if source_content is None:
            continue
        source_lines = source_content.splitlines()
        outside_fences = non_fenced_line_numbers(source_content)
        new_line_number = None
        for raw_line in patch.splitlines():
            hunk_match = HUNK_RE.match(raw_line)
            if hunk_match:
                new_line_number = int(hunk_match.group(1))
                continue
            if new_line_number is None or raw_line.startswith(("---", "+++")):
                continue
            if raw_line.startswith("-"):
                continue
            if not raw_line.startswith(("+", " ")):
                continue
            if (
                raw_line.startswith("+")
                and new_line_number in outside_fences
                and new_line_number <= len(source_lines)
                and raw_line[1:] == source_lines[new_line_number - 1]
            ):
                yield file_path, raw_line[1:]
            new_line_number += 1


def capture_target_snapshots(changed_files, target_repo_path):
    snapshots = {}
    for file in changed_files or []:
        file_path = getattr(file, "filename", "")
        status = getattr(file, "status", "")
        if not file_path.endswith(".md") or status == "removed":
            continue
        if file_path not in snapshots:
            snapshots[file_path] = read_safe_target_text(target_repo_path, file_path)
    return snapshots


def changed_line_numbers(before_content, after_content):
    after_lines = str(after_content or "").splitlines()
    if before_content is None:
        return set(range(1, len(after_lines) + 1))

    before_lines = str(before_content or "").splitlines()
    matcher = difflib.SequenceMatcher(None, before_lines, after_lines)
    changed = set()
    for tag, _, _, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        changed.update(range(j1 + 1, j2 + 1))
    return changed


def line_ranges_for_file(before_content, after_content):
    changed = changed_line_numbers(before_content, after_content)
    return changed


def build_link_title_candidates(changed_files, source_loader, target_loader):
    candidates = {}

    @lru_cache(maxsize=None)
    def source_content(path):
        return source_loader(path)

    @lru_cache(maxsize=None)
    def source_heading_index(path):
        content = source_content(path)
        if content is None:
            return None
        return extract_heading_index(content)

    @lru_cache(maxsize=None)
    def target_heading_index(path):
        content = target_loader(path)
        if content is None:
            return None
        return extract_heading_index(content)

    for file_path, line in iter_changed_added_lines_by_file(
        changed_files, source_content
    ):
        current_file = PurePosixPath(file_path)
        for link in iter_links_in_segment(line):
            target_relative, fragment = resolve_internal_markdown_target(
                current_file,
                link.destination,
            )
            if target_relative is None:
                continue

            source_index = source_heading_index(target_relative.as_posix())
            target_index = target_heading_index(target_relative.as_posix())
            if source_index is None or target_index is None:
                continue

            source_heading = source_index.resolve(fragment)
            target_heading = target_index.resolve(fragment)
            if source_heading is None or target_heading is None:
                continue
            if normalize_comparison_text(link.label) != normalize_comparison_text(
                source_heading.title
            ):
                continue

            candidates[(target_relative, fragment)] = target_heading.title

    return candidates


def replace_line_link_titles(line, current_file, changed_line_candidates):
    replacements = []
    for link in iter_links_in_segment(line):
        target_relative, fragment = resolve_internal_markdown_target(
            current_file,
            link.destination,
        )
        if target_relative is None:
            continue
        target_title = changed_line_candidates.get((target_relative, fragment))
        if not target_title or target_title == link.label:
            continue
        replacements.append((link.label_start, link.label_end, target_title))

    if not replacements:
        return line, 0

    updated = line
    for start, end, target_title in sorted(replacements, reverse=True):
        updated = updated[:start] + target_title + updated[end:]
    return updated, len(replacements)


def finalize_internal_link_titles(
    changed_files,
    target_repo_path,
    before_snapshots,
    source_loader,
    printer=print,
):
    """Synchronize link labels only on target lines changed in this run."""
    if not target_repo_path:
        return 0

    def target_loader(path):
        return read_safe_target_text(target_repo_path, path)

    candidates = build_link_title_candidates(
        changed_files,
        source_loader,
        target_loader,
    )
    if not candidates:
        return 0

    replacements = 0
    for file_path, before_content in sorted((before_snapshots or {}).items()):
        try:
            target_file_path = safe_target_path(target_repo_path, file_path)
        except ValueError:
            continue
        if not os.path.exists(target_file_path):
            continue

        # Preserve original line endings when rewriting only selected lines.
        with open(target_file_path, "r", encoding="utf-8", newline="") as file:
            after_content = file.read()
        changed_lines = line_ranges_for_file(before_content, after_content)
        if not changed_lines:
            continue

        lines = after_content.splitlines(keepends=True)
        outside_fences = non_fenced_line_numbers(after_content)
        updated_lines = list(lines)
        current_file = PurePosixPath(file_path)
        file_replacements = 0
        for index, line in enumerate(lines, 1):
            if index not in changed_lines or index not in outside_fences:
                continue
            updated_line, count = replace_line_link_titles(
                line,
                current_file,
                candidates,
            )
            if count:
                updated_lines[index - 1] = updated_line
                file_replacements += count

        if not file_replacements:
            continue

        atomic_write_text(target_file_path, "".join(updated_lines), newline="")
        printer(
            f"   🔗 Internal link title finalizer: {file_path} "
            f"({file_replacements} replacement(s))"
        )
        replacements += file_replacements

    return replacements

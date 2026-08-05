"""Validate translated Markdown document structure against source HEAD."""

from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
import re
from typing import Optional

from heading_anchor_utils import (
    EXPLICIT_HEADING_ANCHOR_RE,
    build_heading_anchor_slug,
)


FENCE_RE = re.compile(r"^ {0,3}(```+|~~~+)")
HEADING_RE = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.+)")


@dataclass
class StructureValidationIssue:
    file_path: str
    reason: str
    source_compact: str = ""
    target_compact: str = ""
    first_difference: str = ""

    def to_dict(self):
        return {
            "file_path": self.file_path,
            "reason": self.reason,
            "source_compact": self.source_compact,
            "target_compact": self.target_compact,
            "first_difference": self.first_difference,
        }


@dataclass(frozen=True)
class CustomContentTag:
    line_number: int
    text: str
    kind: str


CUSTOM_CONTENT_TAG_RE = re.compile(r"</?CustomContent\b[^<>]*>")


def iter_markdown_content_lines(content):
    """Yield non-fenced Markdown lines as ``(line_number, line)`` tuples."""
    in_code_block = False
    code_block_marker = None

    for line_number, line in enumerate((content or "").splitlines(), 1):
        fence_match = FENCE_RE.match(line)
        if fence_match:
            marker = fence_match.group(1)
            if not in_code_block:
                in_code_block = True
                code_block_marker = marker
            elif (
                marker[0] == code_block_marker[0]
                and len(marker) >= len(code_block_marker)
                and not line[fence_match.end():].strip()
            ):
                in_code_block = False
                code_block_marker = None
            continue

        if not in_code_block:
            yield line_number, line


def extract_heading_levels(content):
    """Extract Markdown heading levels while skipping fenced code blocks."""
    return [level for _, level in extract_heading_positions(content)]


def extract_heading_positions(content):
    """Extract ``(line_number, level)`` heading anchors outside code fences."""
    positions = []
    for line_number, line in iter_markdown_content_lines(content):
        heading_match = HEADING_RE.match(line)
        if heading_match:
            positions.append((line_number, len(heading_match.group(1))))

    return positions


def compact_heading_levels(levels):
    """Run-length encode heading levels for concise reports."""
    if not levels:
        return "(no headings)"

    runs = []
    index = 0
    while index < len(levels):
        level = levels[index]
        count = 1
        while index + count < len(levels) and levels[index + count] == level:
            count += 1
        runs.append(f"{'#' * level}x{count}")
        index += count
    return " ".join(runs)


def describe_first_difference(source_levels, target_levels):
    max_len = max(len(source_levels), len(target_levels))
    for index in range(max_len):
        source_level = source_levels[index] if index < len(source_levels) else None
        target_level = target_levels[index] if index < len(target_levels) else None
        if source_level == target_level:
            continue

        source_label = "#" * source_level if source_level else "(missing)"
        target_label = "#" * target_level if target_level else "(missing)"
        return f"heading {index + 1}: source {source_label}, target {target_label}"

    return ""


def compare_heading_structure(file_path, source_content, target_content):
    """Return a structure issue when source and target heading sequences differ."""
    source_levels = extract_heading_levels(source_content)
    target_levels = extract_heading_levels(target_content)

    if source_levels == target_levels:
        return None

    return StructureValidationIssue(
        file_path=file_path,
        reason="heading level sequence differs",
        source_compact=compact_heading_levels(source_levels),
        target_compact=compact_heading_levels(target_levels),
        first_difference=describe_first_difference(source_levels, target_levels),
    )


def extract_source_heading_identities(content):
    """Return source heading identities from explicit or implied anchors."""
    identities = []
    for _, line in iter_markdown_content_lines(content):
        heading_match = HEADING_RE.match(line)
        if not heading_match or len(heading_match.group(1)) == 1:
            continue

        heading_text = heading_match.group(2).strip()
        anchor_match = EXPLICIT_HEADING_ANCHOR_RE.search(heading_text)
        identity = (
            anchor_match.group(1).strip()
            if anchor_match
            else build_heading_anchor_slug(heading_text)
        )
        if identity:
            identities.append(identity)
    return identities


def extract_explicit_heading_identities(content):
    """Return explicit non-top-level heading anchors in document order."""
    identities = []
    for _, line in iter_markdown_content_lines(content):
        heading_match = HEADING_RE.match(line)
        if not heading_match or len(heading_match.group(1)) == 1:
            continue

        anchor_match = EXPLICIT_HEADING_ANCHOR_RE.search(
            heading_match.group(2).strip()
        )
        if anchor_match:
            identities.append(anchor_match.group(1).strip())
    return identities


def _compact_heading_identities(identities, limit=8):
    shown = list(identities[:limit])
    if len(identities) > limit:
        shown.append(f"... (+{len(identities) - limit} more)")
    return " -> ".join(shown) if shown else "(no stable headings)"


def compare_stable_heading_order(file_path, source_content, target_content):
    """Check the relative order of unambiguous anchors shared by both files.

    Source English headings have deterministic implied anchors. Translated
    headings are used only when they retain an explicit source-language anchor,
    so translated title text never participates in identity matching.
    """
    source_identities = extract_source_heading_identities(source_content)
    target_identities = extract_explicit_heading_identities(target_content)
    source_counts = Counter(source_identities)
    target_counts = Counter(target_identities)
    shared_unique = {
        identity
        for identity in source_counts.keys() & target_counts.keys()
        if source_counts[identity] == 1 and target_counts[identity] == 1
    }

    if len(shared_unique) < 2:
        return None

    source_order = [
        identity for identity in source_identities if identity in shared_unique
    ]
    target_order = [
        identity for identity in target_identities if identity in shared_unique
    ]
    if source_order == target_order:
        return None

    first_difference = ""
    for index, (source_identity, target_identity) in enumerate(
        zip(source_order, target_order),
        1,
    ):
        if source_identity != target_identity:
            first_difference = (
                f"stable heading {index}: source {source_identity}, "
                f"target {target_identity}"
            )
            break

    return StructureValidationIssue(
        file_path=file_path,
        reason="stable heading relative order differs",
        source_compact=_compact_heading_identities(source_order),
        target_compact=_compact_heading_identities(target_order),
        first_difference=first_difference,
    )


def compare_expected_heading_anchors(
    file_path,
    target_content,
    expected_anchors,
):
    """Report prompt-time changed-heading anchors missing from the target."""
    expected = [anchor for anchor in (expected_anchors or []) if anchor]
    if not expected:
        return None

    target_anchors = extract_explicit_heading_identities(target_content)
    remaining = Counter(target_anchors)
    missing = []
    for anchor in expected:
        if remaining[anchor] > 0:
            remaining[anchor] -= 1
        else:
            missing.append(anchor)

    if not missing:
        return None

    first_missing = missing[0]
    return StructureValidationIssue(
        file_path=file_path,
        reason="expected heading anchors are missing",
        source_compact=_compact_heading_identities(expected),
        target_compact=_compact_heading_identities(target_anchors),
        first_difference=f"expected anchor {{#{first_missing}}} is missing from target",
    )


def _custom_content_tag_kind(tag_text):
    if tag_text.startswith("</"):
        return "closing"
    if tag_text.rstrip().endswith("/>"):
        return "self-closing"
    return "opening"


def extract_custom_content_tags(content):
    """Extract CustomContent tags outside fenced code blocks in document order."""
    tags = []
    for line_number, line in iter_markdown_content_lines(content):
        for match in CUSTOM_CONTENT_TAG_RE.finditer(line):
            tag_text = match.group(0)
            tags.append(
                CustomContentTag(
                    line_number=line_number,
                    text=tag_text,
                    kind=_custom_content_tag_kind(tag_text),
                )
            )
    return tags


def custom_content_tag_signature(tags):
    return [(tag.kind, tag.text) for tag in tags]


def compact_custom_content_tags(tags, limit=6):
    """Summarize CustomContent tags for reports."""
    if not tags:
        return "0 CustomContent tags"

    shown = [tag.text for tag in tags[:limit]]
    if len(tags) > limit:
        shown.append(f"... (+{len(tags) - limit} more)")
    return f"{len(tags)} CustomContent tags: " + " | ".join(shown)


def describe_first_custom_content_difference(source_tags, target_tags):
    max_len = max(len(source_tags), len(target_tags))
    for index in range(max_len):
        source_tag = source_tags[index] if index < len(source_tags) else None
        target_tag = target_tags[index] if index < len(target_tags) else None
        if source_tag and target_tag and (source_tag.kind, source_tag.text) == (target_tag.kind, target_tag.text):
            continue

        source_label = (
            f"{source_tag.text} at line {source_tag.line_number}"
            if source_tag
            else "(missing)"
        )
        target_label = (
            f"{target_tag.text} at line {target_tag.line_number}"
            if target_tag
            else "(missing)"
        )
        return f"CustomContent tag {index + 1}: source {source_label}, target {target_label}"

    return ""


def describe_custom_content_balance_issue(tags):
    stack = []
    for index, tag in enumerate(tags, 1):
        if tag.kind == "self-closing":
            continue
        if tag.kind == "opening":
            stack.append((index, tag))
            continue
        if not stack:
            return f"CustomContent tag {index} at line {tag.line_number}: closing tag without matching opening tag"
        stack.pop()

    if stack:
        index, tag = stack[-1]
        return f"CustomContent tag {index} at line {tag.line_number}: opening tag without matching closing tag"

    return ""


def compare_custom_content_structure(file_path, source_content, target_content):
    """Return a structure issue when CustomContent tags differ or are unbalanced."""
    source_tags = extract_custom_content_tags(source_content)
    target_tags = extract_custom_content_tags(target_content)

    source_balance_issue = describe_custom_content_balance_issue(source_tags)
    if source_balance_issue:
        return StructureValidationIssue(
            file_path=file_path,
            reason="source CustomContent tags are unbalanced",
            source_compact=compact_custom_content_tags(source_tags),
            target_compact=compact_custom_content_tags(target_tags),
            first_difference=source_balance_issue,
        )

    target_balance_issue = describe_custom_content_balance_issue(target_tags)
    if target_balance_issue:
        return StructureValidationIssue(
            file_path=file_path,
            reason="target CustomContent tags are unbalanced",
            source_compact=compact_custom_content_tags(source_tags),
            target_compact=compact_custom_content_tags(target_tags),
            first_difference=target_balance_issue,
        )

    if custom_content_tag_signature(source_tags) == custom_content_tag_signature(target_tags):
        return None

    return StructureValidationIssue(
        file_path=file_path,
        reason="CustomContent tag sequence differs",
        source_compact=compact_custom_content_tags(source_tags),
        target_compact=compact_custom_content_tags(target_tags),
        first_difference=describe_first_custom_content_difference(source_tags, target_tags),
    )


def extract_headings_with_line_numbers(content):
    """Extract headings with 1-based line numbers, skipping fenced code blocks.

    Returns a list of ``(line_number, level, text)`` tuples.
    """
    headings = []
    for line_number, line in iter_markdown_content_lines(content):
        heading_match = HEADING_RE.match(line)
        if heading_match:
            headings.append(
                (line_number, len(heading_match.group(1)), heading_match.group(2).strip())
            )
    return headings


def compare_added_file_line_integrity(source_content, target_content):
    """Check heading line positions and total line count for added-file translations.

    Returns a dict with match booleans, total line counts, and a list of
    heading position mismatches so the caller can report details.
    """
    source_headings = extract_headings_with_line_numbers(source_content or "")
    target_headings = extract_headings_with_line_numbers(target_content or "")

    source_total = len((source_content or "").splitlines())
    target_total = len((target_content or "").splitlines())

    mismatched_headings = []
    max_len = max(len(source_headings), len(target_headings))
    for i in range(max_len):
        src = source_headings[i] if i < len(source_headings) else None
        tgt = target_headings[i] if i < len(target_headings) else None

        src_line = src[0] if src else None
        tgt_line = tgt[0] if tgt else None
        if src_line == tgt_line:
            continue

        mismatched_headings.append({
            "index": i + 1,
            "source_line": src_line,
            "source_heading": f"{'#' * src[1]} {src[2]}" if src else "(missing)",
            "target_line": tgt_line,
            "target_heading": f"{'#' * tgt[1]} {tgt[2]}" if tgt else "(missing)",
        })

    return {
        "heading_lines_match": len(mismatched_headings) == 0,
        "total_lines_match": source_total == target_total,
        "source_total_lines": source_total,
        "target_total_lines": target_total,
        "mismatched_headings": mismatched_headings,
    }


_RELATED_RESOURCES_HEADING_RE = re.compile(r"related\s+resources", re.IGNORECASE)


def _section_has_resource_card_block(section_text):
    """Return True when *section_text* contains a RelatedResources + ResourceCard block."""
    return bool(
        re.search(r"<RelatedResources\b", section_text)
        and re.search(r"<ResourceCard\b", section_text)
    )


def strip_related_resources_sections(content):
    """Remove Related-resources sections that contain RelatedResources/ResourceCard tags.

    In commit-based mode the translation pipeline skips these sections,
    so the verification side should strip them from the source content
    before comparison to avoid false-positive mismatches.
    """
    if not content:
        return content
    if "<RelatedResources" not in content or "<ResourceCard" not in content:
        return content

    lines = content.splitlines()

    headings = []
    for line_number, line in iter_markdown_content_lines(content):
        heading_match = HEADING_RE.match(line)
        if heading_match:
            headings.append(
                (line_number - 1, len(heading_match.group(1)), heading_match.group(2).strip())
            )

    sections_to_remove = []
    for idx, (line_idx, level, text) in enumerate(headings):
        if not _RELATED_RESOURCES_HEADING_RE.search(text):
            continue
        section_end = len(lines)
        for next_idx in range(idx + 1, len(headings)):
            if headings[next_idx][1] <= level:
                section_end = headings[next_idx][0]
                break
        section_text = "\n".join(lines[line_idx:section_end])
        if _section_has_resource_card_block(section_text):
            sections_to_remove.append((line_idx, section_end))

    if not sections_to_remove:
        return content

    removed = set()
    for start, end in sections_to_remove:
        removed.update(range(start, end))

    kept = [line for i, line in enumerate(lines) if i not in removed]
    while kept and not kept[-1].strip():
        kept.pop()

    result = "\n".join(kept)
    if result and content.endswith(("\n", "\r")):
        result += "\n"
    return result


def validate_markdown_heading_structures(
    file_paths: Iterable[str],
    source_content_loader: Callable[[str], Optional[str]],
    target_content_loader: Callable[[str], Optional[str]],
    expected_heading_anchors_loader: Optional[
        Callable[[str], Iterable[str]]
    ] = None,
):
    """Validate Markdown heading levels and CustomContent tags."""
    issues = []

    for file_path in sorted(file_paths):
        if not file_path.lower().endswith(".md"):
            continue

        try:
            source_content = source_content_loader(file_path)
        except Exception as exc:
            issues.append(
                StructureValidationIssue(
                    file_path=file_path,
                    reason=f"could not read source HEAD content: {exc}",
                )
            )
            continue

        try:
            target_content = target_content_loader(file_path)
        except Exception as exc:
            issues.append(
                StructureValidationIssue(
                    file_path=file_path,
                    reason=f"could not read translated target content: {exc}",
                )
            )
            continue

        if source_content is None:
            issues.append(
                StructureValidationIssue(
                    file_path=file_path,
                    reason="could not read source HEAD content",
                )
            )
            continue

        if target_content is None:
            issues.append(
                StructureValidationIssue(
                    file_path=file_path,
                    reason="could not read translated target content",
                )
            )
            continue

        expected_anchor_issue = None
        if expected_heading_anchors_loader:
            try:
                expected_anchors = expected_heading_anchors_loader(file_path)
            except Exception as exc:
                issues.append(
                    StructureValidationIssue(
                        file_path=file_path,
                        reason=f"could not determine expected heading anchors: {exc}",
                    )
                )
                expected_anchors = []
            expected_anchor_issue = compare_expected_heading_anchors(
                file_path,
                target_content,
                expected_anchors,
            )

        for issue in (
            compare_heading_structure(file_path, source_content, target_content),
            compare_stable_heading_order(file_path, source_content, target_content),
            expected_anchor_issue,
            compare_custom_content_structure(file_path, source_content, target_content),
        ):
            if issue:
                issues.append(issue)

    return issues

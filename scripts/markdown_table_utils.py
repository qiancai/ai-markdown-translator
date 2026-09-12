"""Markdown table parsing and untranslated-content detection helpers."""

from dataclasses import dataclass
import re


FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
SEPARATOR_CELL_RE = re.compile(r"^:?-{3,}:?$")
ENGLISH_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")
CJK_RE = re.compile(r"[\u3400-\u9fff]")
JAPANESE_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
INLINE_CODE_RE = re.compile(r"`+[^`]*`+")
HTML_TAG_RE = re.compile(r"<[^>]+>")
DOC_VARIABLE_RE = re.compile(r"\{\{\{.*?\}\}\}")
MARKDOWN_LINK_TARGET_RE = re.compile(r"\]\([^)]+\)")
MARKDOWN_LINK_TARGET_CAPTURE_RE = re.compile(r"\]\(([^)]+)\)")
URL_RE = re.compile(r"https?://\S+")
NUMBER_RE = re.compile(r"(?<![A-Za-z_])\d+(?:[.,]\d+)*(?:%|x)?")
UPPER_IDENTIFIER_RE = re.compile(r"\b[A-Z][A-Z0-9_]{1,}\b")


@dataclass(frozen=True)
class MarkdownTableBlock:
    """A standard pipe table and its zero-based, end-exclusive line range."""

    start_line: int
    end_line: int
    lines: tuple[str, ...]

    @property
    def text(self):
        return "\n".join(self.lines)

    @property
    def header(self):
        return self.lines[0]

    @property
    def separator(self):
        return self.lines[1]

    @property
    def column_counts(self):
        return tuple(count_markdown_table_columns(line) for line in self.lines)


@dataclass(frozen=True)
class SuspectedTableOmission:
    """A source/target table pair that appears not to be fully translated."""

    table_index: int
    reason: str
    source: MarkdownTableBlock
    target: MarkdownTableBlock


def split_markdown_table_cells(line):
    """Split a pipe-table row while preserving escaped pipe characters."""
    stripped = (line or "").strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        return []
    body = stripped[1:-1]
    return [cell.strip() for cell in re.split(r"(?<!\\)\|", body)]


def count_markdown_table_columns(line):
    return len(split_markdown_table_cells(line))


def is_markdown_table_separator(line):
    cells = split_markdown_table_cells(line)
    return bool(cells) and all(SEPARATOR_CELL_RE.fullmatch(cell) for cell in cells)


def extract_markdown_tables(content):
    """Extract standard pipe tables outside fenced code blocks."""
    lines = (content or "").splitlines()
    tables = []
    current_start = None
    current_lines = []
    fence_character = None
    fence_length = 0

    def close_current(end_line):
        nonlocal current_start, current_lines
        if (
            current_start is not None
            and len(current_lines) >= 2
            and is_markdown_table_separator(current_lines[1])
        ):
            tables.append(
                MarkdownTableBlock(
                    start_line=current_start,
                    end_line=end_line,
                    lines=tuple(current_lines),
                )
            )
        current_start = None
        current_lines = []

    for line_index, line in enumerate(lines):
        fence_match = FENCE_RE.match(line)
        if fence_match:
            close_current(line_index)
            marker = fence_match.group(1)
            if fence_character is None:
                fence_character = marker[0]
                fence_length = len(marker)
            elif marker[0] == fence_character and len(marker) >= fence_length:
                fence_character = None
                fence_length = 0
            continue

        if fence_character is None and TABLE_ROW_RE.fullmatch(line):
            if current_start is None:
                current_start = line_index
            current_lines.append(line)
            continue

        close_current(line_index)

    close_current(len(lines))
    return tables


def contains_translatable_natural_language(text, source_language):
    """Return whether text contains likely natural language for the source."""
    candidate = INLINE_CODE_RE.sub(" ", text or "")
    candidate = HTML_TAG_RE.sub(" ", candidate)
    candidate = DOC_VARIABLE_RE.sub(" ", candidate)
    candidate = MARKDOWN_LINK_TARGET_RE.sub("]", candidate)
    candidate = URL_RE.sub(" ", candidate)
    normalized_language = (source_language or "").strip().lower()

    if normalized_language == "english":
        return any(
            len(word) >= 2 and any(character.islower() for character in word)
            for word in ENGLISH_WORD_RE.findall(candidate)
        )
    if normalized_language == "chinese":
        return bool(CJK_RE.search(candidate))
    if normalized_language == "japanese":
        return bool(JAPANESE_RE.search(candidate))
    return any(character.isalpha() for character in candidate)


def contains_likely_prose_cell(cell, source_language):
    """Return whether a table cell looks like natural-language prose."""
    candidate = INLINE_CODE_RE.sub(" ", cell)
    candidate = HTML_TAG_RE.sub(" ", candidate)
    candidate = DOC_VARIABLE_RE.sub(" ", candidate)
    candidate = MARKDOWN_LINK_TARGET_RE.sub("]", candidate)
    candidate = URL_RE.sub(" ", candidate)
    normalized_language = (source_language or "").strip().lower()
    if normalized_language == "english":
        words = [
            word
            for word in ENGLISH_WORD_RE.findall(candidate)
            if len(word) >= 2 and any(character.islower() for character in word)
        ]
        return len(words) >= 2
    if normalized_language == "chinese":
        return bool(CJK_RE.search(candidate))
    if normalized_language == "japanese":
        return bool(JAPANESE_RE.search(candidate))
    return sum(character.isalpha() for character in candidate) >= 4


def table_shapes_match(source_table, target_table):
    return (
        len(source_table.lines) == len(target_table.lines)
        and source_table.column_counts == target_table.column_counts
    )


def restore_table_indentation(source_table, candidate_table):
    """Apply each source row's leading indentation to the candidate row."""
    restored_lines = []
    for source_line, candidate_line in zip(source_table.lines, candidate_table.lines):
        source_indent = source_line[: len(source_line) - len(source_line.lstrip(" \t"))]
        restored_lines.append(source_indent + candidate_line.lstrip(" \t"))
    return MarkdownTableBlock(
        start_line=candidate_table.start_line,
        end_line=candidate_table.end_line,
        lines=tuple(restored_lines),
    )


def find_suspected_table_omissions(
    source_content,
    target_content,
    source_language="English",
):
    """Find unchanged tables or unchanged headers with translated body rows."""
    source_tables = extract_markdown_tables(source_content)
    target_tables = extract_markdown_tables(target_content)
    if len(source_tables) != len(target_tables):
        return []

    omissions = []
    for table_index, (source_table, target_table) in enumerate(
        zip(source_tables, target_tables),
        1,
    ):
        if not table_shapes_match(source_table, target_table):
            continue
        if (
            source_table.text == target_table.text
            and contains_translatable_natural_language(
                source_table.text,
                source_language,
            )
        ):
            omissions.append(
                SuspectedTableOmission(
                    table_index=table_index,
                    reason="entire table is unchanged from the source",
                    source=source_table,
                    target=target_table,
                )
            )
            continue
        if (
            source_table.header == target_table.header
            and contains_translatable_natural_language(
                source_table.header,
                source_language,
            )
        ):
            omissions.append(
                SuspectedTableOmission(
                    table_index=table_index,
                    reason="table header is unchanged from the source",
                    source=source_table,
                    target=target_table,
                )
            )
            continue

        unchanged_prose_rows = [
            row_index
            for row_index, (source_row, target_row) in enumerate(
                zip(source_table.lines[2:], target_table.lines[2:]),
                1,
            )
            if any(
                source_cell == target_cell
                and contains_likely_prose_cell(source_cell, source_language)
                for source_cell, target_cell in zip(
                    split_markdown_table_cells(source_row),
                    split_markdown_table_cells(target_row),
                )
            )
        ]
        if unchanged_prose_rows:
            omissions.append(
                SuspectedTableOmission(
                    table_index=table_index,
                    reason=(
                        f"{len(unchanged_prose_rows)} natural-language body row(s) "
                        "are unchanged from the source"
                    ),
                    source=source_table,
                    target=target_table,
                )
            )

    return omissions


def validate_repaired_table(source_table, candidate_content):
    """Validate that a repair contains exactly one shape-preserving table."""
    candidate = (candidate_content or "").strip("\r\n")
    candidate_tables = extract_markdown_tables(candidate)
    if len(candidate_tables) != 1 or candidate_tables[0].text != candidate:
        return None, "repair response is not exactly one Markdown table"

    candidate_table = candidate_tables[0]
    if not table_shapes_match(source_table, candidate_table):
        return None, "repair changed the table row count or column count"
    candidate_table = restore_table_indentation(source_table, candidate_table)
    if candidate_table.separator != source_table.separator:
        return None, "repair changed the Markdown table separator row"
    protected_patterns = {
        "inline code": INLINE_CODE_RE,
        "document placeholders": DOC_VARIABLE_RE,
        "link targets": MARKDOWN_LINK_TARGET_CAPTURE_RE,
        "URLs": URL_RE,
        "numbers": NUMBER_RE,
        "uppercase identifiers": UPPER_IDENTIFIER_RE,
    }
    for label, pattern in protected_patterns.items():
        if pattern.findall(source_table.text) != pattern.findall(candidate_table.text):
            return None, f"repair changed protected {label}"
    return candidate_table.text, ""

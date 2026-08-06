"""
Index File Processor Module
Handles special processing logic for _index.md files.

_index.md files are section landing pages that contain YAML frontmatter and
LearningPath components with markdown links.  They have no level-1 heading
and are short, making heading-based section matching unreliable.

The processing model mirrors the TOC snapshot-sync approach: compare source
base vs source head, reuse existing target translations for unchanged lines,
and batch-translate only new or changed lines via AI.
"""

import json
import os
import re
import threading

from file_io import atomic_write_text
from log_sanitizer import sanitize_exception_message, safe_target_path

print_lock = threading.Lock()


def thread_safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)


FRONTMATTER_FENCE_RE = re.compile(r"^---\s*$")
LEARNING_PATH_CONTAINER_RE = re.compile(
    r"^<LearningPathContainer\b"
)
LEARNING_PATH_OPEN_RE = re.compile(
    r'^<LearningPath\b[^>]*label="([^"]*)"'
)
LEARNING_PATH_CLOSE_RE = re.compile(r"^</LearningPath>")
LEARNING_CONTAINER_CLOSE_RE = re.compile(r"^</LearningPathContainer>")
MARKDOWN_LINK_RE = re.compile(r"^\[([^\]]+)\]\(([^)]+)\)\s*$")
DOCS_ABSOLUTE_LINK_RE = re.compile(
    r"(https?://docs\.pingcap\.com)(?:/(zh|ja))?(?=/|$)"
)

TAG_ATTR_RE = re.compile(r'(title|subTitle|label)="([^"]*)"')


def parse_index_line(line):
    """Classify an _index.md line for translation purposes.

    Returns a dict with at least a ``type`` key:
      - ``frontmatter``: inside YAML frontmatter
      - ``tag``: an HTML-like component tag that may contain translatable attributes
      - ``link``: a markdown link line ``[text](url)``
      - ``blank``: empty or whitespace-only
      - ``other``: anything else (copied verbatim)
    """
    stripped = line.strip()

    if not stripped:
        return {"type": "blank", "text": line}

    if FRONTMATTER_FENCE_RE.match(stripped):
        return {"type": "frontmatter_fence", "text": line}

    container_match = LEARNING_PATH_CONTAINER_RE.match(stripped)
    if container_match:
        attrs = TAG_ATTR_RE.findall(stripped)
        return {
            "type": "tag",
            "tag_kind": "container_open",
            "attrs": {name: value for name, value in attrs},
            "text": line,
        }

    path_match = LEARNING_PATH_OPEN_RE.match(stripped)
    if path_match:
        attrs = TAG_ATTR_RE.findall(stripped)
        return {
            "type": "tag",
            "tag_kind": "path_open",
            "attrs": {name: value for name, value in attrs},
            "text": line,
        }

    if LEARNING_PATH_CLOSE_RE.match(stripped):
        return {"type": "tag", "tag_kind": "path_close", "text": line}

    if LEARNING_CONTAINER_CLOSE_RE.match(stripped):
        return {"type": "tag", "tag_kind": "container_close", "text": line}

    link_match = MARKDOWN_LINK_RE.match(stripped)
    if link_match:
        return {
            "type": "link",
            "display_text": link_match.group(1),
            "url": link_match.group(2),
            "text": line,
        }

    return {"type": "other", "text": line}


def is_frontmatter_line(line_index, lines):
    """Return True when line_index falls inside the YAML frontmatter block."""
    fence_count = 0
    for i in range(len(lines)):
        if FRONTMATTER_FENCE_RE.match(lines[i].strip()):
            fence_count += 1
            if fence_count == 2:
                return i >= line_index
    return False


def _extract_frontmatter(lines):
    """Return (frontmatter_end_index, frontmatter_lines).

    frontmatter_end_index is the 0-based index of the closing ``---`` line
    (inclusive).  If no valid frontmatter is found, returns (-1, []).
    """
    if not lines or not FRONTMATTER_FENCE_RE.match(lines[0].strip()):
        return -1, []
    for i in range(1, len(lines)):
        if FRONTMATTER_FENCE_RE.match(lines[i].strip()):
            return i, lines[: i + 1]
    return -1, []


def _target_docs_locale_prefix(target_language):
    normalized = (target_language or "").strip().lower()
    if normalized == "chinese":
        return "/zh"
    if normalized == "japanese":
        return "/ja"
    return ""


def localize_docs_absolute_links(content, target_language):
    """Rewrite docs.pingcap.com absolute links to the target-language prefix."""
    locale_prefix = _target_docs_locale_prefix(target_language)
    if not locale_prefix or not content:
        return content

    return DOCS_ABSOLUTE_LINK_RE.sub(
        lambda match: f"{match.group(1)}{locale_prefix}",
        content,
    )


def _needs_translation(entry):
    """Return True when a parsed index line contains human-readable text."""
    if entry["type"] == "link":
        return True
    if entry["type"] == "tag" and entry.get("attrs"):
        for attr_name in ("title", "subTitle", "label"):
            value = entry["attrs"].get(attr_name, "")
            if value and re.search(r"[A-Za-z\u4e00-\u9fff]", value):
                return True
    return False


def _line_translation_key(entry):
    """Build a stable identity key for matching source lines across versions.

    Links are keyed by (URL, display_text) to avoid collisions when multiple
    links share the same URL (e.g. a typo where both CSV and Parquet entries
    point to the same path).  Tags are keyed by their attribute set.
    """
    if entry["type"] == "link":
        return ("link", entry["url"], entry.get("display_text", ""))
    if entry["type"] == "tag" and entry.get("tag_kind") == "container_open":
        return ("container_open",)
    if entry["type"] == "tag" and entry.get("tag_kind") == "path_open":
        return ("path_open", entry["attrs"].get("label", ""))
    return None


def build_index_translation_memory(source_base_lines, target_lines):
    """Build a mapping from source-base translation keys to target lines.

    Uses positional alignment between source-base and target: the Nth
    translatable line in the source base maps to the Nth translatable line in
    the target.  This handles translated tag attributes (e.g. label="学习"
    in the target corresponding to label="Learn" in the source base).
    """
    base_entries = [parse_index_line(l) for l in source_base_lines]
    target_entries_lines = [
        (parse_index_line(l), l) for l in target_lines
    ]

    base_key_to_entry = {}
    for i, entry in enumerate(base_entries):
        key = _line_translation_key(entry)
        if key:
            base_key_to_entry.setdefault(key, (i, entry, source_base_lines[i]))

    base_translatable = [
        (i, entry, source_base_lines[i])
        for i, entry in enumerate(base_entries)
        if _needs_translation(entry)
    ]
    target_translatable = [
        (entry, line)
        for entry, line in target_entries_lines
        if _needs_translation(entry)
    ]

    base_key_to_target_line = {}
    for (_, base_entry, _), (_, target_line) in zip(
        base_translatable, target_translatable
    ):
        key = _line_translation_key(base_entry)
        if key:
            base_key_to_target_line[key] = target_line

    return base_key_to_entry, base_key_to_target_line


def plan_synced_index_lines(
    source_base_content, source_head_content, target_content
):
    """Plan a full _index.md rewrite mirroring source HEAD structure.

    Returns (planned_lines, lines_to_translate) where:
      - planned_lines: list of strings (or None for slots awaiting translation)
      - lines_to_translate: list of (slot_index, source_line) tuples
    """
    source_base_lines = source_base_content.split("\n")
    source_head_lines = source_head_content.split("\n")
    target_lines = target_content.split("\n")

    base_key_to_entry, target_key_to_line = build_index_translation_memory(
        source_base_lines, target_lines
    )

    head_fm_end, head_fm_lines = _extract_frontmatter(source_head_lines)
    base_fm_end, base_fm_lines = _extract_frontmatter(source_base_lines)
    target_fm_end, target_fm_lines = _extract_frontmatter(target_lines)

    planned_lines = []
    lines_to_translate = []

    fm_lengths_match = len(base_fm_lines) == len(head_fm_lines)

    for line_idx, source_line in enumerate(source_head_lines):
        if head_fm_end >= 0 and line_idx <= head_fm_end:
            if fm_lengths_match:
                base_fm_line = base_fm_lines[line_idx] if line_idx < len(base_fm_lines) else None
                if base_fm_line == source_line and target_fm_end >= 0 and line_idx < len(target_fm_lines):
                    planned_lines.append(target_fm_lines[line_idx])
                else:
                    planned_lines.append(source_line)
            else:
                planned_lines.append(source_line)
            continue

        entry = parse_index_line(source_line)

        if not _needs_translation(entry):
            planned_lines.append(source_line)
            continue

        key = _line_translation_key(entry)
        if key is None:
            planned_lines.append(None)
            lines_to_translate.append((len(planned_lines) - 1, source_line))
            continue

        base_record = base_key_to_entry.get(key)
        target_line = target_key_to_line.get(key)

        if target_line and base_record:
            _, base_entry, base_raw = base_record
            if entry["type"] == "link":
                base_display = parse_index_line(base_raw).get("display_text", "")
                if base_display == entry["display_text"]:
                    target_entry = parse_index_line(target_line)
                    if target_entry["type"] == "link":
                        recomposed = (
                            f"[{target_entry['display_text']}]({target_entry['url']})"
                        )
                        planned_lines.append(recomposed)
                        continue
            elif entry["type"] == "tag":
                base_attrs = base_entry.get("attrs", {})
                head_attrs = entry.get("attrs", {})
                if base_attrs == head_attrs:
                    planned_lines.append(target_line)
                    continue

        planned_lines.append(None)
        lines_to_translate.append((len(planned_lines) - 1, source_line))

    return planned_lines, lines_to_translate


def _build_glossary_prompt(glossary_matcher, repo_config, *content_parts):
    if not glossary_matcher:
        return "", ""
    from glossary import filter_terms_for_content, format_terms_for_prompt

    matched_terms = filter_terms_for_content(
        glossary_matcher,
        *content_parts,
        source_language=repo_config["source_language"],
    )
    if not matched_terms:
        return "", ""

    glossary_text = format_terms_for_prompt(
        matched_terms,
        source_language=repo_config["source_language"],
        target_language=repo_config["target_language"],
    )
    thread_safe_print(
        f"   📚 Matched {len(matched_terms)} glossary terms for _index.md translation"
    )
    section = (
        f"\nGlossary for terms in {repo_config['source_language']} and "
        f"{repo_config['target_language']}:\n{glossary_text}\n"
    )
    instruction = (
        "\n6. When translating terms listed in the glossary above, "
        "use the provided translations for consistency."
    )
    return section, instruction


def translate_index_lines(
    lines_to_translate, ai_client, repo_config, glossary_matcher=None
):
    """Translate new or changed _index.md lines via AI."""
    if not lines_to_translate:
        return {}

    thread_safe_print(
        f"   🤖 Translating {len(lines_to_translate)} _index.md lines..."
    )

    content_dict = {
        f"line_{i}": line for i, (_, line) in enumerate(lines_to_translate)
    }
    source_lang = repo_config["source_language"]
    target_lang = repo_config["target_language"]
    glossary_section, glossary_instruction = _build_glossary_prompt(
        glossary_matcher,
        repo_config,
        *(line for _, line in lines_to_translate),
    )

    prompt = f"""You are a professional translator. Please translate the following documentation index page lines from {source_lang} to {target_lang}.

IMPORTANT INSTRUCTIONS:
1. Preserve ALL HTML-like tags, their attribute names, and their structure exactly. Only translate the values of translatable attributes (title, subTitle, label).
2. For markdown link lines like [Display Text](URL), translate only the display text inside brackets. Keep the URL unchanged.
3. Preserve all formatting, indentation, and blank lines exactly.
4. Keep URLs, anchors, and technical identifiers unchanged.{glossary_instruction}

Input lines to translate:
{json.dumps(content_dict, indent=2, ensure_ascii=False)}
{glossary_section}

Return the translated lines in the same JSON format:
{{
  "line_0": "translated line",
  "line_1": "translated line"
}}"""

    try:
        ai_response = ai_client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        json_start = ai_response.find("{")
        json_end = ai_response.rfind("}") + 1
        if json_start == -1 or json_end <= json_start:
            thread_safe_print(
                "   ❌ No valid JSON found in _index.md translation response"
            )
            return {}

        translated = json.loads(ai_response[json_start:json_end])
        mapping = {}
        for i, (slot_index, _) in enumerate(lines_to_translate):
            key = f"line_{i}"
            if key in translated:
                mapping[slot_index] = translated[key]

        thread_safe_print(
            f"   ✅ Successfully translated {len(mapping)} _index.md lines"
        )
        return mapping
    except Exception as e:
        thread_safe_print(
            f"   ❌ _index.md translation failed: {sanitize_exception_message(e)}"
        )
        return {}


def process_index_file_by_source_snapshot(
    file_path,
    index_data,
    ai_client,
    repo_config,
    target_file_path,
    glossary_matcher=None,
):
    """Rewrite target _index.md to mirror source HEAD with target translations.

    When the target file does not exist yet, the entire source content is
    translated from scratch.

    Returns True on success, False on failure.
    """
    source_base_content = localize_docs_absolute_links(
        index_data["source_base_content"],
        repo_config.get("target_language"),
    )
    source_head_content = localize_docs_absolute_links(
        index_data["source_head_content"],
        repo_config.get("target_language"),
    )

    target_dir = os.path.dirname(target_file_path)
    if target_dir:
        os.makedirs(target_dir, exist_ok=True)

    if os.path.exists(target_file_path):
        with open(target_file_path, "r", encoding="utf-8") as f:
            target_content = f.read()
    else:
        target_content = ""
        thread_safe_print(
            f"   ℹ️  Target file does not exist; will translate from scratch: {file_path}"
        )

    if target_content.strip():
        target_content = localize_docs_absolute_links(
            target_content,
            repo_config.get("target_language"),
        )

    planned_lines, lines_to_translate = plan_synced_index_lines(
        source_base_content,
        source_head_content,
        target_content,
    )

    translations = translate_index_lines(
        lines_to_translate,
        ai_client,
        repo_config,
        glossary_matcher=glossary_matcher,
    )

    missing = []
    for slot_index, source_line in lines_to_translate:
        if slot_index in translations:
            planned_lines[slot_index] = localize_docs_absolute_links(
                translations[slot_index],
                repo_config.get("target_language"),
            )
        else:
            planned_lines[slot_index] = source_line
            missing.append(source_line.strip())

    if missing:
        thread_safe_print(
            f"   ❌ {len(missing)} _index.md line(s) were not translated"
        )
        for line in missing[:10]:
            thread_safe_print(f"      - {line}")
        if len(missing) > 10:
            thread_safe_print(f"      ... and {len(missing) - 10} more")
        return False

    atomic_write_text(
        target_file_path,
        localize_docs_absolute_links(
            "\n".join(planned_lines),
            repo_config.get("target_language"),
        )
    )

    thread_safe_print(
        f"   ✅ _index.md file synced from source snapshot: {file_path}"
    )
    return True


def process_index_file(
    file_path,
    index_data,
    source_context,
    github_client,
    ai_client,
    repo_config,
    glossary_matcher=None,
):
    """Process a single _index.md file.

    Returns True on success, False on failure.
    """
    thread_safe_print(f"\n📄 Processing _index.md file: {file_path}")

    try:
        target_local_path = repo_config["target_local_path"]
        target_file_path = safe_target_path(target_local_path, file_path)

        if index_data.get("source_base_content") is not None and index_data.get(
            "source_head_content"
        ):
            return process_index_file_by_source_snapshot(
                file_path,
                index_data,
                ai_client,
                repo_config,
                target_file_path,
                glossary_matcher=glossary_matcher,
            )

        thread_safe_print(
            f"   ❌ _index.md processing requires source base and head content"
        )
        return False
    except Exception as e:
        thread_safe_print(
            f"   ❌ Error processing _index.md file {file_path}: "
            f"{sanitize_exception_message(e)}"
        )
        return False


def process_index_files(
    index_files,
    source_context,
    github_client,
    ai_client,
    repo_config,
    glossary_matcher=None,
):
    """Process all _index.md files.

    Returns True if all files are processed successfully, else False.
    """
    if not index_files:
        return True

    thread_safe_print(f"\n📄 Processing {len(index_files)} _index.md files...")
    all_success = True

    for file_path, index_data in index_files.items():
        if index_data.get("type") != "index":
            thread_safe_print(
                f"   ⚠️  Unknown index data type: {index_data.get('type')} for {file_path}"
            )
            all_success = False
            continue

        success = process_index_file(
            file_path,
            index_data,
            source_context,
            github_client,
            ai_client,
            repo_config,
            glossary_matcher=glossary_matcher,
        )
        if not success:
            all_success = False

    if all_success:
        thread_safe_print("   ✅ All _index.md files processed")
    else:
        thread_safe_print("   ⚠️  Some _index.md files failed to process")
    return all_success

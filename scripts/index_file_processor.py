"""
Index File Processor Module
Handles special processing logic for _index.md files.

_index.md files are section landing pages that contain YAML frontmatter and
LearningPath components with markdown links.  They have no level-1 heading
and are short, making heading-based section matching unreliable.

The processing model mirrors the TOC snapshot-sync approach: compare source
base vs source head, match existing translations by component and URL, and
translate only text fields. Source HEAD owns the output structure.
"""

import html
import json
import os
import re
import threading
from collections import Counter, defaultdict
from urllib.parse import urlsplit, urlunsplit

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
    r"^<LearningPath\b"
)
LEARNING_PATH_CLOSE_RE = re.compile(r"^</LearningPath>")
LEARNING_CONTAINER_CLOSE_RE = re.compile(r"^</LearningPathContainer>")
TAG_ATTR_RE = re.compile(r'(?<![\w:-])(title|subTitle|label)="([^"]*)"')
DEFAULT_DOCS_LOCALES = {"chinese": "zh", "japanese": "ja"}
ALL_ATTR_RE = re.compile(r'([\w:-]+)="([^"]*)"')


def _code_spans(text):
    """Yield matched Markdown code spans with exact backtick-run lengths."""
    runs = list(re.finditer(r"`+", text))
    index = 0
    while index < len(runs):
        opening = runs[index]
        # An escaped opening backtick is plain text outside a code span.
        preceding = len(text[:opening.start()]) - len(text[:opening.start()].rstrip("\\"))
        if preceding % 2:
            index += 1
            continue
        closing = next((i for i in range(index + 1, len(runs))
                        if len(runs[i].group(0)) == len(opening.group(0))), None)
        if closing is None:
            index += 1
            continue
        yield opening.start(), runs[closing].end()
        index = closing + 1


def _transform_link_text(text, transform):
    """Transform prose without changing literal code-span contents."""
    parts, cursor = [], 0
    for start, end in _code_spans(text):
        parts.extend((transform(text[cursor:start]), text[start:end]))
        cursor = end
    parts.append(transform(text[cursor:]))
    return "".join(parts)


def _escape_link_prose(text):
    parts, index = [], 0
    punctuation = r"!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    while index < len(text):
        char = text[index]
        if char == "\\" and index + 1 < len(text) and text[index + 1] in punctuation:
            parts.append(text[index:index + 2])
            index += 2
            continue
        escaped = {"[": "&#91;", "]": "&#93;", "\\": "&#92;", "`": "&#96;"}.get(char)
        parts.append(escaped if escaped is not None else html.escape(char, quote=False))
        index += 1
    return "".join(parts)


def _split_index_lines(content):
    """Treat a multiline component opening tag as one translation unit."""
    lines = content.split("\n")
    fm_end, _ = _extract_frontmatter(lines)
    units, index = [], 0
    while index < len(lines):
        line = lines[index]
        if index <= fm_end or not re.match(r"^\s*<(?:LearningPathContainer|LearningPath)\b", line):
            units.append(line)
            index += 1
            continue
        quote, complete, cursor = None, False, 0
        while not complete:
            while cursor < len(line):
                char = line[cursor]
                if char in ("\"", "'"):
                    if quote == char:
                        quote = None
                    elif quote is None:
                        quote = char
                elif char == ">" and quote is None:
                    complete = True
                    break
                cursor += 1
            if complete:
                break
            index += 1
            if index == len(lines):
                raise ValueError("Unterminated LearningPath opening tag")
            line += "\n" + lines[index]
        units.append(line)
        index += 1
    return units


def _closing_delimiter(text, start, opening, closing, code_spans=False):
    """Find a balanced delimiter, respecting escapes and link-label code spans."""
    depth, index = 1, start + 1
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if code_spans and char == "`":
            run = re.match(r"`+", text[index:]).group(0)
            closing_run = next(
                (match for match in re.finditer(r"`+", text[index + len(run):])
                 if len(match.group(0)) == len(run)), None
            )
            if closing_run is not None:
                index += len(run) + closing_run.end()
                continue
            index += len(run)
            continue
        if char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return None


def _parse_markdown_link(line):
    start = len(line) - len(line.lstrip())
    if start == len(line) or line[start] != "[":
        return None
    end = _closing_delimiter(line, start, "[", "]", code_spans=True)
    if end is None or line[end + 1:end + 2] != "(":
        return None
    url_end = _closing_delimiter(line, end + 1, "(", ")")
    if url_end is None or line[url_end + 1:].strip():
        return None
    display, url = line[start + 1:end], line[end + 2:url_end]
    if not display.strip() or not url.strip():
        return None
    return {
        "display_text": display, "url": url,
        "display_start": start + 1, "display_end": end,
    }


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

    link = _parse_markdown_link(line)
    if link:
        return {"type": "link", "text": line, **link}

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


def _target_docs_locale_prefix(target_language, target_locale=None):
    # Empty optional CI variables mean unset, preserving established routes.
    if target_locale is not None and not isinstance(target_locale, str):
        raise ValueError("target_locale must be a string")
    locale = (target_locale or "").strip().strip("/")
    if not locale:
        locale = DEFAULT_DOCS_LOCALES.get((target_language or "").strip().lower(), "")
    if locale and not re.fullmatch(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*", locale):
        raise ValueError("Invalid target_locale")
    return f"/{locale}" if locale else ""


def _index_locale_options(repo_config):
    explicit = repo_config.get("target_locale")
    language = (repo_config.get("target_language") or "").strip().lower()
    if explicit is None or (isinstance(explicit, str) and not explicit.strip()):
        explicit = None if language in DEFAULT_DOCS_LOCALES else os.getenv("TARGET_LOCALE")
    return {"target_locale": explicit, "known_locales": repo_config.get("docs_locales", ())}


def localize_docs_absolute_links(
    content, target_language, target_locale=None, known_locales=()
):
    """Localize only the configured docs host and known locale prefixes."""
    prefix = _target_docs_locale_prefix(target_language, target_locale)
    if not prefix or not content:
        return content
    locales = {"zh", "ja", prefix[1:]}
    for locale in known_locales:
        normalized = _target_docs_locale_prefix("", locale)
        if normalized:
            locales.add(normalized[1:])
    locale_pattern = "|".join(re.escape(locale) for locale in sorted(locales))
    pattern = re.compile(
        rf"(https?://docs\.pingcap\.com)(?:/(?:{locale_pattern}))?(?=/|$)"
    )
    return pattern.sub(lambda match: f"{match.group(1)}{prefix}", content)


def _text_fields(entry):
    """Select text by schema, independently of its alphabet or language."""
    if entry["type"] == "link":
        return {"display_text": entry["display_text"]}
    if entry["type"] == "tag":
        return {
            key: value for key, value in entry.get("attrs", {}).items()
            if value.strip()
        }
    return {}


def _needs_translation(entry):
    return bool(_text_fields(entry))


def _line_translation_key(entry):
    if entry["type"] == "link":
        return ("link", entry["url"], entry["display_text"])
    if entry.get("tag_kind") == "container_open":
        return ("container_open",)
    if entry.get("tag_kind") == "path_open":
        return ("path_open", entry["attrs"].get("label", ""))
    return None


def _url_identity(url):
    """Ignore established site locales, preserving other paths and anchors."""
    parts = urlsplit(url)
    if parts.netloc == "docs.pingcap.com":
        path = re.sub(r"^/(?:zh|ja)(?=/|$)", "", parts.path)
        return urlunsplit(parts._replace(path=path))
    return url


def _index_blocks(lines):
    """Keep component ancestry and collect descendant URLs as matching evidence."""
    blocks = [{"entry": None, "index": None, "links": [], "children": [], "urls": set()}]
    stack = [0]
    fm_end, _ = _extract_frontmatter(lines)
    for index, line in enumerate(lines):
        if index <= fm_end:
            continue
        entry = parse_index_line(line)
        kind = entry.get("tag_kind")
        if kind in ("container_open", "path_open"):
            block = {"entry": entry, "index": index, "links": [], "children": [], "urls": set()}
            blocks[stack[-1]]["children"].append(len(blocks))
            if not line.rstrip().endswith("/>"):
                stack.append(len(blocks))
            blocks.append(block)
        elif kind in ("path_close", "container_close"):
            if len(stack) > 1:
                stack.pop()
        elif entry["type"] == "link":
            blocks[stack[-1]]["links"].append((index, entry))
            for ancestor in stack:
                blocks[ancestor]["urls"].add(_url_identity(entry["url"]))
    return blocks


def _component_identity(block):
    entry = block["entry"]
    if entry is None:
        return ("root",)
    attrs = dict(ALL_ATTR_RE.findall(entry["text"]))
    return (entry["tag_kind"], tuple(sorted(
        (key, value) for key, value in attrs.items()
        if key not in ("title", "subTitle", "label")
    )))


def _match_sibling_blocks(source_blocks, target_blocks, source_ids, target_ids):
    source_identity = {i: _component_identity(source_blocks[i]) for i in source_ids}
    target_identity = {i: _component_identity(target_blocks[i]) for i in target_ids}
    source_counts, target_counts = Counter(source_identity.values()), Counter(target_identity.values())
    matches = {}
    for si, identity in source_identity.items():
        if source_counts[identity] == target_counts[identity] == 1:
            matches[si] = next(ti for ti in target_ids if target_identity[ti] == identity)

    # Link overlap resolves duplicate identities or changed attributes, but must
    # never steal a component already identified by stable source attributes.
    scores = {}
    used_targets = set(matches.values())
    for si in source_ids:
        if si in matches:
            continue
        for ti in target_ids:
            if ti in used_targets:
                continue
            source, target = source_blocks[si], target_blocks[ti]
            if source["entry"]["tag_kind"] != target["entry"]["tag_kind"]:
                continue
            same = source_identity[si] == target_identity[ti]
            if source["entry"]["tag_kind"] == "container_open" and not same:
                continue
            if not same and (target_counts[source_identity[si]] or source_counts[target_identity[ti]]):
                continue
            overlap = len(source["urls"] & target["urls"])
            if overlap:
                scores[si, ti] = overlap
    source_best, target_best = {}, {}
    source_winners, target_winners = defaultdict(list), defaultdict(list)
    for (si, ti), score in scores.items():
        if score > source_best.get(si, 0):
            source_best[si], source_winners[si] = score, [ti]
        elif score == source_best[si]:
            source_winners[si].append(ti)
        if score > target_best.get(ti, 0):
            target_best[ti], target_winners[ti] = score, [si]
        elif score == target_best[ti]:
            target_winners[ti].append(si)
    for si, winners in source_winners.items():
        if len(winners) == 1 and target_winners[winners[0]] == [si]:
            matches[si] = winners[0]
    return matches


def _match_index_lines(source_lines, target_lines):
    """Match within verified parent components, leaving ambiguity untranslated."""
    source_blocks, target_blocks = _index_blocks(source_lines), _index_blocks(target_lines)
    pending, matches = [(0, 0)], {}
    while pending:
        si, ti = pending.pop()
        source, target = source_blocks[si], target_blocks[ti]
        if source["index"] is not None:
            matches[source["index"]] = target["index"]
        source_urls, target_urls = defaultdict(list), defaultdict(list)
        for index, entry in source["links"]:
            source_urls[_url_identity(entry["url"])].append(index)
        for index, entry in target["links"]:
            target_urls[_url_identity(entry["url"])].append(index)
        for url, indices in source_urls.items():
            targets = target_urls[url]
            if len(indices) == len(targets) == 1:
                matches[indices[0]] = targets[0]
        children = _match_sibling_blocks(
            source_blocks, target_blocks, source["children"], target["children"]
        )
        pending.extend(children.items())
    return matches


def build_index_translation_memory(source_base_lines, target_lines):
    """Expose reusable translations for unambiguous source identities."""
    entries, memory = {}, {}
    keys = [_line_translation_key(parse_index_line(line)) for line in source_base_lines]
    counts = Counter(keys)
    matches = _match_index_lines(source_base_lines, target_lines)
    for index, key in enumerate(keys):
        if key is None or counts[key] != 1:
            continue
        entries[key] = (
            index, parse_index_line(source_base_lines[index]), source_base_lines[index]
        )
        if index in matches:
            memory[key] = target_lines[matches[index]]
    return entries, memory


def _render_text_fields(source_line, fields, escape=True):
    """Insert text into the source structure, escaping delimiter characters."""
    entry = parse_index_line(source_line)
    if entry["type"] == "link":
        value = fields["display_text"]
        if escape:
            value = _transform_link_text(value, _escape_link_prose)
        start, end = entry["display_start"], entry["display_end"]
        return source_line[:start] + value + source_line[end:]

    def replace_attribute(match):
        name = match.group(1)
        if name not in fields:
            return match.group(0)
        value = html.escape(fields[name], quote=True) if escape else fields[name]
        return f'{name}="{value}"'

    return TAG_ATTR_RE.sub(replace_attribute, source_line)


def plan_synced_index_lines(source_base_content, source_head_content, target_content):
    """Build source HEAD structure, reusing only verified unchanged text."""
    # Commit cursors precede frontmatter and are maintained by the caller.
    target_content = re.sub(
        r"^<!--\s*Corresponding EN commit:\s*[0-9a-fA-F]{7,64}\s*-->\r?\n",
        "", target_content,
    )
    base, head, target = [
        _split_index_lines(content) for content in
        (source_base_content, source_head_content, target_content)
    ]
    base_to_target = _match_index_lines(base, target)
    head_to_base = _match_index_lines(head, base)
    head_fm_end, head_fm = _extract_frontmatter(head)
    _, base_fm = _extract_frontmatter(base)
    _, target_fm = _extract_frontmatter(target)
    planned, queue = [], []
    for index, line in enumerate(head):
        if index <= head_fm_end:
            if (len(base_fm) == len(head_fm) and index < len(target_fm)
                    and base_fm[index] == line):
                planned.append(target_fm[index])
            else:
                planned.append(line)
            continue
        fields = _text_fields(parse_index_line(line))
        if not fields:
            planned.append(line)
            continue
        base_index = head_to_base.get(index)
        target_index = base_to_target.get(base_index)
        if base_index is not None and target_index is not None:
            base_fields = _text_fields(parse_index_line(base[base_index]))
            target_fields = _text_fields(parse_index_line(target[target_index]))
            if fields == base_fields and fields.keys() == target_fields.keys():
                planned.append(_render_text_fields(line, target_fields, escape=False))
                continue
        planned.append(None)
        queue.append((index, line))
    return planned, queue


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
        "\n4. When translating terms listed in the glossary above, "
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
        f"line_{i}": {
            key: (_transform_link_text(value, html.unescape) if key == "display_text"
                  else html.unescape(value))
            for key, value in _text_fields(parse_index_line(line)).items()
        }
        for i, (_, line) in enumerate(lines_to_translate)
    }
    source_lang = repo_config["source_language"]
    target_lang = repo_config["target_language"]
    glossary_section, glossary_instruction = _build_glossary_prompt(
        glossary_matcher,
        repo_config,
        *(line for _, line in lines_to_translate),
    )

    prompt = f"""You are a professional translator. Please translate the following documentation index text fields from {source_lang} to {target_lang}.

IMPORTANT INSTRUCTIONS:
1. Return exactly the same JSON keys and nested field names, with string values.
2. Translate only text values. Preserve inline code spans and Markdown escapes exactly. Do not return whole component tags or whole Markdown links.
3. Keep technical identifiers unchanged. Each value must remain on a single line.{glossary_instruction}

Input lines to translate:
{json.dumps(content_dict, indent=2, ensure_ascii=False)}
{glossary_section}

Return the translated lines in the same JSON format:
{{
  "line_0": {{"display_text": "translated text"}},
  "line_1": {{"label": "translated label"}}
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
        if not isinstance(translated, dict):
            thread_safe_print("   ❌ Index translation response must be a JSON object")
            return {}
        mapping = {}
        for i, (slot_index, source_line) in enumerate(lines_to_translate):
            fields = translated.get(f"line_{i}")
            expected = _text_fields(parse_index_line(source_line))
            if (not isinstance(fields, dict) or fields.keys() != expected.keys()
                    or any(not isinstance(value, str) or not value.strip()
                           or any(char in value for char in "\r\n")
                           for value in fields.values())):
                continue
            mapping[slot_index] = _render_text_fields(source_line, fields)

        if len(mapping) == len(lines_to_translate):
            thread_safe_print(f"   ✅ Successfully translated {len(mapping)} _index.md lines")
        else:
            thread_safe_print(
                f"   ❌ Valid Index translations: {len(mapping)}/{len(lines_to_translate)}; "
                "missing or invalid fields prevent this file from being written"
            )
        return mapping
    except Exception as e:
        thread_safe_print(
            f"   ❌ _index.md translation failed: {sanitize_exception_message(e)}"
        )
        return {}


def _valid_index_structure(source_content, target_content):
    source, target = _split_index_lines(source_content), _split_index_lines(target_content)
    if len(source) != len(target):
        return False
    fm_end, _ = _extract_frontmatter(source)
    stack = []
    for index, (source_line, target_line) in enumerate(zip(source, target)):
        if index <= fm_end:
            continue
        source_entry, target_entry = parse_index_line(source_line), parse_index_line(target_line)
        fields = _text_fields(source_entry)
        if fields:
            if _text_fields(target_entry).keys() != fields.keys():
                return False
            placeholders = {key: "TEXT" for key in fields}
            if _render_text_fields(source_line, placeholders) != _render_text_fields(target_line, placeholders):
                return False
        elif source_line != target_line:
            return False
        kind = source_entry.get("tag_kind")
        if kind in ("container_open", "path_open"):
            if not source_line.rstrip().endswith("/>"):
                stack.append(kind)
        elif kind in ("container_close", "path_close"):
            expected = "container_open" if kind == "container_close" else "path_open"
            if not stack or stack.pop() != expected:
                return False
    return not stack


def process_index_file_by_source_snapshot(
    file_path,
    index_data,
    ai_client,
    repo_config,
    target_file_path,
    glossary_matcher=None,
):
    """Rewrite target _index.md to mirror source HEAD with target translations.

    When the target file does not exist yet, all component and link text
    fields are translated from scratch. Frontmatter keeps its existing policy.

    Returns True on success, False on failure.
    """
    try:
        locale_options = _index_locale_options(repo_config)
        source_base_content = localize_docs_absolute_links(
            index_data["source_base_content"], repo_config.get("target_language"),
            **locale_options,
        )
        source_head_content = localize_docs_absolute_links(
            index_data["source_head_content"], repo_config.get("target_language"),
            **locale_options,
        )
    except ValueError as error:
        thread_safe_print(
            f"   ❌ Invalid Index locale configuration for {file_path}: {error}"
        )
        return False

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
            **locale_options,
        )

    try:
        planned_lines, lines_to_translate = plan_synced_index_lines(
            source_base_content, source_head_content, target_content,
        )
    except ValueError as error:
        thread_safe_print(f"   ❌ Invalid Index component syntax: {error}")
        return False

    translations = translate_index_lines(
        lines_to_translate,
        ai_client,
        repo_config,
        glossary_matcher=glossary_matcher,
    )

    missing = []
    for slot_index, source_line in lines_to_translate:
        if slot_index in translations:
            planned_lines[slot_index] = translations[slot_index]
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

    content = "\n".join(planned_lines)
    if not _valid_index_structure(source_head_content, content):
        thread_safe_print("   ❌ Index structure validation failed; target retained")
        return False
    atomic_write_text(target_file_path, content)

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

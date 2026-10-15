"""
TOC Processor Module
Handles special processing logic for TOC.md files
"""

import os
import re
import json
from github import Github
from openai import OpenAI
from file_io import atomic_write_text
from log_sanitizer import sanitize_exception_message, safe_target_path

from workflow_logging import thread_safe_print


TOC_LIST_TEXT_LINE_RE = re.compile(r'^(\s*-\s+)(.*)$')
TOC_HEADING_LINE_RE = re.compile(r'^(#{1,10}\s+)(.+)$')


def find_matching_bracket(text, open_index):
    depth = 0
    index = open_index
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return -1


def parse_toc_markdown_link_line(line):
    prefix_match = re.match(r'^(\s*-\s*)', line)
    if not prefix_match:
        return None

    prefix = prefix_match.group(1)
    rest = line[len(prefix):]
    if not rest.startswith("["):
        return None

    close_bracket_index = find_matching_bracket(rest, 0)
    if close_bracket_index == -1:
        return None

    if close_bracket_index + 1 >= len(rest) or rest[close_bracket_index + 1] != "(":
        return None

    close_paren_index = rest.find(")", close_bracket_index + 2)
    if close_paren_index == -1:
        return None

    return {
        "type": "link",
        "prefix": prefix,
        "text": rest[1:close_bracket_index],
        "link": rest[close_bracket_index + 2:close_paren_index],
        "suffix": rest[close_paren_index + 1:],
    }


def parse_toc_line(line):
    """Parse a TOC line into the small subset we need for stable syncing."""
    link_entry = parse_toc_markdown_link_line(line)
    if link_entry:
        return link_entry

    list_match = TOC_LIST_TEXT_LINE_RE.match(line)
    if list_match:
        return {
            "type": "list_text",
            "prefix": list_match.group(1),
            "text": list_match.group(2),
        }

    heading_match = TOC_HEADING_LINE_RE.match(line)
    if heading_match:
        return {
            "type": "heading",
            "prefix": heading_match.group(1),
            "text": heading_match.group(2),
        }

    return {"type": "other", "text": line}


def compose_toc_line_with_target_text(source_line, target_line):
    """Use source structure while preserving the translated target display text."""
    source_entry = parse_toc_line(source_line)
    target_entry = parse_toc_line(target_line)

    if source_entry["type"] == "link" and target_entry["type"] == "link":
        return (
            f"{source_entry['prefix']}[{target_entry['text']}]"
            f"({source_entry['link']}){source_entry['suffix']}"
        )

    if source_entry["type"] == "list_text" and target_entry["type"] == "list_text":
        return f"{source_entry['prefix']}{target_entry['text']}"

    if source_entry["type"] == "heading" and target_entry["type"] == "heading":
        return f"{source_entry['prefix']}{target_entry['text']}"

    return target_line


def plain_translation_key(entry):
    if entry["type"] == "heading":
        return ("heading", entry["prefix"].strip(), entry["text"])
    if entry["type"] == "list_text":
        return ("list_text", entry["text"])
    return None


def build_toc_translation_memory(source_base_lines, target_lines):
    """Build link and plain-line translation memory from the current target TOC."""
    base_link_entries = {}
    target_link_lines = {}

    for line in source_base_lines:
        entry = parse_toc_line(line)
        if entry["type"] == "link":
            base_link_entries.setdefault(entry["link"], entry)

    for line in target_lines:
        entry = parse_toc_line(line)
        if entry["type"] == "link":
            target_link_lines.setdefault(entry["link"], line)

    base_plain_entries = []
    target_plain_lines = []

    for line in source_base_lines:
        entry = parse_toc_line(line)
        if plain_translation_key(entry):
            base_plain_entries.append((entry, line))

    for line in target_lines:
        entry = parse_toc_line(line)
        if plain_translation_key(entry):
            target_plain_lines.append((entry, line))

    plain_line_map = {}
    for (base_entry, _), (target_entry, target_line) in zip(base_plain_entries, target_plain_lines):
        if base_entry["type"] != target_entry["type"]:
            continue
        key = plain_translation_key(base_entry)
        if key:
            plain_line_map.setdefault(key, target_line)

    return {
        "base_link_entries": base_link_entries,
        "target_link_lines": target_link_lines,
        "plain_line_map": plain_line_map,
    }


def plan_synced_toc_lines(source_base_content, source_head_content, target_content, source_added_line_numbers=None):
    """Plan a full TOC rewrite that follows source head structure.

    Existing target translations are reused when the source display text did not
    change. New or renamed TOC rows are returned for batched AI translation.
    """
    source_base_lines = source_base_content.split("\n")
    source_head_lines = source_head_content.split("\n")
    target_lines = target_content.split("\n")
    memory = build_toc_translation_memory(source_base_lines, target_lines)
    source_added_line_numbers = set(source_added_line_numbers or [])

    planned_lines = []
    lines_to_translate = []

    for source_line_num, source_line in enumerate(source_head_lines, 1):
        entry = parse_toc_line(source_line)

        if entry["type"] == "link":
            base_entry = memory["base_link_entries"].get(entry["link"])
            target_line = memory["target_link_lines"].get(entry["link"])
            if target_line and base_entry and base_entry["text"] == entry["text"]:
                planned_lines.append(compose_toc_line_with_target_text(source_line, target_line))
            else:
                planned_lines.append(None)
                lines_to_translate.append((len(planned_lines) - 1, source_line))
            continue

        key = plain_translation_key(entry)
        if key and source_line_num in source_added_line_numbers:
            planned_lines.append(None)
            lines_to_translate.append((len(planned_lines) - 1, source_line))
        elif key and key in memory["plain_line_map"]:
            planned_lines.append(compose_toc_line_with_target_text(source_line, memory["plain_line_map"][key]))
        elif entry["type"] in ("heading", "list_text"):
            planned_lines.append(None)
            lines_to_translate.append((len(planned_lines) - 1, source_line))
        else:
            planned_lines.append(source_line)

    return planned_lines, lines_to_translate


def extract_toc_link_from_line(line):
    """Extract the link part (including parentheses) from a TOC line"""
    entry = parse_toc_line(line)
    if entry["type"] == "link":
        return f"({entry['link']})"
    return None

def is_toc_translation_needed(line):
    """Return True when a TOC line likely contains human-readable text."""
    entry = parse_toc_line(line)
    text = (entry.get("text") or "").strip()

    if entry["type"] == "other":
        return False
    if not text:
        return False
    if re.search(r'[\u4e00-\u9fff]', text):
        return True
    if re.search(r'[A-Za-z]', text):
        return True
    if " " in text:
        return True
    return False

def find_best_toc_match(target_link, target_lines, source_line_num):
    """Find the best matching line in target TOC based on link content and line proximity"""
    matches = []
    
    for i, line in enumerate(target_lines):
        line_link = extract_toc_link_from_line(line.strip())
        if line_link and line_link == target_link:
            matches.append({
                'line_num': i + 1,  # Convert to 1-based
                'line': line.strip(),
                'distance': abs((i + 1) - source_line_num)
            })
    
    if not matches:
        return None
    
    # Sort by distance to source line number, choose the closest one
    matches.sort(key=lambda x: x['distance'])
    return matches[0]


def find_best_toc_plain_match(target_line, target_lines, source_line_num):
    """Find the best matching plain TOC line in the target by translated text."""
    if not target_line:
        return None

    target_anchor_entry = parse_toc_line(target_line)
    if target_anchor_entry["type"] not in ("list_text", "heading"):
        return None

    matches = []
    for i, line in enumerate(target_lines):
        entry = parse_toc_line(line)
        if entry["type"] != target_anchor_entry["type"]:
            continue
        if line != target_line:
            continue
        matches.append({
            "line_num": i + 1,
            "line": line.strip(),
            "distance": abs((i + 1) - source_line_num),
        })

    if not matches:
        return None

    matches.sort(key=lambda x: x["distance"])
    return matches[0]


def build_toc_plain_anchor_map(source_base_lines, target_lines):
    """Map unchanged plain source TOC anchors to their translated target lines."""
    if source_base_lines is None:
        return {}

    base_plain_entries = []
    target_plain_entries = []

    for line in source_base_lines:
        entry = parse_toc_line(line)
        key = plain_translation_key(entry)
        if key:
            base_plain_entries.append((entry, key))

    for line in target_lines:
        entry = parse_toc_line(line)
        if plain_translation_key(entry):
            target_plain_entries.append((entry, line))

    plain_anchor_map = {}
    for (base_entry, key), (target_entry, target_line) in zip(base_plain_entries, target_plain_entries):
        if base_entry["type"] != target_entry["type"]:
            continue
        plain_anchor_map.setdefault(key, target_line)

    return plain_anchor_map


def find_toc_plain_anchor_match(source_line, target_lines, source_line_num, plain_anchor_map):
    entry = parse_toc_line(source_line)
    key = plain_translation_key(entry)
    if not key:
        return None

    target_line = plain_anchor_map.get(key)
    if not target_line:
        return None

    match = find_best_toc_plain_match(target_line, target_lines, source_line_num)
    if not match:
        return None

    return {
        "target_line": target_line,
        "source_line": source_line_num,
        "match": match,
    }

def group_consecutive_lines(lines):
    """Group consecutive lines together"""
    if not lines:
        return []
    
    # Sort lines by line number
    sorted_lines = sorted(lines, key=lambda x: x['line_number'])
    
    groups = []
    current_group = [sorted_lines[0]]
    
    for i in range(1, len(sorted_lines)):
        current_line = sorted_lines[i]
        prev_line = sorted_lines[i-1]
        
        # Consider lines consecutive if they are within 2 lines of each other
        if current_line['line_number'] - prev_line['line_number'] <= 2:
            current_group.append(current_line)
        else:
            groups.append(current_group)
            current_group = [current_line]
    
    groups.append(current_group)
    return groups


def find_toc_group_anchors(source_lines, target_lines, start_line_num, end_line_num, source_base_lines=None):
    """Find the nearest surrounding link anchors for a TOC diff group."""
    previous_anchor = None
    next_anchor = None
    plain_anchor_map = build_toc_plain_anchor_map(source_base_lines, target_lines)

    for line_num in range(start_line_num - 1, 0, -1):
        source_line = source_lines[line_num - 1]
        previous_link = extract_toc_link_from_line(source_line)
        if previous_link and find_best_toc_match(previous_link, target_lines, line_num):
            previous_anchor = {
                "link": previous_link,
                "source_line": line_num,
            }
            break

        previous_plain_anchor = find_toc_plain_anchor_match(
            source_line,
            target_lines,
            line_num,
            plain_anchor_map,
        )
        if previous_plain_anchor:
            previous_anchor = previous_plain_anchor
            break

    for line_num in range(end_line_num + 1, len(source_lines) + 1):
        source_line = source_lines[line_num - 1]
        next_link = extract_toc_link_from_line(source_line)
        if next_link and find_best_toc_match(next_link, target_lines, line_num):
            next_anchor = {
                "link": next_link,
                "source_line": line_num,
            }
            break

        next_plain_anchor = find_toc_plain_anchor_match(
            source_line,
            target_lines,
            line_num,
            plain_anchor_map,
        )
        if next_plain_anchor:
            next_anchor = next_plain_anchor
            break

    return previous_anchor, next_anchor


def describe_toc_anchor(anchor):
    if not anchor:
        return ""
    if anchor.get("link"):
        return f"link: {anchor['link']}"
    return f"plain line: {anchor['target_line'].strip()}"


def add_toc_anchor_fields(operation, position, anchor):
    if not anchor:
        return

    operation[f"anchor_{position}_source_line"] = anchor["source_line"]
    if anchor.get("link"):
        operation[f"anchor_{position}_link"] = anchor["link"]
    elif anchor.get("target_line") is not None:
        operation[f"anchor_{position}_target_line"] = anchor["target_line"]


def build_toc_line_signature(line):
    """Build a small structural signature for matching source TOC lines to target lines."""
    entry = parse_toc_line(line)
    if entry["type"] == "link":
        return {
            "type": "link",
            "prefix": entry["prefix"],
            "link": entry["link"],
        }
    if entry["type"] in ("list_text", "heading"):
        return {
            "type": entry["type"],
            "prefix": entry["prefix"],
        }
    return {"type": entry["type"]}


def toc_line_matches_signature(line, signature):
    """Return True when a target TOC line is structurally compatible with a source line."""
    entry = parse_toc_line(line)
    if entry["type"] != signature["type"]:
        return False
    if entry["type"] == "link":
        return (
            entry["prefix"] == signature["prefix"]
            and entry["link"] == signature["link"]
        )
    if entry["type"] in ("list_text", "heading"):
        return entry["prefix"] == signature["prefix"]
    return True


def find_verified_toc_group_ranges(expected_lines, target_lines, previous_match=None, next_match=None):
    """Find structurally compatible target ranges for a TOC group between resolved anchors."""
    group_len = len(expected_lines)
    if group_len == 0:
        return []

    start_index = previous_match["line_num"] if previous_match else 0
    end_index = next_match["line_num"] - 1 if next_match else len(target_lines)
    if end_index - start_index < group_len:
        return []

    signatures = [build_toc_line_signature(line) for line in expected_lines]
    candidates = []
    for candidate_start in range(start_index, end_index - group_len + 1):
        candidate_lines = target_lines[candidate_start:candidate_start + group_len]
        if all(
            toc_line_matches_signature(candidate_lines[i], signatures[i])
            for i in range(group_len)
        ):
            candidates.append((candidate_start, candidate_start + group_len))

    return candidates


def log_ambiguous_toc_group(group_label, candidates):
    """Log ambiguous TOC group matches in a compact way."""
    rendered = ", ".join(
        f"{start + 1}-{end}" if end - start > 1 else f"{start + 1}"
        for start, end in candidates[:5]
    )
    if len(candidates) > 5:
        rendered += f", ... (+{len(candidates) - 5} more)"
    thread_safe_print(
        f"      ⚠️  Skipping {group_label}: found {len(candidates)} matching target ranges ({rendered})"
    )

def process_toc_operations(file_path, operations, source_lines, target_lines, target_local_path, source_base_lines=None):
    """Process TOC.md file operations with special logic"""
    thread_safe_print(f"\n📋 Processing TOC.md with special logic...")
    
    results = {
        'added': [],
        'modified': [],
        'deleted': []
    }
    
    # Process deleted linked lines first
    for deleted_line in operations['deleted_lines']:
        deleted_content = deleted_line['content']
        deleted_link = extract_toc_link_from_line(deleted_content)

        if deleted_link:
            thread_safe_print(f"   🗑️  Processing deleted TOC line with link: {deleted_link}")
            
            # Find matching line in target
            match = find_best_toc_match(deleted_link, target_lines, deleted_line['line_number'])
            if match:
                thread_safe_print(f"      ✅ Found target line {match['line_num']}: {match['line']}")
                results['deleted'].append({
                    'source_line': deleted_line['line_number'],
                    'target_line': match['line_num'],
                    'content': deleted_content
                })
            else:
                thread_safe_print(f"      ❌ No matching line found for {deleted_link}")

    plain_deleted_lines = [
        line
        for line in operations["deleted_lines"]
        if parse_toc_line(line["content"])["type"] in ("list_text", "heading")
        and not extract_toc_link_from_line(line["content"])
    ]
    deleted_groups = group_consecutive_lines(plain_deleted_lines)
    for group in deleted_groups:
        if source_base_lines is None:
            thread_safe_print(
                "      ⚠️  Skipping deleted TOC plain-text group: base TOC content is unavailable"
            )
            continue

        first_deleted_line = group[0]
        last_deleted_line = group[-1]
        thread_safe_print(
            f"   🗑️  Processing deleted TOC group starting at line {first_deleted_line['line_number']}"
        )

        previous_anchor, next_anchor = find_toc_group_anchors(
            source_base_lines,
            target_lines,
            first_deleted_line["line_number"],
            last_deleted_line["line_number"],
            source_base_lines=source_base_lines,
        )

        if previous_anchor:
            thread_safe_print(f"      📍 Using previous anchor {describe_toc_anchor(previous_anchor)}")
        if next_anchor:
            thread_safe_print(f"      📍 Using next anchor {describe_toc_anchor(next_anchor)}")
        if not previous_anchor and not next_anchor:
            thread_safe_print("      ❌ No target anchor found for deleted TOC group")
            continue

        previous_match = None
        next_match = None
        if previous_anchor and previous_anchor.get("link"):
            previous_match = find_best_toc_match(
                previous_anchor["link"],
                target_lines,
                previous_anchor["source_line"],
            )
        elif previous_anchor:
            previous_match = previous_anchor["match"]
        if next_anchor and next_anchor.get("link"):
            next_match = find_best_toc_match(
                next_anchor["link"],
                target_lines,
                next_anchor["source_line"],
            )
        elif next_anchor:
            next_match = next_anchor["match"]

        expected_lines = [line["content"] for line in group]
        candidate_ranges = find_verified_toc_group_ranges(
            expected_lines,
            target_lines,
            previous_match=previous_match,
            next_match=next_match,
        )
        if not candidate_ranges:
            thread_safe_print(
                "      ⚠️  Skipping deleted TOC group: no verified target range found"
            )
            continue
        if len(candidate_ranges) > 1:
            log_ambiguous_toc_group("deleted TOC group", candidate_ranges)
            continue

        start_index, end_index = candidate_ranges[0]
        for offset, source_line in enumerate(group):
            target_line = start_index + offset + 1
            results["deleted"].append({
                "source_line": source_line["line_number"],
                "target_line": target_line,
                "content": source_line["content"],
            })
            thread_safe_print(f"      ✅ Queued delete for target line {target_line}")
    
    # Process added lines
    added_groups = group_consecutive_lines(operations['added_lines'])
    for group in added_groups:
        if group:  # Skip empty groups
            first_added_line = group[0]
            last_added_line = group[-1]
            thread_safe_print(f"   ➕ Processing added TOC group starting at line {first_added_line['line_number']}")

            previous_anchor, next_anchor = find_toc_group_anchors(
                source_lines,
                target_lines,
                first_added_line['line_number'],
                last_added_line['line_number'],
                source_base_lines=source_base_lines,
            )

            if previous_anchor:
                thread_safe_print(f"      📍 Using previous anchor {describe_toc_anchor(previous_anchor)}")
            if next_anchor:
                thread_safe_print(f"      📍 Using next anchor {describe_toc_anchor(next_anchor)}")
            if not previous_anchor and not next_anchor:
                thread_safe_print("      ❌ No target anchor found for added TOC group")
                continue

            group_id = f"added:{first_added_line['line_number']}"

            for index, added_line in enumerate(group):
                added_content = added_line['content']
                operation = {
                    'group_id': group_id,
                    'group_offset': index,
                    'source_line': added_line['line_number'],
                    'content': added_content,
                    'needs_translation': is_toc_translation_needed(added_content),
                }
                add_toc_anchor_fields(operation, "previous", previous_anchor)
                add_toc_anchor_fields(operation, "next", next_anchor)

                results['added'].append(operation)
                if operation['needs_translation']:
                    thread_safe_print(f"         📝 Added for translation: {added_content.strip()}")
                else:
                    thread_safe_print(f"         ⏭️  Added without translation: {added_content.strip()}")
    
    # Process modified lines  
    modified_groups = group_consecutive_lines(operations['modified_lines'])
    for group in modified_groups:
        if group:  # Skip empty groups
            first_modified_line = group[0]
            last_modified_line = group[-1]
            thread_safe_print(f"   ✏️  Processing modified TOC group starting at line {first_modified_line['line_number']}")

            previous_anchor, next_anchor = find_toc_group_anchors(
                source_lines,
                target_lines,
                first_modified_line['line_number'],
                last_modified_line['line_number'],
                source_base_lines=source_base_lines,
            )

            if previous_anchor:
                thread_safe_print(f"      📍 Using previous anchor {describe_toc_anchor(previous_anchor)}")
            if next_anchor:
                thread_safe_print(f"      📍 Using next anchor {describe_toc_anchor(next_anchor)}")
            if not previous_anchor and not next_anchor:
                thread_safe_print("      ❌ No target anchor found for modified TOC group")
                continue

            group_id = f"modified:{first_modified_line['line_number']}"

            for index, modified_line in enumerate(group):
                modified_content = modified_line['content']
                operation = {
                    'group_id': group_id,
                    'group_offset': index,
                    'source_line': modified_line['line_number'],
                    'content': modified_content,
                    'original_content': modified_line.get('original_content'),
                    'needs_translation': is_toc_translation_needed(modified_content),
                }
                add_toc_anchor_fields(operation, "previous", previous_anchor)
                add_toc_anchor_fields(operation, "next", next_anchor)

                results['modified'].append(operation)
                if operation['needs_translation']:
                    thread_safe_print(f"         📝 Modified for translation: {modified_content.strip()}")
                else:
                    thread_safe_print(f"         ⏭️  Modified without translation: {modified_content.strip()}")
    
    return results

def find_toc_modification_line(mod_op, target_lines):
    """Find the actual line number to modify in target TOC based on context"""
    # This function helps find the exact line to modify in target TOC
    # based on the modification operation context
    
    target_line_context = mod_op.get('target_line_context', 0)
    
    # Look for the line after the context line that should be modified
    # This is a simplified approach - in practice, you might need more sophisticated logic
    
    if target_line_context > 0 and target_line_context < len(target_lines):
        # Check if the next line is the one to modify
        return target_line_context + 1
    
    return target_line_context


def group_toc_ops_by_group_id(ops, reverse=False):
    """Group TOC operations with explicit group ordering by source line."""
    grouped = {}
    for op in ops:
        group_id = op.get("group_id") or f"legacy:{op.get('source_line', 0)}"
        grouped.setdefault(group_id, []).append(op)

    def group_sort_line(group):
        line_numbers = [item.get("source_line", 0) for item in group]
        return max(line_numbers) if reverse else min(line_numbers)

    sorted_groups = sorted(
        grouped.values(),
        key=group_sort_line,
        reverse=reverse,
    )
    return [
        sorted(group, key=lambda item: item.get("group_offset", 0))
        for group in sorted_groups
    ]


def resolve_toc_group_anchor_matches(group_ops, target_lines):
    """Resolve anchor links against the current target lines."""
    first_op = group_ops[0]
    previous_match = None
    next_match = None

    previous_link = first_op.get("anchor_previous_link")
    if previous_link:
        previous_match = find_best_toc_match(
            previous_link,
            target_lines,
            first_op.get("anchor_previous_source_line", 0),
        )
    elif first_op.get("anchor_previous_target_line") is not None:
        previous_match = find_best_toc_plain_match(
            first_op["anchor_previous_target_line"],
            target_lines,
            first_op.get("anchor_previous_source_line", 0),
        )

    next_link = first_op.get("anchor_next_link")
    if next_link:
        next_match = find_best_toc_match(
            next_link,
            target_lines,
            first_op.get("anchor_next_source_line", 0),
        )
    elif first_op.get("anchor_next_target_line") is not None:
        next_match = find_best_toc_plain_match(
            first_op["anchor_next_target_line"],
            target_lines,
            first_op.get("anchor_next_source_line", 0),
        )

    return previous_match, next_match

def build_toc_glossary_prompt(glossary_matcher, repo_config, *content_parts):
    """Build the glossary prompt section for TOC translation."""
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
    thread_safe_print(f"   📚 Matched {len(matched_terms)} glossary terms for TOC translation")
    glossary_prompt_section = f"\nGlossary for terms in {repo_config['source_language']} and {repo_config['target_language']}:\n{glossary_text}\n"
    glossary_instruction = "\n5. When translating terms listed in the glossary above, use the provided translations for consistency."
    return glossary_prompt_section, glossary_instruction


def log_toc_modified_group_verification_context(group):
    """Log verification details when a modified TOC group cannot be matched cleanly."""
    fallback_lines = [
        str(op.get("source_line", 0))
        for op in group
        if not op.get("original_content")
    ]
    if fallback_lines:
        thread_safe_print(
            "      ℹ️  original_content is unavailable for source line(s) "
            f"{', '.join(fallback_lines)}; verification fell back to post-change TOC content"
        )

    if any(
        parse_toc_line(op.get("original_content") or op["content"])["type"] in ("list_text", "heading")
        for op in group
    ):
        thread_safe_print(
            "      ℹ️  Plain-text TOC groups are matched by anchor window and structure only; "
            "ambiguous target ranges are skipped to avoid overwriting target-only content"
        )


def translate_toc_lines(toc_operations, ai_client, repo_config, glossary_matcher=None):
    """Translate multiple TOC lines at once"""
    lines_to_translate = []
    
    # Collect all lines that need translation
    for op in toc_operations:
        if op.get('needs_translation', False):
            lines_to_translate.append({
                'operation_type': 'added' if (
                    'target_insertion_after' in op
                    or op.get("group_id", "").startswith("added:")
                ) else 'modified',
                'content': op['content'],
                'source_line': op['source_line']
            })
    
    if not lines_to_translate:
        thread_safe_print(f"   ⏭️  No TOC lines need translation")
        return {}
    
    thread_safe_print(f"   🤖 Translating {len(lines_to_translate)} TOC lines...")
    
    # Prepare content for AI translation
    content_dict = {}
    for i, line_info in enumerate(lines_to_translate):
        content_dict[f"line_{i}"] = line_info['content']
    
    source_lang = repo_config['source_language']
    target_lang = repo_config['target_language']
    glossary_prompt_section, glossary_instruction = build_toc_glossary_prompt(
        glossary_matcher,
        repo_config,
        *(line_info["content"] for line_info in lines_to_translate),
    )
    
    prompt = f"""You are a professional translator. Please translate the following TOC (Table of Contents) lines from {source_lang} to {target_lang}.

IMPORTANT INSTRUCTIONS:
1. Preserve ALL formatting, indentation, spaces, dashes, markdown links, parentheses, and special characters exactly as they appear.
2. Translate only the human-readable TOC text. This includes link text in [brackets] and plain TOC group titles such as list items or headings.
3. Keep URLs, anchors, placeholders such as {{{{ .dedicated }}}}, product names, and technical terms unchanged when they should not be localized.
4. Maintain the exact same indentation and spacing structure.{glossary_instruction}

Input lines to translate:
{json.dumps(content_dict, indent=2, ensure_ascii=False)}
{glossary_prompt_section}

Please return the translated lines in the same JSON format, preserving all formatting and translating only the human-readable TOC text.

Return format:
{{
  "line_0": "translated line with preserved formatting",
  "line_1": "translated line with preserved formatting"
}}"""

    #print(prompt) #DEBUG
    # Add token estimation
    try:
        from main import print_token_estimation
        print_token_estimation(prompt, "TOC translation")
    except ImportError:
        # Fallback if import fails - use tiktoken
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            tokens = enc.encode(prompt)
            actual_tokens = len(tokens)
            char_count = len(prompt)
            thread_safe_print(
                f"   💰 TOC translation | Input: {char_count:,} characters | "
                f"Actual tokens: {actual_tokens:,} (tiktoken cl100k_base)"
            )
        except Exception:
            # Final fallback to character approximation
            estimated_tokens = len(prompt) // 4
            char_count = len(prompt)
            thread_safe_print(
                f"   💰 TOC translation | Input: {char_count:,} characters | "
                f"Estimated tokens: ~{estimated_tokens:,} (fallback: 4 chars/token approximation)"
            )
    
    try:
        ai_response = ai_client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        #print(ai_response) #DEBUG
        thread_safe_print(f"   📝 AI translation response received")
        
        # Parse AI response
        try:
            json_start = ai_response.find('{')
            json_end = ai_response.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = ai_response[json_start:json_end]
                translated_lines = json.loads(json_str)
                
                # Map back to original operations
                translation_mapping = {}
                for i, line_info in enumerate(lines_to_translate):
                    key = f"line_{i}"
                    if key in translated_lines:
                        translation_mapping[line_info['source_line']] = translated_lines[key]
                
                thread_safe_print(f"   ✅ Successfully translated {len(translation_mapping)} TOC lines")
                return translation_mapping
                
        except json.JSONDecodeError as e:
            thread_safe_print(
                f"   ❌ Failed to parse AI translation response: {sanitize_exception_message(e)}"
            )
            return {}
            
    except Exception as e:
        thread_safe_print(f"   ❌ AI translation failed: {sanitize_exception_message(e)}")
        return {}


def translate_toc_full_lines(lines_to_translate, ai_client, repo_config, glossary_matcher=None):
    """Translate full TOC lines while preserving list structure and links."""
    if not lines_to_translate:
        return {}

    thread_safe_print(f"   🤖 Translating {len(lines_to_translate)} new or renamed TOC lines...")

    content_dict = {
        f"line_{i}": line
        for i, (_, line) in enumerate(lines_to_translate)
    }
    source_lang = repo_config["source_language"]
    target_lang = repo_config["target_language"]
    glossary_prompt_section, glossary_instruction = build_toc_glossary_prompt(
        glossary_matcher,
        repo_config,
        *(line for _, line in lines_to_translate),
    )

    prompt = f"""You are a professional translator. Please translate the following Table of Contents lines from {source_lang} to {target_lang}.

IMPORTANT INSTRUCTIONS:
1. Preserve indentation, list markers, markdown links, image links, URLs, anchors, and trailing markers exactly.
2. Translate only human-readable TOC text.
3. Keep template placeholders such as {{{{ .dedicated }}}} unchanged.
4. Keep product names and technical terms unchanged when they should not be localized.
5. Return valid JSON only.{glossary_instruction}

Input lines to translate:
{json.dumps(content_dict, indent=2, ensure_ascii=False)}
{glossary_prompt_section}

Return format:
{{
  "line_0": "translated line with preserved formatting",
  "line_1": "translated line with preserved formatting"
}}"""

    try:
        ai_response = ai_client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        json_start = ai_response.find("{")
        json_end = ai_response.rfind("}") + 1
        if json_start == -1 or json_end <= json_start:
            return {}

        translated_lines = json.loads(ai_response[json_start:json_end])
        translation_mapping = {}
        for i, (line_index, _) in enumerate(lines_to_translate):
            key = f"line_{i}"
            if key in translated_lines:
                translation_mapping[line_index] = translated_lines[key]

        thread_safe_print(f"   ✅ Successfully translated {len(translation_mapping)} TOC lines")
        return translation_mapping
    except Exception as e:
        thread_safe_print(f"   ❌ TOC full-line translation failed: {sanitize_exception_message(e)}")
        return {}


def process_toc_file_by_source_snapshot(file_path, toc_data, ai_client, repo_config, target_file_path, glossary_matcher=None):
    """Rewrite target TOC to mirror source head structure with target translations."""
    with open(target_file_path, "r", encoding="utf-8") as f:
        target_content = f.read()

    planned_lines, lines_to_translate = plan_synced_toc_lines(
        toc_data["source_base_content"],
        toc_data["source_head_content"],
        target_content,
        source_added_line_numbers=toc_data.get("source_added_line_numbers", []),
    )
    translations = translate_toc_full_lines(
        lines_to_translate,
        ai_client,
        repo_config,
        glossary_matcher=glossary_matcher,
    )

    missing_translations = []
    for line_index, source_line in lines_to_translate:
        if line_index in translations:
            planned_lines[line_index] = translations[line_index]
        else:
            planned_lines[line_index] = source_line
            missing_translations.append(source_line.strip())

    if missing_translations:
        thread_safe_print(
            f"   ❌ {len(missing_translations)} TOC line(s) were not translated"
        )
        for missing_line in missing_translations[:10]:
            thread_safe_print(f"      - {missing_line}")
        if len(missing_translations) > 10:
            thread_safe_print(f"      ... and {len(missing_translations) - 10} more")
        return False

    atomic_write_text(target_file_path, "\n".join(planned_lines))

    thread_safe_print(f"   ✅ TOC file synced from source snapshot: {file_path}")
    return True


def process_toc_file(file_path, toc_data, source_context_or_pr_url, github_client, ai_client, repo_config, glossary_matcher=None):
    """Process a single TOC.md file with special logic.

    Returns:
      bool: True if the TOC file is processed successfully, else False.
    """
    thread_safe_print(f"\n📋 Processing TOC file: {file_path}")
    
    try:
        target_local_path = repo_config['target_local_path']
        target_file_path = safe_target_path(target_local_path, file_path)

        if toc_data.get("source_base_content") and toc_data.get("source_head_content"):
            return process_toc_file_by_source_snapshot(
                file_path,
                toc_data,
                ai_client,
                repo_config,
                target_file_path,
                glossary_matcher=glossary_matcher,
            )
        
        # Read current target file
        with open(target_file_path, 'r', encoding='utf-8') as f:
            target_content = f.read()
        
        target_lines = target_content.split('\n')
        operations = toc_data['operations']
        
        # Separate operations by type
        deleted_ops = [op for op in operations if 'target_line' in op]
        added_ops = [op for op in operations if op.get("group_id", "").startswith("added:") or 'target_insertion_after' in op]
        modified_ops = [op for op in operations if op.get("group_id", "").startswith("modified:") or 'target_line_context' in op]
        
        thread_safe_print(f"   📊 TOC operations: {len(deleted_ops)} deleted, {len(added_ops)} added, {len(modified_ops)} modified")
        
        # Process deletions first (work backwards to maintain line numbers)
        if deleted_ops:
            thread_safe_print(f"   🗑️  Processing {len(deleted_ops)} deletions...")
            deleted_ops.sort(key=lambda x: x['target_line'], reverse=True)
            
            for del_op in deleted_ops:
                target_line_num = del_op['target_line'] - 1  # Convert to 0-based
                if 0 <= target_line_num < len(target_lines):
                    thread_safe_print(f"      ❌ Deleting line {del_op['target_line']}: {target_lines[target_line_num].strip()}")
                    del target_lines[target_line_num]
        
        # Process modifications
        if modified_ops:
            thread_safe_print(f"   ✏️  Processing {len(modified_ops)} modifications...")
            
            # Get translations for operations that need them
            translations = translate_toc_lines(
                modified_ops,
                ai_client,
                repo_config,
                glossary_matcher=glossary_matcher,
            )

            for group in group_toc_ops_by_group_id(modified_ops):
                previous_match, next_match = resolve_toc_group_anchor_matches(group, target_lines)
                expected_lines = [
                    op.get("original_content") or op["content"]
                    for op in group
                ]
                candidate_ranges = find_verified_toc_group_ranges(
                    expected_lines,
                    target_lines,
                    previous_match=previous_match,
                    next_match=next_match,
                )

                if group[0].get("group_id"):
                    if not candidate_ranges:
                        log_toc_modified_group_verification_context(group)
                        thread_safe_print(
                            "      ⚠️  Skipping modified TOC group: no verified target range found"
                        )
                        continue
                    if len(candidate_ranges) > 1:
                        log_toc_modified_group_verification_context(group)
                        log_ambiguous_toc_group("modified TOC group", candidate_ranges)
                        continue
                    start_index, _ = candidate_ranges[0]
                else:
                    if previous_match:
                        start_index = previous_match["line_num"]
                    elif next_match:
                        start_index = max(0, next_match["line_num"] - 1 - len(group))
                    else:
                        start_index = find_toc_modification_line(group[0], target_lines) - 1

                for offset, mod_op in enumerate(group):
                    target_line_num = start_index + offset

                    if 0 <= target_line_num < len(target_lines):
                        if mod_op.get('needs_translation', False) and mod_op['source_line'] in translations:
                            new_content = translations[mod_op['source_line']]
                            thread_safe_print(f"      ✏️  Modifying line {target_line_num + 1} with translation")
                        else:
                            new_content = mod_op['content']
                            thread_safe_print(f"      ✏️  Modifying line {target_line_num + 1} without translation")

                        target_lines[target_line_num] = new_content
        
        # Process additions last
        if added_ops:
            thread_safe_print(f"   ➕ Processing {len(added_ops)} additions...")
            
            # Get translations for operations that need them
            translations = translate_toc_lines(
                added_ops,
                ai_client,
                repo_config,
                glossary_matcher=glossary_matcher,
            )

            for group in group_toc_ops_by_group_id(added_ops, reverse=True):
                previous_match, next_match = resolve_toc_group_anchor_matches(group, target_lines)

                if previous_match:
                    insertion_index = previous_match["line_num"]
                elif next_match:
                    insertion_index = max(0, next_match["line_num"] - 1)
                else:
                    insertion_index = group[0].get("target_insertion_after", len(target_lines))

                prev_label = previous_match["line_num"] if previous_match else "start"
                next_label = next_match["line_num"] if next_match else "end"
                thread_safe_print(
                    f"      📍 Resolved insertion window prev={prev_label} next={next_label}; "
                    f"starting at line {insertion_index + 1}"
                )

                for add_op in group:
                    if add_op.get('needs_translation', False) and add_op['source_line'] in translations:
                        new_content = translations[add_op['source_line']]
                        thread_safe_print(f"      ➕ Inserting at line {insertion_index + 1} with translation")
                    else:
                        new_content = add_op['content']
                        thread_safe_print(f"      ➕ Inserting at line {insertion_index + 1} without translation")
                    
                    # Insert the new line while preserving the group's source order.
                    if insertion_index < len(target_lines):
                        target_lines.insert(insertion_index, new_content)
                    else:
                        target_lines.append(new_content)
                    insertion_index += 1
        
        # Write updated content back to file
        updated_content = '\n'.join(target_lines)
        atomic_write_text(target_file_path, updated_content)
        
        thread_safe_print(f"   ✅ TOC file updated: {file_path}")
        return True
        
    except Exception as e:
        thread_safe_print(
            f"   ❌ Error processing TOC file {file_path}: {sanitize_exception_message(e)}"
        )
        return False

def process_toc_files(toc_files, source_context_or_pr_url, github_client, ai_client, repo_config, glossary_matcher=None):
    """Process all TOC files.

    Returns:
      bool: True if all TOC files are processed successfully, else False.
    """
    if not toc_files:
        return True
    
    thread_safe_print(f"\n📋 Processing {len(toc_files)} TOC files...")
    all_success = True
    
    for file_path, toc_data in toc_files.items():
        if toc_data['type'] == 'toc':
            success = process_toc_file(
                file_path,
                toc_data,
                source_context_or_pr_url,
                github_client,
                ai_client,
                repo_config,
                glossary_matcher=glossary_matcher,
            )
            if not success:
                all_success = False
        else:
            thread_safe_print(f"   ⚠️  Unknown TOC data type: {toc_data['type']} for {file_path}")
            all_success = False
    
    if all_success:
        thread_safe_print(f"   ✅ All TOC files processed")
    else:
        thread_safe_print(f"   ⚠️  Some TOC files failed to process")
    return all_success

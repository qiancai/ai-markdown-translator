"""
File Adder Module
Handles processing and translation of newly added files
"""

import os
import re
import json
import threading
from github import Github
from openai import OpenAI
from file_io import atomic_write_text
from log_sanitizer import sanitize_exception_message, safe_target_path
from markdown_table_utils import (
    extract_markdown_tables,
    find_suspected_table_omissions,
    table_shapes_match,
    validate_repaired_table,
)
from workflow_outcome import FileOutcomes
from product_specific_handler import get_product_name, rewrite_tidb_version_anchors_in_text
from svg_preprocessor import strip_svgs, restore_svgs

# Thread-safe printing
print_lock = threading.Lock()


class TranslatedBatchText(str):
    """Translated batch text with non-fatal completeness warnings."""

    def __new__(cls, value, partial_reasons=None):
        obj = super().__new__(cls, value or "")
        obj.partial_reasons = list(partial_reasons or [])
        return obj

def thread_safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)


def _find_markdown_section_starts(lines):
    """Return ATX heading positions, ignoring heading-like lines in code fences."""
    section_starts = []
    fence_character = None
    fence_length = 0

    for index, line in enumerate(lines):
        stripped = line.lstrip()
        fence_match = re.match(r'^(`{3,}|~{3,})', stripped)
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

        indentation = len(line) - len(stripped)
        if indentation <= 3 and re.match(r'^#{1,6}\s+\S', stripped):
            section_starts.append(index)

    return section_starts


def _find_safe_blank_line_boundaries(lines, start, end):
    """Find paragraph boundaries outside fenced code within lines[start:end]."""
    boundaries = []
    fence_character = None
    fence_length = 0

    for index in range(start, end):
        stripped = lines[index].lstrip()
        fence_match = re.match(r'^(`{3,}|~{3,})', stripped)
        if fence_match:
            marker = fence_match.group(1)
            if fence_character is None:
                fence_character = marker[0]
                fence_length = len(marker)
            elif marker[0] == fence_character and len(marker) >= fence_length:
                fence_character = None
                fence_length = 0
            continue

        boundary = index + 1
        if fence_character is None and not stripped and start < boundary < end:
            boundaries.append(boundary)

    return boundaries


def _split_oversized_range_at_blank_lines(lines, start, end, max_lines):
    """Split an oversized range at the nearest safe paragraph boundaries."""
    if end - start <= max_lines:
        return [(start, end)]

    boundaries = _find_safe_blank_line_boundaries(lines, start, end)
    if not boundaries:
        return [(start, end)]

    ranges = []
    current_start = start
    while end - current_start > max_lines:
        target = current_start + max_lines
        candidates = [boundary for boundary in boundaries if boundary > current_start]
        if not candidates:
            break
        split_at = min(candidates, key=lambda boundary: abs(boundary - target))
        ranges.append((current_start, split_at))
        current_start = split_at

    if current_start < end:
        ranges.append((current_start, end))
    return ranges


def create_section_batches(file_content, max_lines_per_batch=200):
    """Create batches of file content for translation, respecting section boundaries.

    Prefer real Markdown section headings. If one section is itself oversized,
    split it only at a blank line outside fenced code, never at an arbitrary line.
    """
    lines = file_content.split('\n')
    total_lines = len(lines)
    heading_indices = _find_markdown_section_starts(lines)

    raw_batches = []
    if heading_indices:
        # The normal path closes batches only at real section headings.
        batch_start = 0
        for idx, heading_pos in enumerate(heading_indices):
            if heading_pos <= batch_start:
                continue
            look_ahead_end = (
                heading_indices[idx + 1]
                if idx + 1 < len(heading_indices)
                else total_lines
            )
            if look_ahead_end - batch_start > max_lines_per_batch:
                raw_batches.append((batch_start, heading_pos))
                batch_start = heading_pos

        if batch_start < total_lines:
            raw_batches.append((batch_start, total_lines))
    elif file_content.strip():
        raw_batches.append((0, total_lines))

    safe_batches = []
    for start, end in raw_batches:
        safe_batches.extend(
            _split_oversized_range_at_blank_lines(
                lines,
                start,
                end,
                max_lines_per_batch,
            )
        )

    return [
        '\n'.join(lines[start:end])
        for start, end in safe_batches
        if '\n'.join(lines[start:end]).strip()
    ]


def ensure_blank_lines_before_headings(content):
    """Ensure every markdown heading (lines starting with #) is preceded by a blank line.

    AI translations sometimes omit the blank line that Markdown best-practice
    requires before a heading, especially at batch boundaries.  This function
    adds one where missing while leaving already-correct spacing untouched.
    """
    if not content:
        return content
    content_lines = content.split('\n')
    heading_indices = set(_find_markdown_section_starts(content_lines))
    result_lines = []
    for index, line in enumerate(content_lines):
        if index in heading_indices and result_lines:
            prev = result_lines[-1]
            if prev.strip():
                result_lines.append('')
        result_lines.append(line)
    return '\n'.join(result_lines)


def _join_translated_batches(source_batches, translated_batches):
    """Join translations without losing blank paragraph boundaries from source."""
    if not translated_batches:
        return ""

    combined = translated_batches[0]
    for index, translated_batch in enumerate(translated_batches[1:], start=1):
        source_has_blank_boundary = source_batches[index - 1].endswith('\n')
        trailing_newlines = len(combined) - len(combined.rstrip('\n'))
        leading_newlines = len(translated_batch) - len(translated_batch.lstrip('\n'))
        required_newlines = 2 if source_has_blank_boundary else 1
        separator_length = max(
            0,
            required_newlines - trailing_newlines - leading_newlines,
        )
        combined += '\n' * separator_length + translated_batch

    return combined


def strip_ai_markdown_wrapper(text):
    """Strip markdown code-fence wrappers the AI sometimes adds around its response."""
    if not text:
        return text
    stripped = text.strip()
    if stripped.startswith('```'):
        first_nl = stripped.find('\n')
        if first_nl == -1:
            return stripped
        header = stripped[:first_nl].strip().lstrip('`').strip()
        if not header or header.lower() in ('markdown', 'md', 'text', 'txt'):
            stripped = stripped[first_nl + 1:]
            if stripped.rstrip().endswith('```'):
                stripped = stripped.rstrip()
                stripped = stripped[:-3].rstrip('\n')
            return stripped
    return text


def preprocess_added_file_batch_for_heading_anchor_stability(batch_content, source_language, target_language, source_mode=""):
    """Add prompt-only stability tweaks for commit-based English -> non-English file additions."""
    if not batch_content:
        return batch_content

    if (source_language or "").lower() != "english":
        return batch_content

    normalized_target = (target_language or "").lower()
    if normalized_target == "english":
        return batch_content

    from file_updater import (
        add_heading_anchor_if_needed,
        get_language_alias_prefix,
        preprocess_aliases_line,
        preprocess_tidb_cloud_links_in_line,
        should_apply_tidb_cloud_link_rewrite,
    )

    enable_commit_only_preprocessing = (source_mode or "").lower() == "commit"
    enable_tidb_cloud_link_rewrite = should_apply_tidb_cloud_link_rewrite(
        source_language,
        target_language,
        source_mode=source_mode,
    )

    if not enable_commit_only_preprocessing and not enable_tidb_cloud_link_rewrite:
        return batch_content

    lang_prefix = get_language_alias_prefix(target_language)

    processed_lines = []
    for line in batch_content.splitlines():
        if enable_commit_only_preprocessing:
            line = add_heading_anchor_if_needed(line)
            line = preprocess_aliases_line(line, lang_prefix, diff_added_only=False)
        if enable_tidb_cloud_link_rewrite:
            line = preprocess_tidb_cloud_links_in_line(line, diff_added_only=False)
        processed_lines.append(line)
    return "\n".join(processed_lines)


def translate_markdown_table(
    source_table,
    current_target_table,
    ai_client,
    source_language="English",
    target_language="Chinese",
    glossary_matcher=None,
):
    """Translate one suspected table while preserving its Markdown structure."""
    glossary_prompt_section = ""
    if glossary_matcher:
        from glossary import filter_terms_for_content, format_terms_for_prompt

        matched_terms = filter_terms_for_content(
            glossary_matcher,
            source_table,
            source_language=source_language,
        )
        if matched_terms:
            glossary_prompt_section = (
                "\nUse these glossary translations when applicable:\n"
                + format_terms_for_prompt(
                    matched_terms,
                    source_language=source_language,
                    target_language=target_language,
                )
                + "\n"
            )

    prompt = f"""You are correcting one Markdown table translated from {source_language} to {target_language}.

Translate every natural-language phrase that still needs translation, including text in header cells and body cells. Reuse correct text that is already translated in the current target table.

STRICT STRUCTURE RULES:
1. Return exactly one complete Markdown table and nothing else.
2. Preserve every pipe delimiter, the exact separator row, indentation, row count, and column count.
3. Preserve code spans, SQL, commands, URLs, file paths, placeholders, product names, API names, identifiers, parameter names, numbers, and symbols exactly.
4. When a column headed "Parameter", "Field", "Setting", "Option", or a similar label contains parameter names, configuration names, or UI field names, preserve those body-cell names exactly as written in the source, even if they contain spaces or lowercase words. Also preserve exact UI action names, status labels, mode names, and option/default values. Translate the column header and descriptive cells normally.
5. Do not treat the table as a code block. Translate natural-language descriptions even when they appear next to protected technical text.
6. Do not add explanations or wrap the table in a code fence.
{glossary_prompt_section}
Source table:
{source_table}

Current target table:
{current_target_table}
"""

    response = ai_client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    incomplete_reason = ""
    if getattr(response, "completion_status", "complete") == "incomplete":
        incomplete_reason = (
            "table repair response incomplete: "
            + (getattr(response, "completion_reason", "") or "unknown reason")
        )
    return strip_ai_markdown_wrapper(response), incomplete_reason


def repair_suspected_untranslated_tables(
    source_content,
    translated_content,
    ai_client,
    source_language="English",
    target_language="Chinese",
    glossary_matcher=None,
):
    """Run a focused second pass for tables apparently skipped by the model."""
    source_tables = extract_markdown_tables(source_content)
    target_tables = extract_markdown_tables(translated_content)
    if not source_tables and not target_tables:
        return translated_content, []
    if len(source_tables) != len(target_tables):
        return translated_content, [
            "Markdown table count changed during translation: "
            f"source {len(source_tables)}, target {len(target_tables)}"
        ]

    shape_mismatches = [
        index
        for index, (source_table, target_table) in enumerate(
            zip(source_tables, target_tables),
            1,
        )
        if not table_shapes_match(source_table, target_table)
    ]
    partial_reasons = [
        f"Markdown table {index} changed row count or column count"
        for index in shape_mismatches
    ]
    omissions = find_suspected_table_omissions(
        source_content,
        translated_content,
        source_language=source_language,
    )
    if not omissions:
        return translated_content, partial_reasons

    has_trailing_newline = translated_content.endswith(("\n", "\r"))
    translated_lines = translated_content.splitlines()
    replacements = []
    for omission in omissions:
        thread_safe_print(
            f"   🔁 Retrying Markdown table {omission.table_index}: "
            f"{omission.reason}"
        )
        try:
            repaired_response, incomplete_reason = translate_markdown_table(
                omission.source.text,
                omission.target.text,
                ai_client,
                source_language=source_language,
                target_language=target_language,
                glossary_matcher=glossary_matcher,
            )
        except Exception as e:
            partial_reasons.append(
                f"Markdown table {omission.table_index} repair failed: "
                f"{sanitize_exception_message(e)}"
            )
            continue

        if incomplete_reason:
            partial_reasons.append(
                f"Markdown table {omission.table_index}: {incomplete_reason}"
            )
            continue

        repaired_table, validation_error = validate_repaired_table(
            omission.source,
            repaired_response,
        )
        if validation_error:
            partial_reasons.append(
                f"Markdown table {omission.table_index} repair rejected: "
                f"{validation_error}"
            )
            continue

        replacements.append(
            (
                omission.target.start_line,
                omission.target.end_line,
                repaired_table.splitlines(),
            )
        )

    for start_line, end_line, replacement_lines in reversed(replacements):
        translated_lines[start_line:end_line] = replacement_lines

    repaired_content = "\n".join(translated_lines)
    if has_trailing_newline:
        repaired_content += "\n"
    remaining_omissions = find_suspected_table_omissions(
        source_content,
        repaired_content,
        source_language=source_language,
    )
    partial_reasons.extend(
        f"Markdown table {omission.table_index} remains suspicious after repair: "
        f"{omission.reason}"
        for omission in remaining_omissions
    )
    return repaired_content, partial_reasons


def translate_file_batch(batch_content, ai_client, source_language="English", target_language="Chinese", glossary_matcher=None, source_mode="", context_reference=None, prior_translation_reference=None):
    """Translate a single batch of file content using AI.

    ``context_reference`` is optional surrounding source text included in the
    prompt as read-only context (not translated, not returned). It lets callers
    translate a small fragment while still giving the model the full section for
    accurate terminology and tone.

    ``prior_translation_reference`` is the existing target-language translation of
    the surrounding section. When provided, the model is told to keep its wording
    for anything that did not change and only re-render what the source actually
    changed -- i.e. a minimal edit rather than a fresh translation.
    """
    if not batch_content.strip():
        return batch_content

    thread_safe_print(f"   🤖 Translating batch ({len(batch_content.split())} words)...")
    prompt_batch_content = preprocess_added_file_batch_for_heading_anchor_stability(
        batch_content,
        source_language,
        target_language,
        source_mode=source_mode,
    )

    table_source_content = prompt_batch_content
    prompt_batch_content, svg_map = strip_svgs(prompt_batch_content)
    if svg_map:
        thread_safe_print(f"   🖼️  Replaced {len(svg_map)} SVG(s) with placeholders for AI translation")

    # Build glossary section for prompt if matcher is provided
    glossary_prompt_section = ""
    glossary_instruction = ""
    if glossary_matcher:
        from glossary import filter_terms_for_content, format_terms_for_prompt
        matched_terms = filter_terms_for_content(glossary_matcher, prompt_batch_content, source_language=source_language)
        if matched_terms:
            glossary_text = format_terms_for_prompt(
                matched_terms,
                source_language=source_language,
                target_language=target_language,
            )
            glossary_prompt_section = f"\n{glossary_text}\n"
            glossary_instruction = "\n6. When translating terms listed in the glossary, use the provided translations for consistency."
            thread_safe_print(f"   📚 Matched {len(matched_terms)} glossary terms for batch translation")

    doc_variable_example = "{{{ .starter }}}"
    product_name = get_product_name()
    document_description = f"{product_name} document" if product_name else "document"

    context_block = ""
    if context_reference and context_reference.strip():
        context_block = (
            "Surrounding section, for context only. Do NOT translate it and do NOT "
            "include it in your output; use it only to keep terminology, tone, and "
            f"meaning consistent:\n{context_reference}\n\n"
        )

    prior_block = ""
    if prior_translation_reference and prior_translation_reference.strip():
        prior_block = (
            f"Existing {target_language} translation of this section. For any text "
            "whose meaning did not change, reuse its exact wording, terminology, and "
            "phrasing; only re-render the parts whose source meaning actually "
            "changed. Do NOT output this reference itself:\n"
            f"{prior_translation_reference}\n\n"
        )

    prompt = f"""You are an expert technical writer in the database domain, proficient in writing clear, concise, and easy-to-understand user documentation.

Your task is to translate the following {document_description} content from {source_language} to {target_language}.

IMPORTANT INSTRUCTIONS:
1. Preserve ALL Markdown formatting (headers, links, code blocks, tables, etc.)
   - For Markdown tables, translate natural-language text in every header cell and body cell if the translating it helps users in the target language better understand the table. Preserve the table syntax, not the source-language prose.
        - Preserve every pipe delimiter, separator row, indentation, row count, and column count. Keep code spans, identifiers, parameter names, product names, API names, numbers, and symbols unchanged.
        - In columns headed "Parameter", "Field", "Setting", "Option", or similar, preserve body-cell parameter names, configuration names, and UI field names exactly as written in the source, even when they contain spaces or lowercase words. Also preserve exact UI action names, status labels, mode names, and option/default values. Translate the column header and descriptions normally.
        - In some scenarios, if keeping certain words in {source_language} is much better and common than translating them to {target_language}, keep them in {source_language} is OK. For example, "AWS access key" is better to keep in English.
    - For internal Markdown links such as [link text](/path/to/file.md#section-name), only translate the natural-language part in the link text to the {target_language}, and keep the linked file name and section name exactly the same as written in the source.
2. Do NOT translate:
   - Code examples, SQL queries, configuration values, doc variables/placeholders such as {doc_variable_example}, and Mermaid diagram code blocks (```mermaid ... ```). Preserve doc variables exactly as they appear, including triple braces and when they appear inside HTML attributes or tab labels.
   - `<CustomContent ...>` and `</CustomContent>` tags. Preserve each tag, its attributes, and its placement exactly, while translating natural-language text inside inline `CustomContent` normally.
   - Explicit anchors such as {{#example-test}} in the section titles.
   - File paths, URLs, and command line examples
   - Variable names and system configuration parameters
   - Some text wrapped in ** (such as **Create Resource** on the **Project** page) are UI button or label names, keep them in English if the context of that paragraph indicates that it is UI text.
3. Translate only the descriptive text and explanations (for such content, you can rewrite it from {source_language} to {target_language} in a more natural and fluent way without changing its original meaning).

    - If the {target_language} is English, use title case for #-level titles and sentence case for titles at ## level or deeper. Otherwise, skip this rule.

4. Maintain the exact structure and indentation
5. Keep all special characters and formatting intact
{glossary_instruction}

Input:

Glossary for terms in {source_language} and {target_language}:
{glossary_prompt_section}

{context_block}{prior_block}Content to translate:
{prompt_batch_content}
"""

    # Add token estimation
    try:
        from main import print_token_estimation
        print_token_estimation(prompt, "File addition translation")
    except ImportError:
        # Fallback if import fails - use tiktoken
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            tokens = enc.encode(prompt)
            actual_tokens = len(tokens)
            char_count = len(prompt)
            print(f"   💰 File addition translation")
            print(f"      📝 Input: {char_count:,} characters")
            print(f"      🔢 Actual tokens: {actual_tokens:,} (using tiktoken cl100k_base)")
        except Exception:
            # Final fallback to character approximation
            estimated_tokens = len(prompt) // 4
            char_count = len(prompt)
            print(f"   💰 File addition translation")
            print(f"      📝 Input: {char_count:,} characters")
            print(f"      🔢 Estimated tokens: ~{estimated_tokens:,} (fallback: 4 chars/token approximation)")
    
    # Batch separators belong to the caller and are commonly omitted by the
    # model, then restored when translated blocks are reassembled. Exclude only
    # trailing line breaks from this truncation heuristic so a one-line
    # paragraph ending in ``\n\n`` is compared as one line, not three.
    source_line_count = len(batch_content.rstrip("\r\n").splitlines())

    ai_response = ai_client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    partial_reasons = []
    if getattr(ai_response, "completion_status", "complete") == "incomplete":
        partial_reasons.append(
            "AI response incomplete: "
            + (getattr(ai_response, "completion_reason", "") or "unknown reason")
        )
    translated_content = ai_response
    translated_content = strip_ai_markdown_wrapper(translated_content)
    translated_content = restore_svgs(translated_content, svg_map)
    translated_content = rewrite_tidb_version_anchors_in_text(
        translated_content,
        source_language,
        target_language,
        source_mode=source_mode,
    )
    translated_content, table_partial_reasons = repair_suspected_untranslated_tables(
        table_source_content,
        translated_content,
        ai_client,
        source_language=source_language,
        target_language=target_language,
        glossary_matcher=glossary_matcher,
    )
    partial_reasons.extend(table_partial_reasons)

    translated_line_count = len(
        translated_content.rstrip("\r\n").splitlines()
    )
    if translated_line_count < source_line_count * 0.6:
        thread_safe_print(
            f"   ⚠️  Translated batch has significantly fewer lines "
            f"({translated_line_count}) than source ({source_line_count}). "
            f"The AI response may have been truncated."
        )
        partial_reasons.append(
            f"Translated batch has {translated_line_count} lines versus {source_line_count} source lines"
        )

    thread_safe_print(f"   ✅ Batch translation completed")
    return TranslatedBatchText(translated_content, partial_reasons)

def process_added_files(
    added_files,
    source_context_or_pr_url,
    github_client,
    ai_client,
    repo_config,
    glossary_matcher=None,
    return_details=False,
    return_outcomes=False,
    overwrite_existing=False,
):
    """Process newly added files by translating and creating them in target repository.

    source_context_or_pr_url is currently unused here, but the shared signature keeps
    PR-based and commit-based call sites aligned.
    """
    if not added_files:
        thread_safe_print("\n📄 No new files to process")
        if return_outcomes:
            return FileOutcomes()
        return (True, {}) if return_details else True
    
    thread_safe_print(f"\n📄 Processing {len(added_files)} newly added files...")
    all_success = True
    failure_reasons = {}
    outcomes = FileOutcomes()
    
    target_local_path = repo_config['target_local_path']
    source_language = repo_config['source_language']
    target_language = repo_config['target_language']
    from file_updater import get_source_mode
    source_mode = get_source_mode(source_context_or_pr_url)
    
    for file_path, file_content in added_files.items():
        thread_safe_print(f"\n📝 Processing new file: {file_path}")
        
        # Create target file path
        try:
            target_file_path = safe_target_path(target_local_path, file_path)
        except ValueError as e:
            reason = sanitize_exception_message(e)
            thread_safe_print(f"   ❌ {reason}")
            failure_reasons[file_path] = reason
            all_success = False
            outcomes.add(file_path, "failed", reason)
            continue
        target_dir = os.path.dirname(target_file_path)
        
        # Create directory if it doesn't exist
        if not os.path.exists(target_dir):
            try:
                os.makedirs(target_dir, exist_ok=True)
                thread_safe_print(f"   📁 Created directory: {target_dir}")
            except Exception as e:
                reason = (
                    f"Error creating target directory {target_dir}: "
                    f"{sanitize_exception_message(e)}"
                )
                thread_safe_print(f"   ❌ {reason}")
                failure_reasons[file_path] = reason
                all_success = False
                outcomes.add(file_path, "failed", reason)
                continue
        
        # Check if file already exists
        target_file_exists = os.path.exists(target_file_path)
        if target_file_exists:
            if overwrite_existing:
                thread_safe_print(f"   ♻️  Target file exists; overwriting: {target_file_path}")
            else:
                reason = f"Target file already exists: {target_file_path}"
                thread_safe_print(f"   ⚠️  {reason}")
                failure_reasons[file_path] = reason
                all_success = False
                outcomes.add(file_path, "failed", reason)
                continue

        # Create section batches for translation
        batches = create_section_batches(file_content, max_lines_per_batch=200)
        thread_safe_print(f"   📦 Created {len(batches)} batches for translation")
        
        # Translate each batch
        translated_batches = []
        file_partial_reasons = []
        translation_failed = False
        for i, batch in enumerate(batches):
            thread_safe_print(f"   🔄 Processing batch {i+1}/{len(batches)}")
            try:
                translated_batch = translate_file_batch(
                    batch, 
                    ai_client, 
                    source_language, 
                    target_language,
                    glossary_matcher=glossary_matcher,
                    source_mode=source_mode,
                )
            except Exception as e:
                reason = f"Translation failed for {file_path} batch {i+1}: {sanitize_exception_message(e)}"
                thread_safe_print(f"   ❌ {reason}")
                failure_reasons[file_path] = reason
                all_success = False
                outcomes.add(file_path, "failed", reason)
                translation_failed = True
                break
            translated_batches.append(translated_batch)
            file_partial_reasons.extend(
                getattr(translated_batch, "partial_reasons", [])
            )

        if translation_failed:
            continue
        
        # Combine translated batches while preserving source paragraph boundaries.
        translated_content = _join_translated_batches(batches, translated_batches)
        translated_content = ensure_blank_lines_before_headings(translated_content)

        # Write translated content to target file
        try:
            atomic_write_text(target_file_path, translated_content)
            
            action = "Updated" if overwrite_existing and target_file_exists else "Created"
            thread_safe_print(f"   ✅ {action} translated file: {target_file_path}")
            if file_partial_reasons:
                outcomes.add(
                    file_path,
                    "partial",
                    "; ".join(dict.fromkeys(file_partial_reasons)),
                )
            else:
                outcomes.add(file_path, "success")
            
        except Exception as e:
            reason = f"Error creating file {target_file_path}: {sanitize_exception_message(e)}"
            thread_safe_print(
                f"   ❌ {reason}"
            )
            failure_reasons[file_path] = reason
            all_success = False
            outcomes.add(file_path, "failed", reason)
    
    thread_safe_print(f"\n✅ Completed processing all new files")
    if return_outcomes:
        return outcomes
    return (all_success, failure_reasons) if return_details else all_success

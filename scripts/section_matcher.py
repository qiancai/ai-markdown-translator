"""
Section Matcher Module
Handles section hierarchy matching including direct matching and AI matching
"""

import os
import re
import json
import threading
from github import Github
from openai import OpenAI
from log_sanitizer import sanitize_exception_message
from product_specific_handler import get_product_name

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

def is_markdown_heading(line):
    """True only for real markdown headings at column 0."""
    if not line or not isinstance(line, str):
        return False
    if line != line.lstrip():
        return False
    return re.match(r'^#{1,10}\s+\S', line) is not None

def extract_first_heading_from_content(content):
    """Extract first markdown heading line from section content."""
    if not content:
        return None
    for line in content.splitlines():
        if is_markdown_heading(line):
            return line.strip()
    return None


def trim_content_by_end_marker(content, end_marker):
    """Trim content before the first line containing end_marker."""
    if not end_marker or not isinstance(content, str) or not content:
        return content

    lines = content.splitlines()
    for idx, line in enumerate(lines):
        if end_marker in line:
            return "\n".join(lines[:idx]).rstrip()
    return content

_HEADING_ANCHOR_RE = re.compile(r'\s*\{#[^}]+\}\s*$')


def strip_heading_anchor(text):
    """Strip trailing {#anchor-slug} from a heading or hierarchy string.
    
    In commit-based mode, documents often contain explicit anchors like:
      #### 2.2 データファイル用のTiDB Lightning構成ファイルを作成する {#2-2-create-the-...}
    
    For hierarchy strings (containing ' > '), anchors are stripped from every
    segment, not just the leaf.
    """
    if not text:
        return text
    if ' > ' in text:
        parts = text.split(' > ')
        return ' > '.join(_HEADING_ANCHOR_RE.sub('', p) for p in parts)
    return _HEADING_ANCHOR_RE.sub('', text)


def clean_title_for_matching(title):
    """Clean title for matching by removing markdown formatting and span elements"""
    if not title:
        return ""
    
    # Remove span elements like <span class="version-mark">New in v5.0</span>
    title = re.sub(r'<span[^>]*>.*?</span>', '', title)
    
    # Remove explicit heading anchors like {#some-anchor-slug}
    title = re.sub(r'\s*\{#[^}]+\}\s*$', '', title)
    
    # Remove markdown header prefix (# ## ### etc.)
    title = re.sub(r'^#{1,6}\s*', '', title.strip())
    
    # Remove backticks
    title = title.replace('`', '')
    
    # Strip whitespace
    title = title.strip()
    
    return title

def is_bottom_section_marker(title):
    """Return True for internal bottom-section markers, not user-facing headings."""
    return bool(title and re.match(r'^bottom-(?:added|modified)-\d+$', title))

def is_system_variable_or_config(title):
    """Check if a title represents a system variable or configuration item"""
    cleaned_title = clean_title_for_matching(title)
    
    if not cleaned_title:
        return False
    if is_bottom_section_marker(cleaned_title):
        return False
    
    # Check if original title had backticks (indicating code/config item)
    original_has_backticks = '`' in title
    
    # System variables and config items are typically:
    # 1. Alphanumeric characters with underscores, hyphens, dots, or percent signs
    # 2. No spaces in the middle
    # 3. Often contain underscores, hyphens, dots, or percent signs
    # 4. May contain uppercase letters (like alert rule names)
    # 5. Single words wrapped in backticks (like `capacity`, `engine`)
    
    # Check if it contains only allowed characters (including % for metrics/alerts)
    allowed_chars = re.match(r'^[a-zA-Z0-9_\-\.%]+$', cleaned_title)
    
    # Check if it contains at least one separator (common in system vars/config/alerts)
    has_separator = ('_' in cleaned_title or '-' in cleaned_title or 
                    '.' in cleaned_title or '%' in cleaned_title)
    
    # Check if it doesn't contain spaces (spaces would indicate it's likely a regular title)
    no_spaces = ' ' not in cleaned_title
    
    # Additional patterns for alert rules and metrics
    is_alert_rule = (cleaned_title.startswith('PD_') or 
                    cleaned_title.startswith('TiDB_') or
                    cleaned_title.startswith('TiKV_') or
                    cleaned_title.endswith('_alert') or
                    '%' in cleaned_title)
    
    # NEW: Check if it's a single word in backticks (config/variable name)
    # Examples: `capacity`, `engine`, `enable`, `dirname` etc.
    is_single_backticked_word = (original_has_backticks and 
                                allowed_chars and 
                                no_spaces and 
                                len(cleaned_title.split()) == 1)
    
    return bool(allowed_chars and (has_separator or is_alert_rule or is_single_backticked_word) and no_spaces)

def find_toplevel_title_matches(source_sections, target_lines):
    """Find matches for top-level titles (# Level) by direct pattern matching"""
    matched_dict = {}
    failed_matches = []
    skipped_sections = []
    
    thread_safe_print(f"🔍 Searching for top-level title matches")
    
    for source_line_num, source_hierarchy in source_sections.items():
        # Extract the leaf title from hierarchy
        source_leaf_title = source_hierarchy.split(' > ')[-1] if ' > ' in source_hierarchy else source_hierarchy
        
        # Only process top-level titles
        if not source_leaf_title.startswith('# '):
            skipped_sections.append({
                'line_num': source_line_num,
                'hierarchy': source_hierarchy,
                'reason': 'Not a top-level title'
            })
            continue
        
        thread_safe_print(f"   📝 Looking for top-level match: {source_leaf_title}")
        
        # Find the first top-level title in target document
        target_match = None
        for line_num, line in enumerate(target_lines, 1):
            line = line.strip()
            if line.startswith('# '):
                target_match = {
                    'line_num': line_num,
                    'title': line,
                    'hierarchy_string': line[2:].strip()  # Remove '# ' prefix for hierarchy
                }
                thread_safe_print(f"      ✓ Found target top-level at line {line_num}: {line}")
                break
        
        if target_match:
            matched_dict[str(target_match['line_num'])] = target_match['hierarchy_string']
            thread_safe_print(f"      ✅ Top-level match: line {target_match['line_num']}")
        else:
            thread_safe_print(f"      ❌ No top-level title found in target")
            failed_matches.append({
                'line_num': source_line_num,
                'hierarchy': source_hierarchy,
                'reason': 'No top-level title found in target'
            })
    
    thread_safe_print(f"📊 Top-level matching result: {len(matched_dict)} matches found")
    if failed_matches:
        thread_safe_print(f"⚠️  {len(failed_matches)} top-level sections failed to match:")
        for failed in failed_matches:
            thread_safe_print(f"      ❌ Line {failed['line_num']}: {failed['hierarchy']} - {failed['reason']}")
    
    return matched_dict, failed_matches, skipped_sections


def find_direct_matches_for_special_files(source_sections, target_hierarchy, target_lines):
    """Find direct matches for system variables/config items without using AI"""
    matched_dict = {}
    failed_matches = []
    skipped_sections = []
    
    # Build target headers with hierarchy paths
    target_headers = {}
    for line_num, raw_line in enumerate(target_lines, 1):
        line = raw_line.strip()
        if is_markdown_heading(raw_line):
            match = re.match(r'^(#{1,10})\s+(.+)', line)
            if match:
                level = len(match.group(1))
                title = match.group(2).strip()
                target_headers[line_num] = {
                    'level': level,
                    'title': title,
                    'line': line
                }
    
    thread_safe_print(f"   🔍 Searching for direct matches among {len(target_headers)} target headers")
    
    for source_line_num, source_hierarchy in source_sections.items():
        # Extract the leaf title from hierarchy
        source_leaf_title = source_hierarchy.split(' > ')[-1] if ' > ' in source_hierarchy else source_hierarchy
        source_clean_title = clean_title_for_matching(source_leaf_title)
        
        thread_safe_print(f"   📝 Looking for match: {source_clean_title}")
        
        if not is_system_variable_or_config(source_leaf_title):
            thread_safe_print(f"      ⚠️  Not a system variable/config, skipping direct match")
            skipped_sections.append({
                'line_num': source_line_num,
                'hierarchy': source_hierarchy,
                'reason': 'Not a system variable or config item'
            })
            continue
        
        # Find potential matches in target
        potential_matches = []
        for target_line_num, target_header in target_headers.items():
            target_clean_title = clean_title_for_matching(target_header['title'])
            
            if source_clean_title == target_clean_title:
                # Build hierarchy path for this target header
                hierarchy_path = build_hierarchy_path(target_lines, target_line_num, target_headers)
                potential_matches.append({
                    'line_num': target_line_num,
                    'header': target_header,
                    'hierarchy_path': hierarchy_path,
                    'hierarchy_string': ' > '.join([f"{'#' * h['level']} {h['title']}" for h in hierarchy_path if h['level'] > 1 or len(hierarchy_path) == 1])
                })
                thread_safe_print(f"      ✓ Found potential match at line {target_line_num}: {target_header['title']}")
        
        if len(potential_matches) == 1:
            # Single match found
            match = potential_matches[0]
            matched_dict[str(match['line_num'])] = match['hierarchy_string']
            thread_safe_print(f"      ✅ Direct match: line {match['line_num']}")
        elif len(potential_matches) > 1:
            # Multiple matches, need to use parent hierarchy to disambiguate
            thread_safe_print(f"      🔀 Multiple matches found ({len(potential_matches)}), using parent hierarchy")
            
            # Extract parent hierarchy from source
            source_parts = source_hierarchy.split(' > ')
            if len(source_parts) > 1:
                source_parent_titles = [clean_title_for_matching(part) for part in source_parts[:-1]]
                
                best_match = None
                best_score = -1
                
                for match in potential_matches:
                    # Compare parent hierarchy
                    target_parent_titles = [clean_title_for_matching(h['title']) for h in match['hierarchy_path'][:-1]]
                    
                    # Calculate similarity score
                    score = 0
                    min_len = min(len(source_parent_titles), len(target_parent_titles))
                    
                    for i in range(min_len):
                        if i < len(source_parent_titles) and i < len(target_parent_titles):
                            if source_parent_titles[-(i+1)] == target_parent_titles[-(i+1)]:  # Compare from end
                                score += 1
                            else:
                                break
                    
                    thread_safe_print(f"        📊 Match at line {match['line_num']} score: {score}")
                    
                    if score > best_score:
                        best_score = score
                        best_match = match
                
                if best_match and best_score > 0:
                    matched_dict[str(best_match['line_num'])] = best_match['hierarchy_string']
                    thread_safe_print(f"      ✅ Best match: line {best_match['line_num']} (score: {best_score})")
                else:
                    thread_safe_print(f"      ❌ No good parent hierarchy match found")
                    failed_matches.append({
                        'line_num': source_line_num,
                        'hierarchy': source_hierarchy,
                        'reason': 'Multiple matches found but no good parent hierarchy match'
                    })
            else:
                thread_safe_print(f"      ⚠️  No parent hierarchy in source, cannot disambiguate")
                failed_matches.append({
                    'line_num': source_line_num,
                    'hierarchy': source_hierarchy,
                    'reason': 'Multiple matches found but no parent hierarchy to disambiguate'
                })
        else:
            thread_safe_print(f"      ❌ No matches found for: {source_clean_title}")
            # Try fuzzy matching for similar titles (e.g., --host vs --hosts)
            fuzzy_matched = False
            source_clean_lower = source_clean_title.lower()
            for target_header in target_headers:
                # Handle both dict and tuple formats
                if isinstance(target_header, dict):
                    target_clean = clean_title_for_matching(target_header['title'])
                elif isinstance(target_header, (list, tuple)) and len(target_header) >= 2:
                    target_clean = clean_title_for_matching(target_header[1])  # title is at index 1
                else:
                    continue  # Skip invalid entries
                target_clean_lower = target_clean.lower()
                # Check for similar titles (handle plural/singular and minor differences)
                # Case 1: One is substring of another (e.g., --host vs --hosts)
                # Case 2: Small character difference (1-2 characters)
                len_diff = abs(len(source_clean_lower) - len(target_clean_lower))
                if (len_diff <= 2 and 
                    (source_clean_lower in target_clean_lower or 
                     target_clean_lower in source_clean_lower)):
                        thread_safe_print(f"      ≈ Fuzzy match found: {source_clean_title} ≈ {target_clean}")
                        if isinstance(target_header, dict):
                            matched_dict[str(target_header['line_num'])] = target_header['hierarchy_string']
                            thread_safe_print(f"      ✅ Fuzzy match: line {target_header['line_num']}")
                        elif isinstance(target_header, (list, tuple)) and len(target_header) >= 3:
                            matched_dict[str(target_header[0])] = target_header[2]  # line_num at index 0, hierarchy at index 2
                            thread_safe_print(f"      ✅ Fuzzy match: line {target_header[0]}")
                        fuzzy_matched = True
                        break
            
            if not fuzzy_matched:
                failed_matches.append({
                    'line_num': source_line_num,
                    'hierarchy': source_hierarchy,
                    'reason': 'No matching section found in target'
                })
    
    thread_safe_print(f"   📊 Direct matching result: {len(matched_dict)} matches found")
    
    if failed_matches:
        thread_safe_print(f"   ⚠️  {len(failed_matches)} sections failed to match:")
        for failed in failed_matches:
            thread_safe_print(f"      ❌ Line {failed['line_num']}: {failed['hierarchy']} - {failed['reason']}")
    
    if skipped_sections:
        thread_safe_print(f"   ℹ️  {len(skipped_sections)} sections skipped (not system variables/config):")
        for skipped in skipped_sections:
            thread_safe_print(f"      ⏭️  Line {skipped['line_num']}: {skipped['hierarchy']} - {skipped['reason']}")
    
    return matched_dict, failed_matches, skipped_sections

def filter_non_system_sections(target_hierarchy):
    """Filter out system variable/config sections from target hierarchy for AI mapping"""
    filtered_hierarchy = {}
    system_sections_count = 0
    
    for line_num, hierarchy in target_hierarchy.items():
        # Extract the leaf title from hierarchy
        leaf_title = hierarchy.split(' > ')[-1] if ' > ' in hierarchy else hierarchy
        
        if is_system_variable_or_config(leaf_title):
            system_sections_count += 1
        else:
            filtered_hierarchy[line_num] = hierarchy
    
    thread_safe_print(f"   🔧 Filtered target hierarchy: {len(filtered_hierarchy)} non-system sections (removed {system_sections_count} system sections)")
    
    return filtered_hierarchy

def format_hierarchy_list_for_prompt(hierarchy):
    """Render a hierarchy dict/list in stable line order for AI prompts."""
    if not hierarchy:
        return ""

    if isinstance(hierarchy, dict):
        def sort_key(item):
            line_num, _ = item
            try:
                return int(line_num)
            except (TypeError, ValueError):
                return 0

        return "\n".join(
            f"{line_num}: {section}"
            for line_num, section in sorted(hierarchy.items(), key=sort_key)
        )

    return "\n".join(str(section) for section in hierarchy)


def build_changed_sections_context(source_sections, source_diff_dict=None):
    """Render changed source sections with operation and old/new hierarchy hints."""
    if not isinstance(source_sections, dict):
        return "\n".join(strip_heading_anchor(str(section)) for section in source_sections)

    rows = []
    source_diff_dict = source_diff_dict or {}

    for key, hierarchy in source_sections.items():
        source_info = source_diff_dict.get(key, {}) if isinstance(source_diff_dict, dict) else {}
        operation = source_info.get("operation", "unknown")
        old_heading = extract_first_heading_from_content(source_info.get("old_content", ""))
        new_heading = extract_first_heading_from_content(source_info.get("new_content", ""))
        original_hierarchy = source_info.get("original_hierarchy", hierarchy)

        details = [
            f"- key: {key}",
            f"  operation: {operation}",
            f"  matching source hierarchy: {strip_heading_anchor(hierarchy)}",
            f"  original source hierarchy: {strip_heading_anchor(original_hierarchy)}",
        ]
        if old_heading:
            details.append(f"  old heading: {strip_heading_anchor(old_heading)}")
        if new_heading:
            details.append(f"  new heading: {strip_heading_anchor(new_heading)}")
        rows.append("\n".join(details))

    return "\n".join(rows)


def get_corresponding_sections(
    source_sections,
    target_sections,
    ai_client,
    source_language,
    target_language,
    max_tokens=20000,
    source_base_hierarchy=None,
    source_head_hierarchy=None,
    source_diff_dict=None,
    source_mode="",
):
    """Use AI to find corresponding sections between different languages"""
    
    # Strip heading anchors from all sections for cleaner AI prompt
    if isinstance(source_sections, dict):
        source_text = "\n".join(strip_heading_anchor(v) for v in source_sections.values())
        changed_sections_text = build_changed_sections_context(source_sections, source_diff_dict)
        number_of_sections = len(source_sections)
    else:
        source_text = "\n".join(strip_heading_anchor(s) for s in source_sections)
        changed_sections_text = source_text
        number_of_sections = len(source_sections)
    target_text = "\n".join(strip_heading_anchor(s) for s in target_sections)
    source_base_hierarchy_for_prompt = {
        k: strip_heading_anchor(v) for k, v in source_base_hierarchy.items()
    } if source_base_hierarchy else source_base_hierarchy
    source_head_hierarchy_for_prompt = {
        k: strip_heading_anchor(v) for k, v in source_head_hierarchy.items()
    } if source_head_hierarchy else source_head_hierarchy
    source_base_text = format_hierarchy_list_for_prompt(source_base_hierarchy_for_prompt)
    source_head_text = format_hierarchy_list_for_prompt(source_head_hierarchy_for_prompt)
    normalized_source_mode = (source_mode or "").lower()
    product_name = get_product_name()
    product_context = f" for {product_name}" if product_name else ""

    if source_base_text or source_head_text:
        mode_guidance = (
            "This is commit-based mode. The target file is expected to correspond to the source BASE file. "
            "First align the source BASE section structure to the target-language section structure, then map the changed sections. "
            "For added sections, return the target-language reference section that should be used as the insertion anchor."
            if normalized_source_mode == "commit"
            else
            "This is PR mode. Use the full source structure only as local context to disambiguate parents and siblings. "
            "Do not assume every source section must have a target-language counterpart."
        )

        prompt = f"""I am aligning the {source_language} and {target_language} documentation{product_context}.

{mode_guidance}

Full {source_language} BASE section structure:

{source_base_text or "(not available)"}

Full {source_language} HEAD section structure:

{source_head_text or "(not available)"}

Changed {source_language} section(s) to map, in order:

{changed_sections_text}

Here is the section structure of the corresponding {target_language} file.
Please select the corresponding {number_of_sections} section(s) in {target_language} from the list below.

⚠️ Strict rules:
1. Return **only** the exact same number ({number_of_sections}) of target-language section titles, in the same order as the changed sections above.
2. Return only section titles that appear in the target-language list below, preserving the exact text and # prefix.
3. For modified or deleted sections, prefer the target section with the same heading level and equivalent parent/sibling position.
4. For added sections, return the existing target-language reference section used as the insertion anchor.
5. Do **not** output any explanations, comments, summaries, keys, numbering, or extra lines.
6. Wrap your answer in a Markdown code block enclosed in three backticks.

{target_text}"""
    else:
        prompt = f"""I am aligning the {source_language} and {target_language} documentation{product_context}.

I have modified the following {number_of_sections} section(s) in the {source_language} file:

{source_text}

Here is the section structure of the corresponding {target_language} file.
Please select the corresponding {number_of_sections} section(s) in {target_language} from the list below that I should modify.

⚠️ Strict rules:
1. Return **only** the exact same number ({number_of_sections}) of section titles that best match both the given {source_language} sections and structures.
2. Return only the matched section titles in the original format (keeping # prefix in the title), each in its own line.
3. Do **not** output any explanations, comments, summaries, or extra lines.
4. Wrap your answer in a Markdown code block enclosed in three backticks.

{target_text}"""

    thread_safe_print(
        f"\n   📤 AI mapping request ({source_language} → {target_language}): "
        f"{number_of_sections} section(s), {len(target_sections)} target candidate(s), "
        f"{len(prompt):,} prompt chars"
    )
    verbose_thread_safe_print(f"   " + "="*80)
    verbose_thread_safe_print(f"   {prompt}")
    verbose_thread_safe_print(f"   " + "="*80)

    # Import token estimation function from main
    try:
        from main import print_token_estimation
        print_token_estimation(prompt, f"Section mapping ({source_language} → {target_language})")
    except ImportError:
        # Fallback if import fails - use tiktoken
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            tokens = enc.encode(prompt)
            actual_tokens = len(tokens)
            char_count = len(prompt)
            thread_safe_print(f"   💰 Section mapping ({source_language} → {target_language})")
            thread_safe_print(f"      📝 Input: {char_count:,} characters")
            thread_safe_print(f"      🔢 Actual tokens: {actual_tokens:,} (using tiktoken cl100k_base)")
        except Exception:
            # Final fallback to character approximation
            estimated_tokens = len(prompt) // 4
            char_count = len(prompt)
            thread_safe_print(f"   💰 Section mapping ({source_language} → {target_language})")
            thread_safe_print(f"      📝 Input: {char_count:,} characters")
            thread_safe_print(f"      🔢 Estimated tokens: ~{estimated_tokens:,} (fallback: 4 chars/token approximation)")

    try:
        ai_response = ai_client.chat_completion(
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=max_tokens
        )
        
        thread_safe_print(f"\n   📥 AI mapping response received: {len(ai_response or ''):,} chars")
        verbose_thread_safe_print(f"   " + "-"*80)
        verbose_thread_safe_print(f"   {ai_response}")
        verbose_thread_safe_print(f"   " + "-"*80)
        
        return ai_response
    except Exception as e:
        print(f"   ❌ AI mapping error: {sanitize_exception_message(e)}")
        return None


def get_corresponding_sections_json(
    source_sections,
    target_hierarchy,
    ai_client,
    repo_config,
    max_tokens=20000,
    source_base_hierarchy=None,
    source_head_hierarchy=None,
    source_diff_dict=None,
    source_mode="",
):
    """Use full-context AI mapping and ask for key-based JSON output."""
    source_language = repo_config['source_language']
    target_language = repo_config['target_language']
    changed_sections_text = build_changed_sections_context(source_sections, source_diff_dict)
    source_base_hierarchy_for_prompt = {
        k: strip_heading_anchor(v) for k, v in source_base_hierarchy.items()
    } if source_base_hierarchy else source_base_hierarchy
    source_head_hierarchy_for_prompt = {
        k: strip_heading_anchor(v) for k, v in source_head_hierarchy.items()
    } if source_head_hierarchy else source_head_hierarchy
    source_base_text = format_hierarchy_list_for_prompt(source_base_hierarchy_for_prompt)
    source_head_text = format_hierarchy_list_for_prompt(source_head_hierarchy_for_prompt)
    target_hierarchy_for_prompt = {
        k: strip_heading_anchor(v) for k, v in target_hierarchy.items()
    } if target_hierarchy else target_hierarchy
    target_text = format_hierarchy_list_for_prompt(target_hierarchy_for_prompt)
    expected_keys = list(source_sections.keys())
    expected_keys_json = json.dumps(expected_keys, ensure_ascii=False)
    normalized_source_mode = (source_mode or "").lower()
    product_name = get_product_name()
    product_context = f" for {product_name}" if product_name else ""
    mode_guidance = (
        "This is commit-based mode. The target file is expected to correspond to the source BASE file. "
        "First align the source BASE section structure to the target-language section structure, then map each changed section by key. "
        "For added sections, the matching/original source hierarchy is the insertion reference section, not the new section being inserted; "
        "return the target-language counterpart of that reference section."
        if normalized_source_mode == "commit"
        else
        "This is PR mode. Use the full source structure only as local context to disambiguate parents and siblings. "
        "Do not assume every source section must have a target-language counterpart."
    )

    prompt = f"""I am aligning the {source_language} and {target_language} documentation{product_context}.

{mode_guidance}

Full {source_language} BASE section structure:

{source_base_text or "(not available)"}

Full {source_language} HEAD section structure:

{source_head_text or "(not available)"}

Changed {source_language} section(s) to map:

{changed_sections_text}

Here is the section structure of the corresponding {target_language} file:

{target_text}

Return a JSON object that maps every changed section key to the exact target-language section hierarchy.

⚠️ Strict rules:
1. Return exactly these keys and no others: {expected_keys_json}
2. Every JSON value must be one exact hierarchy string from the target-language list above.
3. For modified or deleted sections, prefer the target section with the same heading level and equivalent parent/sibling position.
4. For added sections, map the shown matching/original source hierarchy to its existing target-language counterpart and use that as the insertion anchor; do not map to the parent of the new heading.
5. Do not return null, arrays, comments, Markdown headings, explanations, or extra text.
6. Wrap your answer in a Markdown code block enclosed in three backticks.

Example shape:
```json
{{"modified_12": "## 目标章节", "added_20": "## 插入锚点"}}
```"""

    thread_safe_print(
        f"\n   📤 AI JSON mapping request ({source_language} → {target_language}): "
        f"{len(source_sections)} section(s), {len(target_hierarchy)} target candidate(s), "
        f"{len(prompt):,} prompt chars"
    )
    verbose_thread_safe_print(f"   " + "="*80)
    verbose_thread_safe_print(f"   {prompt}")
    verbose_thread_safe_print(f"   " + "="*80)

    try:
        from main import print_token_estimation
        print_token_estimation(prompt, f"Section JSON mapping ({source_language} → {target_language})")
    except ImportError:
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            actual_tokens = len(enc.encode(prompt))
            thread_safe_print(f"   💰 Section JSON mapping ({source_language} → {target_language})")
            thread_safe_print(f"      📝 Input: {len(prompt):,} characters")
            thread_safe_print(f"      🔢 Actual tokens: {actual_tokens:,} (using tiktoken cl100k_base)")
        except Exception:
            thread_safe_print(f"   💰 Section JSON mapping ({source_language} → {target_language})")
            thread_safe_print(f"      📝 Input: {len(prompt):,} characters")
            thread_safe_print(f"      🔢 Estimated tokens: ~{len(prompt) // 4:,} (fallback: 4 chars/token approximation)")

    try:
        ai_response = ai_client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=max_tokens,
        )
        thread_safe_print(f"\n   📥 AI JSON mapping response received: {len(ai_response or ''):,} chars")
        verbose_thread_safe_print(f"   " + "-"*80)
        verbose_thread_safe_print(f"   {ai_response}")
        verbose_thread_safe_print(f"   " + "-"*80)
        return ai_response
    except Exception as e:
        thread_safe_print(f"   ❌ AI JSON mapping error: {sanitize_exception_message(e)}")
        return None


def parse_ai_json_mapping_response(ai_response):
    """Parse a fenced or raw JSON object from AI mapping output."""
    if not ai_response:
        return {}

    text = ai_response.strip()
    fenced_match = re.search(r'```(?:json)?\s*(.*?)```', text, flags=re.DOTALL | re.IGNORECASE)
    if fenced_match:
        text = fenced_match.group(1).strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        thread_safe_print(f"      ❌ Could not parse AI JSON mapping response: {sanitize_exception_message(e)}")
        return {}

    if not isinstance(parsed, dict):
        thread_safe_print("      ❌ AI JSON mapping response is not an object")
        return {}

    return {str(key): value for key, value in parsed.items() if isinstance(value, str)}

def parse_ai_response(ai_response):
    """Parse AI response to extract section names"""
    fenced_sections = []
    fallback_sections = []
    in_code_block = False
    saw_code_fence = False

    for raw_line in ai_response.splitlines():
        line = raw_line.strip()
        if line.startswith('```'):
            saw_code_fence = True
            in_code_block = not in_code_block
            continue

        if not line:
            continue

        if line.startswith('- '):
            # Handle cases where AI returns a list
            line = line[2:].strip()

        if in_code_block:
            fenced_sections.append(line)
        elif not saw_code_fence:
            fallback_sections.append(line)

    return fenced_sections if saw_code_fence else fallback_sections

def find_matching_line_numbers(ai_sections, target_hierarchy_dict):
    """Find line numbers in target hierarchy dict that match AI sections"""
    matched_dict = {}
    
    for ai_section in ai_sections:
        ai_section_stripped = strip_heading_anchor(ai_section)
        # Look for exact matches first (with and without anchor)
        found = False
        for line_num, hierarchy in target_hierarchy_dict.items():
            hierarchy_stripped = strip_heading_anchor(hierarchy)
            if hierarchy == ai_section or hierarchy_stripped == ai_section_stripped:
                matched_dict[str(line_num)] = hierarchy
                found = True
                break
        
        if not found:
            # Look for partial matches (in case of slight differences)
            partial_candidates = {}
            for line_num, hierarchy in target_hierarchy_dict.items():
                ai_clean = ai_section_stripped.replace('### ', '').replace('## ', '').strip()
                hierarchy_clean = strip_heading_anchor(hierarchy).replace('### ', '').replace('## ', '').strip()
                
                if ai_clean in hierarchy_clean or hierarchy_clean in ai_clean:
                    partial_candidates[str(line_num)] = hierarchy
                    thread_safe_print(f"      ≈ Partial match candidate at line {line_num}: {hierarchy}")
            if partial_candidates:
                matched_dict.update(partial_candidates)
                found = True
        
        if not found:
            thread_safe_print(f"      ✗ No match found for: {ai_section}")
    
    return matched_dict


def find_candidate_line_numbers_for_ai_section(ai_section, target_hierarchy_dict, used_lines=None):
    """Find candidate target line numbers for a single AI-returned section."""
    used_lines = used_lines or set()
    matched_dict = {}

    ai_section_stripped = strip_heading_anchor(ai_section)

    for line_num, hierarchy in target_hierarchy_dict.items():
        line_num = str(line_num)
        if line_num in used_lines:
            continue
        hierarchy_stripped = strip_heading_anchor(hierarchy)
        if hierarchy == ai_section or hierarchy_stripped == ai_section_stripped:
            matched_dict[line_num] = hierarchy

    if matched_dict:
        return matched_dict

    ai_clean = (
        ai_section_stripped.replace('### ', '')
        .replace('## ', '')
        .replace('# ', '')
        .strip()
    )
    for line_num, hierarchy in target_hierarchy_dict.items():
        line_num = str(line_num)
        if line_num in used_lines:
            continue
        hierarchy_clean = (
            strip_heading_anchor(hierarchy)
            .replace('### ', '')
            .replace('## ', '')
            .replace('# ', '')
            .strip()
        )
        if ai_clean == hierarchy_clean or ai_clean in hierarchy_clean or hierarchy_clean in ai_clean:
            matched_dict[line_num] = hierarchy

    return matched_dict


def build_target_match_result(key, source_operation, target_line, target_hierarchy_str):
    """Create a normalized match result entry for downstream processing."""
    result = {
        "target_line": str(target_line),
        "target_hierarchy": format_target_hierarchy(target_hierarchy_str),
    }
    if source_operation == "added":
        result["insertion_type"] = "before_reference"
    return result


def resolve_batched_ai_matches(source_sections, ai_sections, target_hierarchy_dict, source_diff_dict):
    """Validate and resolve batched AI mapping results back to source keys."""
    expected = len(source_sections)
    if len(ai_sections) != expected:
        thread_safe_print(
            f"      ❌ Batched AI result count mismatch: expected {expected}, got {len(ai_sections)}"
        )
        return {}, list(source_sections.keys())

    used_lines = set()
    resolved = {}
    failed_keys = []
    last_target_line = "0"

    for (key, source_hierarchy), ai_section in zip(source_sections.items(), ai_sections):
        source_operation = source_diff_dict.get(key, {}).get("operation", "")
        allow_reused_line = source_operation == "added"
        candidates = find_candidate_line_numbers_for_ai_section(
            ai_section, target_hierarchy_dict, None if allow_reused_line else used_lines
        )
        if not candidates:
            thread_safe_print(f"      ✗ No target hierarchy match found for batched AI section: {ai_section}")
            failed_keys.append(key)
            continue

        target_line, target_hierarchy_str = choose_best_ai_match(
            candidates,
            source_hierarchy,
            previous_target_line=last_target_line,
        )
        if not target_line or (target_line in used_lines and not allow_reused_line):
            thread_safe_print(f"      ✗ Duplicate or invalid batched match for {key}: {ai_section}")
            failed_keys.append(key)
            continue

        if not allow_reused_line:
            used_lines.add(target_line)
        last_target_line = target_line
        resolved[key] = build_target_match_result(
            key, source_operation, target_line, target_hierarchy_str
        )

    return resolved, failed_keys


def resolve_keyed_ai_matches(source_sections, keyed_sections, target_hierarchy_dict, source_diff_dict):
    """Validate and resolve a key -> target hierarchy AI mapping result."""
    expected_keys = set(source_sections.keys())
    actual_keys = set(keyed_sections.keys())
    missing_keys = [key for key in source_sections if key not in actual_keys]
    extra_keys = sorted(actual_keys - expected_keys)

    if missing_keys:
        thread_safe_print(
            f"      ❌ AI JSON mapping is missing {len(missing_keys)} key(s): {', '.join(missing_keys)}"
        )
    if extra_keys:
        thread_safe_print(
            f"      ❌ AI JSON mapping returned unexpected key(s): {', '.join(extra_keys)}"
        )

    used_lines = set()
    resolved = {}
    failed_keys = list(missing_keys)
    last_target_line = "0"

    for key, source_hierarchy in source_sections.items():
        if key not in keyed_sections:
            continue

        source_operation = source_diff_dict.get(key, {}).get("operation", "")
        allow_reused_line = source_operation == "added"
        ai_section = keyed_sections[key]
        candidates = find_candidate_line_numbers_for_ai_section(
            ai_section, target_hierarchy_dict, None if allow_reused_line else used_lines
        )
        if not candidates:
            thread_safe_print(f"      ✗ No target hierarchy match found for {key}: {ai_section}")
            failed_keys.append(key)
            continue

        target_line, target_hierarchy_str = choose_best_ai_match(
            candidates,
            source_hierarchy,
            previous_target_line=last_target_line,
        )
        if not target_line or (target_line in used_lines and not allow_reused_line):
            thread_safe_print(f"      ✗ Duplicate or invalid keyed match for {key}: {ai_section}")
            failed_keys.append(key)
            continue

        if not allow_reused_line:
            used_lines.add(target_line)
        last_target_line = target_line
        resolved[key] = build_target_match_result(
            key, source_operation, target_line, target_hierarchy_str
        )

    return resolved, failed_keys


def batch_match_sections_with_ai_json(
    source_sections,
    target_hierarchy,
    ai_client,
    repo_config,
    max_tokens=20000,
    source_diff_dict=None,
    source_base_hierarchy=None,
    source_head_hierarchy=None,
    source_mode="",
):
    """Match multiple sections in one full-context AI call with key-based JSON."""
    if not source_sections:
        return {}, []

    ai_response = get_corresponding_sections_json(
        source_sections,
        target_hierarchy,
        ai_client,
        repo_config,
        max_tokens=max_tokens,
        source_base_hierarchy=source_base_hierarchy,
        source_head_hierarchy=source_head_hierarchy,
        source_diff_dict=source_diff_dict,
        source_mode=source_mode,
    )
    if not ai_response:
        return {}, list(source_sections.keys())

    keyed_sections = parse_ai_json_mapping_response(ai_response)
    if not keyed_sections:
        return {}, list(source_sections.keys())

    return resolve_keyed_ai_matches(
        source_sections,
        keyed_sections,
        target_hierarchy,
        source_diff_dict or {},
    )


def batch_match_sections_with_ai(
    source_sections,
    target_hierarchy,
    ai_client,
    repo_config,
    max_tokens=20000,
    source_diff_dict=None,
    source_base_hierarchy=None,
    source_head_hierarchy=None,
    source_mode="",
):
    """Match multiple non-direct sections in one AI call."""
    if not source_sections:
        return [], []

    ai_response = get_corresponding_sections(
        source_sections,
        list(target_hierarchy.values()),
        ai_client,
        repo_config['source_language'],
        repo_config['target_language'],
        max_tokens,
        source_base_hierarchy=source_base_hierarchy,
        source_head_hierarchy=source_head_hierarchy,
        source_diff_dict=source_diff_dict,
        source_mode=source_mode,
    )

    if not ai_response:
        return [], list(source_sections.keys())

    return parse_ai_response(ai_response), []

def get_heading_level(hierarchy):
    """Get markdown heading level from hierarchy string, if present."""
    if not hierarchy:
        return None
    leaf = hierarchy.split(' > ')[-1] if ' > ' in hierarchy else hierarchy
    leaf = leaf.strip()
    m = re.match(r'^(#{1,10})\s+\S', leaf)
    if m:
        return len(m.group(1))
    return None


def extract_step_number(title):
    """Extract a tutorial step number from English/Chinese step headings."""
    if not title:
        return None
    leaf_title = title.split(' > ')[-1] if ' > ' in title else title
    cleaned = clean_title_for_matching(leaf_title)
    match = re.match(r'^(?:Step|步骤)\s+(\d+)\b', cleaned, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def is_direct_match_candidate(hierarchy):
    """Return True if the hierarchy should bypass batched AI matching."""
    if not hierarchy:
        return False
    leaf_title = hierarchy.split(' > ')[-1] if ' > ' in hierarchy else hierarchy
    return (
        hierarchy == "frontmatter"
        or leaf_title.startswith('# ')
        or is_system_variable_or_config(leaf_title)
    )


def get_bottom_modified_matching_hierarchy(key, hierarchy, source_info):
    """Resolve a bottom-modified marker to a real heading when content provides one."""
    if not hierarchy or not hierarchy.startswith('bottom-modified-'):
        return hierarchy

    source_hierarchy = source_info.get('matching_hierarchy')
    if source_hierarchy:
        thread_safe_print(
            f"   🔚 Bottom-modified section {key}: using source dict hierarchy for matching: {source_hierarchy}"
        )
        return source_hierarchy

    source_content = source_info.get('new_content') or source_info.get('old_content') or ''
    inferred_heading = extract_first_heading_from_content(source_content)
    if inferred_heading:
        thread_safe_print(
            f"   🔚 Bottom-modified section {key}: inferred matching hierarchy from content: {inferred_heading}"
        )
        return inferred_heading

    thread_safe_print(
        f"   ⚠️  Bottom-modified section {key} has no inferred heading; keeping marker for fallback matching"
    )
    return hierarchy


def find_step_heading_fallback_match(source_hierarchy, target_hierarchy):
    """Fallback-match tutorial step headings by level and step number."""
    source_step = extract_step_number(source_hierarchy)
    source_level = get_heading_level(source_hierarchy)
    if source_step is None or source_level is None:
        return None

    candidates = []
    for line_num, candidate_hierarchy in target_hierarchy.items():
        if get_heading_level(candidate_hierarchy) != source_level:
            continue
        candidate_step = extract_step_number(candidate_hierarchy)
        if candidate_step == source_step:
            candidates.append((str(line_num), candidate_hierarchy))

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        thread_safe_print(
            f"      ⚠️ Ambiguous step-heading fallback for {source_hierarchy}: "
            f"{len(candidates)} candidates"
        )

    return None

def choose_best_ai_match(ai_matched, source_hierarchy, previous_target_line=None):
    """Pick the best line when AI returns multiple possible sections.

    When previous_target_line is set (batch/keyed resolve context), uses document-
    order heuristics: prefer the first candidate strictly after previous_target_line.
    When previous_target_line is None (single-section fallback), preserves the legacy
    heuristic of picking the largest line number to avoid top-level titles.
    """
    if not ai_matched:
        return None, None
    if len(ai_matched) == 1:
        line, hierarchy = next(iter(ai_matched.items()))
        return line, hierarchy

    source_level = get_heading_level(source_hierarchy)
    items = list(ai_matched.items())

    if source_level is not None:
        same_level = []
        for line, hierarchy in items:
            h_level = get_heading_level(hierarchy)
            if h_level == source_level:
                same_level.append((line, hierarchy))
        if same_level:
            if previous_target_line is not None:
                forward_candidates = [
                    (line, hierarchy)
                    for line, hierarchy in same_level
                    if int(line) > int(previous_target_line)
                ]
                if forward_candidates:
                    forward_candidates.sort(key=lambda x: int(x[0]))
                    return forward_candidates[0]
                # No forward candidates; pick earliest available.
                same_level.sort(key=lambda x: int(x[0]))
                return same_level[0]

            # Single-section mode (no ordering context): pick largest line number
            # to avoid accidentally choosing the top-level document title.
            same_level.sort(key=lambda x: int(x[0]))
            return same_level[-1]

    # Fallback to largest line number to avoid accidentally choosing top-level title.
    items.sort(key=lambda x: int(x[0]))
    return items[-1]


def validate_mapping_for_retry(matched_sections, source_diff_dict=None):
    """Return deterministic mapping-risk errors that merit full-context retry."""
    errors = []
    claimed_lines = {}
    source_diff_dict = source_diff_dict or {}
    matched_keys = set(matched_sections or {})

    for key, section_data in source_diff_dict.items():
        operation = section_data.get("operation", "")
        if operation in ("modified", "deleted") and key not in matched_keys:
            errors.append(f"{key} ({operation}) did not map to any target section")

    for key, section_data in matched_sections.items():
        operation = section_data.get("source_operation", "")
        if operation not in ("modified", "deleted"):
            continue

        target_line = section_data.get("target_line")
        target_hierarchy = section_data.get("target_hierarchy", "")
        source_hierarchy = section_data.get("source_matching_hierarchy") or section_data.get("source_original_hierarchy", "")

        if not target_line or target_line in ("unknown", "-1"):
            continue

        previous_key = claimed_lines.get(target_line)
        if previous_key:
            errors.append(
                f"{key} and {previous_key} both map to target line {target_line} ({target_hierarchy})"
            )
        else:
            claimed_lines[target_line] = key

        source_level = get_heading_level(source_hierarchy)
        target_level = get_heading_level(target_hierarchy)
        old_heading_level = get_heading_level(
            extract_first_heading_from_content(section_data.get("source_old_content", ""))
        )
        new_heading_level = get_heading_level(
            extract_first_heading_from_content(section_data.get("source_new_content", ""))
        )
        source_heading_level_changed = (
            old_heading_level is not None
            and new_heading_level is not None
            and old_heading_level != new_heading_level
        )
        if (
            not source_heading_level_changed
            and source_level is not None
            and target_level is not None
            and source_level != target_level
        ):
            errors.append(
                f"{key} maps source level {source_level} ({source_hierarchy}) "
                f"to target level {target_level} ({target_hierarchy})"
            )

    return errors


def validate_commit_mode_matches(matched_sections, source_diff_dict=None):
    """Return validation errors that should block commit-mode writes."""
    return validate_mapping_for_retry(matched_sections, source_diff_dict)


def build_enhanced_match_result(key, hierarchy, matching_hierarchy, result, source_info, target_lines):
    """Attach target/source content metadata to a raw target match result."""
    target_line = result.get('target_line', 'unknown')
    target_content = ""

    if hierarchy == "intro_section" or key == "intro_section":
        target_content, _ = extract_intro_section_content_from_lines(target_lines)
    elif target_line == '-1':
        target_content = ""
    elif target_line != 'unknown' and target_line != '0':
        try:
            target_line_num = int(target_line)
            target_content = extract_section_direct_content(target_line_num, target_lines)
        except (ValueError, IndexError):
            target_content = ""
    elif target_line == '0':
        target_content = extract_frontmatter_content(target_lines)

    target_end_marker = source_info.get('target_end_marker')
    if target_end_marker:
        target_content = trim_content_by_end_marker(target_content, target_end_marker)

    return {
        **result,
        'target_content': target_content,
        'source_original_hierarchy': source_info.get('original_hierarchy', ''),
        'source_matching_hierarchy': matching_hierarchy,
        'source_operation': source_info.get('operation', ''),
        'source_old_content': source_info.get('old_content', ''),
        'source_new_content': source_info.get('new_content', ''),
        'target_end_marker': target_end_marker
    }

def build_hierarchy_path(lines, line_num, all_headers):
    """Build the full hierarchy path for a header at given line (from auto-sync-pr-changes.py)"""
    if line_num not in all_headers:
        return []
    
    current_header = all_headers[line_num]
    current_level = current_header['level']
    hierarchy_path = []
    
    # Find all parent headers
    for check_line in sorted(all_headers.keys()):
        if check_line >= line_num:
            break
        
        header = all_headers[check_line]
        if header['level'] < current_level:
            # This is a potential parent
            # Remove any headers at same or deeper level
            while hierarchy_path and hierarchy_path[-1]['level'] >= header['level']:
                hierarchy_path.pop()
            hierarchy_path.append(header)
    
    # Add current header
    hierarchy_path.append(current_header)
    
    return hierarchy_path

def map_insertion_points_to_target(insertion_points, target_hierarchy, target_lines, file_path, source_context_or_pr_url, github_client, ai_client, repo_config, max_non_system_sections=120):
    """Map source insertion points to target language locations"""
    target_insertion_points = {}
    
    thread_safe_print(f"   📍 Mapping {len(insertion_points)} insertion points to target...")
    
    for group_key, point_info in insertion_points.items():
        previous_section_hierarchy = point_info['previous_section_hierarchy']
        thread_safe_print(f"      🔍 Finding target location for: {previous_section_hierarchy}")
        
        # Extract title for system variable checking
        if ' > ' in previous_section_hierarchy:
            title = previous_section_hierarchy.split(' > ')[-1]
        else:
            title = previous_section_hierarchy
        
        # Check if this is a system variable/config that can be directly matched
        cleaned_title = clean_title_for_matching(title)
        if is_system_variable_or_config(cleaned_title):
            thread_safe_print(f"         🎯 Direct matching for system var/config: {cleaned_title}")
            
            # Direct matching for system variables
            temp_source = {point_info['previous_section_line']: previous_section_hierarchy}
            matched_dict, failed_matches, skipped_sections = find_direct_matches_for_special_files(
                temp_source, target_hierarchy, target_lines
            )
            
            if matched_dict:
                # Get the first (and should be only) matched target line
                target_line = list(matched_dict.keys())[0]
                
                # Find the end of this section
                target_line_num = int(target_line)
                insertion_after_line = find_section_end_line(target_line_num, target_hierarchy, target_lines)
                
                target_insertion_points[group_key] = {
                    'insertion_after_line': insertion_after_line,
                    'target_hierarchy': target_hierarchy.get(str(target_line_num), ''),
                    'insertion_type': point_info['insertion_type'],
                    'new_sections': point_info['new_sections']
                }
                thread_safe_print(f"         ✅ Direct match found, insertion after line {insertion_after_line}")
                continue
        
        # If not a system variable or direct matching failed, use AI
        thread_safe_print(f"         🤖 Using AI mapping for: {cleaned_title}")
        
        # Filter target hierarchy for AI (remove system sections)
        filtered_target_hierarchy = filter_non_system_sections(target_hierarchy)
        
        # Check if filtered hierarchy is too large for AI
        # Use provided max_non_system_sections parameter
        if len(filtered_target_hierarchy) > max_non_system_sections:
            thread_safe_print(f"         ❌ Target hierarchy too large for AI: {len(filtered_target_hierarchy)} > {max_non_system_sections}")
            continue
        
        # Prepare source for AI mapping
        temp_source = {str(point_info['previous_section_line']): previous_section_hierarchy}
        
        ai_response = get_corresponding_sections(
            list(temp_source.values()), 
            list(filtered_target_hierarchy.values()), 
            ai_client, 
            repo_config['source_language'], 
            repo_config['target_language'],
            max_tokens=20000
        )
        
        if ai_response:
            # Parse AI response and find matching line numbers
            ai_sections = parse_ai_response(ai_response)
            ai_matched = find_matching_line_numbers(ai_sections, target_hierarchy)
            
            if ai_matched and len(ai_matched) > 0:
                # Get the first match (we only have one source section)
                target_line = list(ai_matched.keys())[0]
                target_line_num = int(target_line)
                
                # Find the end of this section
                insertion_after_line = find_section_end_line(target_line_num, target_hierarchy, target_lines)
                
                target_insertion_points[group_key] = {
                    'insertion_after_line': insertion_after_line,
                    'target_hierarchy': target_hierarchy.get(target_line, ''),
                    'insertion_type': point_info['insertion_type'],
                    'new_sections': point_info['new_sections']
                }
                thread_safe_print(f"         ✅ AI match found, insertion after line {insertion_after_line}")
            else:
                thread_safe_print(f"         ❌ No AI matching sections found for: {previous_section_hierarchy}")
        else:
            thread_safe_print(f"         ❌ No AI response received for: {previous_section_hierarchy}")
    
    return target_insertion_points

def extract_hierarchies_from_diff_dict(source_diff_dict):
    """Extract original_hierarchy from source_diff_dict for section matching"""
    extracted_hierarchies = {}
    
    for key, diff_info in source_diff_dict.items():
        operation = diff_info.get('operation', '')
        original_hierarchy = diff_info.get('original_hierarchy', '')
        
        # Process all sections: modified, deleted, and added
        if operation in ['modified', 'deleted', 'added'] and original_hierarchy:
            # Use the key as the identifier for the hierarchy
            extracted_hierarchies[key] = original_hierarchy
    
    thread_safe_print(f"📄 Extracted {len(extracted_hierarchies)} hierarchies from source diff dict")
    for key, hierarchy in extracted_hierarchies.items():
        verbose_thread_safe_print(f"   {key}: {hierarchy}")
    
    return extracted_hierarchies

def match_source_diff_to_target(
    source_diff_dict,
    target_hierarchy,
    target_lines,
    ai_client,
    repo_config,
    max_non_system_sections=120,
    max_tokens=20000,
    source_mode=None,
    source_base_hierarchy=None,
    source_head_hierarchy=None,
    source_hierarchy_provider=None,
):
    """
    Match source_diff_dict original_hierarchy to target file sections
    Uses direct matching for system variables/config and AI matching for others
    
    Returns:
        dict: Matched sections with enhanced information including:
            - target_line: Line number in target file
            - target_hierarchy: Target section hierarchy 
            - insertion_type: For added sections only
            - source_original_hierarchy: Original hierarchy from source
            - source_operation: Operation type (modified/added/deleted)
            - source_old_content: Old content from source diff
            - source_new_content: New content from source diff
    """
    thread_safe_print(f"🔗 Starting source diff to target matching...")
    normalized_source_mode = (source_mode or repo_config.get("source_mode") or "pr").lower()
    full_context_max_sections = int(os.getenv("FULL_CONTEXT_MAPPING_MAX_SECTIONS", "100"))
    
    # Extract hierarchies from source diff dict
    source_hierarchies = extract_hierarchies_from_diff_dict(source_diff_dict)
    
    if not source_hierarchies:
        thread_safe_print(f"⚠️  No hierarchies to match")
        return {}
    
    # Process sections in original order to maintain consistency
    # Initialize final matching results with ordered dict to preserve order
    from collections import OrderedDict
    all_matched_sections = OrderedDict()
    source_entries = OrderedDict(
        (key, dict(source_diff_dict.get(key, {})))
        for key in source_hierarchies
    )
    
    # Categorize sections for processing strategy but maintain order.
    intro_section_sections = OrderedDict()
    direct_match_sections = OrderedDict()
    batched_ai_sections = OrderedDict()
    bottom_sections = OrderedDict()  # Only bottom-added sections should be here
    matching_hierarchies = OrderedDict()
    
    for key, hierarchy in source_hierarchies.items():
        source_info = source_entries.get(key, {})
        if hierarchy.startswith('bottom-modified-'):
            matching_hierarchy = get_bottom_modified_matching_hierarchy(key, hierarchy, source_info)
        else:
            matching_hierarchy = hierarchy
        matching_hierarchies[key] = matching_hierarchy

        # Check if this is an intro section (highest priority)
        if hierarchy == "intro_section" or key == "intro_section":
            intro_section_sections[key] = matching_hierarchy
        # Only bottom-added sections should skip matching and append to end.
        elif hierarchy.startswith('bottom-added-'):
            bottom_sections[key] = hierarchy
        elif is_direct_match_candidate(matching_hierarchy):
            direct_match_sections[key] = matching_hierarchy
        else:
            batched_ai_sections[key] = matching_hierarchy
    
    thread_safe_print(f"📊 Section categorization:")
    thread_safe_print(f"   📄 Intro section: {len(intro_section_sections)} section(s)")
    thread_safe_print(f"   🎯 Direct matching: {len(direct_match_sections)} sections")
    thread_safe_print(f"   🤖 Batched AI matching: {len(batched_ai_sections)} sections")
    thread_safe_print(f"   🔚 Bottom-added sections: {len(bottom_sections)} sections (no matching needed)")

    batched_ai_results = {}
    batched_ai_failures = set()
    filtered_target_hierarchy = None

    if batched_ai_sections:
        filtered_target_hierarchy = filter_non_system_sections(target_hierarchy)
        if len(filtered_target_hierarchy) <= max_non_system_sections:
            thread_safe_print(
                f"\n🤖 Batch-matching {len(batched_ai_sections)} non-direct sections in one AI call..."
            )
            ai_sections, batch_call_failures = batch_match_sections_with_ai(
                batched_ai_sections,
                filtered_target_hierarchy,
                ai_client,
                repo_config,
                max_tokens,
                source_diff_dict=source_entries,
                source_base_hierarchy=None,
                source_head_hierarchy=None,
                source_mode=normalized_source_mode,
            )
            if batch_call_failures:
                batched_ai_failures.update(batch_call_failures)
            elif ai_sections:
                batched_ai_results, validation_failures = resolve_batched_ai_matches(
                    batched_ai_sections,
                    ai_sections,
                    filtered_target_hierarchy,
                    source_entries,
                )
                batched_ai_failures.update(validation_failures)
                thread_safe_print(
                    f"   ✅ Batched AI matching resolved {len(batched_ai_results)} section(s)"
                )
                if validation_failures:
                    thread_safe_print(
                        f"   ↪ {len(validation_failures)} section(s) will fall back to individual matching"
                    )
            else:
                batched_ai_failures.update(batched_ai_sections.keys())
        else:
            thread_safe_print(
                f"   ❌ Target hierarchy too large for batched AI matching: "
                f"{len(filtered_target_hierarchy)} > {max_non_system_sections}"
            )
            batched_ai_failures.update(batched_ai_sections.keys())
    
    # Process each section in original order
    verbose_thread_safe_print(f"\n🔄 Processing sections in original order...")
    
    for key, hierarchy in source_hierarchies.items():
        verbose_thread_safe_print(f"   🔍 Processing {key}: {hierarchy}")
        matching_hierarchy = matching_hierarchies.get(key, hierarchy)
        
        # Determine processing strategy based on section type and content
        if hierarchy == "intro_section" or key == "intro_section":
            # Intro section - match to target's content from first # heading to first ## heading
            verbose_thread_safe_print(f"      📄 Intro section - matching to target (# to ##)")
            # Find the first top-level heading in target
            first_heading_line = 0
            for i, line in enumerate(target_lines, 1):
                if line.strip().startswith('# '):
                    first_heading_line = i
                    break
            # Find the first level-2 header in target
            first_level2_line = 0
            for i, line in enumerate(target_lines, 1):
                if line.strip().startswith('## '):
                    first_level2_line = i
                    break
            if first_level2_line == 0:
                first_level2_line = len(target_lines) + 1
            
            result = {
                "target_line": str(first_heading_line) if first_heading_line else "1",
                "target_hierarchy": "intro_section",
                "intro_section_end_line": first_level2_line
            }
        elif hierarchy.startswith('bottom-added-'):
            # Bottom-added section - no matching needed, append to end
            verbose_thread_safe_print(f"      🔚 Bottom-added section - append to end of document")
            result = {
                "target_line": "-1",  # Special marker for bottom-added sections
                "target_hierarchy": hierarchy
            }
        elif key in batched_ai_results:
            verbose_thread_safe_print(f"      🤖 Using batched AI match result")
            result = batched_ai_results[key]
        elif key in batched_ai_sections:
            verbose_thread_safe_print(f"      ↪ Falling back to individual matching")
            if key.startswith('added_'):
                result = process_added_section(
                    key,
                    matching_hierarchy,
                    target_hierarchy,
                    target_lines,
                    ai_client,
                    repo_config,
                    max_non_system_sections,
                    max_tokens,
                )
            else:
                operation = source_entries.get(key, {}).get('operation', 'unknown')
                verbose_thread_safe_print(f"      {operation.capitalize()} section - finding target match")
                result = process_modified_or_deleted_section(
                    key,
                    matching_hierarchy,
                    target_hierarchy,
                    target_lines,
                    ai_client,
                    repo_config,
                    max_non_system_sections,
                    max_tokens,
                )
        else:
            # Direct-match path.
            operation = source_entries.get(key, {}).get('operation', 'unknown')
            if key.startswith('added_'):
                verbose_thread_safe_print(f"      ➕ Added section - finding insertion point")
                result = process_added_section(
                    key,
                    matching_hierarchy,
                    target_hierarchy,
                    target_lines,
                    ai_client,
                    repo_config,
                    max_non_system_sections,
                    max_tokens,
                )
            else:
                verbose_thread_safe_print(f"      {operation.capitalize()} section - finding target match")
                result = process_modified_or_deleted_section(
                    key,
                    matching_hierarchy,
                    target_hierarchy,
                    target_lines,
                    ai_client,
                    repo_config,
                    max_non_system_sections,
                    max_tokens,
                )
        
        if result:
            # Add source language information from source_diff_dict
            source_info = source_entries.get(key, {})
            all_matched_sections[key] = build_enhanced_match_result(
                key,
                hierarchy,
                matching_hierarchy,
                result,
                source_info,
                target_lines,
            )
            verbose_thread_safe_print(f"      ✅ {key}: -> line {result.get('target_line', 'unknown')}")
        else:
            thread_safe_print(f"      ❌ {key}: matching failed")

    validation_errors = validate_mapping_for_retry(all_matched_sections, source_entries)
    if validation_errors:
        thread_safe_print("\n⚠️  Section mapping risk detected after lightweight mapping:")
        for error in validation_errors:
            thread_safe_print(f"   - {error}")

        can_retry_with_full_context = (
            len(target_hierarchy or {}) <= full_context_max_sections
            and bool(batched_ai_sections)
        )
        if can_retry_with_full_context:
            if not source_base_hierarchy and not source_head_hierarchy and source_hierarchy_provider:
                try:
                    source_base_hierarchy, source_head_hierarchy = source_hierarchy_provider()
                except Exception as e:
                    thread_safe_print(
                        f"   ⚠️  Could not load source hierarchy context for retry: "
                        f"{sanitize_exception_message(e)}"
                    )

            if source_base_hierarchy or source_head_hierarchy:
                thread_safe_print(
                    f"   🧭 Retrying section mapping with full context "
                    f"({normalized_source_mode} mode, {len(target_hierarchy or {})} target sections)"
                )
                retry_batched_results, retry_validation_failures = batch_match_sections_with_ai_json(
                    batched_ai_sections,
                    filtered_target_hierarchy or filter_non_system_sections(target_hierarchy),
                    ai_client,
                    repo_config,
                    max_tokens,
                    source_diff_dict=source_entries,
                    source_base_hierarchy=source_base_hierarchy,
                    source_head_hierarchy=source_head_hierarchy,
                    source_mode=normalized_source_mode,
                )

                if retry_validation_failures:
                    thread_safe_print(
                        f"   ⚠️  Full-context JSON mapping left "
                        f"{len(retry_validation_failures)} unresolved section(s)"
                    )

                if retry_batched_results:
                    retried_sections = all_matched_sections.copy()
                    for key, result in retry_batched_results.items():
                        hierarchy = source_hierarchies.get(key, "")
                        matching_hierarchy = matching_hierarchies.get(key, hierarchy)
                        retried_sections[key] = build_enhanced_match_result(
                            key,
                            hierarchy,
                            matching_hierarchy,
                            result,
                            source_entries.get(key, {}),
                            target_lines,
                        )

                    retry_validation_errors = validate_mapping_for_retry(
                        retried_sections,
                        source_entries,
                    )
                    if normalized_source_mode == "pr":
                        if len(retry_validation_errors) <= len(validation_errors):
                            thread_safe_print(
                                f"   ✅ Using full-context retry result for PR-mode mapping "
                                f"({len(retry_validation_errors)} remaining risk item(s))"
                            )
                            all_matched_sections = retried_sections
                            validation_errors = retry_validation_errors
                        else:
                            thread_safe_print(
                                f"   ↪ Keeping lightweight PR-mode mapping because retry increased risk "
                                f"({len(validation_errors)} -> {len(retry_validation_errors)})"
                            )
                    else:
                        all_matched_sections = retried_sections
                        validation_errors = retry_validation_errors
                        if validation_errors:
                            thread_safe_print("\n❌ Commit-mode full-context mapping validation failed:")
                            for error in validation_errors:
                                thread_safe_print(f"   - {error}")
                            return {}
                elif normalized_source_mode == "commit":
                    thread_safe_print("   ❌ Commit-mode full-context mapping produced no usable repair result")
                    return {}
            else:
                thread_safe_print("   ⚠️  No source hierarchy context available for full-context retry")
        else:
            thread_safe_print("   ⚠️  Full-context retry is not available for this file")

    if normalized_source_mode == "commit":
        validation_errors = validate_commit_mode_matches(all_matched_sections, source_entries)
        if validation_errors:
            thread_safe_print("\n❌ Commit-mode section mapping validation failed:")
            for error in validation_errors:
                thread_safe_print(f"   - {error}")
            return {}

    thread_safe_print(f"\n📊 Final matching results: {len(all_matched_sections)} total matches")
    return all_matched_sections

def process_modified_or_deleted_section(key, hierarchy, target_hierarchy, target_lines, ai_client, repo_config, max_non_system_sections, max_tokens=20000):
    """Process modified or deleted sections to find target matches"""
    # Extract the leaf title from hierarchy for checking
    leaf_title = hierarchy.split(' > ')[-1] if ' > ' in hierarchy else hierarchy
    
    # Check if this is suitable for direct matching
    if (hierarchy == "frontmatter" or 
        leaf_title.startswith('# ') or  # Top-level titles
        is_system_variable_or_config(leaf_title)):  # System variables/config
        
        if hierarchy == "frontmatter":
            return {"target_line": "0", "target_hierarchy": "frontmatter"}
            
        elif leaf_title.startswith('# '):
            # Top-level title matching
            temp_sections = {key: hierarchy}
            matched_dict, failed_matches, skipped_sections = find_toplevel_title_matches(
                temp_sections, target_lines
            )
            if matched_dict:
                target_line = list(matched_dict.keys())[0]
                # For top-level titles, add # prefix to the hierarchy
                return {
                    "target_line": target_line, 
                    "target_hierarchy": f"# {matched_dict[target_line]}"
                }
                
        else:
            # System variable/config matching
            temp_sections = {key: hierarchy}
            matched_dict, failed_matches, skipped_sections = find_direct_matches_for_special_files(
                temp_sections, target_hierarchy, target_lines
            )
            if matched_dict:
                target_line = list(matched_dict.keys())[0]
                target_hierarchy_str = list(matched_dict.values())[0]
                formatted_hierarchy = format_target_hierarchy(target_hierarchy_str)
                
                return {
                    "target_line": target_line,
                    "target_hierarchy": formatted_hierarchy
                }
    else:
        # AI matching for non-system sections
        filtered_target_hierarchy = filter_non_system_sections(target_hierarchy)
        
        if len(filtered_target_hierarchy) <= max_non_system_sections:
            temp_sections = {key: hierarchy}
            
            ai_response = get_corresponding_sections(
                list(temp_sections.values()),
                list(filtered_target_hierarchy.values()),
                ai_client,
                repo_config['source_language'],
                repo_config['target_language'],
                max_tokens
            )
            
            if ai_response:
                ai_sections = parse_ai_response(ai_response)
                ai_matched = find_matching_line_numbers(ai_sections, target_hierarchy)
                
                if ai_matched:
                    target_line, target_hierarchy_str = choose_best_ai_match(ai_matched, hierarchy)
                    
                    # Format AI matched hierarchy with # prefix and remove top-level title
                    formatted_hierarchy = format_target_hierarchy(target_hierarchy_str)
                    
                    return {
                        "target_line": target_line,
                        "target_hierarchy": formatted_hierarchy
                    }

        fallback_match = find_step_heading_fallback_match(hierarchy, target_hierarchy)
        if fallback_match:
            target_line, target_hierarchy_str = fallback_match
            thread_safe_print(
                f"      ↪ Fallback step-heading match: {hierarchy} -> {target_hierarchy_str}"
            )
            return {
                "target_line": target_line,
                "target_hierarchy": format_target_hierarchy(target_hierarchy_str),
            }
    
    return None

def format_target_hierarchy(target_hierarchy_str):
    """Format target hierarchy to preserve complete hierarchy structure"""
    if target_hierarchy_str.startswith('##') or target_hierarchy_str.startswith('#'):
        # Already formatted, return as is
        return target_hierarchy_str
    elif ' > ' in target_hierarchy_str:
        # Keep complete hierarchy structure, just ensure proper formatting
        return target_hierarchy_str
    else:
        # Single level, add ## prefix for compatibility
        return f"## {target_hierarchy_str}"

def process_added_section(key, reference_hierarchy, target_hierarchy, target_lines, ai_client, repo_config, max_non_system_sections, max_tokens=20000):
    """Process added sections to find insertion points"""
    # For added sections, hierarchy points to the next section (where to insert before)
    reference_leaf = reference_hierarchy.split(' > ')[-1] if ' > ' in reference_hierarchy else reference_hierarchy
    
    if (reference_hierarchy == "frontmatter" or 
        reference_leaf.startswith('# ') or 
        is_system_variable_or_config(reference_leaf)):
        
        # Use direct matching for the reference section
        temp_reference = {f"ref_{key}": reference_hierarchy}
        
        if reference_hierarchy == "frontmatter":
            return {
                "target_line": "0",
                "target_hierarchy": "frontmatter",
                "insertion_type": "before_reference"
            }
            
        elif reference_leaf.startswith('# '):
            matched_dict, failed_matches, skipped_sections = find_toplevel_title_matches(
                temp_reference, target_lines
            )
            if matched_dict:
                target_line = list(matched_dict.keys())[0]
                formatted_hierarchy = f"# {matched_dict[target_line]}"
                return {
                    "target_line": target_line,
                    "target_hierarchy": formatted_hierarchy,
                    "insertion_type": "before_reference"
                }
                
        else:
            # System variable/config
            matched_dict, failed_matches, skipped_sections = find_direct_matches_for_special_files(
                temp_reference, target_hierarchy, target_lines
            )
            if matched_dict:
                target_line = list(matched_dict.keys())[0]
                target_hierarchy_str = list(matched_dict.values())[0]
                formatted_hierarchy = format_target_hierarchy(target_hierarchy_str)
                return {
                    "target_line": target_line,
                    "target_hierarchy": formatted_hierarchy,
                    "insertion_type": "before_reference"
                }
    else:
        # Use AI matching for the reference section
        filtered_target_hierarchy = filter_non_system_sections(target_hierarchy)
        
        if len(filtered_target_hierarchy) <= max_non_system_sections:
            temp_reference = {f"ref_{key}": reference_hierarchy}
            
            ai_response = get_corresponding_sections(
                list(temp_reference.values()),
                list(filtered_target_hierarchy.values()),
                ai_client,
                repo_config['source_language'],
                repo_config['target_language'],
                max_tokens
            )
            
            if ai_response:
                ai_sections = parse_ai_response(ai_response)
                ai_matched = find_matching_line_numbers(ai_sections, target_hierarchy)
                
                if ai_matched:
                    target_line = list(ai_matched.keys())[0]
                    target_hierarchy_str = list(ai_matched.values())[0]
                    formatted_hierarchy = format_target_hierarchy(target_hierarchy_str)
                    return {
                        "target_line": target_line,
                        "target_hierarchy": formatted_hierarchy,
                        "insertion_type": "before_reference"
                    }
    
    return None

def extract_target_section_content(target_line_num, target_lines):
    """Extract target section content from target_lines (includes sub-sections)"""
    if target_line_num >= len(target_lines):
        return ""
    
    start_line = target_line_num - 1  # Convert to 0-based index
    
    # Find the end of the section by looking for the next header
    raw_current_line = target_lines[start_line]
    current_line = raw_current_line.strip()
    if not is_markdown_heading(raw_current_line):
        return current_line
    
    current_level = len(current_line.split()[0])  # Count # characters
    end_line = len(target_lines)  # Default to end of file
    
    # For top-level headers (# level 1), stop at first sublevel (## level 2)
    # For other headers, stop at same or higher level
    in_code_block = False
    code_block_delimiter = None
    if current_level == 1:
        # Top-level header: stop at first ## (level 2) or higher
        for i in range(start_line + 1, len(target_lines)):
            raw_line = target_lines[i]
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
                line_level = len(line.split()[0])
                if line_level >= 2:  # Stop at ## or higher level
                    end_line = i
                    break
    else:
        # Sub-level header: stop at same or higher level (traditional behavior)
        for i in range(start_line + 1, len(target_lines)):
            raw_line = target_lines[i]
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
                line_level = len(line.split()[0])
                if line_level <= current_level:
                    end_line = i
                    break
    
    # Extract content from start_line to end_line
    section_content = '\n'.join(target_lines[start_line:end_line])
    return section_content.strip()

def extract_section_direct_content(target_line_num, target_lines):
    """Extract ONLY the direct content of a section (excluding sub-sections)"""
    if target_line_num >= len(target_lines):
        return ""
    
    start_line = target_line_num - 1  # Convert to 0-based index
    
    # Find the end of the section by looking for the next header
    raw_current_line = target_lines[start_line]
    current_line = raw_current_line.strip()
    if not is_markdown_heading(raw_current_line):
        return current_line
    
    current_level = len(current_line.split()[0])  # Count # characters
    end_line = len(target_lines)  # Default to end of file
    
    # Only extract until the first header (any level)
    # This means we stop at ANY header - whether it's a sub-section OR same/higher level
    in_code_block = False
    code_block_delimiter = None
    for i in range(start_line + 1, len(target_lines)):
        raw_line = target_lines[i]
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
            end_line = i
            break
    
    # Extract content from start_line to end_line
    section_content = '\n'.join(target_lines[start_line:end_line])
    return section_content.strip()

def extract_frontmatter_content(target_lines):
    """Extract frontmatter content from beginning to first header"""
    if not target_lines:
        return ""
    
    frontmatter_lines = []
    for i, line in enumerate(target_lines):
        line_stripped = line.strip()
        # Stop when we hit the first top-level header
        if line_stripped.startswith('# '):
            break
        frontmatter_lines.append(line.rstrip())
    
    return '\n'.join(frontmatter_lines)


def extract_intro_section_content_from_lines(target_lines):
    """
    Extract intro section content from target lines: from the first
    top-level heading (#) to the line before the first level-2 heading (##).
    Excludes frontmatter.
    Returns: (intro_content, first_level2_line)
    """
    start_idx = None
    for i, line in enumerate(target_lines):
        if line.strip().startswith('# '):
            start_idx = i
            break

    if start_idx is None:
        return "", len(target_lines) + 1

    intro_lines = []
    first_level2_line = len(target_lines) + 1
    for i in range(start_idx, len(target_lines)):
        if target_lines[i].strip().startswith('## '):
            first_level2_line = i + 1  # 1-based
            break
        intro_lines.append(target_lines[i].rstrip())

    intro_content = '\n'.join(intro_lines)
    return intro_content, first_level2_line


def find_section_end_line(section_start_line, target_hierarchy, target_lines):
    """Find the end line of a section to determine insertion point (from auto-sync-pr-changes.py)"""
    
    # Get the current section's level
    raw_current_section_line = target_lines[section_start_line - 1]
    current_section_line = raw_current_section_line.strip()
    current_level = len(current_section_line.split()[0]) if is_markdown_heading(raw_current_section_line) else 5
    
    # Find the next section at the same level or higher (lower number)
    next_section_line = None
    for line_num_str in sorted(target_hierarchy.keys(), key=int):
        line_num = int(line_num_str)
        if line_num > section_start_line:
            # Check the level of this section
            raw_section_line = target_lines[line_num - 1]
            section_line = raw_section_line.strip()
            if is_markdown_heading(raw_section_line):
                section_level = len(section_line.split()[0])
                if section_level <= current_level:
                    next_section_line = line_num
                    break
    
    if next_section_line:
        # Insert before the next same-level or higher-level section
        return next_section_line - 1
    else:
        # This is the last section at this level, insert at the end of the file
        return len(target_lines)

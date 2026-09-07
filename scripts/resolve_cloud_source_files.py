#!/usr/bin/env python3
"""Resolve commit-sync SOURCE_FILES for Cloud translation workflows."""

import os
import re
import subprocess
from pathlib import Path

try:
    from markdown_it import MarkdownIt
except ImportError:
    MarkdownIt = None

MARKDOWN_LINK_RE = re.compile(r'!?\[[^\]]*\]\(([^)]+)\)')


def parse_list(value):
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def normalize_doc_path(value):
    rel = str(value or "").strip().replace("\\", "/")
    if not rel:
        return ""
    rel = rel.split("#", 1)[0].split("?", 1)[0].strip()
    if rel.startswith("<") and rel.endswith(">"):
        rel = rel[1:-1].strip()
    rel = rel.lstrip("/")
    if rel.startswith("./"):
        rel = rel[2:]
    if rel.startswith("docs/"):
        rel = rel[5:]
    return rel


def extract_markdown_doc_links_with_regex(markdown):
    links = []
    for match in MARKDOWN_LINK_RE.finditer(markdown or ""):
        url = match.group(1).strip()
        if " " in url:
            url = url.split(None, 1)[0]
        rel = normalize_doc_path(url)
        if rel.endswith(".md"):
            links.append(rel)
    return links


def get_token_attr(token, name):
    attrs = getattr(token, "attrs", None)
    if isinstance(attrs, dict):
        return attrs.get(name)
    for key, value in attrs or []:
        if key == name:
            return value
    return None


def extract_markdown_doc_links(markdown):
    if MarkdownIt is None:
        return extract_markdown_doc_links_with_regex(markdown)

    links = []
    parser = MarkdownIt("commonmark")
    for token in parser.parse(markdown or ""):
        if token.type != "inline":
            continue
        for child in token.children or []:
            if child.type != "link_open":
                continue
            rel = normalize_doc_path(get_token_attr(child, "href"))
            if rel.endswith(".md"):
                links.append(rel)
    return links


def build_allowed_files(docs_source_path, toc_files, extra_files=None):
    allowed = set(toc_files)
    source_root = Path(docs_source_path)

    for toc_file in toc_files:
        toc_path = source_root / toc_file
        if not toc_path.exists():
            raise FileNotFoundError(f"Cloud TOC file not found: {toc_path}")
        allowed.update(extract_markdown_doc_links(toc_path.read_text(encoding="utf-8")))

    for extra_file in extra_files or []:
        rel = normalize_doc_path(extra_file)
        if rel:
            allowed.add(rel)

    return allowed


def resolve_requested_file(value, allowed):
    rel = normalize_doc_path(value)
    if not rel or rel in allowed or "/" in rel:
        return rel

    candidates = sorted(item for item in allowed if os.path.basename(item) == rel)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError(f'Ambiguous Cloud file name "{rel}": {", ".join(candidates)}')
    return rel


def parse_git_name_status(output):
    rows = []
    for line in (output or "").splitlines():
        parts = line.split("\t")
        if not parts or not parts[0]:
            continue
        status = parts[0][0]
        if status == "R" and len(parts) >= 3:
            rows.append(
                {
                    "filename": normalize_doc_path(parts[2]),
                    "previous_filename": normalize_doc_path(parts[1]),
                }
            )
        elif len(parts) >= 2:
            rows.append(
                {
                    "filename": normalize_doc_path(parts[1]),
                    "previous_filename": "",
                }
            )
    return rows


def list_changed_files(docs_source_path, base_ref, head_ref):
    output = subprocess.check_output(
        ["git", "-C", docs_source_path, "diff", "--name-status", "-M", base_ref, head_ref],
        text=True,
        stderr=subprocess.STDOUT,
    )
    return parse_git_name_status(output)


def read_git_file(docs_source_path, ref, file_path):
    try:
        return subprocess.check_output(
            ["git", "-C", docs_source_path, "show", f"{ref}:{file_path}"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return None


def collect_toc_scope_added_files(docs_source_path, toc_files, base_ref, head_ref):
    """Return Markdown files newly linked to the aggregate Cloud TOC scope."""
    base_links = set()
    head_links = set()

    for toc_file in toc_files:
        head_content = read_git_file(docs_source_path, head_ref, toc_file)
        if head_content is None:
            continue

        base_content = read_git_file(docs_source_path, base_ref, toc_file)
        base_links.update(extract_markdown_doc_links(base_content or ""))
        head_links.update(extract_markdown_doc_links(head_content))

    return head_links - base_links


def collect_new_toc_full_translation_files(
    docs_source_path,
    docs_target_path,
    toc_files,
    base_ref,
    head_ref,
    extra_files=None,
):
    """Return files that need a one-time full translation for a new TOC scope.

    A configured TOC is new when it exists at the source HEAD but not in the
    checked-out target branch. Links from all new TOCs are deduplicated before
    subtracting the aggregate scope of TOCs that already exist in the target.
    """
    if not docs_target_path:
        return set(), set()

    target_root = Path(docs_target_path)
    new_toc_files = {
        toc_file
        for toc_file in toc_files
        if read_git_file(docs_source_path, head_ref, toc_file) is not None
        and not (target_root / toc_file).is_file()
    }
    existing_toc_files = set(toc_files) - new_toc_files

    new_toc_links = set()
    for toc_file in new_toc_files:
        head_content = read_git_file(docs_source_path, head_ref, toc_file)
        new_toc_links.update(extract_markdown_doc_links(head_content or ""))

    previously_allowed_links = set()
    for toc_file in existing_toc_files:
        # BASE is intentional: a link added to an existing TOC during this
        # compare range was not previously covered by weekly translation. If
        # a new TOC also links it, it still needs one complete HEAD translation.
        base_content = read_git_file(docs_source_path, base_ref, toc_file)
        previously_allowed_links.update(extract_markdown_doc_links(base_content or ""))

    full_translation_files = set(new_toc_files)
    full_translation_files.update(new_toc_links - previously_allowed_links)

    # Configured index files are not necessarily linked from a TOC. A missing
    # target index therefore needs the same one-time HEAD translation.
    for extra_file in extra_files or []:
        rel = normalize_doc_path(extra_file)
        if (
            rel
            and read_git_file(docs_source_path, head_ref, rel) is not None
            and not (target_root / rel).is_file()
        ):
            full_translation_files.add(rel)

    return full_translation_files, new_toc_files


def add_unique(items, item):
    if item and item not in items:
        items.append(item)


def resolve_source_files(
    allowed,
    input_file_names="",
    changed_rows=None,
    always_include_files=None,
):
    normalized_always_include_files = set()
    for item in always_include_files or []:
        rel = normalize_doc_path(item)
        if rel:
            normalized_always_include_files.add(rel)
    always_include_files = normalized_always_include_files
    requested = [
        resolve_requested_file(item, allowed)
        for item in parse_list(input_file_names)
    ]
    requested = [item for item in requested if item]

    resolved = []
    if requested:
        invalid = [
            item for item in requested
            if item not in allowed and item not in always_include_files
        ]
        if invalid:
            raise ValueError(
                "The following files are not in the configured Cloud TOC scope: "
                + ", ".join(invalid)
            )
        for rel in requested:
            add_unique(resolved, rel)
        return resolved

    for row in changed_rows or []:
        filename = row.get("filename", "")
        previous_filename = row.get("previous_filename", "")
        if filename in allowed or filename in always_include_files:
            add_unique(resolved, filename)
        if previous_filename in allowed or previous_filename in always_include_files:
            add_unique(resolved, previous_filename)

    for rel in sorted(always_include_files):
        add_unique(resolved, rel)

    return resolved


def append_github_output(output_path, values):
    if not output_path:
        return
    with open(output_path, "a", encoding="utf-8") as f:
        for key, value in values.items():
            f.write(f"{key}={value}\n")


def main():
    docs_source_path = os.environ["DOCS_SOURCE_PATH"]
    docs_target_path = os.getenv("DOCS_TARGET_PATH", "")
    toc_files = parse_list(os.getenv("TOC_FILES") or os.environ.get("CLOUD_TOC_FILES", ""))
    cloud_index_files = parse_list(os.getenv("INDEX_FILES") or os.getenv("CLOUD_INDEX_FILES", ""))
    input_file_names = os.getenv("INPUT_FILE_NAMES", "")
    base_ref = os.environ["BASE_REF"]
    head_ref = os.environ["HEAD_REF"]

    allowed = build_allowed_files(docs_source_path, toc_files, extra_files=cloud_index_files)
    if input_file_names.strip():
        changed_rows = []
        toc_scope_added_files = set()
        full_translation_files = set()
        new_toc_files = set()
    else:
        changed_rows = list_changed_files(docs_source_path, base_ref, head_ref)
        toc_scope_added_files = collect_toc_scope_added_files(
            docs_source_path,
            toc_files,
            base_ref,
            head_ref,
        )
        new_toc_full_translation_files, new_toc_files = (
            collect_new_toc_full_translation_files(
                docs_source_path,
                docs_target_path,
                toc_files,
                base_ref,
                head_ref,
                extra_files=cloud_index_files,
            )
        )
        # A file newly linked from an existing TOC also enters translation
        # scope for the first time and must use its complete HEAD content.
        full_translation_files = (
            set(toc_scope_added_files) | new_toc_full_translation_files
        )
    resolved = resolve_source_files(
        allowed,
        input_file_names=input_file_names,
        changed_rows=changed_rows,
        always_include_files=full_translation_files,
    )

    append_github_output(
        os.getenv("GITHUB_OUTPUT", ""),
        {
            "files": ",".join(resolved),
            "full_files": ",".join(sorted(full_translation_files)),
            "new_toc_files": ",".join(sorted(new_toc_files)),
            "has_source_changes": "true" if resolved else "false",
            "allowed_count": str(len(allowed)),
        },
    )

    if resolved:
        print(f"Resolved {len(resolved)} source file(s): {','.join(resolved)}")
        if new_toc_files:
            print(f"Detected new target TOC file(s): {','.join(sorted(new_toc_files))}")
        if full_translation_files:
            print(
                "Resolved "
                f"{len(full_translation_files)} full-translation bootstrap file(s)."
            )
    else:
        print("No Cloud TOC-scoped source changes detected.")


if __name__ == "__main__":
    main()

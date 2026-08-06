"""
Product-specific post-processing helpers for translation output.
"""

import os
import re

# Fallback when the PRODUCT environment variable is unset or blank.
DEFAULT_PRODUCT_NAME = ""

TIDB_ZH_VERSION_ANCHOR_RE = re.compile(
    r'#(?P<prefix>[^)\s#]*?)-从-(?P<version>v\d{2,3})-版本开始引入'
)
TIDB_EN_VERSION_ANCHOR_RE = re.compile(
    r'#(?P<prefix>[^)\s#]*?)-new-in-(?P<version>v\d{2,3})'
)
MARKDOWN_LINK_RE = re.compile(r'(?<!!)\[([^\]]+)\]\(([^)]+)\)')


def get_product_name():
    """Return the configured product name, or an empty string when omitted."""
    configured_name = (os.getenv("PRODUCT") or "").strip()
    return configured_name or DEFAULT_PRODUCT_NAME


def should_apply_tidb_version_anchor_rewrite(source_language, target_language, source_mode=""):
    """Return True when PR-mode TiDB version anchors should be normalized."""
    if get_product_name().lower() != "tidb":
        return False
    if (source_mode or "").lower() != "pr":
        return False

    normalized_source = (source_language or "").lower()
    normalized_target = (target_language or "").lower()
    return (
        normalized_source == "chinese" and normalized_target == "english"
    ) or (
        normalized_source == "english" and normalized_target == "chinese"
    )


def rewrite_tidb_version_anchors_in_text(text, source_language, target_language, source_mode=""):
    """Rewrite TiDB version anchors in Markdown link URLs from AI-translated PR output."""
    if not isinstance(text, str) or not should_apply_tidb_version_anchor_rewrite(
        source_language,
        target_language,
        source_mode=source_mode,
    ):
        return text

    normalized_source = (source_language or "").lower()
    normalized_target = (target_language or "").lower()

    if normalized_source == "chinese" and normalized_target == "english":
        anchor_re = TIDB_ZH_VERSION_ANCHOR_RE
        anchor_replacement = lambda match: f"#{match.group('prefix')}-new-in-{match.group('version')}"
    elif normalized_source == "english" and normalized_target == "chinese":
        anchor_re = TIDB_EN_VERSION_ANCHOR_RE
        anchor_replacement = lambda match: f"#{match.group('prefix')}-从-{match.group('version')}-版本开始引入"
    else:
        return text

    def replace_link(match):
        label, url = match.groups()
        if ".md#" not in url:
            return match.group(0)
        updated_url = anchor_re.sub(anchor_replacement, url)
        return f"[{label}]({updated_url})"

    return MARKDOWN_LINK_RE.sub(replace_link, text)


def _new_mapping_like(mapping):
    """Create an empty mapping while preserving dict subclasses when possible."""
    if type(mapping) is dict:
        return {}

    try:
        return mapping.__class__()
    except Exception:
        return {}


def rewrite_tidb_version_anchors_in_sections(sections, source_language, target_language, source_mode=""):
    """Rewrite TiDB version anchors in a dict of translated section content."""
    if not isinstance(sections, dict) or not should_apply_tidb_version_anchor_rewrite(
        source_language,
        target_language,
        source_mode=source_mode,
    ):
        return sections

    rewritten = _new_mapping_like(sections)
    for key, value in sections.items():
        rewritten[key] = rewrite_tidb_version_anchors_in_text(
            value,
            source_language,
            target_language,
            source_mode=source_mode,
        )

    if hasattr(sections, "failures"):
        try:
            rewritten.failures = list(sections.failures)
        except AttributeError:
            pass
    return rewritten

"""Shared helpers for stable Markdown heading anchors."""

import re


EXPLICIT_HEADING_ANCHOR_RE = re.compile(r"\s+\{#([^}]+)\}\s*$")


def build_heading_anchor_slug(heading_text):
    """Build a stable slug for an English Markdown heading."""
    text = EXPLICIT_HEADING_ANCHOR_RE.sub("", (heading_text or "").strip())
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Ignore HTML tags themselves while preserving their visible text content.
    text = re.sub(r"</?[^>]+>", " ", text)
    # Keep dotted version numbers compact in anchors, e.g. v4.0.10 -> v4010.
    text = re.sub(r"(?<=\d)\.(?=\d)", "", text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    text = re.sub(r"\s+", "-", text.strip())
    text = re.sub(r"-+", "-", text)
    return text.strip("-")

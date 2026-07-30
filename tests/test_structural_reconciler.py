import sys
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from structural_reconciler import (
    _deterministic_version_mark_inner,
    _protect_custom_content_tags,
    _restore_markdown_indentation,
    _section_similarity_text,
    _split_into_heading_sections,
    _translate_preserving_custom_content,
    reconcile_restructured_file,
    reconcile_version_mark_only_change,
    split_into_blocks,
)
from diff_analyzer import detect_structural_change
from translation_structure_validator import (
    custom_content_tag_signature,
    extract_custom_content_tags,
)


REPO_CONFIG = {
    "source_language": "English",
    "target_language": "Chinese",
    "target_local_path": "/tmp",
}


class MockAI:
    """Identity 'translator': echoes back the content portion of the prompt.

    This keeps CustomContent tags and heading levels intact so the reconciler's
    structure self-check passes, while letting tests assert which blocks were
    actually sent to the AI.
    """

    def __init__(self):
        self.calls = []

    @staticmethod
    def _content_portion(prompt):
        marker = "Content to translate:\n"
        index = prompt.rfind(marker)
        return prompt[index + len(marker):] if index != -1 else prompt

    def chat_completion(self, messages, temperature=0.1):
        self.calls.append(messages)
        return self._content_portion(messages[0]["content"])

    def contents(self):
        """The block content actually sent for translation (excludes the fixed
        instruction boilerplate, which mentions things like 'Some text ...')."""
        return [self._content_portion(call[0]["content"]) for call in self.calls]

    def prompts(self):
        return [call[0]["content"] for call in self.calls]


def _tags(content):
    return custom_content_tag_signature(extract_custom_content_tags(content))


class SplitBlocksTest(unittest.TestCase):
    def test_join_reproduces_content(self):
        content = "# A\n\nintro\n\n## B\n\nbody\n"
        self.assertEqual("".join(split_into_blocks(content)), content)

    def test_blank_lines_separate_paragraphs(self):
        content = "para one\n\npara two\n"
        blocks = split_into_blocks(content)
        self.assertEqual(len(blocks), 2)

    def test_code_fence_is_atomic_even_with_internal_blank_line(self):
        content = "intro\n\n```sql\nA\n\nB\n```\n\noutro\n"
        blocks = split_into_blocks(content)
        # intro / the whole fence / outro -> 3 blocks; the blank line inside the
        # fence must NOT split it.
        self.assertEqual(len(blocks), 3)
        self.assertIn("```sql\nA\n\nB\n```", blocks[1])

    def test_section_splitting_uses_validator_heading_rules(self):
        indented = "# Title\n\n   ## CommonMark heading\n\nbody\n"
        too_deep = "# Title\n\n####### Not an ATX heading\n\nbody\n"

        self.assertEqual(len(_split_into_heading_sections(indented)), 2)
        self.assertEqual(len(_split_into_heading_sections(too_deep)), 1)

    def test_section_splitting_does_not_require_blank_line_before_heading(self):
        content = "# Title\n\n> A note.\n#### Steps\n\nbody\n"

        sections = _split_into_heading_sections(content)

        self.assertEqual(2, len(sections))
        self.assertTrue("".join(sections[1]).startswith("#### Steps\n"))

    def test_preamble_similarity_ignores_heading_syntax_inside_fence(self):
        content = "```text\n# code comment\n```\n\nPreamble body.\n"

        similarity_text = _section_similarity_text(split_into_blocks(content))

        self.assertEqual(content.rstrip("\n"), similarity_text)


class ReconcilePureMoveTest(unittest.TestCase):
    """A section moved verbatim must be reused, not duplicated or dropped."""

    base = (
        "# Title\n\nIntro paragraph.\n\n"
        "## Execution process\n\nExec body.\n\n"
        "## Restrictions\n\n<CustomContent platform=\"tidb\">\n\ntidb body.\n\n</CustomContent>\n\n"
        "## Example\n\nExample body.\n"
    )
    head = (
        "# Title\n\nIntro paragraph.\n\n"
        "## Example\n\nExample body.\n\n"
        "## Execution process\n\nExec body.\n\n"
        "## Restrictions\n\n<CustomContent platform=\"tidb\">\n\ntidb body.\n\n</CustomContent>\n"
    )
    target = (
        "# 标题\n\n简介段落。\n\n"
        "## 执行流程 {#execution-process}\n\n执行正文。\n\n"
        "## 限制 {#restrictions}\n\n<CustomContent platform=\"tidb\">\n\ntidb 正文。\n\n</CustomContent>\n\n"
        "## 示例 {#example}\n\n示例正文。\n"
    )

    def test_pure_move_reuses_all_without_ai(self):
        ai = MockAI()
        out = reconcile_restructured_file(
            "tiflash.md", self.head, self.base, self.target, ai, REPO_CONFIG, source_mode="commit"
        )
        self.assertIsNotNone(out)
        self.assertEqual(len(ai.calls), 0, "pure move must not invoke the AI")
        self.assertEqual(out.count("## 执行流程"), 1)
        self.assertEqual(out.count("## 限制"), 1)
        self.assertEqual(out.count("## 示例"), 1)
        self.assertEqual(_tags(out), _tags(self.head))
        self.assertLess(out.index("## 示例"), out.index("## 执行流程"))
        self.assertIn("执行正文。", out)
        self.assertIn("tidb 正文。", out)


class ReconcileWrappingTest(unittest.TestCase):
    """Existing content newly wrapped in CustomContent must stay balanced, and
    the wrapped section's prose/heading must be reused rather than re-translated."""

    base = (
        "# Coprocessor Cache\n\nIntro.\n\n"
        "## Monitoring\n\nSome text.\n\n"
        "### View the Grafana monitoring panel\n\nPanel text.\n\n"
        "## Faraway\n\nUntouched faraway section.\n"
    )
    head = (
        "# Coprocessor Cache\n\nIntro.\n\n"
        "## Monitoring\n\nSome text.\n\n"
        "<CustomContent platform=\"tidb-cloud\" plan=\"dedicated\">\n\n"
        "### View the Grafana monitoring panel\n\nPanel text.\n\n"
        "</CustomContent>\n\n"
        "<CustomContent platform=\"tidb\">\n\n"
        "### View the Grafana panel\n\nPanel text 2.\n\n"
        "</CustomContent>\n\n"
        "## Faraway\n\nUntouched faraway section.\n"
    )
    target = (
        "# Coprocessor 缓存\n\n简介。\n\n"
        "## 监控\n\n一些文本。\n\n"
        "### 查看 Grafana 监控面板 {#view-the-grafana-monitoring-panel}\n\n面板文本。\n\n"
        "## 远方章节 {#faraway}\n\n未改动的远方章节。\n"
    )

    def test_wrapping_balances_tags_and_reuses_existing_translation(self):
        ai = MockAI()
        out = reconcile_restructured_file(
            "coprocessor.md", self.head, self.base, self.target, ai, REPO_CONFIG, source_mode="commit"
        )
        self.assertIsNotNone(out)
        # Core fix: tag sequence matches HEAD exactly (no duplicate opener).
        self.assertEqual(_tags(out), _tags(self.head))
        # Everything that did not change in English is reused verbatim, including
        # the now-wrapped monitoring panel heading + body.
        self.assertIn("简介。", out)
        self.assertIn("一些文本。", out)
        self.assertIn("### 查看 Grafana 监控面板 {#view-the-grafana-monitoring-panel}", out)
        self.assertIn("面板文本。", out)
        self.assertIn("## 远方章节 {#faraway}", out)
        self.assertIn("未改动的远方章节。", out)
        # The wrapped/unchanged blocks must not be re-translated.
        for needle in ("Faraway", "Some text", "Grafana monitoring panel"):
            self.assertFalse(
                any(needle in prompt for prompt in ai.contents()),
                f"unchanged block containing {needle!r} must be reused, not retranslated",
            )
        # CustomContent tag lines are emitted verbatim, never sent to the AI.
        self.assertFalse(
            any("CustomContent" in prompt for prompt in ai.contents()),
            "CustomContent tag-only blocks must not be sent to the AI",
        )
        # The brand-new section is translated and carries a source-derived anchor.
        self.assertIn("{#view-the-grafana-panel}", out)

    def test_cross_section_wrapper_is_valid_with_block_count_drift(self):
        base = (
            "# Title\n\nIntro.\n\n"
            "## A\n\nA one.\n\nA two.\n\n"
            "## C\n\nC stable.\n"
        )
        head = (
            "# Title\n\nIntro.\n\n"
            "## A\n\nA one.\n\nA two.\n\n"
            '<CustomContent plan="byoc">\n\n'
            "## B\n\nB new.\n\n"
            "</CustomContent>\n\n"
            "## C\n\nC stable.\n"
        )
        # Merging the two translated A paragraphs forces heading-section
        # alignment, the path that previously rejected the cross-section tags.
        target = (
            "# 标题\n\n简介。\n\n"
            "## 甲\n\n甲一。\n甲二。\n\n"
            "## 丙\n\n丙保持不变。\n"
        )
        ai = MockAI()

        out = reconcile_restructured_file(
            "cross-section.md", head, base, target, ai, REPO_CONFIG,
            source_mode="commit",
        )

        self.assertIsNotNone(out)
        self.assertEqual(_tags(head), _tags(out))
        self.assertIn("## B", out)
        self.assertIn("## 丙\n\n丙保持不变。", out)
        self.assertFalse(any("<CustomContent" in content for content in ai.contents()))

    def test_inline_custom_content_tags_are_sent_to_ai_unchanged(self):
        base = "# Title\n\nPremium instances are supported.\n"
        head = (
            "# Title\n\nPremium "
            '<CustomContent plan="byoc">and BYOC </CustomContent>'
            "instances are supported.\n"
        )
        target = "# 标题\n\n支持 Premium 实例。\n"

        class SyntheticPlaceholderDroppingAI(MockAI):
            def chat_completion(self, messages, temperature=0.1):
                self.calls.append(messages)
                content = self._content_portion(messages[0]["content"])
                return content.replace(
                    "{{{ .__ai_custom_content_tag_0_0000 }}}", ""
                ).replace(
                    "{{{ .__ai_custom_content_tag_0_0001 }}}", ""
                )

        ai = SyntheticPlaceholderDroppingAI()
        out = reconcile_restructured_file(
            "inline-tags.md", head, base, target, ai, REPO_CONFIG,
            source_mode="commit",
        )

        self.assertIsNotNone(out)
        self.assertEqual(_tags(head), _tags(out))
        self.assertTrue(any("CustomContent" in content for content in ai.contents()))
        self.assertFalse(
            any("__ai_custom_content_tag_" in content for content in ai.contents())
        )

    def test_changed_inline_custom_content_tag_declines_on_final_validation(self):
        base = "# Title\n\nPremium instances are supported.\n"
        head = (
            "# Title\n\nPremium "
            '<CustomContent plan="byoc">and BYOC </CustomContent>'
            "instances are supported.\n"
        )
        target = "# 标题\n\n支持 Premium 实例。\n"

        class TagManglingAI(MockAI):
            def chat_completion(self, messages, temperature=0.1):
                self.calls.append(messages)
                content = self._content_portion(messages[0]["content"])
                return content.replace("</CustomContent>", "")

        out = reconcile_restructured_file(
            "mangled-inline-tag.md", head, base, target,
            TagManglingAI(), REPO_CONFIG, source_mode="commit",
        )

        self.assertIsNone(out)

    def test_only_standalone_custom_content_tags_are_protected(self):
        content = (
            'Premium <CustomContent plan="byoc">or BYOC</CustomContent> instance.\n\n'
            '    <CustomContent plan="premium">\n\n'
            "    Premium-only content.\n\n"
            "    </CustomContent>\n"
        )

        protected, tags, placeholders, _ = _protect_custom_content_tags(content)

        self.assertIn(
            '<CustomContent plan="byoc">or BYOC</CustomContent>',
            protected,
        )
        self.assertNotIn('<CustomContent plan="premium">', protected)
        self.assertEqual(
            ['<CustomContent plan="premium">', "</CustomContent>"],
            tags,
        )
        self.assertEqual(2, len(placeholders))

    def test_changed_custom_content_placeholder_declines_safely(self):
        content = (
            '<CustomContent plan="byoc">\n\n'
            "BYOC instances are supported.\n\n"
            "</CustomContent>\n"
        )

        class PlaceholderDroppingAI(MockAI):
            def chat_completion(self, messages, temperature=0.1):
                self.calls.append(messages)
                content = self._content_portion(messages[0]["content"])
                start = content.index("{{{ .__ai_custom_content_tag_")
                end = content.index("}}}", start) + 3
                return content[:start] + content[end:]

        out = _translate_preserving_custom_content(
            content,
            PlaceholderDroppingAI(),
            "English",
            "Chinese",
            source_mode="commit",
        )

        self.assertIsNone(out)

    def test_translation_failure_is_not_reported_as_placeholder_mutation(self):
        content = '<CustomContent plan="byoc">text</CustomContent>\n'

        with patch(
            "structural_reconciler.translate_file_batch", return_value=None
        ):
            with patch(
                "structural_reconciler.thread_safe_print"
            ) as print_mock:
                translated = _translate_preserving_custom_content(content, MockAI())

        self.assertIsNone(translated)
        self.assertFalse(
            any(
                "changed protected CustomContent placeholders" in str(call)
                for call in print_mock.call_args_list
            )
        )

    def test_nested_note_keeps_indentation_after_client_or_model_strip(self):
        base = (
            "# Title\n\n"
            "3. Choose a resource type:\n\n"
            "    - Create a TiDB X instance.\n\n"
            "        > **Note:**\n"
            "        >\n"
            "        > Premium settings apply per instance.\n\n"
            "    - Create a Dedicated cluster.\n"
        )
        head = (
            "# Title\n\n"
            "3. Choose a resource type:\n\n"
            "    - Create a TiDB X instance.\n\n"
            '        <CustomContent plan="premium">\n\n'
            "        > **Note:**\n"
            "        >\n"
            "        > Premium settings apply per instance.\n\n"
            "        </CustomContent>\n\n"
            '        <CustomContent plan="byoc">\n\n'
            "        > **Note:**\n"
            "        >\n"
            "        > - For BYOC instances, projects are optional.\n"
            "        > - Premium settings apply per instance.\n\n"
            "        </CustomContent>\n\n"
            "    - Create a Dedicated cluster.\n"
        )
        target = (
            "# 标题\n\n"
            "3. 选择资源类型：\n\n"
            "    - 创建 TiDB X 实例。\n\n"
            "        > **Note:**\n"
            "        >\n"
            "        > Premium 设置按实例生效。\n\n"
            "    - 创建 Dedicated 集群。\n"
        )

        class StrippingAI(MockAI):
            def chat_completion(self, messages, temperature=0.1):
                self.calls.append(messages)
                return self._content_portion(messages[0]["content"]).strip()

        class EveryLineStrippingAI(MockAI):
            def chat_completion(self, messages, temperature=0.1):
                self.calls.append(messages)
                content = self._content_portion(messages[0]["content"]).strip()
                return "\n".join(line.lstrip() for line in content.splitlines())

        for ai in (StrippingAI(), EveryLineStrippingAI()):
            with self.subTest(ai=type(ai).__name__):
                out = reconcile_restructured_file(
                    "nested-note.md",
                    head,
                    base,
                    target,
                    ai,
                    REPO_CONFIG,
                    source_mode="commit",
                )

                self.assertIsNotNone(out)
                self.assertIn(
                    "        > - For BYOC instances, projects are optional.",
                    out,
                )
                self.assertNotIn("\n> **Note:**", out)
                self.assertEqual(2, out.count("        > **Note:**"))

    def test_translation_does_not_duplicate_preserved_first_line_indentation(self):
        content = "        > **Note:**\n        >\n        > Body.\n"

        with patch(
            "structural_reconciler.translate_file_batch",
            return_value=content,
        ):
            translated = _translate_preserving_custom_content(content, MockAI())

        self.assertEqual(content, translated)

    def test_top_level_translation_is_unchanged_by_indentation_restore(self):
        content = "> **Note:**\n>\n> Body.\n"

        with patch(
            "structural_reconciler.translate_file_batch",
            return_value=content,
        ):
            translated = _translate_preserving_custom_content(content, MockAI())

        self.assertEqual(content, translated)

    def test_restore_markdown_indentation_replaces_different_line_prefixes(self):
        source = "        > **Note:**\n        > Body.\n"
        translated = "    > **Note:**\n> 正文。\n"

        self.assertEqual(
            "        > **Note:**\n        > 正文。\n",
            _restore_markdown_indentation(source, translated),
        )

    def test_restore_markdown_indentation_leaves_leading_newline_unmapped(self):
        source = "        > **Note:**\n"
        translated = "\n> **Note:**\n"

        self.assertEqual(
            translated,
            _restore_markdown_indentation(source, translated),
        )

    def test_restore_markdown_indentation_only_pairs_first_line_on_count_drift(self):
        source = "        > First.\n        > Second.\n"
        translated = "> 第一行。\n> 新增行。\n> 第二行。\n"

        self.assertEqual(
            "        > 第一行。\n> 新增行。\n> 第二行。\n",
            _restore_markdown_indentation(source, translated),
        )

    def test_restore_markdown_indentation_handles_empty_content(self):
        self.assertEqual("", _restore_markdown_indentation("", ""))
        self.assertEqual("", _restore_markdown_indentation("    source", ""))


class ReconcileCodeBlockTest(unittest.TestCase):
    """An unchanged code block adjacent to a changed paragraph must be reused
    byte-for-byte and never sent to the AI (regression: AI mangled code fences)."""

    base = (
        "# T\n\nIntro.\n\n"
        "## Syntax\n\n"
        "```sql\nINSERT INTO t\n    [ON DUP KEY UPDATE x]\n\nvalue:\n    {expr}\n```\n\n"
        "Done.\n"
    )
    head = (
        "# T\n\nIntro changed.\n\n"
        "## Syntax\n\n"
        "```sql\nINSERT INTO t\n    [ON DUP KEY UPDATE x]\n\nvalue:\n    {expr}\n```\n\n"
        "Done.\n"
    )
    target = (
        "# T\n\n简介。\n\n"
        "## 语法 {#syntax}\n\n"
        "```sql\nINSERT INTO t\n    [ON DUP KEY UPDATE x]\n\nvalue:\n    {expr}\n```\n\n"
        "完成。\n"
    )

    def test_unchanged_code_block_is_preserved_and_not_translated(self):
        ai = MockAI()
        out = reconcile_restructured_file(
            "syntax.md", self.head, self.base, self.target, ai, REPO_CONFIG, source_mode="commit"
        )
        self.assertIsNotNone(out)
        # The code block (with its internal blank line) survives byte-for-byte.
        self.assertIn(
            "```sql\nINSERT INTO t\n    [ON DUP KEY UPDATE x]\n\nvalue:\n    {expr}\n```",
            out,
        )
        # Only the changed paragraph was sent to the AI.
        self.assertEqual(len(ai.calls), 1)
        self.assertIn("Intro changed.", ai.contents()[0])
        # The code fence content was never sent to the AI.
        self.assertFalse(
            any("```sql" in prompt or "ON DUP KEY UPDATE" in prompt for prompt in ai.contents()),
            "the code block must never be sent to the AI",
        )
        # Unchanged surrounding translations are reused.
        self.assertIn("完成。", out)
        self.assertIn("## 语法 {#syntax}", out)


class ReconcileSectionContextTest(unittest.TestCase):
    """A changed block is translated with its enclosing section as context, but
    only the changed block itself is the content to translate."""

    base = (
        "# T\n\nIntro.\n\n"
        "## Details\n\nFirst paragraph about widgets.\n\nSecond paragraph about gadgets.\n\nThird paragraph about gizmos.\n"
    )
    head = (
        "# T\n\nIntro.\n\n"
        "## Details\n\nFirst paragraph about widgets.\n\nSecond paragraph about gadgets, now revised.\n\nThird paragraph about gizmos.\n"
    )
    target = (
        "# T\n\n简介。\n\n"
        "## 详情 {#details}\n\n关于 widgets 的第一段。\n\n关于 gadgets 的第二段。\n\n关于 gizmos 的第三段。\n"
    )

    def test_changed_block_translated_with_section_context(self):
        ai = MockAI()
        out = reconcile_restructured_file(
            "ctx.md", self.head, self.base, self.target, ai, REPO_CONFIG, source_mode="commit"
        )
        self.assertIsNotNone(out)
        # Only the one changed paragraph is sent as the content to translate.
        self.assertEqual(len(ai.calls), 1)
        content = ai.contents()[0]
        self.assertIn("Second paragraph about gadgets, now revised.", content)
        self.assertNotIn("First paragraph", content)
        self.assertNotIn("Third paragraph", content)
        # ... but the full enclosing section IS provided as reference context.
        prompt = ai.prompts()[0]
        self.assertIn("for context only", prompt)
        self.assertIn("First paragraph about widgets.", prompt)
        self.assertIn("Third paragraph about gizmos.", prompt)
        self.assertIn("## Details", prompt)
        # Unchanged sibling paragraphs keep their existing translation verbatim.
        self.assertIn("关于 widgets 的第一段。", out)
        self.assertIn("关于 gizmos 的第三段。", out)

    def test_modified_block_gets_existing_translation_for_minimal_edit(self):
        ai = MockAI()
        reconcile_restructured_file(
            "ctx.md", self.head, self.base, self.target, ai, REPO_CONFIG, source_mode="commit"
        )
        prompt = ai.prompts()[0]
        # The existing translation of the enclosing section is supplied so the
        # model can minimally edit instead of re-translating from scratch.
        self.assertIn("Existing Chinese translation of this section", prompt)
        self.assertIn("关于 widgets 的第一段。", prompt)
        self.assertIn("关于 gizmos 的第三段。", prompt)


class ReconcileNewSectionTest(unittest.TestCase):
    """A brand-new section has no prior translation, so it is translated fresh
    (no existing-translation reference is supplied)."""

    base = "# T\n\n## A\n\nBody A.\n"
    head = "# T\n\n## A\n\nBody A.\n\n## B\n\nBody B brand new.\n"
    target = "# T\n\n## A {#a}\n\n正文 A。\n"

    def test_new_section_has_no_prior_translation_reference(self):
        ai = MockAI()
        out = reconcile_restructured_file(
            "new.md", self.head, self.base, self.target, ai, REPO_CONFIG, source_mode="commit"
        )
        self.assertIsNotNone(out)
        self.assertEqual(len(ai.calls), 1)
        prompt = ai.prompts()[0]
        self.assertIn("Body B brand new.", prompt)
        self.assertNotIn("Existing Chinese translation of this section", prompt)
        # The unchanged section A keeps its existing translation.
        self.assertIn("正文 A。", out)


class ReconcileVersionMarkMinimalUpdateTest(unittest.TestCase):
    base = (
        "# Variables\n\n"
        "### `tidb_opt_partial_ordered_index_for_topn` "
        "<span class=\"version-mark\">New in v8.5.6</span>\n\n"
        "The first paragraph is unchanged.\n\n"
        "The second paragraph is also unchanged.\n"
    )
    head = base.replace("New in v8.5.6", "New in v8.5.7")
    target = (
        "# 系统变量\n\n"
        "### `tidb_opt_partial_ordered_index_for_topn` "
        "<span class=\"version-mark\">从 v8.5.6 开始引入</span> "
        "{#stable-variable-anchor}\n\n"
        "第一段保持不变。\n\n"
        "第二段也保持不变。\n"
    )

    def test_version_number_change_reuses_entire_target_section_without_ai(self):
        ai = MockAI()
        out = reconcile_restructured_file(
            "version.md", self.head, self.base, self.target, ai, REPO_CONFIG,
            source_mode="commit",
        )

        self.assertIsNotNone(out)
        self.assertEqual(ai.calls, [])
        self.assertIn("<span class=\"version-mark\">从 v8.5.7 开始引入</span>", out)
        self.assertIn("{#stable-variable-anchor}", out)
        self.assertIn("第一段保持不变。\n\n第二段也保持不变。", out)
        self.assertNotIn("v8.5.6", out)

    def test_version_number_change_survives_target_block_drift(self):
        target = self.target.replace(
            "第一段保持不变。\n\n第二段也保持不变。",
            "第一段保持不变。\n第二段也保持不变。",
        )
        ai = MockAI()
        out = reconcile_restructured_file(
            "version-drift.md", self.head, self.base, target, ai, REPO_CONFIG,
            source_mode="commit",
        )

        self.assertIsNotNone(out)
        self.assertEqual(ai.calls, [])
        self.assertIn("从 v8.5.7 开始引入", out)
        self.assertIn("第一段保持不变。\n第二段也保持不变。", out)

    def test_non_version_span_change_translates_only_span_inner_text(self):
        base = self.base.replace("New in v8.5.6", "Experimental")
        head = self.base.replace("New in v8.5.6", "Generally available")
        target = self.target.replace("从 v8.5.6 开始引入", "实验特性")
        ai = MockAI()
        out = reconcile_restructured_file(
            "version-wording.md", head, base, target, ai, REPO_CONFIG,
            source_mode="commit",
        )

        self.assertIsNotNone(out)
        self.assertEqual(
            [content.strip() for content in ai.contents()],
            ["Generally available"],
        )
        self.assertIn(
            "<span class=\"version-mark\">Generally available</span> "
            "{#stable-variable-anchor}",
            out,
        )
        self.assertIn("第一段保持不变。\n\n第二段也保持不变。", out)

    def test_mixed_version_mark_and_regular_section_changes(self):
        base = (
            "# Variables\n\n"
            "## `var_a` <span class=\"version-mark\">New in v8.5.6</span>\n\n"
            "Variable A body.\n\n"
            "## Regular section\n\n"
            "Old regular body.\n"
        )
        head = base.replace("v8.5.6", "v8.5.7").replace(
            "Old regular body.", "Changed regular body."
        )
        target = (
            "# 系统变量\n\n"
            "## `var_a` <span class=\"version-mark\">从 v8.5.6 开始引入</span>\n\n"
            "变量 A 正文保持不变。\n\n"
            "## 普通章节\n\n"
            "原有普通正文。\n"
        )
        ai = MockAI()

        out = reconcile_restructured_file(
            "mixed.md", head, base, target, ai, REPO_CONFIG,
            source_mode="commit",
        )

        self.assertIsNotNone(out)
        self.assertEqual(
            [content.strip() for content in ai.contents()],
            ["Changed regular body."],
        )
        self.assertIn("从 v8.5.7 开始引入", out)
        self.assertIn("变量 A 正文保持不变。", out)
        self.assertIn("## 普通章节", out)
        self.assertIn("Changed regular body.", out)

    def test_semantically_distinct_duplicate_markers_do_not_cross_pair(self):
        base = (
            "# Variables\n\n"
            "## `var_name` <span class=\"version-mark\">New in v7.0</span>\n\n"
            "Same source body.\n\n"
            "## `var_name` <span class=\"version-mark\">Deprecated in v8.0</span>\n\n"
            "Same source body.\n"
        )
        head = base.replace("v7.0", "v7.1").replace("v8.0", "v8.1")
        target = (
            "# 系统变量\n\n"
            "## `var_name` <span class=\"version-mark\">从 v7.0 开始引入</span>\n\n"
            "新增语义对应的译文。\n\n"
            "## `var_name` <span class=\"version-mark\">从 v8.0 开始弃用</span>\n\n"
            "弃用语义对应的译文。\n"
        )
        ai = MockAI()

        out = reconcile_restructured_file(
            "duplicate-markers.md", head, base, target, ai, REPO_CONFIG,
            source_mode="commit",
        )

        self.assertIsNotNone(out)
        self.assertEqual(ai.calls, [])
        self.assertLess(out.index("从 v7.1 开始引入"), out.index("新增语义对应的译文"))
        self.assertLess(out.index("从 v8.1 开始弃用"), out.index("弃用语义对应的译文"))

    def test_nested_span_is_balanced_during_minimal_update(self):
        base = (
            "### `var_a` <span class=\"version-mark\">New in "
            "<span class=\"highlight\">v8.5.6</span></span>\n\nBody.\n"
        )
        head = base.replace("v8.5.6", "v8.5.7")
        target = (
            "### `var_a` <span class=\"version-mark\">从 "
            "<span class=\"highlight\">v8.5.6</span> 开始引入</span>\n\n正文。\n"
        )

        updated = reconcile_version_mark_only_change(
            head, base, target, MockAI(), "English", "Chinese",
            source_mode="commit",
        )

        self.assertEqual(
            updated,
            target.replace("v8.5.6", "v8.5.7"),
        )
        self.assertEqual(updated.count("<span"), updated.count("</span>"))

    def test_ai_span_update_that_changes_target_structure_is_rejected(self):
        base = (
            "### `var_a` <span class=\"version-mark\">Experimental\nstatus</span>\n"
        )
        head = base.replace("Experimental", "Generally available")
        target = (
            "### `var_a` <span class=\"version-mark\">实验性\n状态</span>\n"
        )

        class StructureChangingAI(MockAI):
            def chat_completion(self, messages, temperature=0.1):
                self.calls.append(messages)
                return "正式可用\n## 注入的标题"

        updated = reconcile_version_mark_only_change(
            head, base, target, StructureChangingAI(), "English", "Chinese",
            source_mode="commit",
        )

        self.assertIsNone(updated)

    def test_version_token_replacement_respects_target_boundaries(self):
        change = {
            "base_inner": "New in 8.5.6",
            "head_inner": "New in 8.5.7",
            "target_inner": "对应 18.5.6 client",
        }

        self.assertIsNone(_deterministic_version_mark_inner(change))


class ReconcileWhitespaceTest(unittest.TestCase):
    def test_trailing_whitespace_only_change_is_reused(self):
        base = "# A\n\nIntro.\n\n## B\n\nLine with trailing space.   \n"
        head = "# A\n\nIntro.\n\n## B\n\nLine with trailing space.\n"
        target = "# A\n\n简介。\n\n## B {#b}\n\n带行尾空格的行。\n"
        ai = MockAI()
        out = reconcile_restructured_file("ws.md", head, base, target, ai, REPO_CONFIG, source_mode="commit")
        self.assertIsNotNone(out)
        self.assertEqual(len(ai.calls), 0, "whitespace-only change must be reused")
        self.assertIn("带行尾空格的行。", out)


class ReconcileFallbackTest(unittest.TestCase):
    def test_heading_mismatch_returns_none(self):
        base = "# A\n\nx\n\n## B\n\ny\n"
        head = "# A\n\nx\n\n## B\n\ny\n"
        target = "# A\n\nx\n"
        ai = MockAI()
        out = reconcile_restructured_file("drift.md", head, base, target, ai, REPO_CONFIG, source_mode="commit")
        self.assertIsNone(out)

    def test_section_count_mismatch_returns_none_after_heading_check_passes(self):
        base = "# A\n\nx\n"
        head = "# A\n\nx changed\n"
        target = "目标前言\n\n# 甲\n\nx\n"
        ai = MockAI()

        out = reconcile_restructured_file(
            "preamble-drift.md", head, base, target, ai, REPO_CONFIG,
            source_mode="commit",
        )

        self.assertIsNone(out)
        self.assertEqual(ai.calls, [])

    def test_section_aligned_structure_self_check_failure_returns_none(self):
        base = "# T\n\n## A\n\nA one.\n\nA two.\n"
        head = base + "\n## B\n\nB new.\n"
        target = "# 标题\n\n## 甲\n\n甲一。\n甲二。\n"

        class BadHeadingAI(MockAI):
            def chat_completion(self, messages, temperature=0.1):
                self.calls.append(messages)
                return "缺少标题的新内容。\n"

        ai = BadHeadingAI()
        out = reconcile_restructured_file(
            "bad-output.md", head, base, target, ai, REPO_CONFIG,
            source_mode="commit",
        )

        self.assertIsNone(out)
        self.assertEqual(len(ai.calls), 1)

    def test_section_translation_repairs_heading_level_drift(self):
        base = "# T\n\n## A\n\nA one.\n\nA two.\n"
        head = base + "\n## B\n\nB new.\n"
        target = "# 标题\n\n## 甲\n\n甲一。\n甲二。\n"

        class WrongLevelAI(MockAI):
            def chat_completion(self, messages, temperature=0.1):
                self.calls.append(messages)
                return "### 新章节\n\n新内容。\n"

        out = reconcile_restructured_file(
            "repaired-output.md", head, base, target, WrongLevelAI(), REPO_CONFIG,
            source_mode="commit",
        )

        self.assertIsNotNone(out)
        self.assertIn("## 新章节", out)
        self.assertNotIn("### 新章节", out)

    def test_new_section_restores_trailing_newlines_before_next_heading(self):
        base = (
            "# T\n\n## A\n\nA.\n\n## C\n\nC.\n"
        )
        head = (
            "# T\n\n## A\n\nA.\n\n## B\n\nB new.\n\n## C\n\nC.\n"
        )
        # Target-only intro forces heading-section reconciliation.
        target = (
            "# 标题\n\n目标简介。\n\n## 甲\n\n甲。\n\n## 丙\n\n丙。\n"
        )

        class NoTrailingNewlineAI(MockAI):
            def chat_completion(self, messages, temperature=0.1):
                self.calls.append(messages)
                return "## 乙\n\n乙的新内容。"  # Deliberately no final newline.

        out = reconcile_restructured_file(
            "newline.md", head, base, target, NoTrailingNewlineAI(), REPO_CONFIG,
            source_mode="commit",
        )

        self.assertIsNotNone(out)
        self.assertIn("乙的新内容。\n\n## 丙", out)

    def test_missing_inputs_return_none(self):
        ai = MockAI()
        self.assertIsNone(
            reconcile_restructured_file("x.md", "", "base", "target", ai, REPO_CONFIG)
        )
        self.assertIsNone(
            reconcile_restructured_file("x.md", "head", "base", "", ai, REPO_CONFIG)
        )


class ReconcileBlockDriftBySectionTest(unittest.TestCase):
    def test_heading_after_blockquote_without_blank_line_still_aligns(self):
        base = (
            "# Restore\n\n## Cloud storage\n\n"
            "> **Note:**\n>\n> Existing restriction.\n\n"
            "#### Steps\n\nOld steps.\n"
        )
        head = base.replace("Old steps.", "Updated steps.")
        target = (
            "# 恢复\n\n## 云存储\n\n"
            "> **注意：**\n>\n> 现有限制。\n"
            "#### 步骤\n\n旧步骤。\n"
        )

        out = reconcile_restructured_file(
            "blockquote-heading.md", head, base, target, MockAI(), REPO_CONFIG,
            source_mode="commit",
        )

        self.assertIsNotNone(out)
        self.assertIn("#### 步骤\n\nUpdated steps.", out)

    def test_move_with_addition_reuses_unchanged_section_despite_block_drift(self):
        base = (
            "# Title\n\n"
            "## A\n\nA source paragraph one.\n\nA source paragraph two.\n\n"
            "## B\n\nB old.\n"
        )
        head = (
            "# Title\n\n"
            "## B\n\nB changed.\n\n"
            "## A\n\nA source paragraph one.\n\nA source paragraph two.\n\n"
            "## C\n\nC new.\n"
        )
        # Section A has one target prose block for two source prose blocks, so
        # whole-file block parity is intentionally broken.
        target = (
            "# 标题\n\n"
            "## 甲 {#a}\n\n甲译文第一段。\n甲译文第二段。\n\n"
            "## 乙 {#b}\n\n乙的旧译文。\n"
        )
        ai = MockAI()

        out = reconcile_restructured_file(
            "move.md", head, base, target, ai, REPO_CONFIG, source_mode="commit"
        )

        self.assertIsNotNone(out)
        self.assertLess(out.index("## 乙 {#b}"), out.index("## 甲 {#a}"))
        self.assertEqual(out.count("甲译文第一段。\n甲译文第二段。"), 1)
        self.assertTrue(any("B changed." in content for content in ai.contents()))
        self.assertTrue(any("C new." in content for content in ai.contents()))
        self.assertFalse(any("A source paragraph" in content for content in ai.contents()))

    def test_local_block_mismatch_retranslates_only_changed_section(self):
        base = "# Title\n\n## A\n\nA one.\n\nA two.\n\n## B\n\nB stable.\n"
        head = "# Title\n\n## A\n\nA one changed.\n\nA two.\n\n## B\n\nB stable.\n"
        target = "# 标题\n\n## 甲\n\n甲一。\n甲二。\n\n## 乙\n\n乙保持不变。\n"
        ai = MockAI()

        out = reconcile_restructured_file(
            "local-drift.md", head, base, target, ai, REPO_CONFIG, source_mode="commit"
        )

        self.assertIsNotNone(out)
        self.assertIn("## 乙\n\n乙保持不变。", out)
        self.assertFalse(any("B stable." in content for content in ai.contents()))
        self.assertTrue(any("A one changed." in content for content in ai.contents()))
        self.assertNotIn("Surrounding section, for context only", ai.prompts()[0])

    def test_duplicate_headings_match_exact_section_before_changed_sibling(self):
        base = (
            "# Title\n\n"
            "## Note\n\nAlpha source.\n\n"
            "## Note\n\nBeta source.\n"
        )
        head = (
            "# Title\n\n"
            "## Note\n\nBeta source revised.\n\n"
            "## Note\n\nAlpha source.\n"
        )
        # The target-only intro makes global block counts differ and exercises
        # heading-section reconciliation. The first HEAD Note corresponds to
        # the second BASE Note, despite sharing the same heading text.
        target = (
            "# 标题\n\n目标简介。\n\n"
            "## 注意\n\nAlpha 旧译文。\n\n"
            "## 注意\n\nBeta 旧译文。\n"
        )
        ai = MockAI()

        out = reconcile_restructured_file(
            "duplicate.md", head, base, target, ai, REPO_CONFIG,
            source_mode="commit",
        )

        self.assertIsNotNone(out)
        self.assertEqual(len(ai.calls), 1)
        prompt = ai.prompts()[0]
        self.assertIn("Beta 旧译文。", prompt)
        self.assertNotIn("Alpha 旧译文。", prompt)
        self.assertIn("Alpha 旧译文。", out)


class DetectStructuralChangeTest(unittest.TestCase):
    def test_detects_section_move(self):
        base = "# T\n\n## A\n\na\n\n## B\n\nb\n"
        head = "# T\n\n## B\n\nb\n\n## A\n\na\n"
        self.assertTrue(detect_structural_change(base, head))

    def test_detects_section_move_with_added_and_deleted_headings(self):
        base = "# T\n\n## A\n\na\n\n## B\n\nb\n\n## Removed\n\nold\n"
        head = "# T\n\n## B\n\nb\n\n## Added\n\nnew\n\n## A\n\na\n"
        self.assertTrue(detect_structural_change(base, head))

    def test_add_and_delete_without_reorder_is_not_structural(self):
        base = "# T\n\n## A\n\na\n\n## Removed\n\nold\n\n## B\n\nb\n"
        head = "# T\n\n## A\n\na\n\n## Added\n\nnew\n\n## B\n\nb\n"
        self.assertFalse(detect_structural_change(base, head))

    def test_duplicate_heading_deletion_without_reorder_is_not_structural(self):
        base = (
            "# T\n\n## Note\n\na\n\n## A\n\nx\n\n"
            "## Note\n\nb\n\n## B\n\ny\n"
        )
        head = "# T\n\n## Note\n\na\n\n## A\n\nx\n\n## B\n\ny\n"
        self.assertFalse(detect_structural_change(base, head))

    def test_duplicate_headings_with_surviving_reorder_are_structural(self):
        base = (
            "# T\n\n## Note\n\na\n\n## A\n\nx\n\n"
            "## Note\n\nb\n\n## B\n\ny\n"
        )
        head = (
            "# T\n\n## B\n\ny\n\n## Note\n\na\n\n"
            "## A\n\nx\n\n## Note\n\nb\n"
        )
        self.assertTrue(detect_structural_change(base, head))

    def test_detects_customcontent_change(self):
        base = "# T\n\n## A\n\na\n"
        head = "# T\n\n<CustomContent platform=\"tidb\">\n\n## A\n\na\n\n</CustomContent>\n"
        self.assertTrue(detect_structural_change(base, head))

    def test_plain_content_edit_is_not_structural(self):
        base = "# T\n\n## A\n\nold body\n\n## B\n\nb\n"
        head = "# T\n\n## A\n\nnew body\n\n## B\n\nb\n"
        self.assertFalse(detect_structural_change(base, head))

    def test_version_mark_only_heading_edit_is_not_structural(self):
        base = (
            "# Variables\n\n"
            "### `tidb_opt_partial_ordered_index_for_topn` "
            "<span class=\"version-mark\">New in v8.5.6</span>\n\n"
            "Unchanged body.\n"
        )
        head = base.replace("v8.5.6", "v8.5.7")

        self.assertFalse(detect_structural_change(base, head))

    def test_code_block_hashes_do_not_trigger_move(self):
        base = "# T\n\n## A\n\n```\n# comment\n```\n\n## B\n\nb\n"
        head = "# T\n\n## A\n\n```\n# other comment\n```\n\n## B\n\nb\n"
        self.assertFalse(detect_structural_change(base, head))


if __name__ == "__main__":
    unittest.main()

import sys
import unittest
from pathlib import Path
from unittest import mock


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from markdown_table_utils import (
    extract_markdown_tables,
    find_suspected_table_omissions,
    validate_repaired_table,
)


class MarkdownTableUtilsTest(unittest.TestCase):
    def test_extracts_standard_tables_outside_code_fences(self):
        content = """# Guide

| Concept | Description |
| --- | --- |
| Role | Controls access |

```markdown
| Not | A table |
| --- | --- |
```
"""

        tables = extract_markdown_tables(content)

        self.assertEqual(1, len(tables))
        self.assertEqual("| Concept | Description |", tables[0].header)
        self.assertEqual((2, 2, 2), tables[0].column_counts)

    def test_detects_an_entire_unchanged_table(self):
        source = """| Question | Recommended option |
| --- | --- |
| Can SQL produce the result? | SQL scalar UDF |"""

        omissions = find_suspected_table_omissions(source, source)

        self.assertEqual(1, len(omissions))
        self.assertEqual(
            "entire table is unchanged from the source",
            omissions[0].reason,
        )

    def test_detects_an_untranslated_header_with_translated_body(self):
        source = """| Concept | Description |
| --- | --- |
| Role | Controls access |"""
        target = """| Concept | Description |
| --- | --- |
| 角色 | 控制访问 |"""

        omissions = find_suspected_table_omissions(source, target)

        self.assertEqual(1, len(omissions))
        self.assertEqual(
            "table header is unchanged from the source",
            omissions[0].reason,
        )

    def test_does_not_flag_a_translated_table(self):
        source = """| Concept | Description |
| --- | --- |
| Role | Controls access |"""
        target = """| 概念 | 描述 |
| --- | --- |
| 角色 | 控制访问 |"""

        self.assertEqual([], find_suspected_table_omissions(source, target))

    def test_detects_an_untranslated_natural_language_body_row(self):
        source = """| Concept | Description |
| --- | --- |
| Role | Controls access |
| User | Runs queries |"""
        target = """| 概念 | 描述 |
| --- | --- |
| 角色 | Controls access |
| 用户 | 运行查询 |"""

        omissions = find_suspected_table_omissions(source, target)

        self.assertEqual(1, len(omissions))
        self.assertIn("body row", omissions[0].reason)

    def test_does_not_flag_uppercase_technical_identifiers(self):
        source = """| SQL | API |
| --- | --- |
| DDL | HTTP |"""

        self.assertEqual([], find_suspected_table_omissions(source, source))

    def test_exact_names_and_readme_links_do_not_trigger_body_omissions(self):
        cases = [
            ("Provider", "Location", "Google Cloud", "US Central", "提供商", "位置", "美国中部"),
            ("Credential", "Access", "Owner token", "Access one file system", "凭证", "访问范围", "访问一个文件系统"),
            ("Metric name", "Labels", "Query duration", "Avg, P99, P99-{SQL type}", "指标名称", "标签", "Avg, P99, P99-{SQL type}"),
            ("Language", "Documentation", "Python", "[drive9-py README](https://example.com/README.md)", "语言", "文档", "[drive9-py README](https://example.com/README.md)"),
        ]
        for first, second, name, value, first_zh, second_zh, translated in cases:
            with self.subTest(column=first):
                source = f"| {first} | {second} |\n| --- | --- |\n| {name} | {value} |"
                target = f"| {first_zh} | {second_zh} |\n| --- | --- |\n| {name} | {translated} |"
                self.assertEqual([], find_suspected_table_omissions(source, target))

    def test_name_column_does_not_hide_an_untranslated_description(self):
        source = "| Credential | Description |\n| --- | --- |\n| Owner token | Controls access |"
        target = source.replace("Credential", "凭证").replace("Description", "描述")
        omissions = find_suspected_table_omissions(source, target)
        self.assertEqual(1, len(omissions))
        self.assertIn("body row", omissions[0].reason)

    def test_partial_header_omission_is_still_detected(self):
        source = "| Function | Description |\n| --- | --- |\n| `UPPER(text)` | Uppercase text |"
        target = source.replace("Description", "描述").replace("Uppercase text", "转为大写")
        omissions = find_suspected_table_omissions(source, target)
        self.assertEqual(1, len(omissions))
        self.assertIn("header", omissions[0].reason)

    def test_regular_document_links_are_not_exempt_from_translation(self):
        source = "| Documentation |\n| --- |\n| [Installation guide](/guide.md) |"
        target = source.replace("Documentation", "文档")
        self.assertEqual(1, len(find_suspected_table_omissions(source, target)))

    def test_protected_content_can_reorder_within_a_cell(self):
        source = extract_markdown_tables(
            "| Function | Description |\n| --- | --- |\n"
            "| `POSITION(x IN y)` | SQL equivalent of STRPOS: `a`, `b`, 10 and 20. |"
        )[0]
        candidate = (
            "| 函数 | 描述 |\n| --- | --- |\n"
            "| `POSITION(x IN y)` | STRPOS 的 SQL 写法：`b`、`a`，20 和 10。 |"
        )
        self.assertEqual((candidate, ""), validate_repaired_table(source, candidate))
        for damaged in [candidate.replace("STRPOS", "OTHER"), candidate.replace("`b`", "`a`")]:
            self.assertIn("protected", validate_repaired_table(source, damaged)[1])

    def test_protected_content_cannot_move_between_rows_or_columns(self):
        source = extract_markdown_tables(
            "| Name | Description |\n| --- | --- |\n| `a` | 10 |\n| `b` | 20 |"
        )[0]
        for candidate in [
            "| 名称 | 描述 |\n| --- | --- |\n| `a` | 20 |\n| `b` | 10 |",
            "| 名称 | 描述 |\n| --- | --- |\n| 10 | `a` |\n| `b` | 20 |",
        ]:
            self.assertIn("protected", validate_repaired_table(source, candidate)[1])

    def test_geographic_prose_can_translate_without_relaxing_code_protection(self):
        source = extract_markdown_tables(
            "| Provider | Location | Code |\n| --- | --- | --- |\n"
            "| Microsoft Azure | US Central (`centralus`) | `azure-centralus` |"
        )[0]
        candidate = source.text.replace("Provider", "提供商").replace("Location", "位置").replace("Code", "代码").replace("US Central", "美国中部")
        self.assertEqual((candidate, ""), validate_repaired_table(source, candidate))
        self.assertIn("protected", validate_repaired_table(source, candidate.replace("`centralus`", "`eastus`"))[1])

    def test_header_omission_is_repaired_despite_normal_sentence_reordering(self):
        from file_adder import repair_suspected_untranslated_tables

        source = (
            "| Function | Description |\n| --- | --- |\n"
            "| `POSITION(x IN y)` | SQL equivalent of STRPOS. |\n"
        )
        initial = source.replace("SQL equivalent of STRPOS.", "STRPOS 的 SQL 标准写法。")
        repaired = initial.replace("Function", "函数").replace("Description", "描述")
        ai = mock.Mock()
        ai.chat_completion.return_value = repaired
        output, reasons = repair_suspected_untranslated_tables(source, initial, ai)
        self.assertEqual(repaired, output)
        self.assertEqual([], reasons)
        ai.chat_completion.assert_called_once()

    def test_repair_validation_requires_the_same_shape_and_separator(self):
        source = extract_markdown_tables(
            """| Concept | Description |
| --- | --- |
| Role | Controls access |"""
        )[0]
        candidate = """| 概念 | 描述 |
| --- | --- |
| 角色 | 控制访问 |"""

        repaired, error = validate_repaired_table(source, candidate)

        self.assertEqual(candidate, repaired)
        self.assertEqual("", error)

    def test_repair_validation_rejects_a_changed_separator(self):
        source = extract_markdown_tables(
            """| Concept | Description |
| --- | --- |
| Role | Controls access |"""
        )[0]
        candidate = """| 概念 | 描述 |
| :--- | ---: |
| 角色 | 控制访问 |"""

        repaired, error = validate_repaired_table(source, candidate)

        self.assertIsNone(repaired)
        self.assertIn("separator row", error)

    def test_repair_validation_restores_source_indentation_for_every_row(self):
        source = extract_markdown_tables(
            """    | Concept | Description |
    | --- | --- |
    | Role | Controls access |"""
        )[0]
        candidate = """| 概念 | 描述 |
| --- | --- |
| 角色 | 控制访问 |"""

        repaired, error = validate_repaired_table(source, candidate)

        self.assertEqual(
            """    | 概念 | 描述 |
    | --- | --- |
    | 角色 | 控制访问 |""",
            repaired,
        )
        self.assertEqual("", error)

    def test_repair_validation_rejects_changed_protected_content(self):
        source = extract_markdown_tables(
            """| Setting | Description | Reference | Default |
| --- | --- | --- | --- |
| `MAX_ROWS` | Maximum row count | [Guide](/guide.md) | 10 |"""
        )[0]
        candidates = {
            "inline code": """| 设置 | 描述 | 参考 | 默认值 |
| --- | --- | --- | --- |
| `MIN_ROWS` | 最大行数 | [指南](/guide.md) | 10 |""",
            "link targets": """| 设置 | 描述 | 参考 | 默认值 |
| --- | --- | --- | --- |
| `MAX_ROWS` | 最大行数 | [指南](/other.md) | 10 |""",
            "numbers": """| 设置 | 描述 | 参考 | 默认值 |
| --- | --- | --- | --- |
| `MAX_ROWS` | 最大行数 | [指南](/guide.md) | 20 |""",
        }

        for protected_kind, candidate in candidates.items():
            with self.subTest(protected_kind=protected_kind):
                repaired, error = validate_repaired_table(source, candidate)

                self.assertIsNone(repaired)
                self.assertIn("protected", error)


if __name__ == "__main__":
    unittest.main()

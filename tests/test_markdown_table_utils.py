import sys
import unittest
from pathlib import Path


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

import os
import sys
import unittest
from pathlib import Path
from unittest import mock


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from section_matcher import (
    batch_match_sections_with_ai,
    get_corresponding_sections,
    get_corresponding_sections_json,
    is_direct_match_candidate,
    is_system_variable_or_config,
    match_source_diff_to_target,
    parse_ai_response,
    process_modified_or_deleted_section,
)


class FakeAIClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def chat_completion(self, messages, temperature, max_tokens):
        self.prompts.append(messages[0]["content"])
        if not self.responses:
            raise AssertionError("No fake AI responses left")
        return self.responses.pop(0)


class SectionMatcherRegressionTest(unittest.TestCase):
    def capture_ai_mapping_prompts(self):
        ai_client = FakeAIClient(
            [
                "```\n## Target\n```",
                "```\n## Target\n```",
                '```json\n{"modified_1": "## Target"}\n```',
            ]
        )

        get_corresponding_sections(
            ["## Source"],
            ["## Target"],
            ai_client,
            "English",
            "Japanese",
        )
        get_corresponding_sections(
            {"modified_1": "## Source"},
            ["## Target"],
            ai_client,
            "English",
            "Japanese",
            source_base_hierarchy={1: "## Source"},
            source_head_hierarchy={1: "## Source"},
            source_diff_dict={
                "modified_1": {
                    "operation": "modified",
                    "new_content": "## Source",
                }
            },
            source_mode="commit",
        )
        get_corresponding_sections_json(
            {"modified_1": "## Source"},
            {1: "## Target"},
            ai_client,
            {
                "source_language": "English",
                "target_language": "Japanese",
            },
            source_base_hierarchy={1: "## Source"},
            source_head_hierarchy={1: "## Source"},
            source_diff_dict={
                "modified_1": {
                    "operation": "modified",
                    "new_content": "## Source",
                }
            },
            source_mode="commit",
        )
        return ai_client.prompts

    def test_ai_mapping_prompts_use_configured_product_name(self):
        with mock.patch.dict(os.environ, {"PRODUCT": "ExampleDB"}, clear=False):
            prompts = self.capture_ai_mapping_prompts()

        self.assertEqual(len(prompts), 3)
        for prompt in prompts:
            self.assertIn("documentation for ExampleDB", prompt)
            self.assertNotIn("documentation for TiDB", prompt)

    def test_ai_mapping_prompts_use_product_neutral_wording_when_unset(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            prompts = self.capture_ai_mapping_prompts()

        self.assertEqual(len(prompts), 3)
        for prompt in prompts:
            self.assertIn("English and Japanese documentation.", prompt)
            self.assertNotIn("documentation for .", prompt)

    def build_target_fixture(self):
        target_lines = [
            "# 集成 TiDB 向量搜索与 Jina AI Embeddings API",
            "",
            "## 前置条件",
            "",
            "## 运行示例应用",
            "",
            "### 步骤 1. 克隆仓库",
            "",
            "### 步骤 2. 创建虚拟环境",
            "",
            "### 步骤 3. 安装依赖",
            "",
            "### 步骤 4. 配置环境变量",
            "旧内容",
            "",
            "### 步骤 5. 运行示例",
            "",
            "## 示例代码片段",
            "",
            "### 从 Jina AI 获取 embedding",
            "",
            "### 连接 TiDB 集群",
            "旧连接内容",
            "",
            "### 定义向量表结构",
            "旧结构内容",
            "",
            "### 使用 Jina AI 生成 embedding 并存储到 TiDB",
            "",
            "### 在 TiDB 中基于 Jina AI embedding 进行语义搜索",
            "",
            "## 另请参阅",
        ]
        target_hierarchy = {
            "1": "# 集成 TiDB 向量搜索与 Jina AI Embeddings API",
            "3": "## 前置条件",
            "5": "## 运行示例应用",
            "7": "## 运行示例应用 > ### 步骤 1. 克隆仓库",
            "9": "## 运行示例应用 > ### 步骤 2. 创建虚拟环境",
            "11": "## 运行示例应用 > ### 步骤 3. 安装依赖",
            "13": "## 运行示例应用 > ### 步骤 4. 配置环境变量",
            "16": "## 运行示例应用 > ### 步骤 5. 运行示例",
            "18": "## 示例代码片段",
            "20": "## 示例代码片段 > ### 从 Jina AI 获取 embedding",
            "22": "## 示例代码片段 > ### 连接 TiDB 集群",
            "25": "## 示例代码片段 > ### 定义向量表结构",
            "28": "## 示例代码片段 > ### 使用 Jina AI 生成 embedding 并存储到 TiDB",
            "30": "## 示例代码片段 > ### 在 TiDB 中基于 Jina AI embedding 进行语义搜索",
            "32": "## 另请参阅",
        }
        return target_lines, target_hierarchy

    def test_step_heading_falls_back_by_level_and_step_number(self):
        repo_config = {
            "source_language": "English",
            "target_language": "Chinese",
        }
        target_hierarchy = {
            "29": "# 集成 TiDB 向量搜索与 Jina AI Embeddings API > ## 运行示例应用",
            "45": "# 集成 TiDB 向量搜索与 Jina AI Embeddings API > ## 运行示例应用 > ### 步骤 3. 连接到 TiDB",
            "59": "# 集成 TiDB 向量搜索与 Jina AI Embeddings API > ## 运行示例应用 > ### 步骤 4. 配置环境变量",
            "127": "# 集成 TiDB 向量搜索与 Jina AI Embeddings API > ## 运行示例应用 > ### 步骤 5. 运行示例",
        }

        result = process_modified_or_deleted_section(
            "modified_59",
            "## Run the sample app > ### Step 4. Configure the environment variables",
            target_hierarchy,
            [],
            ai_client=None,
            repo_config=repo_config,
            max_non_system_sections=120,
        )

        self.assertEqual(
            result,
            {
                "target_line": "59",
                "target_hierarchy": "# 集成 TiDB 向量搜索与 Jina AI Embeddings API > ## 运行示例应用 > ### 步骤 4. 配置环境变量",
            },
        )

    def test_batch_match_uses_source_order_for_duplicate_step_titles(self):
        repo_config = {
            "source_language": "English",
            "target_language": "Chinese",
        }
        target_lines = [
            "# 文档标题",
            "",
            "## 使用 AWS 设置私有端点（端点共享模型）",
            "",
            "### Step 2. 创建 AWS interface 端点",
            "",
            "## 使用 AWS 设置私有端点（端点独占模型）",
            "",
            "### Step 2. 创建 AWS interface 端点",
            "",
        ]
        target_hierarchy = {
            "1": "# 文档标题",
            "3": "## 使用 AWS 设置私有端点（端点共享模型）",
            "5": "## 使用 AWS 设置私有端点（端点共享模型） > ### Step 2. 创建 AWS interface 端点",
            "7": "## 使用 AWS 设置私有端点（端点独占模型）",
            "9": "## 使用 AWS 设置私有端点（端点独占模型） > ### Step 2. 创建 AWS interface 端点",
        }
        source_diff_dict = {
            "modified_50": {
                "operation": "modified",
                "original_hierarchy": "## Set up a private endpoint with AWS (endpoint shared model) > ### Step 2. Create an AWS interface endpoint",
                "old_content": "old",
                "new_content": "new",
            },
            "modified_100": {
                "operation": "modified",
                "original_hierarchy": "## Set up a private endpoint with AWS (endpoint exclusive model) > ### Step 2. Create an AWS interface endpoint",
                "old_content": "old",
                "new_content": "new",
            },
        }
        ai_client = FakeAIClient(
            [
                "```\n### Step 2. 创建 AWS interface 端点\n### Step 2. 创建 AWS interface 端点\n```"
            ]
        )

        result = match_source_diff_to_target(
            source_diff_dict,
            target_hierarchy,
            target_lines,
            ai_client,
            repo_config,
            max_non_system_sections=120,
        )

        self.assertEqual(result["modified_50"]["target_line"], "5")
        self.assertEqual(result["modified_100"]["target_line"], "9")

    def test_batch_match_added_sections_disambiguated_by_forward_logic(self):
        """Added sections allow line reuse, so used_lines can't disambiguate.

        Forward logic (previous_target_line) must advance through duplicates.
        """
        repo_config = {
            "source_language": "English",
            "target_language": "Chinese",
        }
        target_lines = [
            "# Title",
            "",
            "## Parent A",
            "",
            "### Configure",
            "",
            "## Parent B",
            "",
            "### Configure",
            "",
        ]
        target_hierarchy = {
            "1": "# Title",
            "3": "## Parent A",
            "5": "## Parent A > ### Configure",
            "7": "## Parent B",
            "9": "## Parent B > ### Configure",
        }
        source_diff_dict = {
            "added_10": {
                "operation": "added",
                "original_hierarchy": "## Parent A > ### Configure",
                "old_content": "",
                "new_content": "new",
            },
            "added_20": {
                "operation": "added",
                "original_hierarchy": "## Parent B > ### Configure",
                "old_content": "",
                "new_content": "new",
            },
        }
        ai_client = FakeAIClient(
            [
                "```\n### Configure\n### Configure\n```"
            ]
        )

        result = match_source_diff_to_target(
            source_diff_dict,
            target_hierarchy,
            target_lines,
            ai_client,
            repo_config,
            max_non_system_sections=120,
        )

        self.assertEqual(result["added_10"]["target_line"], "5")
        self.assertEqual(result["added_20"]["target_line"], "9")

    def test_match_source_diff_to_target_batches_non_direct_sections_once(self):
        repo_config = {
            "source_language": "English",
            "target_language": "Chinese",
        }
        target_lines, target_hierarchy = self.build_target_fixture()
        source_diff_dict = {
            "modified_16": {
                "operation": "modified",
                "original_hierarchy": "## Prerequisites",
                "old_content": "old",
                "new_content": "new",
            },
            "modified_59": {
                "operation": "modified",
                "original_hierarchy": "## Run the sample app > ### Step 4. Configure the environment variables",
                "old_content": "old",
                "new_content": "new",
            },
            "added_178": {
                "operation": "added",
                "original_hierarchy": "## Sample code snippets > ### Define the vector table schema",
                "old_content": "",
                "new_content": "new",
            },
            "deleted_178": {
                "operation": "deleted",
                "original_hierarchy": "## Sample code snippets > ### Connect to the TiDB cluster",
                "old_content": "old",
                "new_content": "",
            },
        }
        ai_client = FakeAIClient(
            [
                "\n".join(
                    [
                        "```",
                        "## 前置条件",
                        "运行示例应用 > ### 步骤 4. 配置环境变量",
                        "## 示例代码片段 > ### 定义向量表结构",
                        "## 示例代码片段 > ### 连接 TiDB 集群",
                        "```",
                    ]
                )
            ]
        )

        result = match_source_diff_to_target(
            source_diff_dict,
            target_hierarchy,
            target_lines,
            ai_client,
            repo_config,
            max_non_system_sections=120,
        )

        self.assertEqual(len(ai_client.prompts), 1)
        prompt = ai_client.prompts[0]
        self.assertIn("## Prerequisites", prompt)
        self.assertIn("## Run the sample app > ### Step 4. Configure the environment variables", prompt)
        self.assertIn("## Sample code snippets > ### Define the vector table schema", prompt)
        self.assertIn("## Sample code snippets > ### Connect to the TiDB cluster", prompt)
        self.assertEqual(set(result.keys()), set(source_diff_dict.keys()))
        self.assertEqual(result["modified_16"]["target_line"], "3")
        self.assertEqual(result["modified_59"]["target_line"], "13")
        self.assertEqual(result["added_178"]["target_line"], "25")
        self.assertEqual(result["added_178"]["insertion_type"], "before_reference")
        self.assertEqual(result["deleted_178"]["target_line"], "22")

    def test_added_section_can_reuse_modified_target_anchor(self):
        repo_config = {
            "source_language": "English",
            "target_language": "Chinese",
        }
        target_lines = ["# 标题", "", "## 管理组织访问", ""]
        target_hierarchy = {
            "1": "# 标题",
            "3": "## 管理组织访问",
        }
        source_diff_dict = {
            "added_146": {
                "operation": "added",
                "original_hierarchy": "## Manage organization access",
                "old_content": "",
                "new_content": "### Instance roles\n\nnew",
            },
            "modified_172": {
                "operation": "modified",
                "original_hierarchy": "## Manage organization access",
                "old_content": "## Manage organization access\n\nold",
                "new_content": "## Manage organization access\n\nnew",
            },
        }
        ai_client = FakeAIClient(
            [
                "\n".join(
                    [
                        "```",
                        "## 管理组织访问",
                        "## 管理组织访问",
                        "```",
                    ]
                )
            ]
        )

        result = match_source_diff_to_target(
            source_diff_dict,
            target_hierarchy,
            target_lines,
            ai_client,
            repo_config,
        )

        self.assertEqual(len(ai_client.prompts), 1)
        self.assertEqual(result["added_146"]["target_line"], "3")
        self.assertEqual(result["added_146"]["insertion_type"], "before_reference")
        self.assertEqual(result["modified_172"]["target_line"], "3")

    def test_match_source_diff_to_target_falls_back_when_batch_result_is_incomplete(self):
        repo_config = {
            "source_language": "English",
            "target_language": "Chinese",
        }
        target_lines, target_hierarchy = self.build_target_fixture()
        source_diff_dict = {
            "modified_16": {
                "operation": "modified",
                "original_hierarchy": "## Prerequisites",
                "old_content": "old",
                "new_content": "new",
            },
            "modified_59": {
                "operation": "modified",
                "original_hierarchy": "## Run the sample app > ### Step 4. Configure the environment variables",
                "old_content": "old",
                "new_content": "new",
            },
        }
        ai_client = FakeAIClient(
            [
                "```\n## 前置条件\n```",
                "```\n## 前置条件\n```",
                "```\n运行示例应用 > ### 步骤 4. 配置环境变量\n```",
            ]
        )

        result = match_source_diff_to_target(
            source_diff_dict,
            target_hierarchy,
            target_lines,
            ai_client,
            repo_config,
            max_non_system_sections=120,
        )

        self.assertEqual(len(ai_client.prompts), 3)
        self.assertEqual(set(result.keys()), {"modified_16", "modified_59"})
        self.assertEqual(result["modified_16"]["target_line"], "3")
        self.assertEqual(result["modified_59"]["target_line"], "13")

    def test_bottom_modified_marker_uses_source_dict_hierarchy_for_ai_matching(self):
        repo_config = {
            "source_language": "English",
            "target_language": "Chinese",
        }
        target_lines = [
            "# TiDB 向量搜索集成概览",
            "",
            "## 对象关系映射 (ORM) 库",
            "",
            "旧内容",
        ]
        target_hierarchy = {
            "1": "# TiDB 向量搜索集成概览",
            "3": "## 对象关系映射 (ORM) 库",
        }
        source_diff_dict = {
            "modified_39": {
                "operation": "modified",
                "original_hierarchy": "bottom-modified-39",
                "matching_hierarchy": "## Object Relational Mapping (ORM) libraries",
                "old_content": "old body without heading",
                "new_content": "new body without heading",
            },
        }
        ai_client = FakeAIClient(["```\n## 对象关系映射 (ORM) 库\n```"])

        result = match_source_diff_to_target(
            source_diff_dict,
            target_hierarchy,
            target_lines,
            ai_client,
            repo_config,
            max_non_system_sections=120,
        )

        self.assertFalse(is_direct_match_candidate("bottom-modified-39"))
        combined_prompts = "\n\n".join(ai_client.prompts)
        self.assertIn("## Object Relational Mapping (ORM) libraries", combined_prompts)
        self.assertNotIn("bottom-modified-39", combined_prompts)
        self.assertEqual(result["modified_39"]["target_line"], "3")
        self.assertEqual(result["modified_39"]["source_original_hierarchy"], "bottom-modified-39")

    def test_bottom_modified_source_dict_direct_candidates_use_direct_matching(self):
        repo_config = {
            "source_language": "English",
            "target_language": "Chinese",
        }
        cases = [
            ("# English document title", "# 中文文档标题"),
            ("## `tidb_enable_example`", "## `tidb_enable_example`"),
        ]

        for source_heading, target_heading in cases:
            with self.subTest(source_heading=source_heading):
                target_lines = [target_heading, "", "旧内容"]
                target_hierarchy = {"1": target_heading}
                source_diff_dict = {
                    "modified_39": {
                        "operation": "modified",
                        "original_hierarchy": "bottom-modified-39",
                        "matching_hierarchy": source_heading,
                        "old_content": "old body without heading",
                        "new_content": "new body without heading",
                    },
                }
                ai_client = FakeAIClient([])

                result = match_source_diff_to_target(
                    source_diff_dict,
                    target_hierarchy,
                    target_lines,
                    ai_client,
                    repo_config,
                    max_non_system_sections=120,
                )

                self.assertEqual(ai_client.prompts, [])
                self.assertEqual(result["modified_39"]["target_line"], "1")

    def test_commit_mode_direct_system_variable_match_preserves_target_level(self):
        repo_config = {
            "source_language": "English",
            "target_language": "Chinese",
            "source_mode": "commit",
        }
        target_lines = [
            "# 系统变量",
            "",
            "## 变量参考",
            "",
            '### tidb_ddl_disk_quota <span class="version-mark">从 v6.3.0 版本开始引入</span>',
            "",
            "> **注意：**",
            ">",
            "> 旧链接",
        ]
        target_hierarchy = {
            "1": "# 系统变量",
            "3": "## 变量参考",
            "5": '## 变量参考 > ### tidb_ddl_disk_quota <span class="version-mark">从 v6.3.0 版本开始引入</span>',
        }
        source_diff_dict = {
            "modified_1600": {
                "operation": "modified",
                "original_hierarchy": '## Variable reference > ### tidb_ddl_disk_quota <span class="version-mark">New in v6.3.0</span>',
                "old_content": '### tidb_ddl_disk_quota <span class="version-mark">New in v6.3.0</span>\n\nold link',
                "new_content": '### tidb_ddl_disk_quota <span class="version-mark">New in v6.3.0</span>\n\nnew link',
            },
        }
        ai_client = FakeAIClient([])

        result = match_source_diff_to_target(
            source_diff_dict,
            target_hierarchy,
            target_lines,
            ai_client,
            repo_config,
        )

        self.assertEqual(ai_client.prompts, [])
        self.assertEqual(result["modified_1600"]["target_line"], "5")
        self.assertEqual(
            result["modified_1600"]["target_hierarchy"],
            '## 变量参考 > ### tidb_ddl_disk_quota <span class="version-mark">从 v6.3.0 版本开始引入</span>',
        )

    def test_bottom_modified_marker_without_source_dict_hierarchy_falls_back_to_content(self):
        repo_config = {
            "source_language": "English",
            "target_language": "Chinese",
        }
        target_lines = [
            "# TiDB 向量搜索集成概览",
            "",
            "## 对象关系映射 (ORM) 库",
            "",
            "旧内容",
        ]
        target_hierarchy = {
            "1": "# TiDB 向量搜索集成概览",
            "3": "## 对象关系映射 (ORM) 库",
        }
        source_diff_dict = {
            "modified_39": {
                "operation": "modified",
                "original_hierarchy": "bottom-modified-39",
                "old_content": "## Object Relational Mapping (ORM) libraries\n\nold",
                "new_content": "## Object Relational Mapping (ORM) libraries\n\nnew",
            },
        }
        ai_client = FakeAIClient(["```\n## 对象关系映射 (ORM) 库\n```"])

        result = match_source_diff_to_target(
            source_diff_dict,
            target_hierarchy,
            target_lines,
            ai_client,
            repo_config,
            max_non_system_sections=120,
        )

        combined_prompts = "\n\n".join(ai_client.prompts)
        self.assertIn("## Object Relational Mapping (ORM) libraries", combined_prompts)
        self.assertNotIn("bottom-modified-39", combined_prompts)
        self.assertEqual(result["modified_39"]["target_line"], "3")

    def test_bottom_modified_marker_without_source_dict_or_content_heading_does_not_direct_match(self):
        repo_config = {
            "source_language": "English",
            "target_language": "Chinese",
        }
        target_lines = ["# 文档标题", "", "## bottom-modified-39", "", "旧内容"]
        target_hierarchy = {
            "1": "# 文档标题",
            "3": "## bottom-modified-39",
        }
        source_diff_dict = {
            "modified_39": {
                "operation": "modified",
                "original_hierarchy": "bottom-modified-39",
                "old_content": "old body without a heading",
                "new_content": "new body without a heading",
            },
        }
        ai_client = FakeAIClient(["", ""])

        result = match_source_diff_to_target(
            source_diff_dict,
            target_hierarchy,
            target_lines,
            ai_client,
            repo_config,
            max_non_system_sections=120,
        )

        self.assertFalse(is_direct_match_candidate("bottom-modified-39"))
        self.assertFalse(is_system_variable_or_config("bottom-modified-39"))
        self.assertEqual(result, {})
        self.assertEqual(len(ai_client.prompts), 2)

    def test_parse_ai_response_prefers_code_block_content(self):
        response = "\n".join(
            [
                "Here are the matching sections:",
                "```",
                "## 前置条件",
                "运行示例应用 > ### 步骤 4. 配置环境变量",
                "```",
                "Hope this helps.",
            ]
        )

        parsed = parse_ai_response(response)

        self.assertEqual(
            parsed,
            ["## 前置条件", "运行示例应用 > ### 步骤 4. 配置环境变量"],
        )

    def test_batch_match_sections_with_ai_keeps_list_shape_on_failure(self):
        repo_config = {
            "source_language": "English",
            "target_language": "Chinese",
        }

        ai_sections, failed_keys = batch_match_sections_with_ai(
            {"modified_16": "## Prerequisites"},
            {"3": "## 前置条件"},
            ai_client=FakeAIClient([""]),
            repo_config=repo_config,
        )

        self.assertEqual(ai_sections, [])
        self.assertEqual(failed_keys, ["modified_16"])

    def test_pr_mode_retries_with_full_context_without_hard_rejecting(self):
        repo_config = {
            "source_language": "English",
            "target_language": "Chinese",
            "source_mode": "pr",
        }
        target_lines = [
            "# 监控 TiDB",
            "",
            "## 监控指标",
            "",
            "### 指标页面",
            "",
        ]
        target_hierarchy = {
            "1": "# 监控 TiDB",
            "3": "## 监控指标",
            "5": "## 监控指标 > ### 指标页面",
        }
        source_diff_dict = {
            "modified_79": {
                "operation": "modified",
                "original_hierarchy": "## Monitoring metrics",
                "old_content": "## Monitoring metrics\n\nold",
                "new_content": "## Monitoring metrics\n\nnew",
            },
        }
        ai_client = FakeAIClient(
            [
                "```\n## 监控指标 > ### 指标页面\n```",
                '```json\n{"modified_79": "## 监控指标"}\n```',
            ]
        )

        result = match_source_diff_to_target(
            source_diff_dict,
            target_hierarchy,
            target_lines,
            ai_client,
            repo_config,
            source_base_hierarchy={6: "# Monitor TiDB", 79: "## Monitoring metrics"},
            source_head_hierarchy={6: "# Monitor TiDB", 79: "## Monitoring metrics"},
        )

        self.assertEqual(result["modified_79"]["target_line"], "3")
        self.assertEqual(len(ai_client.prompts), 2)
        self.assertNotIn("Full English BASE section structure", ai_client.prompts[0])
        self.assertIn("This is PR mode", ai_client.prompts[1])
        self.assertIn("Full English BASE section structure", ai_client.prompts[1])
        self.assertIn("Return a JSON object", ai_client.prompts[1])

    def test_full_context_provider_is_not_called_without_mapping_risk(self):
        repo_config = {
            "source_language": "English",
            "target_language": "Chinese",
            "source_mode": "commit",
        }
        target_lines = ["# 监控 TiDB", "", "## 监控指标", ""]
        target_hierarchy = {
            "1": "# 监控 TiDB",
            "3": "## 监控指标",
        }
        source_diff_dict = {
            "modified_79": {
                "operation": "modified",
                "original_hierarchy": "## Monitoring metrics",
                "old_content": "## Monitoring metrics\n\nold",
                "new_content": "## Monitoring metrics\n\nnew",
            },
        }
        ai_client = FakeAIClient(["```\n## 监控指标\n```"])

        def fail_if_called():
            raise AssertionError("source hierarchy provider should be lazy")

        result = match_source_diff_to_target(
            source_diff_dict,
            target_hierarchy,
            target_lines,
            ai_client,
            repo_config,
            source_hierarchy_provider=fail_if_called,
        )

        self.assertEqual(result["modified_79"]["target_line"], "3")
        self.assertEqual(len(ai_client.prompts), 1)
        self.assertNotIn("Full English BASE section structure", ai_client.prompts[0])

    def test_commit_mode_retries_full_context_then_rejects_level_mismatch(self):
        repo_config = {
            "source_language": "English",
            "target_language": "Chinese",
            "source_mode": "commit",
        }
        target_lines = [
            "# 监控 TiDB",
            "",
            "## 监控指标",
            "",
            "### 指标页面",
            "",
        ]
        target_hierarchy = {
            "1": "# 监控 TiDB",
            "3": "## 监控指标",
            "5": "## 监控指标 > ### 指标页面",
        }
        source_diff_dict = {
            "modified_79": {
                "operation": "modified",
                "original_hierarchy": "## Monitoring metrics",
                "old_content": "## Monitoring metrics\n\nold",
                "new_content": "## Monitoring metrics\n\nnew",
            },
        }
        ai_client = FakeAIClient(
            [
                "```\n## 监控指标 > ### 指标页面\n```",
                '```json\n{"modified_79": "## 监控指标 > ### 指标页面"}\n```',
            ]
        )

        result = match_source_diff_to_target(
            source_diff_dict,
            target_hierarchy,
            target_lines,
            ai_client,
            repo_config,
            source_base_hierarchy={6: "# Monitor TiDB", 79: "## Monitoring metrics"},
            source_head_hierarchy={6: "# Monitor TiDB", 79: "## Monitoring metrics"},
        )

        self.assertEqual(result, {})
        self.assertEqual(len(ai_client.prompts), 2)
        self.assertNotIn("Full English BASE section structure", ai_client.prompts[0])
        self.assertIn("This is commit-based mode", ai_client.prompts[1])
        self.assertIn("Return a JSON object", ai_client.prompts[1])

    def test_commit_mode_allows_level_mismatch_when_source_heading_level_changed(self):
        repo_config = {
            "source_language": "English",
            "target_language": "Chinese",
            "source_mode": "commit",
        }
        target_lines = [
            "# SQL",
            "",
            "## Statements",
            "",
            "### SQL statement",
            "",
        ]
        target_hierarchy = {
            "1": "# SQL",
            "3": "## Statements",
            "5": "## Statements > ### SQL statement",
        }
        source_diff_dict = {
            "modified_20": {
                "operation": "modified",
                "original_hierarchy": "## SQL statement",
                "old_content": "## SQL statement\n\nold",
                "new_content": "### SQL statement\n\nnew",
            },
        }
        ai_client = FakeAIClient(["```\n## Statements > ### SQL statement\n```"])

        result = match_source_diff_to_target(
            source_diff_dict,
            target_hierarchy,
            target_lines,
            ai_client,
            repo_config,
            source_base_hierarchy={20: "## SQL statement"},
            source_head_hierarchy={20: "### SQL statement"},
        )

        self.assertEqual(result["modified_20"]["target_line"], "5")
        self.assertEqual(len(ai_client.prompts), 1)
        self.assertNotIn("Full English BASE section structure", ai_client.prompts[0])

    def test_commit_mode_rejects_duplicate_modified_target_lines(self):
        repo_config = {
            "source_language": "English",
            "target_language": "Chinese",
            "source_mode": "commit",
        }
        target_lines = ["# 标题", "", "## 同一节", ""]
        target_hierarchy = {
            "1": "# 标题",
            "3": "## 同一节",
        }
        source_diff_dict = {
            "modified_10": {
                "operation": "modified",
                "original_hierarchy": "## First section",
                "old_content": "## First section\n\nold",
                "new_content": "## First section\n\nnew",
            },
            "modified_20": {
                "operation": "modified",
                "original_hierarchy": "## Second section",
                "old_content": "## Second section\n\nold",
                "new_content": "## Second section\n\nnew",
            },
        }
        ai_client = FakeAIClient(
            [
                "```\n## 同一节\n## 同一节\n```",
                "```\n## 同一节\n```",
                '```json\n{"modified_10": "## 同一节", "modified_20": "## 同一节"}\n```',
            ]
        )

        result = match_source_diff_to_target(
            source_diff_dict,
            target_hierarchy,
            target_lines,
            ai_client,
            repo_config,
            source_base_hierarchy={10: "## First section", 20: "## Second section"},
            source_head_hierarchy={10: "## First section", 20: "## Second section"},
        )

        self.assertEqual(result, {})
        self.assertEqual(len(ai_client.prompts), 3)
        self.assertIn("Full English BASE section structure", ai_client.prompts[2])
        self.assertIn("Return a JSON object", ai_client.prompts[2])


if __name__ == "__main__":
    unittest.main()

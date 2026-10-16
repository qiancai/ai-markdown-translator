"""Exercise commit-sync Index routing through the real snapshot processor."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import commit_sync_workflow as workflow


INDEX_PATH = "tidb-cloud/byoc/_index.md"
SOURCE = '''---
title: TiDB Cloud Documentation
---

<LearningPathContainer platform="tidb-cloud" title="TiDB Cloud BYOC">
<LearningPath label="Learn" icon="cloud1">
[Overview](https://docs.pingcap.com/tidbcloud/byoc-onboarding-overview/?plan=byoc#start)
[CLI](/ai/ti/ti-quick-start.md)
[GitHub](https://github.com/pingcap/docs)
</LearningPath>
</LearningPathContainer>
'''


class IndexAI:
    def __init__(self, fail=False, incomplete=False):
        self.calls = []
        self.fail = fail
        self.incomplete = incomplete

    def chat_completion(self, messages, temperature=0.1):
        prompt = messages[0]["content"]
        self.calls.append(prompt)
        # A whole-file translation prompt is a routing failure in this test.
        data, _ = json.JSONDecoder().raw_decode(prompt.split("Input lines to translate:\n", 1)[1])
        if self.fail:
            return "{}"
        for fields in data.values():
            for key, value in fields.items():
                fields[key] = {"Learn": "学习", "Overview": "概览"}.get(value, value)
        response = json.dumps(data, ensure_ascii=False)
        if self.incomplete:
            from ai_client import CompletionText

            return CompletionText(response, status="incomplete", reason="max_output_tokens")
        return response


def run_group(tmp_path, case, client, path=INDEX_PATH, content=SOURCE, dedicated=True):
    changed = []
    added = {}
    if case in {"added", "renamed"}:
        changed = [SimpleNamespace(filename=path, status=case, previous_filename=None)]
        added = {path: content}
    analyzed = ({}, {}, {}, added, [], {}, {}, [], [], [], set(), {}, {})
    stats = workflow.TranslationStats()
    with mock.patch.object(workflow, "TARGET_REPO_PATH", str(tmp_path)), mock.patch.object(
        workflow, "analyze_source_changes", return_value=analyzed,
    ), mock.patch.object(
        workflow, "get_full_translation_source_content", return_value=content,
    ), mock.patch.object(
        workflow, "collect_toc_scope_added_files", return_value={path} if case == "scope" else set(),
    ), mock.patch.object(workflow, "git_add_changes"), mock.patch.object(
        workflow, "git_add_successful_task_changes",
    ), mock.patch.object(
        workflow, "validate_successful_translation_structures",
    ), mock.patch.object(workflow, "process_added_files", return_value=True) as ordinary:
        result = workflow.process_translation_group(
            "test", source_files_translation_mode="full" if case == "full" else "incremental",
            source_files="" if case == "scope" else path, source_folder="",
            diff_context={"mode": "commit", "head_ref": "head", "base_ref": "base"},
            filtered_changed_files=changed, pr_diff="", github_client=None, ai_client=client,
            repo_config={"source_language": "English", "target_language": "Chinese",
                         "target_local_path": str(tmp_path)},
            repo_configs={}, glossary_matcher=None, commit_ignore_files=[],
            translation_stats=stats, source_full_files=path if case == "bootstrap" else "",
        )
    if dedicated:
        ordinary.assert_not_called()
    else:
        ordinary.assert_called_once()
    return result, stats


@pytest.mark.parametrize("case", ["full", "bootstrap", "scope", "added", "renamed"])
def test_new_and_full_indexes_use_snapshot_translation(tmp_path, case):
    client = IndexAI()
    result, stats = run_group(tmp_path, case, client)
    output = (tmp_path / INDEX_PATH).read_text()
    assert 'label="学习"' in output
    assert "[概览](https://docs.pingcap.com/zh/tidbcloud/byoc-onboarding-overview/?plan=byoc#start)" in output
    assert "[CLI](/ai/ti/ti-quick-start.md)" in output
    assert "[GitHub](https://github.com/pingcap/docs)" in output
    assert result["counts"]["index_files"] == 1
    assert result["counts"]["added_files"] == 0
    assert result["successful_file_paths"] == {INDEX_PATH}
    assert not stats.failed
    assert len(client.calls) == 1


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("failure", ["invalid_json", "incomplete"])
def test_failed_bootstrap_index_is_not_written_or_retried_as_whole_file(tmp_path, existing, failure):
    target = tmp_path / INDEX_PATH
    if existing:
        target.parent.mkdir(parents=True)
        target.write_text("Existing translation\n")
    result, stats = run_group(
        tmp_path, "bootstrap", IndexAI(fail=failure == "invalid_json", incomplete=failure == "incomplete"),
    )
    assert len(stats.failed) == 1
    assert result["successful_file_paths"] == set()
    if existing:
        assert target.read_text() == "Existing translation\n"
    else:
        assert not target.exists()


def test_full_translation_retranslates_existing_index_text(tmp_path):
    target = tmp_path / INDEX_PATH
    target.parent.mkdir(parents=True)
    target.write_text(SOURCE.replace('label="Learn"', 'label="旧标签"'))
    client = IndexAI()
    run_group(tmp_path, "full", client)
    assert 'label="学习"' in target.read_text()
    assert "旧标签" not in target.read_text()
    assert client.calls


@pytest.mark.parametrize("path", ["guide.md", "ai/_index.md"])
def test_heading_based_files_remain_in_the_ordinary_translation_queue(tmp_path, path):
    client = IndexAI()
    result, stats = run_group(
        tmp_path, "full", client, path=path, content="# Overview\n\nA normal article.\n",
        dedicated=False,
    )
    assert result["counts"]["index_files"] == 0
    assert result["counts"]["added_files"] == 1
    assert not client.calls
    assert not stats.failed

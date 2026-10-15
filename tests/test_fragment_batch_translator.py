import json
import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import fragment_batch_translator as batching
from ai_client import CompletionText
from structural_reconciler import _collect_fragments, reconcile_restructured_file


def payload(prompt):
    return json.loads(prompt.split("Fragment translation input (JSON):\n", 1)[1])


class BatchAI:
    max_tokens = 65536

    def __init__(self, respond=None):
        self.calls = []
        self.respond = respond

    def chat_completion(self, messages, temperature=0.1):
        prompt = messages[0]["content"]
        self.calls.append(prompt)
        data = payload(prompt)
        if self.respond:
            return self.respond(data)
        # Deliberately return the IDs in reverse order.
        return json.dumps(
            {
                f["id"]: f["source"].replace("Changed", "译文")
                for f in reversed(data["fragments"])
            },
            ensure_ascii=False,
        )


def translate(fragments, client, single=None):
    single = (
        single
        if single is not None
        else mock.Mock(side_effect=lambda f: "retry:" + f.source)
    )
    ok = batching.translate_fragments(
        fragments,
        client,
        "English",
        "Chinese",
        None,
        lambda f: f.source,
        lambda f, text: None if text == "INVALID" else text,
        single,
        lambda f: "single prompt\n" + f.source,
    )
    return ok, single


def test_one_file_combines_18_disjoint_fragments_and_deduplicates_context():
    fragments = [
        batching.TranslationFragment(f"Changed {i}\n", "Shared chapter", "已有译文")
        for i in range(18)
    ]
    client = BatchAI()
    ok, single = translate(fragments, client)
    assert ok
    assert len(client.calls) == 1
    assert len(payload(client.calls[0])["contexts"]) == 1
    single.assert_not_called()
    assert batching.render_fragments(fragments) == "".join(
        f"译文 {i}\n" for i in range(18)
    )


def test_line_budget_splits_groups_without_cutting_fragments():
    fragments = [batching.TranslationFragment("line\n" * 101) for _ in range(2)]
    ok, single = translate(fragments, BatchAI())
    assert ok
    assert single.call_count == 2


def test_complete_prompt_context_counts_toward_input_budget():
    fragments = [
        batching.TranslationFragment("short", "A" * 2000),
        batching.TranslationFragment("short", "B" * 2000),
    ]

    def prepare(fragment):
        return fragment.source

    limit = max(
        batching._tokens(
            batching._build_prompt(
                [("fragment_0001", f)],
                "English",
                "Chinese",
                None,
                prepare,
            )
        )
        for f in fragments
    )
    with mock.patch.object(batching, "MAX_REQUEST_TOKENS", limit):
        ok, single = translate(fragments, BatchAI())
    assert ok
    assert single.call_count == 2


def test_output_budget_also_splits_requests():
    fragments = [
        batching.TranslationFragment("descriptive words " * 40) for _ in range(2)
    ]
    client = BatchAI()
    one = batching._tokens(
        json.dumps({"fragment_0001": fragments[0].source}, ensure_ascii=False)
    )
    client.max_tokens = 2 * one + 256
    ok, single = translate(fragments, client)
    assert ok
    assert single.call_count == 2


def test_only_missing_and_invalid_fragments_are_retried():
    fragments = [batching.TranslationFragment(str(i)) for i in range(3)]
    client = BatchAI(
        lambda data: json.dumps({"fragment_0001": "good", "fragment_0002": "INVALID"})
    )
    ok, single = translate(fragments, client)
    assert ok
    assert fragments[0].translated == "good"
    assert [call.args[0].source for call in single.call_args_list] == ["1", "2"]


def test_duplicate_or_unexpected_ids_do_not_silently_overwrite_results():
    for response in (
        '{"fragment_0001":"a","fragment_0001":"b"}',
        '{"fragment_0001":"a","unknown":"b"}',
        "not JSON",
    ):
        fragments = [
            batching.TranslationFragment("a"),
            batching.TranslationFragment("b"),
        ]
        ok, single = translate(fragments, BatchAI(lambda data: response))
        assert ok
        assert single.call_count == 2


def test_incomplete_response_retries_even_entries_that_look_valid():
    response = CompletionText(
        '{"fragment_0001":"a","fragment_0002":"b"}',
        status="incomplete",
        reason="max_output_tokens",
    )
    fragments = [batching.TranslationFragment("a"), batching.TranslationFragment("b")]
    ok, single = translate(fragments, BatchAI(lambda data: response))
    assert ok
    assert single.call_count == 2


def test_failed_focused_retry_declines_the_file():
    fragments = [batching.TranslationFragment("a"), batching.TranslationFragment("b")]
    ok, single = translate(
        fragments, BatchAI(lambda data: "{}"), mock.Mock(return_value=None)
    )
    assert not ok
    assert single.call_count == 1


def test_over_budget_atomic_fragment_is_not_sent():
    with mock.patch.object(batching, "MAX_REQUEST_TOKENS", 1):
        client = BatchAI()
        ok, single = translate([batching.TranslationFragment("long text")], client)
    assert not ok
    assert client.calls == []
    single.assert_not_called()


def test_singleton_invalid_result_gets_only_one_retry():
    single = mock.Mock(side_effect=[None, "valid"])
    fragments = [batching.TranslationFragment("a")]
    ok, single = translate(fragments, BatchAI(), single)
    assert ok
    assert single.call_count == 2
    assert fragments[0].translated == "valid"


def test_unavailable_tokenizer_uses_conservative_byte_count():
    with mock.patch.object(batching, "_encoding", return_value=None):
        assert batching._tokens("中文") == len("中文".encode("utf-8"))


def test_exactly_200_lines_fit_one_request():
    fragments = [batching.TranslationFragment("line\n" * 100) for _ in range(2)]
    client = BatchAI()
    ok, single = translate(fragments, client)
    assert ok
    assert len(client.calls) == 1
    single.assert_not_called()


def test_large_run_splits_at_paragraph_boundaries_and_keeps_all_bytes():
    source = "".join(f"Paragraph {i}\n\n" for i in range(150))
    fragments = []
    pieces = _collect_fragments(fragments, source, "context", "prior")
    assert len(pieces) == 2
    assert "".join(f.source for f in pieces) == source
    assert all(len(f.source.splitlines()) <= 200 for f in pieces)
    assert all(f.context == "context" and f.prior == "prior" for f in pieces)


def test_indivisible_code_block_is_not_cut_at_200_lines():
    source = "```sql\n" + "SELECT 1;\n" * 205 + "```\n"
    fragments = []
    pieces = _collect_fragments(fragments, source)
    assert len(pieces) == 1
    assert pieces[0].source == source


def test_json_fence_is_accepted_and_non_string_values_are_retried():
    fragments = [batching.TranslationFragment("a"), batching.TranslationFragment("b")]
    client = BatchAI(
        lambda data: '```json\n{"fragment_0001":"valid","fragment_0002":42}\n```'
    )
    ok, single = translate(fragments, client)
    assert ok
    assert fragments[0].translated == "valid"
    assert [call.args[0].source for call in single.call_args_list] == ["b"]


def test_batch_markup_damage_retries_only_affected_fragment():
    class RepairAI(BatchAI):
        def chat_completion(self, messages, temperature=0.1):
            prompt = messages[0]["content"]
            self.calls.append(prompt)
            if "Fragment translation input (JSON):\n" not in prompt:
                return prompt.split("Content to translate:\n", 1)[1]
            data = payload(prompt)
            return json.dumps(
                {
                    f["id"]: f["source"].replace("</CustomContent>", "")
                    for f in data["fragments"]
                }
            )

    base = "# T\n\nOld A\n\nStable\n\nOld B\n"
    head = base.replace(
        "Old A", '<CustomContent plan="byoc">New A</CustomContent>'
    ).replace(
        "Old B",
        "New B",
    )
    target = "# 标题\n\n旧译 A\n\n稳定译文。\n\n旧译 B\n"
    client = RepairAI()
    result = reconcile_restructured_file(
        "tags.md",
        head,
        base,
        target,
        client,
        {"source_language": "English", "target_language": "Chinese"},
        source_mode="commit",
    )
    assert result is not None
    assert len(client.calls) == 2
    assert "New A" in client.calls[1]
    assert "New B" not in client.calls[1].split("Content to translate:\n", 1)[1]
    assert '<CustomContent plan="byoc">New A</CustomContent>' in result
    assert "稳定译文。" in result


def test_reconciliation_keeps_stable_translation_and_original_splice_positions():
    base = "# Title\n\nOld A\n\nStable\n\n## Next\n\nOld B\n"
    head = base.replace("Old A", "Changed A").replace("Old B", "Changed B")
    target = "# 标题\n\n旧译 A\n\n稳定译文。\n\n## 下一节\n\n旧译 B\n"
    client = BatchAI()
    result = reconcile_restructured_file(
        "guide.md",
        head,
        base,
        target,
        client,
        {"source_language": "English", "target_language": "Chinese"},
        source_mode="commit",
    )
    assert len(client.calls) == 1
    assert result == "# 标题\n\n译文 A\n\n稳定译文。\n\n## 下一节\n\n译文 B\n"


def test_heading_alignment_fallback_batches_across_sections_too():
    base = "# T\n\n## A\n\nOld A\n\n## B\n\nOld B\n"
    head = base.replace("Old A", "Changed A").replace("Old B", "Changed B")
    target = "# 标题\n\n目标专有简介。\n\n## 甲\n\n旧译 A\n\n## 乙\n\n旧译 B\n"
    client = BatchAI()
    result = reconcile_restructured_file(
        "fallback.md",
        head,
        base,
        target,
        client,
        {"source_language": "English", "target_language": "Chinese"},
        source_mode="commit",
    )
    assert len(client.calls) == 1
    assert "目标专有简介。" in result
    assert "译文 A" in result and "译文 B" in result


def test_none_and_empty_references_share_one_context():
    fragments = [
        batching.TranslationFragment("Changed A", None, None),
        batching.TranslationFragment("Changed B", "", ""),
    ]
    client = BatchAI()
    ok, single = translate(fragments, client)
    assert ok
    assert len(client.calls) == 1
    data = payload(client.calls[0])
    assert data["contexts"] == [
        {"id": "context_1", "source": "", "existing_translation": ""}
    ]
    assert {f["context_id"] for f in data["fragments"]} == {"context_1"}
    single.assert_not_called()


def test_inline_svg_is_restored_in_singleton_and_batched_paths():
    from structural_reconciler import _resolve_fragments

    class EchoAI(BatchAI):
        def chat_completion(self, messages, temperature=0.1):
            prompt = messages[0]["content"]
            if "Fragment translation input (JSON):\n" in prompt:
                return super().chat_completion(messages, temperature)
            self.calls.append(prompt)
            return prompt.split("Content to translate:\n", 1)[1]

    svg = '<svg viewBox="0 0 1 1"><path d="M0 0"/></svg>'
    source = "Available " + svg
    for count in (1, 2):
        fragments = [
            batching.TranslationFragment(source, inline=True) for _ in range(count)
        ]
        client = EchoAI()
        assert _resolve_fragments(
            fragments, client, "English", "Chinese", None, "commit"
        )
        assert len(client.calls) == 1
        assert all(f.translated == source for f in fragments)
        assert all("MDSvgIcon" not in f.translated for f in fragments)

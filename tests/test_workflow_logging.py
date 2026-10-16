import io
import sys
import threading
from contextlib import redirect_stdout, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import workflow_logging as logs
from parallel_file_processor import make_file_task, run_file_tasks
from file_adder import process_added_files
from file_updater import process_files_in_batches
from diff_analyzer import analyze_source_changes


def test_parallel_logs_keep_file_context_and_report_outcomes():
    barrier = threading.Barrier(2)

    def translate():
        barrier.wait(timeout=5)
        with logs.log_context("batch 1/2"):
            logs.thread_safe_print("request")
        return {"status": "partial"}

    output = io.StringIO()
    with redirect_stdout(output):
        results = run_file_tasks(
            [make_file_task("a.md", translate), make_file_task("b.md", translate)],
            "added files",
            parallel_enabled=True,
            max_workers=2,
        )
        logs.thread_safe_print("outside")

    text = output.getvalue()
    assert "[a.md | batch 1/2] request" in text
    assert "[b.md | batch 1/2] request" in text
    assert "2/2 processed, 0 succeeded, 2 partial, 0 failed" in text
    assert text.endswith("outside\n")
    assert [result["file_path"] for result in results] == ["a.md", "b.md"]


def test_progress_records_exceptions_and_skips_separately():
    def fail():
        raise RuntimeError("API unavailable")

    output = io.StringIO()
    with redirect_stdout(output):
        results = run_file_tasks(
            [
                make_file_task("bad.md", fail),
                make_file_task("old.md", lambda: {"status": "skipped"}),
            ],
            "modified files",
        )
    assert not results[0]["ok"]
    assert (
        "2/2 processed, 0 succeeded, 0 partial, 1 failed, 1 skipped"
        in output.getvalue()
    )


def test_wait_message_keeps_context_and_worker_stops_after_error():
    emitted = threading.Event()
    messages = []
    workers = []

    def record(*args, **kwargs):
        messages.append((logs._context.get(), args))
        workers.append(threading.current_thread())
        emitted.set()

    with mock.patch.object(logs, "thread_safe_print", side_effect=record):
        try:
            with logs.log_context("a.md | batch 2/3"):
                with logs.request_wait_logging(interval=0.01):
                    assert emitted.wait(timeout=5)
                    raise RuntimeError("request failed")
        except RuntimeError:
            pass
    assert messages[0][0] == "a.md | batch 2/3"
    assert "Waiting for AI response" in messages[0][1][0]
    assert all(not worker.is_alive() for worker in workers)


def test_compact_default_and_immediate_flush():
    with mock.patch.dict("os.environ", {}, clear=True):
        with mock.patch("builtins.print") as printer:
            logs.verbose_thread_safe_print("details")
            printer.assert_not_called()
            logs.thread_safe_print("progress")
            printer.assert_called_once_with("progress", flush=True)
    with mock.patch.dict("os.environ", {"VERBOSE_WORKFLOW_LOGS": "true"}):
        with mock.patch("builtins.print") as printer:
            logs.verbose_thread_safe_print("details")
            printer.assert_called_once_with("details", flush=True)


def test_added_file_request_logs_include_path_and_batch_and_write_output(tmp_path):
    client = mock.Mock()
    client.chat_completion.return_value = "译文"
    output = io.StringIO()
    with mock.patch(
        "file_adder.create_section_batches", return_value=["First", "Second"]
    ):
        with redirect_stdout(output):
            outcomes = process_added_files(
                {"guide.md": "First\n\nSecond"},
                {"mode": "commit"},
                None,
                client,
                {
                    "target_local_path": str(tmp_path),
                    "source_language": "English",
                    "target_language": "Chinese",
                },
                return_outcomes=True,
            )
    text = output.getvalue()
    assert "[guide.md | batch 1/2]" in text
    assert "[guide.md | batch 2/2]" in text
    assert "Input:" in text
    assert "Actual tokens:" in text or "Estimated tokens:" in text
    assert "Matched" not in text
    assert outcomes["guide.md"]["status"] == "success"
    assert "译文" in (tmp_path / "guide.md").read_text()


def test_configure_output_supports_redirected_streams():
    stream = mock.Mock()
    with (
        mock.patch.object(sys, "stdout", stream),
        mock.patch.object(sys, "stderr", io.StringIO()),
    ):
        logs.configure_output()
    stream.reconfigure.assert_called_once_with(line_buffering=True)


def test_parallel_workers_inherit_outer_context_and_restore_file_scope():
    barrier = threading.Barrier(2)

    def translate():
        barrier.wait(timeout=5)
        logs.thread_safe_print("inherited request")
        return {"status": "success"}

    output = io.StringIO()
    with redirect_stdout(output), logs.log_context("run-123"):
        results = run_file_tasks(
            [make_file_task("a.md", translate), make_file_task("b.md", translate)],
            "files",
            parallel_enabled=True,
            max_workers=2,
        )
        logs.thread_safe_print("parent restored")
    assert all(result["ok"] for result in results)
    assert "[run-123 | a.md] inherited request" in output.getvalue()
    assert "[run-123 | b.md] inherited request" in output.getvalue()
    assert "[run-123] parent restored" in output.getvalue()
    assert logs._context.get() == ""


def test_legacy_file_pool_also_inherits_submission_context():
    barrier = threading.Barrier(2)

    @logs.file_log_context("file_path")
    def worker(file_path, *args, **kwargs):
        barrier.wait(timeout=5)
        logs.thread_safe_print("legacy request")
        return True, "done"

    output = io.StringIO()
    with (
        mock.patch("file_updater.process_single_file", worker),
        redirect_stdout(output),
    ):
        with logs.log_context("run-456"):
            results = process_files_in_batches(
                {"a.md": {}, "b.md": {}},
                "",
                None,
                None,
                None,
                {},
            )
    assert len(results) == 2
    assert all(result[1] for result in results)
    assert "[run-456 | a.md] legacy request" in output.getvalue()
    assert "[run-456 | b.md] legacy request" in output.getvalue()


def test_file_decorator_binds_named_argument_and_default():
    @logs.file_log_context("file_path")
    def worker(client, file_path="default.md"):
        logs.thread_safe_print("request")
        return client

    output = io.StringIO()
    client = object()
    with redirect_stdout(output):
        assert worker(client, "positional.md") is client
        assert worker(client, file_path="keyword.md") is client
        assert worker(client) is client
    assert "[positional.md] request" in output.getvalue()
    assert "[keyword.md] request" in output.getvalue()
    assert "[default.md] request" in output.getvalue()
    assert logs._context.get() == ""


def test_file_decorator_restores_context_when_worker_raises():
    @logs.file_log_context("file_path")
    def worker(file_path):
        raise ValueError("failed")

    with logs.log_context("parent"):
        try:
            worker(file_path="bad.md")
        except ValueError:
            pass
        assert logs._context.get() == "parent"
    assert logs._context.get() == ""


def test_unknown_or_missing_status_is_failed_without_changing_task_results():
    returned = [
        None,
        {},
        {"status": []},
        {"status": None},
        {"status": "unknown"},
        False,
        True,
    ]
    tasks = [
        make_file_task(str(i), lambda value=value: value)
        for i, value in enumerate(returned)
    ]
    output = io.StringIO()
    with redirect_stdout(output):
        results = run_file_tasks(tasks, "files")
    assert [result["result"] for result in results] == returned
    # ok describes exception-free execution, not the returned business status.
    assert all(result["ok"] for result in results)
    assert (
        "7/7 processed, 1 succeeded, 0 partial, 6 failed, 0 skipped"
        in output.getvalue()
    )


def test_source_description_remains_visible_in_compact_logs():
    context = {
        "mode": "pr",
        "source_repo": "acme/docs",
        "base_ref": "base123",
        "head_ref": "head123",
        "changed_files": [],
        "pr_number": 7,
        "title": "Update guide",
        "source_description": "base123...head123",
        "repo_config": {"target_repo": "acme/docs-cn"},
    }
    for source in (context, "https://github.com/acme/docs/pull/7"):
        output = io.StringIO()
        with mock.patch.dict("os.environ", {"VERBOSE_WORKFLOW_LOGS": "false"}):
            with mock.patch(
                "diff_analyzer.build_pr_diff_context", return_value=context
            ):
                with redirect_stdout(output):
                    analyze_source_changes(source, mock.Mock(), [], [])
        text = output.getvalue()
        expected = (
            "Processing diff: base123...head123"
            if isinstance(source, dict)
            else "Processing PR #7: Update guide"
        )
        assert expected in text


def test_toc_statistics_have_consistent_compact_visibility():
    context = {
        "mode": "pr",
        "source_repo": "acme/docs",
        "base_ref": "base123",
        "head_ref": "head123",
        "changed_files": [
            SimpleNamespace(
                filename="TOC.md",
                status="modified",
                previous_filename=None,
                patch="@@ -1 +1 @@\n-old\n+new",
            )
        ],
        "source_description": "TOC update",
        "repo_config": {"target_repo": "acme/docs-cn"},
    }
    github = mock.Mock()
    github.get_repo.return_value.get_contents.return_value.decoded_content = b"# TOC\n"
    operations = {"added": [{}], "modified": [{}, {}], "deleted": [{}, {}, {}]}
    output = io.StringIO()
    with mock.patch.dict("os.environ", {"VERBOSE_WORKFLOW_LOGS": "false"}):
        with mock.patch(
            "diff_analyzer.get_target_file_content", return_value=("# 目录\n", "local")
        ):
            with mock.patch(
                "toc_processor.process_toc_operations", return_value=operations
            ):
                with redirect_stdout(output):
                    analyze_source_changes(context, github, ["TOC.md"], [])
    assert (
        "TOC TOC.md: 1 added, 2 modified, 3 deleted entries queued" in output.getvalue()
    )
    assert "❌ Deleted:" not in output.getvalue()


def test_progress_io_failure_preserves_task_exception_and_successful_results():
    import parallel_file_processor as processor

    def fail():
        raise RuntimeError("original task failure")

    for error in (BrokenPipeError("stdout unavailable"), ValueError("closed stdout")):
        for parallel in (False, True):
            success = mock.Mock(return_value={"status": "success"})
            with mock.patch.object(processor, "thread_safe_print", side_effect=error):
                results = run_file_tasks(
                    [
                        make_file_task("bad.md", fail),
                        make_file_task("good.md", success),
                    ],
                    "files",
                    parallel_enabled=parallel,
                    max_workers=2,
                )
            assert results[0]["ok"] is False
            assert results[0]["error"] == "original task failure"
            assert results[1]["ok"] is True
            assert results[1]["result"] == {"status": "success"}
            success.assert_called_once_with()


def test_same_file_context_is_not_duplicated():
    output = io.StringIO()
    with redirect_stdout(output), logs.log_context("guide.md"):
        with logs.log_context("guide.md"):
            logs.thread_safe_print("request")
    assert output.getvalue() == "[guide.md] request\n"


def test_same_file_context_is_not_duplicated_beneath_outer_run_scope():
    output = io.StringIO()
    with redirect_stdout(output), logs.log_context("run-123"):
        with logs.log_context("guide.md"):
            with logs.log_context("guide.md"):
                logs.thread_safe_print("request")
            assert logs._context.get() == "run-123 | guide.md"
        assert logs._context.get() == "run-123"
    assert output.getvalue() == "[run-123 | guide.md] request\n"


def test_business_failure_is_not_staged_even_when_execution_is_ok():
    from main_workflow import git_add_successful_task_changes

    results = run_file_tasks(
        [make_file_task("failed.md", lambda: {"status": "failed"})],
        "files",
    )
    assert results[0]["ok"] is True
    with mock.patch("main_workflow.git_add_changes") as stage:
        git_add_successful_task_changes(results, "/tmp/target")
    stage.assert_not_called()


@pytest.mark.parametrize("status", ["complete", "incomplete", "error", "empty"])
def test_ai_elapsed_time_preserves_response_metadata_and_exceptions(status):
    from ai_client import CompletionText

    response = CompletionText("translated", status=status, reason="output limit")
    error = RuntimeError("API unavailable")

    @logs.log_ai_wait
    def request():
        if status == "error":
            raise error
        return "" if status == "empty" else response

    output = io.StringIO()
    with redirect_stdout(output), logs.log_context("guide.md | fragments 1/2"), mock.patch.object(
        logs, "request_wait_logging", return_value=nullcontext()
    ), mock.patch.object(logs.time, "monotonic", side_effect=[100, 132.8]):
        if status == "error":
            with pytest.raises(RuntimeError) as caught:
                request()
            assert caught.value is error
        else:
            assert request() is ("" if status == "empty" else response)
    text = output.getvalue()
    assert "[guide.md | fragments 1/2]" in text
    assert "32.8s" in text
    assert text.count("\n") == 1
    assert ("failed" in text) == (status in {"error", "empty"})
    assert ("incomplete" in text) == (status == "incomplete")


def test_parallel_ai_timing_keeps_each_requests_file_context():
    barrier = threading.Barrier(2)

    @logs.log_ai_wait
    def request():
        barrier.wait(timeout=5)
        if logs._context.get() == "bad.md":
            raise RuntimeError("API unavailable")
        return "translated"

    output = io.StringIO()
    with redirect_stdout(output):
        results = run_file_tasks(
            [make_file_task(path, request) for path in ("good.md", "bad.md")],
            "files", parallel_enabled=True, max_workers=2,
        )
    assert "[good.md] ✅ AI response | " in output.getvalue()
    assert "[bad.md] ❌ AI request failed | " in output.getvalue()
    assert results[0]["ok"] and not results[1]["ok"]


@pytest.mark.parametrize("failure", [None, RuntimeError("failed"), SystemExit(1)])
def test_workflow_total_time_covers_returns_and_exceptional_exits(failure):
    @logs.log_workflow_duration
    def workflow():
        if failure is not None:
            raise failure
        return 1  # A handled failure must not be logged as successful translation.

    output = io.StringIO()
    with redirect_stdout(output), mock.patch.object(logs.time, "monotonic", side_effect=[100, 1277]):
        if failure is None:
            assert workflow() == 1
        else:
            with pytest.raises(type(failure)) as caught:
                workflow()
            assert caught.value is failure
    assert output.getvalue() == "⏱️ Translation workflow elapsed | total 19m 37.0s\n"


@pytest.mark.parametrize("decorator", [logs.log_ai_wait, logs.log_workflow_duration])
def test_timing_output_failure_does_not_replace_original_exception(decorator):
    error = RuntimeError("original failure")

    @decorator
    def fail():
        raise error

    with mock.patch.object(logs, "thread_safe_print", side_effect=ValueError("closed stdout")):
        with pytest.raises(RuntimeError) as caught:
            fail()
    assert caught.value is error


@pytest.mark.parametrize("mode", ["commit", "pr"])
@pytest.mark.parametrize("failure", [None, "source", "reconcile", "write"])
def test_restructured_file_elapsed_includes_snapshots_validation_and_writing(mode, failure, tmp_path):
    import commit_sync_workflow as commit
    import main_workflow as pr

    module = commit if mode == "commit" else pr
    report = commit.TranslationStats() if mode == "commit" else pr.RunReport()
    elapsed = [100.0]

    def source(*args, **kwargs):
        elapsed[0] = 102.0
        if failure == "source":
            raise RuntimeError("source failure")
        return "# Source\n"

    def reconcile(*args, **kwargs):
        elapsed[0] = 105.0
        return None if failure == "reconcile" else "# 译文\n"

    def write(*args, **kwargs):
        elapsed[0] = 106.0
        if failure == "write":
            raise OSError("write failure")

    output = io.StringIO()
    with redirect_stdout(output), mock.patch.object(module, "TARGET_REPO_PATH", str(tmp_path)), mock.patch.object(
        module, "read_target_file_content" if mode == "commit" else "read_safe_target_text",
        return_value="# 旧译文\n",
    ), mock.patch.object(
        module, "get_source_ref_content" if mode == "commit" else "get_source_file_content", side_effect=source,
    ), mock.patch.object(module, "reconcile_restructured_file", side_effect=reconcile), mock.patch.object(
        module, "atomic_write_text", side_effect=write,
    ), mock.patch.object(module, "git_add_changes"), mock.patch.object(
        module.time, "monotonic", side_effect=lambda: elapsed[0],
    ):
        if mode == "commit":
            result = commit.reconcile_restructured_files(
                {"guide.md"}, {}, set(), {"mode": mode}, None, None,
                {"ignore_resource_card_section": False}, None, report, set(),
            )
        else:
            result = pr.reconcile_restructured_files_for_pr(
                {"guide.md"}, {}, {"mode": mode}, None, None, {}, None, report,
            )
    expected = {"source": 2, "reconcile": 5}.get(failure, 6)
    assert f"total {expected:.1f}s" in output.getvalue()
    assert ("✅ Reconciled" in output.getvalue()) == (failure is None)
    assert result == ({"guide.md"} if failure is None else set())
    assert len(report.failed) == (0 if failure is None else 1)

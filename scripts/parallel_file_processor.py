"""Helpers for optional file-level parallel processing."""

import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context

from log_sanitizer import sanitize_exception_message
from workflow_logging import log_context, thread_safe_print


def _print_progress(*args, **kwargs):
    """Progress output must not replace a task's result or original exception."""
    try:
        thread_safe_print(*args, **kwargs)
    except (OSError, ValueError):
        # Broken pipes and closed streams are logging failures, not task failures.
        pass


DEFAULT_PARALLEL_FILE_THRESHOLD = 6
DEFAULT_PARALLEL_FILE_WORKERS = 4


def _positive_int_from_env(name, default):
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError:
        _print_progress(f"⚠️  Invalid {name}={value!r}; using default {default}")
        return default
    return parsed if parsed > 0 else default


def get_parallel_file_threshold():
    return _positive_int_from_env("DIFF_PARALLEL_FILE_THRESHOLD", DEFAULT_PARALLEL_FILE_THRESHOLD)


def get_parallel_file_workers():
    return _positive_int_from_env("DIFF_PARALLEL_WORKERS", DEFAULT_PARALLEL_FILE_WORKERS)


def should_parallelize_file_processing(diff_file_count):
    return diff_file_count > get_parallel_file_threshold() and get_parallel_file_workers() > 1


def count_unique_file_paths(*collections):
    paths = set()
    for collection in collections:
        if not collection:
            continue
        if isinstance(collection, dict):
            paths.update(path for path in collection.keys() if path)
        else:
            paths.update(path for path in collection if path)
    return len(paths)


def make_file_task(file_path, func, resource_key=None):
    return {
        "file_path": file_path,
        "resource_key": resource_key or file_path,
        "func": func,
    }


def make_task_result(status, reason=""):
    return {
        "status": status,
        "reason": reason,
    }


def has_duplicate_task_resources(tasks):
    seen = set()
    duplicates = set()
    for task in tasks:
        key = task.get("resource_key") or task.get("file_path")
        if key in seen:
            duplicates.add(key)
        seen.add(key)
    return duplicates


def _split_into_chunks(items, chunk_count):
    if not items:
        return []
    chunk_count = max(1, min(chunk_count, len(items)))
    chunk_size = math.ceil(len(items) / chunk_count)
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def _run_task(index, task):
    """Capture execution errors; ``ok`` does not describe the business outcome.

    A returned status such as failed or partial must still be checked inside
    ``result``. Keeping both signals preserves the original task return value.
    """
    file_path = task["file_path"]
    try:
        return {
            "index": index,
            "file_path": file_path,
            "ok": True,
            "result": task["func"](),
            "error": None,
        }
    except Exception as e:
        return {
            "index": index,
            "file_path": file_path,
            "ok": False,
            "result": None,
            "error": sanitize_exception_message(e),
        }


def _run_chunk(chunk_index, indexed_tasks):
    _print_progress(f"   ⚡ Chunk {chunk_index}: processing {len(indexed_tasks)} file(s)")
    return [_run_task(index, task) for index, task in indexed_tasks]


def _chunk_failure_results(indexed_tasks, error):
    return [
        {
            "index": index,
            "file_path": task["file_path"],
            "ok": False,
            "result": None,
            "error": error,
        }
        for index, task in indexed_tasks
    ]


def run_file_tasks(tasks, group_name, parallel_enabled=False, max_workers=None):
    """Report explicit status dictionaries or booleans, preserving raw results."""
    if not tasks:
        return []

    progress_lock = threading.Lock()
    counts = {"success": 0, "partial": 0, "failed": 0, "skipped": 0}

    def tracked(task):
        def run():
            started = time.monotonic()
            status = "failed"
            with log_context(task["file_path"]):
                _print_progress(f"📄 Started | {group_name}")
                try:
                    result = task["func"]()
                    # Production tasks return an explicit status dictionary.
                    # Only literal True is accepted as legacy boolean success.
                    status = result.get("status") if isinstance(result, dict) else (
                        "success" if result is True else "failed"
                    )
                    if not isinstance(status, str) or status not in counts:
                        status = "failed"
                    return result
                finally:
                    # _run_task captures exceptions; an exception here retains
                    # the initial failed status and is counted exactly once.
                    with progress_lock:
                        counts[status] += 1
                        _print_progress(
                            f"📊 Finished: {status} | {time.monotonic() - started:.1f}s | "
                            f"{group_name}: {sum(counts.values())}/{len(tasks)} processed, "
                            f"{counts['success']} succeeded, {counts['partial']} partial, "
                            f"{counts['failed']} failed, {counts['skipped']} skipped"
                        )
        return dict(task, func=run)

    tasks = [tracked(task) for task in tasks]
    indexed_tasks = list(enumerate(tasks))

    if not parallel_enabled or len(tasks) == 1:
        return [_run_task(index, task) for index, task in indexed_tasks]

    duplicate_resources = has_duplicate_task_resources(tasks)
    if duplicate_resources:
        _print_progress(
            "⚠️  Parallel processing disabled because multiple tasks share output resources: "
            + ", ".join(sorted(duplicate_resources))
        )
        return [_run_task(index, task) for index, task in indexed_tasks]

    worker_count = min(max_workers or get_parallel_file_workers(), len(tasks))
    chunks = _split_into_chunks(indexed_tasks, worker_count)
    _print_progress(
        f"\n⚡ Parallel processing enabled for {group_name}: "
        f"{len(tasks)} file(s) split into {len(chunks)} chunk(s)"
    )

    results = []
    with ThreadPoolExecutor(max_workers=len(chunks), thread_name_prefix="FileChunk") as executor:
        future_to_chunk = {
            # Each submission needs its own Context: a Context cannot be entered
            # by two worker threads at the same time.
            executor.submit(copy_context().run, _run_chunk, chunk_index, chunk): (chunk_index, chunk)
            for chunk_index, chunk in enumerate(chunks, 1)
        }
        for future in as_completed(future_to_chunk):
            chunk_index, chunk = future_to_chunk[future]
            try:
                chunk_results = future.result()
                results.extend(chunk_results)
            except Exception as e:
                error = sanitize_exception_message(e)
                _print_progress(f"   ❌ Chunk {chunk_index} failed: {error}")
                results.extend(_chunk_failure_results(chunk, error))

    return sorted(results, key=lambda item: item["index"])

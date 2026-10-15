"""Compact, flushed workflow logs with thread-local file and batch context."""

import builtins
import inspect
import os
import sys
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps


_context = ContextVar("workflow_log_context", default="")
_print_lock = threading.Lock()
print_lock = _print_lock


def configure_output():
    """Flush ordinary print calls by line in command-line workflows too."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(line_buffering=True)


def thread_safe_print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    context = _context.get()
    with _print_lock:
        if context:
            builtins.print(f"[{context}]", *args, **kwargs)
        else:
            builtins.print(*args, **kwargs)


def verbose_logging_enabled():
    return os.getenv("VERBOSE_WORKFLOW_LOGS", "false").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def verbose_thread_safe_print(*args, **kwargs):
    if verbose_logging_enabled():
        thread_safe_print(*args, **kwargs)


@contextmanager
def log_context(label):
    parent = _context.get()
    # Layered wrappers for one file should not repeat its innermost scope,
    # including when the file scope already follows an outer run label.
    if parent and (parent == label or str(parent).endswith(f" | {label}")):
        context = parent
    else:
        context = f"{parent} | {label}" if parent else label
    token = _context.set(context)
    try:
        yield
    finally:
        _context.reset(token)


@contextmanager
def request_wait_logging(interval=60):
    """Report long requests without changing their timeout or retry behavior."""
    stopped = threading.Event()
    context = _context.get()
    started = time.monotonic()

    def report():
        token = _context.set(context)
        try:
            while not stopped.wait(interval):
                thread_safe_print(
                    f"⏳ Waiting for AI response: {time.monotonic() - started:.0f}s"
                )
        finally:
            _context.reset(token)

    worker = threading.Thread(target=report, name="AIRequestProgress", daemon=True)
    worker.start()
    try:
        yield
    finally:
        stopped.set()
        worker.join()


def log_ai_wait(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        with request_wait_logging():
            return func(*args, **kwargs)

    return wrapped


def file_log_context(parameter):
    """Scope a file worker by its named path argument, regardless of position."""

    def decorate(func):
        signature = inspect.signature(func)
        if parameter not in signature.parameters:
            raise ValueError(f"Unknown file context parameter: {parameter}")

        @wraps(func)
        def wrapped(*args, **kwargs):
            arguments = signature.bind(*args, **kwargs)
            arguments.apply_defaults()
            path = arguments.arguments[parameter]
            with log_context(path):
                return func(*args, **kwargs)

        return wrapped

    return decorate

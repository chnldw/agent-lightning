# Copyright (c) Microsoft. All rights reserved.

"""Helpers for running async code from synchronous contexts that may already
have a running event loop (e.g. Jupyter, Databricks, IPython kernels)."""

import asyncio
import queue
import threading
from typing import Any, Coroutine, TypeVar

T = TypeVar("T")


def run_coroutine(coro: Coroutine[Any, Any, T]) -> T:
    """Run *coro* to completion and return its result.

    * If no event loop is running in the current thread, delegates to
      :func:`asyncio.run` directly (zero overhead for normal CLI usage).
    * If a loop is already running (Databricks / Jupyter / IPython), spawns a
      short-lived worker thread whose own :func:`asyncio.run` creates and
      tears down an independent loop.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop in this thread – the common case.
        return asyncio.run(coro)

    # Already inside a running loop → offload to a worker thread.
    result_queue: queue.Queue[tuple[bool, Any]] = queue.Queue()

    def _worker() -> None:
        try:
            result = asyncio.run(coro)
            result_queue.put((True, result))
        except BaseException as exc:
            result_queue.put((False, exc))

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    ok, payload = result_queue.get()
    t.join()
    if ok:
        return payload  # type: ignore[no-any-return]
    raise payload

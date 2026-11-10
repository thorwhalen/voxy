"""
Talk to a model running in another Python environment, as if it were local.

The worker itself is ``ps.isolated.Worker`` (re-exported here, with ``WorkerError``):
it starts an adapter file (``voxy/workers/<name>_adapter.py``) with an environment's
interpreter, keeps it alive (weights load once), and turns each call into one JSON
line each way. This module adds what is voxy's: ``env_worker``, the one place an
environment and an adapter become a worker (voxy's thread cap, weights folders and
log file), and ``WorkerPool``, workers shared across threads.

>>> import sys, tempfile, pathlib
>>> adapter = pathlib.Path(tempfile.mkdtemp()) / "echo_adapter.py"
>>> _ = adapter.write_text("def load(**kw): return kw\\ndef echo(state, **kw): return {**state, **kw}")
>>> with Worker(sys.executable, adapter, init={"a": 1}) as w:
...     w.call("echo", b=2)
{'a': 1, 'b': 2}
"""

import os
import threading
from collections.abc import Mapping
from typing import TYPE_CHECKING

from ps.isolated import LOG_TAIL_CHARS, Worker, WorkerError

if TYPE_CHECKING:
    from voxy.envs import EnvSpec

__all__ = ["LOG_TAIL_CHARS", "Worker", "WorkerError", "WorkerPool", "env_worker"]


def env_worker(
    spec: "EnvSpec",
    adapter: str,
    *,
    model: str | None = None,
    init: Mapping | None = None,
    threads: int | None = None,
    rootdir: str | os.PathLike | None = None,
    env_vars: Mapping[str, str] | None = None,
    auto_setup: bool = True,
) -> "Worker":
    """A worker for adapter ``adapter`` in ``spec``'s environment, its log
    ``logs/<adapter>.log`` under voxy's cache: the one place an environment and an
    adapter become a worker.

    Args:
        spec: The environment (built here if missing, unless ``auto_setup`` is False).
        adapter: The adapter's name (``workers/<adapter>_adapter.py``).
        model: Shorthand for ``init={"model": model}``.
        init: Keyword arguments of the adapter's ``load``.
        threads, rootdir: As ``voxy.envs.worker_environ``.
        env_vars: Extra environment variables of the worker.
        auto_setup: Build a missing environment (else raise, naming the setup command).
    """
    from voxy.envs import ensure_env, env_python, is_env_ready, worker_environ
    from voxy.stores import voxy_cache_dir
    from voxy.workers import adapter_path

    if auto_setup:
        python = ensure_env(spec, rootdir=rootdir)
    elif is_env_ready(spec, rootdir=rootdir):
        python = env_python(spec, rootdir=rootdir)
    else:
        raise RuntimeError(
            f"The {spec.name!r} environment isn't built: python -m voxy setup {adapter}"
        )
    return Worker(
        python,
        adapter_path(adapter),
        init={**({"model": model} if model is not None else {}), **(init or {})},
        env={
            **worker_environ(spec, threads=threads, rootdir=rootdir),
            **(env_vars or {}),
        },
        stderr=voxy_cache_dir("logs", f"{adapter}.log"),
    )


class WorkerPool:
    """Shared workers of one adapter, one per ``(model, threads, rootdir)``, each made
    once even when threads ask for it at the same time (so its environment is never
    built twice at once).

    Args:
        spec: The environment the adapter runs in (an ``voxy.envs.EnvSpec``).
        adapter: The adapter's name (``workers/<adapter>_adapter.py``).
        make: ``(spec, adapter, *, model, threads, rootdir) -> Worker``; default
            ``env_worker``.
    """

    def __init__(self, spec: "EnvSpec", adapter: str, *, make=env_worker):
        self.spec, self.adapter, self._make = spec, adapter, make
        self._workers: dict[tuple, Worker] = {}
        self._lock = threading.Lock()

    def get(
        self,
        model: str,
        threads: int | None = None,
        rootdir: str | os.PathLike | None = None,
    ) -> "Worker":
        """The worker for this model, threads and data root (made on first use)."""
        rootdir = rootdir or None  # "" is the default data root, like None
        key = (model, threads, None if rootdir is None else str(rootdir))
        if (worker := self._workers.get(key)) is not None:
            return worker
        # One lock for the whole pool, not one per key: every key builds into the
        # same environment folder, and two uv installs there at once would clash.
        with self._lock:
            if key not in self._workers:
                self._workers[key] = self._make(
                    self.spec,
                    self.adapter,
                    model=model,
                    threads=threads,
                    rootdir=rootdir,
                )
            return self._workers[key]

    def close(self) -> None:
        """Stop every worker (free their memory); the next ``get`` starts afresh."""
        with self._lock:
            workers, self._workers = list(self._workers.values()), {}
        for worker in workers:
            try:
                worker.close()
            except Exception:  # noqa: BLE001 - one stuck worker must not keep the rest
                pass

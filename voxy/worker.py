"""
Talk to a model running in another Python environment, as if it were local.

``Worker`` starts ``voxy/workers/serve.py`` with an environment's interpreter,
keeps it alive (weights load once), and turns each call into one JSON line each
way. Its stderr (model logs, progress bars) goes to a log file, so a chatty
library can't fill a pipe and hang the call.

>>> import sys, tempfile, pathlib
>>> adapter = pathlib.Path(tempfile.mkdtemp()) / "echo_adapter.py"
>>> _ = adapter.write_text("def load(**kw): return kw\\ndef echo(state, **kw): return {**state, **kw}")
>>> with Worker(sys.executable, adapter, init={"a": 1}) as w:
...     w.call("echo", b=2)
{'a': 1, 'b': 2}
"""

import json
import os
import subprocess
import tempfile
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from voxy.workers import SERVE_SCRIPT

if TYPE_CHECKING:
    from voxy.envs import EnvSpec

LOG_TAIL_CHARS = 2000


class WorkerError(RuntimeError):
    """An operation failed inside the worker (the message carries its traceback tail)."""


class Worker:
    """A long-lived adapter process: ``call(op, **request) -> dict``.

    Args:
        python: Interpreter of the environment to run in.
        adapter: The adapter file (``load`` plus operations).
        init: Keyword arguments for the adapter's ``load`` (sent on start).
        env: Environment variables of the process (default: inherited).
        log_path: Where the process's stderr goes (default: a temp file).
        cwd: Working directory (default: a neutral temp dir, so nothing in the
            caller's folder shadows the environment's imports).
    """

    def __init__(
        self,
        python: str | os.PathLike,
        adapter: str | os.PathLike,
        *,
        init: Mapping | None = None,
        env: Mapping[str, str] | None = None,
        log_path: str | os.PathLike | None = None,
        cwd: str | os.PathLike | None = None,
    ):
        self.python, self.adapter = str(python), str(adapter)
        self.init = dict(init or {})
        self.env = dict(env) if env is not None else None
        self.log_path = Path(log_path or tempfile.mkstemp(suffix=".log")[1])
        self.cwd = str(cwd or tempfile.gettempdir())
        self._process = None
        # re-entrant: close() runs inside call() when the process died or failed to
        # load; from another thread it waits for the call in flight to finish
        self._lock = threading.RLock()

    def _start(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log = open(self.log_path, "ab")
        self._process = subprocess.Popen(
            [self.python, str(SERVE_SCRIPT), self.adapter],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log,
            env=self.env,
            cwd=self.cwd,
            text=True,
            bufsize=1,
        )
        log.close()
        try:
            self._request({"op": "load", **self.init})
        except WorkerError:
            self.close()  # a model that failed to load must not serve calls
            raise

    def _log_tail(self) -> str:
        try:
            return self.log_path.read_text(errors="replace")[-LOG_TAIL_CHARS:]
        except OSError:
            return ""

    def _request(self, request: Mapping) -> dict:
        process = self._process
        payload = json.dumps(request) + "\n"  # a bad request is the caller's error
        try:
            process.stdin.write(payload)
            process.stdin.flush()
            line = process.stdout.readline()
        except (OSError, ValueError):  # a broken pipe, or a stream already closed
            line = ""
        if not line:
            self.close()
            raise WorkerError(
                f"The worker for {Path(self.adapter).stem} exited "
                f"(log: {self.log_path}):\n{self._log_tail()}"
            )
        reply = json.loads(line)
        if not reply["ok"]:
            raise WorkerError(
                f"{reply['error']}\n{reply.get('traceback', '')[-LOG_TAIL_CHARS:]}"
            )
        return reply["result"]

    @property
    def running(self) -> bool:
        """True while the process is alive."""
        return self._process is not None and self._process.poll() is None

    def call(self, op: str, **request) -> dict:
        """Run ``op`` in the worker (starting it, and loading the model, on first use)."""
        with self._lock:
            if not self.running:
                self._start()
            return self._request({"op": op, **request})

    def close(self) -> None:
        """Stop the process (its memory goes back to the machine).

        From another thread, this waits for a call in flight to finish first, so it
        cannot abort a hung call (a separate ``kill`` would be needed for that).
        """
        with self._lock:
            process, self._process = self._process, None
        if process is None:
            return
        try:
            process.stdin.close()
            process.wait(timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            process.kill()
            process.wait()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


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
        log_path=voxy_cache_dir("logs", f"{adapter}.log"),
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

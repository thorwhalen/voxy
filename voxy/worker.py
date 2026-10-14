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

from voxy.workers import SERVE_SCRIPT

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
        self._lock = threading.Lock()

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
        try:
            process.stdin.write(json.dumps(request) + "\n")
            process.stdin.flush()
            line = process.stdout.readline()
        except BrokenPipeError:
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
        """Stop the process (its memory goes back to the machine)."""
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

"""
Isolated Python environments for backends whose dependencies clash with yours.

Local voice-cloning models each pin their own torch, transformers and numpy, so
they can't share one environment (or yours). A backend declares an ``EnvSpec``;
``ensure_env`` builds it once with ``uv`` under the data root and returns its
interpreter, which ``voxy.worker`` then runs the model in:

    envs/{name}/            one virtual environment per backend
    models/hf/              model weights (``HF_HOME`` of every worker)
    models/{name}/          a backend's other weights (e.g. Coqui's ``TTS_HOME``)
    models/{name}/source/{commit}/   a backend's code that pip can't install (``GitSource``)

Building and checking an environment is ``ps.isolated``'s (``ensure_env``,
``is_env_ready``); this module adds voxy's layout (where an environment and its
weights live), the ``cpu`` torch default, a backend's own environment variables,
and the thread cap. Nothing here is downloaded at import time; building an
environment is explicit (``ensure_env``, or ``python -m voxy setup <backend>``).

>>> spec = EnvSpec("demo", ("numpy",))
>>> spec.digest() == EnvSpec("demo", ("numpy",)).digest()
True
>>> env_dir(spec, rootdir="/data").as_posix()
'/data/envs/demo'
"""

import os
import re
import shutil
import subprocess
import time
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from ps import isolated

from voxy.stores import voxy_data_dir
from voxy.workers._worker_threads import THREADS_ENVVAR

ENVS_KIND = "envs"
MODELS_KIND = "models"
HF_SUBDIR = "hf"
#: voxy's marker file (environments built before ps.isolated carry it).
ENV_MARKER = ".voxy-env.json"
DFLT_ENV_PYTHON = "3.11"
DFLT_TORCH_BACKEND = "cpu"
#: ``$VOXY_NUM_THREADS``: spelled once, in the helper the adapters share.
VOXY_NUM_THREADS_ENVVAR = THREADS_ENVVAR
#: Torch/ONNX threads per worker: a shared machine keeps headroom for others.
DFLT_NUM_THREADS = 8
THREAD_ENVVARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    VOXY_NUM_THREADS_ENVVAR,
)


@dataclass(frozen=True)
class EnvSpec(isolated.EnvSpec):
    """What a backend's environment holds (``ps.isolated.EnvSpec`` with voxy's defaults).

    Args:
        name: Folder name under ``envs/`` (usually the backend name).
        requirements: ``uv pip install`` arguments (packages, pins, extras).
        python: Python version for ``uv venv``.
        torch_backend: ``uv``'s ``--torch-backend`` ('cpu' keeps CUDA wheels out;
            None leaves the index alone).
        env_vars: Extra environment variables for the worker process (values may
            use ``{models}``, the backend's own weights folder). Not part of the
            digest: changing them needs no rebuild.
    """

    python: str = DFLT_ENV_PYTHON
    torch_backend: str | None = DFLT_TORCH_BACKEND
    env_vars: Mapping[str, str] = field(default_factory=dict)


#: A full commit SHA: what ``GitSource`` pins.
_FULL_SHA = re.compile(r"[0-9a-f]{40}")
#: How long to wait for another process fetching the same source.
DFLT_SOURCE_LOCK_TIMEOUT_S = 900
_LOCK_POLL_S = 1.0


@dataclass(frozen=True)
class GitSource:
    """Code a backend runs that is not pip-installable (no ``pyproject.toml``), at a
    pinned commit: ``ensure_git_source`` fetches it once, and the adapter puts the
    folder on its ``sys.path``.

    Args:
        url: The repository (anything ``git fetch`` takes).
        commit: The full 40-character commit SHA (a branch, tag or short SHA would
            not pin it).

    >>> GitSource("https://x/y", "main")
    Traceback (most recent call last):
      ...
    ValueError: GitSource needs a full 40-character commit SHA, not 'main'
    """

    url: str
    commit: str

    def __post_init__(self):
        if not _FULL_SHA.fullmatch(self.commit):
            raise ValueError(
                f"GitSource needs a full 40-character commit SHA, not {self.commit!r}"
            )


def git_source_dir(
    source: GitSource, name: str, *, rootdir: str | os.PathLike | None = None
) -> Path:
    """Where ``source`` is checked out for backend ``name`` (not created here).

    >>> git_source_dir(GitSource("https://x/y", "0123456789ab" + "0" * 28), "b", rootdir="/d").as_posix()
    '/d/models/b/source/0123456789ab'
    """
    return models_dir(name, "source", source.commit[:12], rootdir=rootdir)


def ensure_git_source(
    source: GitSource,
    name: str,
    *,
    rootdir: str | os.PathLike | None = None,
    run=subprocess.run,
    log=print,
    lock_timeout_s: float = DFLT_SOURCE_LOCK_TIMEOUT_S,
) -> Path:
    """Check out ``source`` for backend ``name`` once (only that commit, no history),
    and return its folder, checked: at that commit, with no local edits.

    The fetch holds a lock (another process fetching the same source waits), goes
    into a ``.partial`` folder (an interrupted one is started afresh), and is renamed
    into place when complete. git runs without the caller's ``GIT_*`` variables.
    """
    path = git_source_dir(source, name, rootdir=rootdir)
    if not path.is_dir():
        path.parent.mkdir(parents=True, exist_ok=True)
        with _dir_lock(path.with_name(path.name + ".lock"), timeout_s=lock_timeout_s):
            if not path.is_dir():
                _fetch_git_source(source, path, run=run, log=log)
    _check_checkout(path, source, run=run)
    return path


def _fetch_git_source(source: GitSource, path: Path, *, run, log) -> None:
    if shutil.which("git") is None:
        raise RuntimeError(f"Fetching {source.url} needs git (https://git-scm.com)")
    partial = path.with_name(path.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    log(f"voxy: fetching {source.url} at {source.commit[:12]} into {path} (one-time)")
    _git(run, partial, "init", "--quiet")
    _git(run, partial, "fetch", "--quiet", "--depth", "1", source.url, source.commit)
    _git(run, partial, "checkout", "--quiet", "--detach", "FETCH_HEAD")
    _check_checkout(partial, source, run=run)
    try:
        partial.rename(path)
    except OSError:
        if not path.is_dir():
            raise
        shutil.rmtree(partial, ignore_errors=True)  # someone else's copy won


def _git(run, folder: Path, *args: str):
    """``git -C folder args...``, without the caller's ``GIT_*`` variables."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    return run(
        ["git", "-C", str(folder), *args],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )


def _check_checkout(path: Path, source: GitSource, *, run) -> None:
    """Refuse a checkout at another commit, or with edits to its tracked files."""
    head = (_git(run, path, "rev-parse", "HEAD").stdout or "").strip()
    if head != source.commit:
        raise RuntimeError(
            f"{path} is at {head[:12] or 'no commit'}, not {source.commit[:12]}: "
            "delete the folder to fetch it again"
        )
    edits = (
        _git(run, path, "status", "--porcelain", "--untracked-files=no").stdout or ""
    ).strip()
    if edits:
        raise RuntimeError(
            f"{path} has local edits ({edits.splitlines()[0].strip()}, ...): "
            "delete the folder to fetch it again"
        )


@contextmanager
def _dir_lock(lock: Path, *, timeout_s: float):
    """Hold ``lock`` (a folder, made atomically) for the block; wait up to ``timeout_s``."""
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            lock.mkdir()
            break
        except FileExistsError:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"{lock} has been held for over {timeout_s:.0f} s; if no voxy "
                    "process is fetching, delete it"
                ) from None
            time.sleep(_LOCK_POLL_S)
    try:
        yield
    finally:
        shutil.rmtree(lock, ignore_errors=True)


def env_dir(spec: EnvSpec, *, rootdir: str | os.PathLike | None = None) -> Path:
    """The folder of ``spec``'s environment (not created here)."""
    return Path(rootdir or voxy_data_dir()) / ENVS_KIND / spec.name


def models_dir(*parts: str, rootdir: str | os.PathLike | None = None) -> Path:
    """Where weights live (``models/``, or a subfolder of it)."""
    return Path(rootdir or voxy_data_dir()).joinpath(MODELS_KIND, *parts)


def env_python(spec: EnvSpec, *, rootdir: str | os.PathLike | None = None) -> Path:
    """The interpreter of ``spec``'s environment (it may not exist yet)."""
    return isolated.venv_python(env_dir(spec, rootdir=rootdir))


def is_env_ready(spec: EnvSpec, *, rootdir: str | os.PathLike | None = None) -> bool:
    """True if the environment exists and was built from this exact spec."""
    return isolated.is_env_ready(
        spec, env_dir(spec, rootdir=rootdir), marker=ENV_MARKER
    )


def ensure_env(
    spec: EnvSpec,
    *,
    rootdir: str | os.PathLike | None = None,
    run=subprocess.run,
    log=print,
) -> Path:
    """Build ``spec``'s environment if missing or stale, and return its interpreter.

    A stale one (built from another spec) is rebuilt from scratch; the host's uv
    pins (``UV_CONSTRAINT`` and co.) are cleared for the build: they are for the
    host, and an isolated environment exists precisely to differ.
    """
    path = env_dir(spec, rootdir=rootdir)
    if not is_env_ready(spec, rootdir=rootdir):
        _remove_unmarked_voxy_env(path, rootdir=rootdir, log=log)
    return isolated.ensure_env(spec, path, run=run, log=log, marker=ENV_MARKER)


def _remove_unmarked_voxy_env(path: Path, *, rootdir, log) -> None:
    """Remove a half-built environment an older voxy left in its own ``envs/`` folder.

    voxy wrote its marker only once a build finished, so an interrupted build left a
    venv with no marker, which ``ps.isolated`` (rightly, in general) refuses to
    delete. Under voxy's own ``envs/`` such a folder is voxy's: it is removed so the
    build can start afresh. Anywhere else, or without a venv interpreter, it is left
    for ps to refuse.
    """
    envs_root = (Path(rootdir or voxy_data_dir()) / ENVS_KIND).resolve()
    if path.resolve().parent != envs_root or (path / ENV_MARKER).exists():
        return
    if isolated.venv_python(path).exists():
        log(f"voxy: removing the half-built environment in {path}")
        shutil.rmtree(path)


def num_threads(threads: int | None = None) -> int:
    """``threads``, else ``$VOXY_NUM_THREADS``, else ``DFLT_NUM_THREADS``.

    >>> num_threads(4)
    4
    """
    return int(threads or os.environ.get(VOXY_NUM_THREADS_ENVVAR) or DFLT_NUM_THREADS)


def worker_environ(
    spec: EnvSpec,
    *,
    threads: int | None = None,
    rootdir: str | os.PathLike | None = None,
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Environment variables for a worker: weights under the data root, threads capped.

    The whole environment (``base``, default ``os.environ``, minus the variables that
    leak the host's packages), so a one-shot command (``subprocess.run``) can use it too.

    >>> env = worker_environ(EnvSpec("x", (), env_vars={"TTS_HOME": "{models}"}),
    ...                      threads=2, rootdir="/d", base={})
    >>> from pathlib import Path
    >>> env["OMP_NUM_THREADS"], Path(env["HF_HOME"]).as_posix(), Path(env["TTS_HOME"]).as_posix()
    ('2', '/d/models/hf', '/d/models/x')
    """
    env = isolated.clean_environ(base)  # no host PYTHONPATH, VIRTUAL_ENV, ...
    n = str(num_threads(threads))
    env.update({var: n for var in THREAD_ENVVARS})
    env["HF_HOME"] = str(models_dir(HF_SUBDIR, rootdir=rootdir))
    own = str(models_dir(spec.name, rootdir=rootdir))
    env.update({k: v.format(models=own) for k, v in spec.env_vars.items()})
    return env


def remove_env(spec: EnvSpec, *, rootdir: str | os.PathLike | None = None) -> None:
    """Delete ``spec``'s environment (weights are kept)."""
    shutil.rmtree(env_dir(spec, rootdir=rootdir), ignore_errors=True)

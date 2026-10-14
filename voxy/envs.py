"""
Isolated Python environments for backends whose dependencies clash with yours.

Local voice-cloning models each pin their own torch, transformers and numpy, so
they can't share one environment (or yours). A backend declares an ``EnvSpec``;
``ensure_env`` builds it once with ``uv`` under the data root and returns its
interpreter, which ``voxy.worker`` then runs the model in:

    envs/{name}/            one virtual environment per backend
    models/hf/              model weights (``HF_HOME`` of every worker)
    models/{name}/          a backend's other weights (e.g. Coqui's ``TTS_HOME``)

Nothing here is downloaded at import time; building an environment is explicit
(``ensure_env``, or ``python -m voxy setup <backend>``).

>>> spec = EnvSpec("demo", ("numpy",))
>>> spec.digest() == EnvSpec("demo", ("numpy",)).digest()
True
>>> env_dir(spec, rootdir="/data").as_posix()
'/data/envs/demo'
"""

import hashlib
import json
import os
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path

from voxy.stores import voxy_data_dir

ENVS_KIND = "envs"
MODELS_KIND = "models"
HF_SUBDIR = "hf"
ENV_MARKER = ".voxy-env.json"
DFLT_ENV_PYTHON = "3.11"
DFLT_TORCH_BACKEND = "cpu"
VOXY_NUM_THREADS_ENVVAR = "VOXY_NUM_THREADS"
#: Torch/ONNX threads per worker: a shared machine keeps headroom for others.
DFLT_NUM_THREADS = 8
THREAD_ENVVARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VOXY_NUM_THREADS",
)


@dataclass(frozen=True)
class EnvSpec:
    """What a backend's environment holds.

    Args:
        name: Folder name under ``envs/`` (usually the backend name).
        requirements: ``uv pip install`` arguments (packages, pins, extras).
        python: Python version for ``uv venv``.
        torch_backend: ``uv``'s ``--torch-backend`` ('cpu' keeps CUDA wheels out;
            None leaves the index alone).
        env_vars: Extra environment variables for the worker process (values may
            use ``{models}``, the backend's own weights folder).
    """

    name: str
    requirements: tuple[str, ...]
    python: str = DFLT_ENV_PYTHON
    torch_backend: str | None = DFLT_TORCH_BACKEND
    env_vars: Mapping[str, str] = field(default_factory=dict)

    def digest(self) -> str:
        """A short hash of what is installed (env vars excluded: they don't need a rebuild)."""
        content = {k: v for k, v in asdict(self).items() if k != "env_vars"}
        return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()[
            :12
        ]


def env_dir(spec: EnvSpec, *, rootdir: str | os.PathLike | None = None) -> Path:
    """The folder of ``spec``'s environment (not created here)."""
    return Path(rootdir or voxy_data_dir()) / ENVS_KIND / spec.name


def models_dir(*parts: str, rootdir: str | os.PathLike | None = None) -> Path:
    """Where weights live (``models/``, or a subfolder of it)."""
    return Path(rootdir or voxy_data_dir()).joinpath(MODELS_KIND, *parts)


def env_python(spec: EnvSpec, *, rootdir: str | os.PathLike | None = None) -> Path:
    """The interpreter of ``spec``'s environment (it may not exist yet)."""
    return env_dir(spec, rootdir=rootdir) / "bin" / "python"


def is_env_ready(spec: EnvSpec, *, rootdir: str | os.PathLike | None = None) -> bool:
    """True if the environment exists and was built from this exact spec."""
    marker = env_dir(spec, rootdir=rootdir) / ENV_MARKER
    try:
        return json.loads(marker.read_text()).get("digest") == spec.digest()
    except (OSError, ValueError):
        return False


def _uv() -> str:
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError(
            "Building a backend environment needs uv: "
            "curl -LsSf https://astral.sh/uv/install.sh | sh "
            "(see https://docs.astral.sh/uv/)"
        )
    return uv


def ensure_env(
    spec: EnvSpec,
    *,
    rootdir: str | os.PathLike | None = None,
    run=subprocess.run,
    log=print,
) -> Path:
    """Build ``spec``'s environment if missing or stale, and return its interpreter.

    The host's ``UV_CONSTRAINT`` is cleared for the install: its pins are for the
    host environment, and an isolated environment exists precisely to differ.
    """
    python = env_python(spec, rootdir=rootdir)
    if is_env_ready(spec, rootdir=rootdir):
        return python
    path = env_dir(spec, rootdir=rootdir)
    uv = _uv()
    env = {**os.environ, "UV_CONSTRAINT": ""}
    log(f"voxy: building the {spec.name!r} environment in {path} (one-time)")
    if not python.exists():
        run([uv, "venv", "--python", spec.python, str(path)], check=True, env=env)
    install = [uv, "pip", "install", "--python", str(python)]
    if spec.torch_backend:
        install += ["--torch-backend", spec.torch_backend]
    run(install + list(spec.requirements), check=True, env=env)
    path.mkdir(parents=True, exist_ok=True)
    (path / ENV_MARKER).write_text(
        json.dumps({"digest": spec.digest(), "spec": asdict(spec)}, indent=2)
    )
    return python


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

    >>> env = worker_environ(EnvSpec("x", (), env_vars={"TTS_HOME": "{models}"}),
    ...                      threads=2, rootdir="/d", base={})
    >>> env["OMP_NUM_THREADS"], env["HF_HOME"], env["TTS_HOME"]
    ('2', '/d/models/hf', '/d/models/x')
    """
    env = dict(os.environ if base is None else base)
    for name in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME"):
        env.pop(name, None)
    n = str(num_threads(threads))
    env.update({var: n for var in THREAD_ENVVARS})
    env["HF_HOME"] = str(models_dir(HF_SUBDIR, rootdir=rootdir))
    own = str(models_dir(spec.name, rootdir=rootdir))
    env.update({k: v.format(models=own) for k, v in spec.env_vars.items()})
    return env


def remove_env(spec: EnvSpec, *, rootdir: str | os.PathLike | None = None) -> None:
    """Delete ``spec``'s environment (weights are kept)."""
    shutil.rmtree(env_dir(spec, rootdir=rootdir), ignore_errors=True)

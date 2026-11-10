# voxy.envs

Isolated Python environments for backends whose dependencies clash with yours.

Local voice-cloning models each pin their own torch, transformers and numpy, so
they can’t share one environment (or yours). A backend declares an `EnvSpec`;
`ensure_env` builds it once with `uv` under the data root and returns its
interpreter, which `voxy.worker` then runs the model in:

> envs/{name}/            one virtual environment per backend
> models/hf/              model weights (`HF_HOME` of every worker)
> models/{name}/          a backend’s other weights (e.g. Coqui’s `TTS_HOME`)
> models/{name}/source/{commit}/   a backend’s code that pip can’t install (`GitSource`)

Building and checking an environment is `ps.isolated`’s (`ensure_env`,
`is_env_ready`); this module adds voxy’s layout (where an environment and its
weights live), the `cpu` torch default, a backend’s own environment variables,
and the thread cap. Nothing here is downloaded at import time; building an
environment is explicit (`ensure_env`, or `python -m voxy setup <backend>`).

```pycon
>>> spec = EnvSpec("demo", ("numpy",))
>>> spec.digest() == EnvSpec("demo", ("numpy",)).digest()
True
>>> env_dir(spec, rootdir="/data").as_posix()
'/data/envs/demo'
```

### Module Attributes

| [`ENV_MARKER`](#voxy.envs.ENV_MARKER)                 | voxy's marker file (environments built before ps.isolated carry it).                                                                                                                                                              |
|-----------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`VOXY_NUM_THREADS_ENVVAR`](#voxy.envs.VOXY_NUM_THREADS_ENVVAR)    | spelled once, in the helper the adapters share.                                                                                                                                                                                   |
| [`DFLT_NUM_THREADS`](#voxy.envs.DFLT_NUM_THREADS)           | a shared machine keeps headroom for others.                                                                                                                                                                                       |
| [`DFLT_SOURCE_LOCK_TIMEOUT_S`](#voxy.envs.DFLT_SOURCE_LOCK_TIMEOUT_S) | How long to wait for another process fetching the same source.                                                                                                                                                                    |
| [`GitSource`](#voxy.envs.GitSource)(url, commit)     | Code a backend runs that is not pip-installable, at a pinned commit (a full commit SHA): [`ensure_git_source()`](#voxy.envs.ensure_git_source) fetches it once, and the adapter puts the folder on its `sys.path`. |

### Functions

| [`ensure_env`](#voxy.envs.ensure_env)(spec, \*[, rootdir, run, log])          | Build `spec`'s environment if missing or stale, and return its interpreter.                                                                                                                                                                           |
|-----------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`ensure_git_source`](#voxy.envs.ensure_git_source)(source, name, \*[, ...])         | Check out `source` for backend `name` once and return its folder, checked (at that commit, no edits to tracked files): `ps.isolated.ensure_git_source()` in voxy's models folder ([`git_source_dir()`](#voxy.envs.git_source_dir)). |
| [`env_dir`](#voxy.envs.env_dir)(spec, \*[, rootdir])                       | The folder of `spec`'s environment (not created here).                                                                                                                                                                                                |
| [`env_python`](#voxy.envs.env_python)(spec, \*[, rootdir])                    | The interpreter of `spec`'s environment (it may not exist yet).                                                                                                                                                                                       |
| [`git_source_dir`](#voxy.envs.git_source_dir)(source, name, \*[, rootdir])        | Where `source` is checked out for backend `name` (not created here).                                                                                                                                                                                  |
| [`is_env_ready`](#voxy.envs.is_env_ready)(spec, \*[, rootdir])                  | True if the environment exists and was built from this exact spec.                                                                                                                                                                                    |
| [`models_dir`](#voxy.envs.models_dir)(\*parts[, rootdir])                     | Where weights live (`models/`, or a subfolder of it).                                                                                                                                                                                                 |
| [`num_threads`](#voxy.envs.num_threads)([threads])                             | `threads`, else `$VOXY_NUM_THREADS`, else `DFLT_NUM_THREADS`.                                                                                                                                                                                         |
| [`remove_env`](#voxy.envs.remove_env)(spec, \*[, rootdir])                    | Delete `spec`'s environment (weights are kept).                                                                                                                                                                                                       |
| [`worker_environ`](#voxy.envs.worker_environ)(spec, \*[, threads, rootdir, base]) | Environment variables for a worker: weights under the data root, threads capped.                                                                                                                                                                      |

### Classes

| [`EnvSpec`](#voxy.envs.EnvSpec)(name, requirements[, python, ...])   | What a backend's environment holds (`ps.isolated.EnvSpec` with voxy's defaults).   |
|-----------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------|

### voxy.envs.DFLT_NUM_THREADS *= 8*

a shared machine keeps headroom for others.

* **Type:**
  Torch/ONNX threads per worker

### voxy.envs.DFLT_SOURCE_LOCK_TIMEOUT_S *= 900*

How long to wait for another process fetching the same source.

### voxy.envs.ENV_MARKER *= '.voxy-env.json'*

voxy’s marker file (environments built before ps.isolated carry it).

### *class* voxy.envs.EnvSpec(name, requirements, python='3.11', torch_backend='cpu', env_vars=dict())

Bases: `EnvSpec`

What a backend’s environment holds (`ps.isolated.EnvSpec` with voxy’s defaults).

* **Parameters:**
  * **name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Folder name under `envs/` (usually the backend name).
  * **requirements** (Tuple[str, …]) – `uv pip install` arguments (packages, pins, extras).
  * **python** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Python version for `uv venv`.
  * **torch_backend** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – `uv`’s `--torch-backend` (‘cpu’ keeps CUDA wheels out;
    None leaves the index alone).
  * **env_vars** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Extra environment variables for the worker process (values may
    use `{models}`, the backend’s own weights folder). Not part of the
    digest: changing them needs no rebuild.

### *class* voxy.envs.GitSource(url, commit)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Code a backend runs that is not pip-installable, at a pinned commit (a full commit
SHA): [`ensure_git_source()`](#voxy.envs.ensure_git_source) fetches it once, and the adapter puts
the folder on its `sys.path`. The mechanism is [`ps.isolated.GitSource`](#voxy.envs.GitSource).

#### fetch_commands(path)

The git commands that check this commit out (only it, no history) into `path`.

```pycon
>>> src = GitSource("https://example.org/r.git", "0" * 40)
>>> [c[:4] for c in src.fetch_commands("/x")]
[('git', 'init', '--quiet', '/x'), ('git', '-C', '/x', 'fetch'), ('git', '-C', '/x', 'checkout')]
```

* **Return type:**
  [`Tuple`](https://docs.python.org/3/library/typing.html#typing.Tuple)[[`Tuple`](https://docs.python.org/3/library/typing.html#typing.Tuple)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis)], [`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis)]

### voxy.envs.VOXY_NUM_THREADS_ENVVAR *= 'VOXY_NUM_THREADS'*

spelled once, in the helper the adapters share.

* **Type:**
  `$VOXY_NUM_THREADS`

### voxy.envs.ensure_env(spec, \*, rootdir=None, run=<function run>, log=<built-in function print>)

Build `spec`’s environment if missing or stale, and return its interpreter.

A stale one (built from another spec) is rebuilt from scratch; the host’s uv
pins (`UV_CONSTRAINT` and co.) are cleared for the build: they are for the
host, and an isolated environment exists precisely to differ.

* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)

### voxy.envs.ensure_git_source(source, name, \*, rootdir=None, run=<function run>, log=None, lock_timeout_s=900)

Check out `source` for backend `name` once and return its folder, checked
(at that commit, no edits to tracked files): `ps.isolated.ensure_git_source()`
in voxy’s models folder ([`git_source_dir()`](#voxy.envs.git_source_dir)).

* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)

### voxy.envs.env_dir(spec, , rootdir=None)

The folder of `spec`’s environment (not created here).

* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)

### voxy.envs.env_python(spec, , rootdir=None)

The interpreter of `spec`’s environment (it may not exist yet).

* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)

### voxy.envs.git_source_dir(source, name, , rootdir=None)

Where `source` is checked out for backend `name` (not created here).

```pycon
>>> git_source_dir(GitSource("https://x/y", "0123456789ab" + "0" * 28), "b", rootdir="/d").as_posix()
'/d/models/b/source/0123456789ab'
```

* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)

### voxy.envs.is_env_ready(spec, , rootdir=None)

True if the environment exists and was built from this exact spec.

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### voxy.envs.models_dir(\*parts, rootdir=None)

Where weights live (`models/`, or a subfolder of it).

* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)

### voxy.envs.num_threads(threads=None)

`threads`, else `$VOXY_NUM_THREADS`, else `DFLT_NUM_THREADS`.

```pycon
>>> num_threads(4)
4
```

* **Return type:**
  [`int`](https://docs.python.org/3/builtins/functions.html#int)

### voxy.envs.remove_env(spec, , rootdir=None)

Delete `spec`’s environment (weights are kept).

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

### voxy.envs.worker_environ(spec, , threads=None, rootdir=None, base=None)

Environment variables for a worker: weights under the data root, threads capped.

The whole environment (`base`, default `os.environ`, minus the variables that
leak the host’s packages), so a one-shot command (`subprocess.run`) can use it too.

```pycon
>>> env = worker_environ(EnvSpec("x", (), env_vars={"TTS_HOME": "{models}"}),
...                      threads=2, rootdir="/d", base={})
>>> from pathlib import Path
>>> env["OMP_NUM_THREADS"], Path(env["HF_HOME"]).as_posix(), Path(env["TTS_HOME"]).as_posix()
('2', '/d/models/hf', '/d/models/x')
```

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

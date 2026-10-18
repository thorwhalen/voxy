# voxy.envs

Isolated Python environments for backends whose dependencies clash with yours.

Local voice-cloning models each pin their own torch, transformers and numpy, so
they can’t share one environment (or yours). A backend declares an `EnvSpec`;
`ensure_env` builds it once with `uv` under the data root and returns its
interpreter, which `voxy.worker` then runs the model in:

> envs/{name}/            one virtual environment per backend
> models/hf/              model weights (`HF_HOME` of every worker)
> models/{name}/          a backend’s other weights (e.g. Coqui’s `TTS_HOME`)

Nothing here is downloaded at import time; building an environment is explicit
(`ensure_env`, or `python -m voxy setup <backend>`).

```pycon
>>> spec = EnvSpec("demo", ("numpy",))
>>> spec.digest() == EnvSpec("demo", ("numpy",)).digest()
True
>>> env_dir(spec, rootdir="/data").as_posix()
'/data/envs/demo'
```

### Module Attributes

| [`DFLT_NUM_THREADS`](#voxy.envs.DFLT_NUM_THREADS)   | a shared machine keeps headroom for others.   |
|---------------------------------------------------------------------|-----------------------------------------------|

### Functions

| [`ensure_env`](#voxy.envs.ensure_env)(spec, \*[, rootdir, run, log])          | Build `spec`'s environment if missing or stale, and return its interpreter.      |
|-----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| [`env_dir`](#voxy.envs.env_dir)(spec, \*[, rootdir])                       | The folder of `spec`'s environment (not created here).                           |
| [`env_python`](#voxy.envs.env_python)(spec, \*[, rootdir])                    | The interpreter of `spec`'s environment (it may not exist yet).                  |
| [`is_env_ready`](#voxy.envs.is_env_ready)(spec, \*[, rootdir])                  | True if the environment exists and was built from this exact spec.               |
| [`models_dir`](#voxy.envs.models_dir)(\*parts[, rootdir])                     | Where weights live (`models/`, or a subfolder of it).                            |
| [`num_threads`](#voxy.envs.num_threads)([threads])                             | `threads`, else `$VOXY_NUM_THREADS`, else `DFLT_NUM_THREADS`.                    |
| [`remove_env`](#voxy.envs.remove_env)(spec, \*[, rootdir])                    | Delete `spec`'s environment (weights are kept).                                  |
| [`worker_environ`](#voxy.envs.worker_environ)(spec, \*[, threads, rootdir, base]) | Environment variables for a worker: weights under the data root, threads capped. |

### Classes

| [`EnvSpec`](#voxy.envs.EnvSpec)(name, requirements[, python, ...])   | What a backend's environment holds.   |
|-----------------------------------------------------------------------------------------------|---------------------------------------|

### voxy.envs.DFLT_NUM_THREADS *= 8*

a shared machine keeps headroom for others.

* **Type:**
  Torch/ONNX threads per worker

### *class* voxy.envs.EnvSpec(name, requirements, python='3.11', torch_backend='cpu', env_vars=dict())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

What a backend’s environment holds.

* **Parameters:**
  * **name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Folder name under `envs/` (usually the backend name).
  * **requirements** ([`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis)]) – `uv pip install` arguments (packages, pins, extras).
  * **python** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Python version for `uv venv`.
  * **torch_backend** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – `uv`’s `--torch-backend` (‘cpu’ keeps CUDA wheels out;
    None leaves the index alone).
  * **env_vars** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Extra environment variables for the worker process (values may
    use `{models}`, the backend’s own weights folder).

#### digest()

A short hash of what is installed (env vars excluded: they don’t need a rebuild).

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### voxy.envs.ensure_env(spec, \*, rootdir=None, run=<function run>, log=<built-in function print>)

Build `spec`’s environment if missing or stale, and return its interpreter.

The host’s `UV_CONSTRAINT` is cleared for the install: its pins are for the
host environment, and an isolated environment exists precisely to differ.

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

```pycon
>>> env = worker_environ(EnvSpec("x", (), env_vars={"TTS_HOME": "{models}"}),
...                      threads=2, rootdir="/d", base={})
>>> from pathlib import Path
>>> env["OMP_NUM_THREADS"], Path(env["HF_HOME"]).as_posix(), Path(env["TTS_HOME"]).as_posix()
('2', '/d/models/hf', '/d/models/x')
```

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

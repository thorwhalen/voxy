# voxy.worker

Talk to a model running in another Python environment, as if it were local.

The worker itself is `ps.isolated.Worker` (`Worker` here, which also takes the
former `log_path=`; `WorkerError` is ps’s):
it starts an adapter file (`voxy/workers/<name>_adapter.py`) with an environment’s
interpreter, keeps it alive (weights load once), and turns each call into one JSON
line each way. This module adds what is voxy’s: `env_worker`, the one place an
environment and an adapter become a worker (voxy’s thread cap, weights folders and
log file), and `WorkerPool`, workers shared across threads.

```pycon
>>> import sys, tempfile, pathlib
>>> adapter = pathlib.Path(tempfile.mkdtemp()) / "echo_adapter.py"
>>> _ = adapter.write_text("def load(**kw): return kw\ndef echo(state, **kw): return {**state, **kw}")
>>> with Worker(sys.executable, adapter, init={"a": 1}) as w:
...     w.call("echo", b=2)
{'a': 1, 'b': 2}
```

### Functions

| [`env_worker`](#voxy.worker.env_worker)(spec, adapter, \*[, model, init, ...])   | A worker for adapter `adapter` in `spec`'s environment, its log `logs/<adapter>.log` under voxy's cache: the one place an environment and an adapter become a worker.   |
|------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

### Classes

| [`Worker`](#voxy.worker.Worker)(python, adapter, \*[, log_path])   | `ps.isolated.Worker`, still accepting voxy's former `log_path=` (now `stderr=`).                                                                                                    |
|--------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`WorkerPool`](#voxy.worker.WorkerPool)(spec, adapter, \*[, make])     | Shared workers of one adapter, one per `(model, threads, rootdir)`, each made once even when threads ask for it at the same time (so its environment is never built twice at once). |

### Exceptions

| [`WorkerError`](#voxy.worker.WorkerError)   | The worker failed a request (its traceback is in the message) or died (its log tail is).   |
|----------------------------------------------------------------|--------------------------------------------------------------------------------------------|

### *class* voxy.worker.Worker(python, adapter, , log_path=None, \*\*kwargs)

Bases: `Worker`

`ps.isolated.Worker`, still accepting voxy’s former `log_path=` (now `stderr=`).

### *exception* voxy.worker.WorkerError

Bases: [`RuntimeError`](https://docs.python.org/3/builtins/exceptions.html#RuntimeError)

The worker failed a request (its traceback is in the message) or died (its log tail is).

### *class* voxy.worker.WorkerPool(spec, adapter, \*, make=<function env_worker>)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Shared workers of one adapter, one per `(model, threads, rootdir)`, each made
once even when threads ask for it at the same time (so its environment is never
built twice at once).

* **Parameters:**
  * **spec** ([`EnvSpec`](voxy.envs.html.md#voxy.envs.EnvSpec)) – The environment the adapter runs in (an `voxy.envs.EnvSpec`).
  * **adapter** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The adapter’s name (`workers/<adapter>_adapter.py`).
  * **make** – `(spec, adapter, *, model, threads, rootdir) -> Worker`; default
    `env_worker`.

#### close()

Stop every worker (free their memory); the next `get` starts afresh.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### get(model, threads=None, rootdir=None)

The worker for this model, threads and data root (made on first use).

* **Return type:**
  [`Worker`](#voxy.worker.Worker)

### voxy.worker.env_worker(spec, adapter, , model=None, init=None, threads=None, rootdir=None, env_vars=None, auto_setup=True)

A worker for adapter `adapter` in `spec`’s environment, its log
`logs/<adapter>.log` under voxy’s cache: the one place an environment and an
adapter become a worker.

* **Parameters:**
  * **spec** ([`EnvSpec`](voxy.envs.html.md#voxy.envs.EnvSpec)) – The environment (built here if missing, unless `auto_setup` is False).
  * **adapter** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The adapter’s name (`workers/<adapter>_adapter.py`).
  * **model** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Shorthand for `init={"model": model}`.
  * **init** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Keyword arguments of the adapter’s `load`.
  * **threads** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – As `voxy.envs.worker_environ`.
  * **rootdir** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – As `voxy.envs.worker_environ`.
  * **env_vars** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Extra environment variables of the worker.
  * **auto_setup** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Build a missing environment (else raise, naming the setup command).
* **Return type:**
  [`Worker`](#voxy.worker.Worker)

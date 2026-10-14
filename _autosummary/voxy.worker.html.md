# voxy.worker

Talk to a model running in another Python environment, as if it were local.

`Worker` starts `voxy/workers/serve.py` with an environment’s interpreter,
keeps it alive (weights load once), and turns each call into one JSON line each
way. Its stderr (model logs, progress bars) goes to a log file, so a chatty
library can’t fill a pipe and hang the call.

```pycon
>>> import sys, tempfile, pathlib
>>> adapter = pathlib.Path(tempfile.mkdtemp()) / "echo_adapter.py"
>>> _ = adapter.write_text("def load(**kw): return kw\ndef echo(state, **kw): return {**state, **kw}")
>>> with Worker(sys.executable, adapter, init={"a": 1}) as w:
...     w.call("echo", b=2)
{'a': 1, 'b': 2}
```

### Classes

| [`Worker`](#voxy.worker.Worker)(python, adapter, \*[, init, env, ...])   | A long-lived adapter process: `call(op, **request) -> dict`.   |
|--------------------------------------------------------------------------------------------------|----------------------------------------------------------------|

### Exceptions

| [`WorkerError`](#voxy.worker.WorkerError)   | An operation failed inside the worker (the message carries its traceback tail).   |
|----------------------------------------------------------------|-----------------------------------------------------------------------------------|

### *class* voxy.worker.Worker(python, adapter, , init=None, env=None, log_path=None, cwd=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A long-lived adapter process: `call(op, **request) -> dict`.

* **Parameters:**
  * **python** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike)) – Interpreter of the environment to run in.
  * **adapter** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike)) – The adapter file (`load` plus operations).
  * **init** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Keyword arguments for the adapter’s `load` (sent on start).
  * **env** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Environment variables of the process (default: inherited).
  * **log_path** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Where the process’s stderr goes (default: a temp file).
  * **cwd** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Working directory (default: a neutral temp dir, so nothing in the
    caller’s folder shadows the environment’s imports).

#### call(op, \*\*request)

Run `op` in the worker (starting it, and loading the model, on first use).

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

#### close()

Stop the process (its memory goes back to the machine).

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### *property* running *: [bool](https://docs.python.org/3/builtins/functions.html#bool)*

True while the process is alive.

### *exception* voxy.worker.WorkerError

Bases: [`RuntimeError`](https://docs.python.org/3/builtins/exceptions.html#RuntimeError)

An operation failed inside the worker (the message carries its traceback tail).

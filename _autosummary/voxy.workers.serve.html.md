# voxy.workers.serve

Worker entry point: run a backend adapter inside its own environment.

Started by `voxy.worker.Worker` with the environment’s interpreter, as
`python serve.py <adapter.py>`. It imports nothing from voxy (the environment
need not have voxy installed), loads the adapter file, then answers one JSON
request per stdin line with one JSON reply per line on the original stdout.
Everything the model libraries print goes to stderr (the worker’s log), so it
can’t corrupt the replies.

An adapter is a plain module with `load(**init) -> state` and any number of
operations `op(state, **request) -> dict` (JSON-ready), e.g. `synthesize`.
Request: `{"op": "load", ...}` or `{"op": "<name>", ...}`.
Reply: `{"ok": true, "result": {...}}` or `{"ok": false, "error": "...", "traceback": "..."}`.

### Functions

| `main`([argv])   |    |
|------------------|----|

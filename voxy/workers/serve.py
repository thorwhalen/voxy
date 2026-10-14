"""
Worker entry point: run a backend adapter inside its own environment.

Started by ``voxy.worker.Worker`` with the environment's interpreter, as
``python serve.py <adapter.py>``. It imports nothing from voxy (the environment
need not have voxy installed), loads the adapter file, then answers one JSON
request per stdin line with one JSON reply per line on the original stdout.
Everything the model libraries print goes to stderr (the worker's log), so it
can't corrupt the replies.

An adapter is a plain module with ``load(**init) -> state`` and any number of
operations ``op(state, **request) -> dict`` (JSON-ready), e.g. ``synthesize``.
Request: ``{"op": "load", ...}`` or ``{"op": "<name>", ...}``.
Reply: ``{"ok": true, "result": {...}}`` or ``{"ok": false, "error": "...", "traceback": "..."}``.
"""

import importlib.util
import json
import os
import sys
import traceback


def _cap_threads():
    """Cap torch's intra-op threads at ``$VOXY_NUM_THREADS`` (if torch is there)."""
    n = os.environ.get("VOXY_NUM_THREADS")
    if not n:
        return
    try:
        import torch
    except ImportError:
        return
    torch.set_num_threads(int(n))
    try:
        torch.set_num_interop_threads(max(1, int(n) // 4))
    except RuntimeError:
        pass  # already set (torch allows it once)


def _load_adapter(path):
    spec = importlib.util.spec_from_file_location("voxy_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    replies = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)  # C-level prints to fd 1 now land in the log too
    sys.stdout = sys.stderr
    adapter = _load_adapter(argv[0])
    _cap_threads()
    state = None
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            op = request.pop("op")
            if op == "load":
                state = adapter.load(**request)
                result = {}
            elif op.startswith("_") or not callable(getattr(adapter, op, None)):
                raise ValueError(f"Unknown operation {op!r}")
            else:
                result = getattr(adapter, op)(state, **request)
            reply = {"ok": True, "result": result or {}}
        except Exception as e:  # report every failure to the client, keep serving
            reply = {
                "ok": False,
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
            }
        replies.write(json.dumps(reply) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

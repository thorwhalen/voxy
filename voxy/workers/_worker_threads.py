"""Thread caps in a worker: a helper the adapters share (it imports nothing from voxy).

``voxy.envs.worker_environ`` sets ``$VOXY_NUM_THREADS`` (:data:`THREADS_ENVVAR`, the
one place that name is spelled); torch adapters call ``cap_torch_threads()`` first
thing in their ``load``, before the model is built, and other runtimes read
``thread_cap()`` (kokoro's onnxruntime), so a shared machine keeps headroom.
"""

import os

THREADS_ENVVAR = "VOXY_NUM_THREADS"
#: Inter-op threads are this fraction of the intra-op ones (at least one).
INTEROP_DIVISOR = 4


def thread_cap(threads=None):
    """``threads``, else ``$VOXY_NUM_THREADS``, as an int (None when neither is set)."""
    n = threads or os.environ.get(THREADS_ENVVAR)
    return int(n) if n else None


def cap_torch_threads(threads=None):
    """Cap torch's intra-op threads at :func:`thread_cap` (and inter-op at a quarter).

    Does nothing without a cap or without torch; inter-op threads can be set once
    per process, so a second call leaves them as they are.
    """
    n = thread_cap(threads)
    if n is None:
        return
    try:
        import torch
    except ImportError:
        return
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(max(1, n // INTEROP_DIVISOR))
    except RuntimeError:
        pass  # already set (torch allows it once)

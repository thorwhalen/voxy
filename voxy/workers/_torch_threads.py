"""Cap torch's threads in a worker: a helper the adapters share (imports nothing from voxy).

Adapters call ``cap_torch_threads()`` at the start of their ``load``, before the
model is built, so a worker keeps to ``$VOXY_NUM_THREADS`` (set by
``voxy.envs.worker_environ``) and a shared machine keeps headroom for others.
"""

import os

THREADS_ENVVAR = "VOXY_NUM_THREADS"
#: Inter-op threads are this fraction of the intra-op ones (at least one).
INTEROP_DIVISOR = 4


def cap_torch_threads(threads=None):
    """Cap torch's intra-op threads at ``threads`` (default ``$VOXY_NUM_THREADS``).

    Does nothing without a cap or without torch; inter-op threads can be set once
    per process, so a second call leaves them as they are.
    """
    n = threads or os.environ.get(THREADS_ENVVAR)
    if not n:
        return
    try:
        import torch
    except ImportError:
        return
    torch.set_num_threads(int(n))
    try:
        torch.set_num_interop_threads(max(1, int(n) // INTEROP_DIVISOR))
    except RuntimeError:
        pass  # already set (torch allows it once)

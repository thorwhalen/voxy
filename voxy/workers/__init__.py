"""
Backend adapters run by ``voxy.worker`` inside each backend's own environment.

Each ``*_adapter.py`` wraps one model library with ``load(**init)`` and operations
such as ``synthesize(state, *, text, ref_audio, ref_text, out_path, **params)``;
``ps.isolated.Worker`` serves it. These files are executed by other interpreters,
so they import nothing from voxy; this folder is on their ``sys.path``, so they
share helpers by plain import (``from _torch_threads import cap_torch_threads``).
"""

from pathlib import Path

WORKERS_DIR = Path(__file__).parent


def adapter_path(name: str) -> Path:
    """The adapter file for ``name`` (e.g. 'f5tts' -> ``f5tts_adapter.py``).

    >>> adapter_path("f5tts").name
    'f5tts_adapter.py'
    """
    return WORKERS_DIR / f"{name}_adapter.py"

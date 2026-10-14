"""
Backend adapters run by ``voxy.worker`` inside each backend's own environment.

``serve.py`` is the entry point; each ``*_adapter.py`` wraps one model library
with ``load(**init)`` and ``synthesize(state, *, text, ref_audio, ref_text,
out_path, **params)``. These files are executed by other interpreters, so they
import nothing from voxy.
"""

from pathlib import Path

WORKERS_DIR = Path(__file__).parent
SERVE_SCRIPT = WORKERS_DIR / "serve.py"


def adapter_path(name: str) -> Path:
    """The adapter file for ``name`` (e.g. 'f5tts' -> ``f5tts_adapter.py``).

    >>> adapter_path("f5tts").name
    'f5tts_adapter.py'
    """
    return WORKERS_DIR / f"{name}_adapter.py"

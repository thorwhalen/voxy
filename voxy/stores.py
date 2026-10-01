"""
Where voxy keeps its data: voice records and audio, as ``MutableMapping`` stores.

Layout under the data root (``$VOXY_DATA_DIR``, default ``~/.local/share/voxy``
via ``config2py``), one subfolder per kind of data:

    voices/{name}.json        voice records: aliases, consent, one profile per backend
    samples/{name}/{file}     prepared audio a clone is made from
    sources/{name}/{file}     raw source media the samples were cut from

Callers take a store as an argument (``voices=``, ``samples=``) and never build
paths themselves, so moving the data elsewhere (another folder, S3, a database)
means passing a different ``MutableMapping``. Folders are created on first write.

>>> import tempfile
>>> root = tempfile.mkdtemp()
>>> voices = voices_store(rootdir=root)
>>> voices["ada"] = {"name": "ada"}
>>> list(voices), voices["ada"]
(['ada'], {'name': 'ada'})
>>> samples = samples_store("ada", rootdir=root)
>>> samples["clip.wav"] = b"RIFF"
>>> list(samples)
['clip.wav']
"""

import os
from collections.abc import MutableMapping
from pathlib import Path

VOXY_DATA_DIR_ENVVAR = "VOXY_DATA_DIR"
APP_NAME = "voxy"
VOICES_KIND = "voices"
SAMPLES_KIND = "samples"
SOURCES_KIND = "sources"
VOICE_RECORD_EXT = ".json"
AUDIO_EXTS = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".webm", ".aac")


def check_voice_name(name: str) -> str:
    """Return ``name`` if it can name a voice (one path segment), else raise.

    >>> check_voice_name("cora")
    'cora'
    >>> check_voice_name("../x")
    Traceback (most recent call last):
      ...
    ValueError: Invalid voice name '../x': use one non-empty segment without '/', '\\' or '..'
    """
    if (
        not isinstance(name, str)
        or not name
        or name in (".", "..")
        or any(c in name for c in "/\\")
        or ".." in name
    ):
        raise ValueError(
            f"Invalid voice name {name!r}: use one non-empty segment without "
            "'/', '\\' or '..'"
        )
    return name


def voxy_data_dir() -> Path:
    """The data root: ``$VOXY_DATA_DIR`` if set, else the platform's app-data folder."""
    if root := os.environ.get(VOXY_DATA_DIR_ENVVAR):
        return Path(root).expanduser().resolve()
    from config2py import get_app_data_folder

    return Path(get_app_data_folder(APP_NAME, ensure_exists=False))


def kind_dir(kind: str, *parts: str, rootdir: str | os.PathLike | None = None) -> Path:
    """The folder for one kind of data, optionally a subfolder (not created here).

    >>> import tempfile
    >>> root = tempfile.mkdtemp()
    >>> kind_dir("samples", "ada", rootdir=root).relative_to(root).as_posix()
    'samples/ada'
    """
    return Path(rootdir or voxy_data_dir()).joinpath(kind, *parts)


def voices_store(*, rootdir: str | os.PathLike | None = None) -> MutableMapping:
    """Voice records, ``name -> dict``, as ``voices/{name}.json`` (other files ignored)."""
    from dol import JsonFiles, KeyCodecs, filt_iter, mk_dirs_if_missing

    store = mk_dirs_if_missing(
        JsonFiles(str(kind_dir(VOICES_KIND, rootdir=rootdir)) + os.sep)
    )
    store = filt_iter(
        store,
        filt=lambda k: (
            k.endswith(VOICE_RECORD_EXT) and not any(c in k for c in "/\\\\")
        ),
    )
    return _checked_names(KeyCodecs.suffixed(VOICE_RECORD_EXT)(store))


def _checked_names(store: MutableMapping) -> MutableMapping:
    """Refuse writes under keys that aren't valid voice names."""
    from dol import wrap_kvs

    return wrap_kvs(store, id_of_key=check_voice_name)


def audio_store(
    kind: str, voice: str, *, rootdir: str | os.PathLike | None = None
) -> MutableMapping:
    """Audio files of one voice and kind, ``filename -> bytes``."""
    from dol import Files, mk_dirs_if_missing

    path = kind_dir(kind, check_voice_name(voice), rootdir=rootdir)
    return mk_dirs_if_missing(Files(str(path) + os.sep))


def samples_store(
    voice: str, *, rootdir: str | os.PathLike | None = None
) -> MutableMapping:
    """The prepared samples a clone of ``voice`` is made from."""
    return audio_store(SAMPLES_KIND, voice, rootdir=rootdir)


def sources_store(
    voice: str, *, rootdir: str | os.PathLike | None = None
) -> MutableMapping:
    """The raw source media ``voice``'s samples were cut from."""
    return audio_store(SOURCES_KIND, voice, rootdir=rootdir)

# voxy.stores

Where voxy keeps its data: voice records and audio, as `MutableMapping` stores.

Layout under the data root (`$VOXY_DATA_DIR`, default `~/.local/share/voxy`
via `config2py`), one subfolder per kind of data:

> voices/{name}.json        voice records: aliases, consent, one profile per backend
> samples/{name}/{file}     prepared audio a clone is made from
> sources/{name}/{file}     raw source media the samples were cut from

Callers take a store as an argument (`voices=`, `samples=`) and never build
paths themselves, so moving the data elsewhere (another folder, S3, a database)
means passing a different `MutableMapping`. Folders are created on first write.

```pycon
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
```

### Functions

| [`audio_store`](#voxy.stores.audio_store)(kind, voice, \*[, rootdir])   | Audio files of one voice and kind, `filename -> bytes`.                       |
|--------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|
| [`check_voice_name`](#voxy.stores.check_voice_name)(name)                    | Return `name` if it can name a voice (one path segment), else raise.          |
| [`kind_dir`](#voxy.stores.kind_dir)(kind, \*parts[, rootdir])        | The folder for one kind of data, optionally a subfolder (not created here).   |
| [`samples_store`](#voxy.stores.samples_store)(voice, \*[, rootdir])       | The prepared samples a clone of `voice` is made from.                         |
| [`sources_store`](#voxy.stores.sources_store)(voice, \*[, rootdir])       | The raw source media `voice`'s samples were cut from.                         |
| [`voices_store`](#voxy.stores.voices_store)(\*[, rootdir])               | Voice records, `name -> dict`, as `voices/{name}.json` (other files ignored). |
| [`voxy_data_dir`](#voxy.stores.voxy_data_dir)()                           | The data root: `$VOXY_DATA_DIR` if set, else the platform's app-data folder.  |

### voxy.stores.audio_store(kind, voice, , rootdir=None)

Audio files of one voice and kind, `filename -> bytes`.

* **Return type:**
  [`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)

### voxy.stores.check_voice_name(name)

Return `name` if it can name a voice (one path segment), else raise.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

```pycon
>>> check_voice_name("ada")
'ada'
>>> check_voice_name("../x")
Traceback (most recent call last):
  ...
ValueError: Invalid voice name '../x': use one non-empty segment without '/', '\' or '..'
```

### voxy.stores.kind_dir(kind, \*parts, rootdir=None)

The folder for one kind of data, optionally a subfolder (not created here).

* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)

```pycon
>>> import tempfile
>>> root = tempfile.mkdtemp()
>>> kind_dir("samples", "ada", rootdir=root).relative_to(root).as_posix()
'samples/ada'
```

### voxy.stores.samples_store(voice, , rootdir=None)

The prepared samples a clone of `voice` is made from.

* **Return type:**
  [`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)

### voxy.stores.sources_store(voice, , rootdir=None)

The raw source media `voice`’s samples were cut from.

* **Return type:**
  [`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)

### voxy.stores.voices_store(, rootdir=None)

Voice records, `name -> dict`, as `voices/{name}.json` (other files ignored).

* **Return type:**
  [`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)

### voxy.stores.voxy_data_dir()

The data root: `$VOXY_DATA_DIR` if set, else the platform’s app-data folder.

* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)

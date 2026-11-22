# voxy.language

Say the language once, and have each engine get it in its own option.

Engines disagree on how a caller pins the language of a take: Seed Speech on fal takes
`language`, ElevenLabs’ newer models take `language_code`, Qwen3-TTS takes
`language` spelled out (“English”), OpenAI’s voices take none. Left unset, an engine
that detects the language per request can flip: an English sentence carrying a year and
two percentages came back read in Tagalog. So the caller says `language="en"` once
(`voxy.text_to_speech(..., language="en")`) and this module maps it, by backend and
model id, the way `voxy.direction` maps delivery directions. A model nobody registered
gets `language` passed through unchanged, as before this seam existed. A new model only
needs [`register_language_option()`](#voxy.language.register_language_option).

```pycon
>>> language_option("fal", "fal-ai/bytedance/seed-speech/tts/v2")
LanguageOption(param='language', detects=True, to_value=None)
>>> language_options("elevenlabs", "eleven_flash_v2_5", "en")
{'language_code': 'en'}
>>> language_options("elevenlabs", "eleven_multilingual_v2", "en")  # takes none: nothing sent
{}
>>> language_options("qwen3tts", None, "fr")
{'language': 'French'}
>>> language_options("someone-new", None, "de")  # unregistered: passed through
{'language': 'de'}
```

### Module Attributes

| [`PASSTHROUGH`](#voxy.language.PASSTHROUGH)    | `language` passed through as given.                                        |
|-----------------------------------------------------------------|----------------------------------------------------------------------------|
| [`LANGUAGE_NAMES`](#voxy.language.LANGUAGE_NAMES) | ISO 639-1 codes to the English names some models want ("en" -> "English"). |

### Functions

| [`language_name`](#voxy.language.language_name)(code)                           | 'English' for 'en' (and 'en-US'); anything else unchanged.                                                                                                                                          |
|------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`language_option`](#voxy.language.language_option)(backend[, model_id])          | The language option of `backend`'s model `model_id`: the registered one, else [`PASSTHROUGH`](#voxy.language.PASSTHROUGH) (`language` sent as given, as before this seam existed). |
| [`language_options`](#voxy.language.language_options)(backend, model_id, language) | The synthesis options that pin `language` on this model: `{}` when no language is asked or the model takes none.                                                                                    |
| [`options_for`](#voxy.language.options_for)(option, language)                 | The synthesis options `option` sends for `language` (`{}` for none).                                                                                                                                |
| [`register_language_option`](#voxy.language.register_language_option)(backend, ...)        | Declare how `backend`'s models matching `model_glob` take a language (checked before the ones already registered).                                                                                  |

### Classes

| [`LanguageOption`](#voxy.language.LanguageOption)([param, detects, to_value])   | How a model takes a language.   |
|-----------------------------------------------------------------------------------------------|---------------------------------|

### Exceptions

| [`LanguageIgnoredWarning`](#voxy.language.LanguageIgnoredWarning)   | `language=` asked of a model that takes no language option (dropped).   |
|---------------------------------------------------------------------------|-------------------------------------------------------------------------|
| [`LanguageUnsetWarning`](#voxy.language.LanguageUnsetWarning)     | No `language=` on a model that guesses the language of each request.    |

### voxy.language.LANGUAGE_NAMES *= {'de': 'German', 'en': 'English', 'es': 'Spanish', 'fr': 'French', 'it': 'Italian', 'ja': 'Japanese', 'ko': 'Korean', 'pt': 'Portuguese', 'ru': 'Russian', 'zh': 'Chinese'}*

ISO 639-1 codes to the English names some models want (“en” -> “English”).

### *exception* voxy.language.LanguageIgnoredWarning

Bases: [`UserWarning`](https://docs.python.org/3/builtins/exceptions.html#UserWarning)

`language=` asked of a model that takes no language option (dropped).

### *class* voxy.language.LanguageOption(param=None, detects=False, to_value=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

How a model takes a language.

`param`: the synthesis option that pins the language (`None`: the model takes
none, so a requested language is dropped, with a warning). `detects`: the model
guesses the language per request when it is not pinned, so an unpinned call can
come back in the wrong language. `to_value`: turns the caller’s code (‘en’) into
what the model wants (‘English’); `None` sends it as given.

### *exception* voxy.language.LanguageUnsetWarning

Bases: [`UserWarning`](https://docs.python.org/3/builtins/exceptions.html#UserWarning)

No `language=` on a model that guesses the language of each request.

### voxy.language.PASSTHROUGH *= LanguageOption(param='language', detects=False, to_value=None)*

`language` passed through as given.

* **Type:**
  What a model nobody registered gets

### voxy.language.language_name(code)

‘English’ for ‘en’ (and ‘en-US’); anything else unchanged.

```pycon
>>> language_name("en-US"), language_name("English")
('English', 'English')
```

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### voxy.language.language_option(backend, model_id=None)

The language option of `backend`’s model `model_id`: the registered one, else
[`PASSTHROUGH`](#voxy.language.PASSTHROUGH) (`language` sent as given, as before this seam existed).

```pycon
>>> language_option("elevenlabs", "eleven_multilingual_v2").param is None
True
```

* **Return type:**
  [`LanguageOption`](#voxy.language.LanguageOption)

### voxy.language.language_options(backend, model_id, language)

The synthesis options that pin `language` on this model: `{}` when no
language is asked or the model takes none.

```pycon
>>> language_options("fal", "fal-ai/bytedance/seed-speech/tts/v2", "en")
{'language': 'en'}
>>> language_options("fal", "fal-ai/bytedance/seed-speech/tts/v2", None)
{}
```

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### voxy.language.options_for(option, language)

The synthesis options `option` sends for `language` (`{}` for none).

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### voxy.language.register_language_option(backend, model_glob, option)

Declare how `backend`’s models matching `model_glob` take a language
(checked before the ones already registered).

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

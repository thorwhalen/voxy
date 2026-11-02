# voxy.workers.align_adapter

Forced alignment with torchaudio’s `MMS_FA` bundle (Meta’s multilingual CTC
aligner): character spans of words already known to be in the audio.

Run by `voxy.worker` in its own environment; audio is read with soundfile. The
aligner’s weights come from Meta’s MMS release, under a non-commercial licence
(CC-BY-NC-4.0, as far as the MMS release states; check before commercial use).

### Module Attributes

| [`LIGATURES`](#voxy.workers.align_adapter.LIGATURES)        | Letters NFKD leaves whole, spelled with the aligner's letters (French œ, ...).   |
|-------------------------------------------------------------------|----------------------------------------------------------------------------------|
| [`STAR`](#voxy.workers.align_adapter.STAR)             | The aligner's token for a word it has no letters for (a number, a symbol).       |
| [`MIN_CROP_SECONDS`](#voxy.workers.align_adapter.MIN_CROP_SECONDS) | The shortest crop the aligner's convolutions take.                               |

### Functions

| [`align`](#voxy.workers.align_adapter.align)(state, \*, audio, words[, max_seconds])     | `{"words": [{"start", "end", "score", "chars": [...]}, ...]}` for `words` (in order) in `audio`; a word with no letter (a number) is placed as the star token.                                                  |
|----------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `load`(\*[, model, device])                                                                        |                                                                                                                                                                                                                 |
| [`score`](#voxy.workers.align_adapter.score)(state, \*, audio, texts[, start, end, ...]) | `{"scores": [...]}`: for each of `texts`, the log-likelihood the aligner gives it over `audio` (cropped to `start`-`end` seconds when given), i.e. minus the CTC loss; `-inf` for a text too long for the crop. |

### voxy.workers.align_adapter.LIGATURES *= {'ß': 'ss', 'æ': 'ae', 'ø': 'o', 'ł': 'l', 'œ': 'oe', 'ʼ': "'", '’': "'"}*

Letters NFKD leaves whole, spelled with the aligner’s letters (French œ, …).

### voxy.workers.align_adapter.MIN_CROP_SECONDS *= 0.1*

The shortest crop the aligner’s convolutions take.

### voxy.workers.align_adapter.STAR *= '\*'*

The aligner’s token for a word it has no letters for (a number, a symbol).

### voxy.workers.align_adapter.align(state, , audio, words, max_seconds=30.0)

`{"words": [{"start", "end", "score", "chars": [...]}, ...]}` for `words`
(in order) in `audio`; a word with no letter (a number) is placed as the star
token. Audio longer than `max_seconds` is refused (memory grows fast: 150 s
took about 6 GB): align it in stretches.

### voxy.workers.align_adapter.score(state, , audio, texts, start=None, end=None, max_seconds=30.0)

`{"scores": [...]}`: for each of `texts`, the log-likelihood the aligner
gives it over `audio` (cropped to `start`-`end` seconds when given), i.e.
minus the CTC loss; `-inf` for a text too long for the crop. Higher is more
likely; compare texts on the same crop only.

Every candidate must be spelled in letters: a number or a symbol would become the
“any sound” token, which fits any audio and would always win.

# voxy.workers.align_adapter

Forced alignment with torchaudio’s `MMS_FA` bundle (Meta’s multilingual CTC
aligner): character spans of words already known to be in the audio.

Run by `voxy.worker` in its own environment; audio is read with soundfile. The
aligner’s weights come from Meta’s MMS release, under a non-commercial licence
(CC-BY-NC-4.0, as far as the MMS release states; check before commercial use).

### Module Attributes

| [`LIGATURES`](#voxy.workers.align_adapter.LIGATURES)   | Letters NFKD leaves whole, spelled with the aligner's letters (French œ, ...).   |
|--------------------------------------------------------------|----------------------------------------------------------------------------------|
| [`STAR`](#voxy.workers.align_adapter.STAR)        | The aligner's token for a word it has no letters for (a number, a symbol).       |

### Functions

| [`align`](#voxy.workers.align_adapter.align)(state, \*, audio, words[, max_seconds])   | `{"words": [{"start", "end", "score", "chars": [...]}, ...]}` for `words` (in order) in `audio`; a word with no letter (a number) is placed as the star token.   |
|--------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `load`(\*[, model, device])                                                                      |                                                                                                                                                                  |

### voxy.workers.align_adapter.LIGATURES *= {'ß': 'ss', 'æ': 'ae', 'ø': 'o', 'ł': 'l', 'œ': 'oe', 'ʼ': "'", '’': "'"}*

Letters NFKD leaves whole, spelled with the aligner’s letters (French œ, …).

### voxy.workers.align_adapter.STAR *= '\*'*

The aligner’s token for a word it has no letters for (a number, a symbol).

### voxy.workers.align_adapter.align(state, , audio, words, max_seconds=30.0)

`{"words": [{"start", "end", "score", "chars": [...]}, ...]}` for `words`
(in order) in `audio`; a word with no letter (a number) is placed as the star
token. Audio longer than `max_seconds` is refused (memory grows fast: 150 s
took about 6 GB): align it in stretches.

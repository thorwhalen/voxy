# voxy.workers.phonemes_adapter

Phoneme recognition (wav2vec2 espeak CTC): frame-wise log-probabilities of a crop, and espeak phonemization.

### Module Attributes

| [`BLANK_SYMBOL`](#voxy.workers.phonemes_adapter.BLANK_SYMBOL)   | the recogniser's padding symbol.   |
|-----------------------------------------------------------------|------------------------------------|

### Functions

| `emissions`(state, \*, audio[, start, end, ...])                                         |                                                                                         |
|------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|
| `load`(\*[, model, device])                                                              |                                                                                         |
| [`phonemize`](#voxy.workers.phonemes_adapter.phonemize)(state, \*, texts[, language]) | espeak IPA of each text (the symbols the recogniser was trained on), stress marks kept. |

### voxy.workers.phonemes_adapter.BLANK_SYMBOL *= '<pad>'*

the recogniser’s padding symbol.

* **Type:**
  The CTC blank

### voxy.workers.phonemes_adapter.phonemize(state, , texts, language='fr-fr')

espeak IPA of each text (the symbols the recogniser was trained on), stress marks kept.

# voxy.textsplit

Split long text (a chapter) into chunks a speech model can say in one breath.

Local cloning models drift, rush or hallucinate past roughly 20-40 seconds of
speech, so long text is spoken chunk by chunk and the audio joined with
controlled pauses. Paragraphs are kept apart (they get a longer pause); inside a
paragraph, whole sentences are packed into chunks of at most `max_chars`, and
only a sentence longer than that is cut, at a clause break if it has one.

```pycon
>>> split_text("One. Two!\n\nThree is a longer sentence, with a clause.", max_chars=20)
[['One. Two!'], ['Three is a longer', 'sentence,', 'with a clause.']]
```

### Module Attributes

| [`ABBREVIATIONS`](#voxy.textsplit.ABBREVIATIONS)   | Words ending in a period that don't end a sentence ("Mr.   |
|------------------------------------------------------------------|------------------------------------------------------------|

### Functions

| [`split_sentences`](#voxy.textsplit.split_sentences)(paragraph)        | The sentences of `paragraph` (whitespace normalised).                        |
|------------------------------------------------------------------------------------|------------------------------------------------------------------------------|
| [`split_text`](#voxy.textsplit.split_text)(text, \*[, max_chars]) | Paragraphs of chunks: `[[chunk, ...], ...]`, each chunk at most `max_chars`. |

### voxy.textsplit.ABBREVIATIONS *= frozenset({'capt', 'ch', 'col', 'dr', 'etc', 'fig', 'gen', 'jr', 'lt', 'mr', 'mrs', 'ms', 'mt', 'no', 'prof', 'rev', 'sgt', 'sr', 'st', 'vol', 'vs'})*

Words ending in a period that don’t end a sentence (“Mr. Darcy”).

### voxy.textsplit.split_sentences(paragraph)

The sentences of `paragraph` (whitespace normalised).

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

```pycon
>>> split_sentences('He said "Go." Then   Mr. Darcy left.')
['He said "Go."', 'Then Mr. Darcy left.']
```

### voxy.textsplit.split_text(text, , max_chars=300)

Paragraphs of chunks: `[[chunk, ...], ...]`, each chunk at most `max_chars`.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]

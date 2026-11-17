# voxy.direction

Delivery directions: one way to write *how* a line is said, rendered for whatever voice says it.

A script marks delivery inline, in the bracket notation ElevenLabs’ v3 models read:
`[dryly] We all believe in agility. [beat] But then...`. That notation is voxy’s
authoring form for any backend. Each backend has a **dialect**, the way it takes
direction, and [`plan_directions()`](#voxy.direction.plan_directions) turns a directed text into what that dialect
accepts:

| dialect       | what happens to `[dryly]`, `[beat]`, `[laughs]`                                                                                                                                                                                 |
|---------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `inline-tags` | sent as written: the model performs them (eleven_v3, eleven_v4)                                                                                                                                                                 |
| `instruction` | the text is cut where the direction changes; each part is sent<br/>with a natural-language instruction (Seed Speech’s<br/>`voice_instruction`, OpenAI’s `instructions`) built from its<br/>tags; pause tags become real silence |
| `plain`       | tags removed, pause tags become real silence (every other model)                                                                                                                                                                |

Which dialect a backend speaks is [`dialect_for()`](#voxy.direction.dialect_for) (`backend`, `model_id`), a
registry you can extend with [`register_dialect()`](#voxy.direction.register_dialect); `SpeechModel.markup_dialect`
asks it. The vocabulary (what each tag means as an instruction, and how long a pause
tag is) is [`DIRECTIONS`](#voxy.direction.DIRECTIONS); an unknown tag still becomes an instruction
(“Deliver this <tag>.”), so writers are not limited to the table.

```pycon
>>> t = "[dryly] We all agree. [beat] Then we disagree."
>>> [(d.text, d.options) for d in plan_directions(t, PLAIN)]
[('We all agree. <break time="0.45s" /> Then we disagree.', {})]
>>> [(d.text, d.options) for d in plan_directions(t, INLINE)]
[('[dryly] We all agree. [beat] Then we disagree.', {})]
>>> for d in plan_directions(t, instruction_dialect("voice_instruction")):
...     print(repr(d.text), d.options)
'We all agree. <break time="0.45s" /> Then we disagree.' {'voice_instruction': 'Delivery: dry, understated, a little ironic.'}
```

### Module Attributes

| [`DIRECTION_RE`](#voxy.direction.DIRECTION_RE)   | `[word]` or `[a few words]`, lowercase, as eleven_v3 reads them.   |
|-----------------------------------------------------------------|--------------------------------------------------------------------|
| [`DIRECTIONS`](#voxy.direction.DIRECTIONS)     | tag -> meaning.                                                    |

### Functions

| [`compose_instruction`](#voxy.direction.compose_instruction)(instruction, tags)            | The instruction for a part: the voice's standing `instruction`, then its tags.                                      |
|----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------|
| [`dialect_for`](#voxy.direction.dialect_for)(backend[, model_id])                  | The dialect of `backend`'s model `model_id` (`PLAIN` when unknown).                                                 |
| [`direction`](#voxy.direction.direction)(tag)                                    | The meaning of `tag` (unknown tags read as a delivery instruction).                                                 |
| [`directions`](#voxy.direction.directions)(text)                                  | The inline direction tags in `text`, in order.                                                                      |
| [`instruction_dialect`](#voxy.direction.instruction_dialect)(param)                        | A dialect taking a natural-language instruction in the option `param`.                                              |
| [`plan_directions`](#voxy.direction.plan_directions)(text, dialect, \*[, instruction]) | The requests that say `text` in `dialect` (see the module docstring).                                               |
| [`register_dialect`](#voxy.direction.register_dialect)(backend, model_glob, dialect)    | Declare that `backend`'s models matching `model_glob` speak `dialect` (checked before the ones already registered). |
| [`strip_directions`](#voxy.direction.strip_directions)(text)                            | `text` with its direction tags removed and spacing tidied.                                                          |

### Classes

| [`Dialect`](#voxy.direction.Dialect)(name[, inline_tags, instruction_param])   | How a backend takes direction.                                                                                                   |
|----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------|
| [`Directed`](#voxy.direction.Directed)(text[, options])                         | One request's worth of a directed text: its text (SSML breaks allowed) and the extra synthesis `options` carrying its direction. |
| [`Direction`](#voxy.direction.Direction)(kind[, instruction, pause_s])           | What one tag asks for: a `pause_s` of silence, or an `instruction` phrase.                                                       |

### voxy.direction.DIRECTIONS *: [Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [Direction](#voxy.direction.Direction)]* *= mappingproxy({'pause': Direction(kind='pause', instruction='', pause_s=0.7), 'short pause': Direction(kind='pause', instruction='', pause_s=0.3), 'long pause': Direction(kind='pause', instruction='', pause_s=1.4), 'beat': Direction(kind='pause', instruction='', pause_s=0.45), 'dryly': Direction(kind='delivery', instruction='dry, understated, a little ironic', pause_s=0.0), 'wryly': Direction(kind='delivery', instruction='wry, with a knowing half-smile', pause_s=0.0), 'deadpan': Direction(kind='delivery', instruction='deadpan: flat, letting the line land by itself', pause_s=0.0), 'mock-solemn': Direction(kind='delivery', instruction='mock-solemn, pretending to be grave', pause_s=0.0), 'emphatic': Direction(kind='delivery', instruction='emphatic, leaning on the key words', pause_s=0.0), 'earnest': Direction(kind='delivery', instruction='earnest and sincere', pause_s=0.0), 'thoughtful': Direction(kind='delivery', instruction='thoughtful and unhurried, thinking it through', pause_s=0.0), 'curious': Direction(kind='delivery', instruction='curious, genuinely asking', pause_s=0.0), 'amused': Direction(kind='delivery', instruction='amused, smiling', pause_s=0.0), 'sarcastic': Direction(kind='delivery', instruction='lightly sarcastic', pause_s=0.0), 'excited': Direction(kind='delivery', instruction='excited, with energy', pause_s=0.0), 'incredulous': Direction(kind='delivery', instruction='incredulous', pause_s=0.0), 'serious': Direction(kind='delivery', instruction='serious and grounded', pause_s=0.0), 'warmly': Direction(kind='delivery', instruction='warm', pause_s=0.0), 'firmly': Direction(kind='delivery', instruction='firm', pause_s=0.0), 'quietly': Direction(kind='delivery', instruction='quieter, softer', pause_s=0.0), 'softly': Direction(kind='delivery', instruction='soft', pause_s=0.0), 'slowly': Direction(kind='delivery', instruction='slow and deliberate', pause_s=0.0), 'rushed': Direction(kind='delivery', instruction='quick, a little rushed', pause_s=0.0), 'whispers': Direction(kind='delivery', instruction='whispered', pause_s=0.0), 'laughs': Direction(kind='nonverbal', instruction='with a laugh in the voice', pause_s=0.0), 'laughs softly': Direction(kind='nonverbal', instruction='with a soft laugh in the voice', pause_s=0.0), 'chuckles': Direction(kind='nonverbal', instruction='with a chuckle in the voice', pause_s=0.0), 'sighs': Direction(kind='nonverbal', instruction='with a sigh', pause_s=0.0)})*

tag -> meaning. Pause tags are silence everywhere but on an
inline-tags model (which performs them itself); the rest become instructions.

* **Type:**
  The vocabulary

### voxy.direction.DIRECTION_RE *= re.compile('\\\\[([a-z][a-z \\\\-]{1,24})\\\\]')*

`[word]` or `[a few words]`, lowercase, as eleven_v3 reads them.

* **Type:**
  An inline direction

### *class* voxy.direction.Dialect(name, inline_tags=False, instruction_param=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

How a backend takes direction.

`inline_tags`: the model performs `[tags]` itself, so the text is sent as written.
`instruction_param`: the synthesis option that takes a natural-language delivery
instruction (`None`: the model takes none).

### *class* voxy.direction.Directed(text, options=dict())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

One request’s worth of a directed text: its text (SSML breaks allowed) and the
extra synthesis `options` carrying its direction.

### *class* voxy.direction.Direction(kind, instruction='', pause_s=0.0)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

What one tag asks for: a `pause_s` of silence, or an `instruction` phrase.

### voxy.direction.compose_instruction(instruction, tags)

The instruction for a part: the voice’s standing `instruction`, then its tags.

```pycon
>>> compose_instruction("A wry essayist.", ["dryly", "laughs softly"])
'A wry essayist. Delivery: dry, understated, a little ironic; with a soft laugh in the voice.'
```

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### voxy.direction.dialect_for(backend, model_id=None)

The dialect of `backend`’s model `model_id` (`PLAIN` when unknown).

```pycon
>>> dialect_for("elevenlabs", "eleven_v3").name, dialect_for("elevenlabs", "eleven_multilingual_v2").name
('inline-tags', 'plain')
>>> dialect_for("fal", "fal-ai/bytedance/seed-speech/tts/v2").instruction_param
'voice_instruction'
```

* **Return type:**
  [`Dialect`](#voxy.direction.Dialect)

### voxy.direction.direction(tag)

The meaning of `tag` (unknown tags read as a delivery instruction).

```pycon
>>> direction("beat").pause_s, direction("tenderly").instruction
(0.45, 'tenderly')
```

* **Return type:**
  [`Direction`](#voxy.direction.Direction)

### voxy.direction.directions(text)

The inline direction tags in `text`, in order.

```pycon
>>> directions("[dryly] Fine. [beat] Really.")
['dryly', 'beat']
```

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### voxy.direction.instruction_dialect(param)

A dialect taking a natural-language instruction in the option `param`.

* **Return type:**
  [`Dialect`](#voxy.direction.Dialect)

### voxy.direction.plan_directions(text, dialect, , instruction=None)

The requests that say `text` in `dialect` (see the module docstring).

`instruction` is a standing delivery instruction (a character, a register) sent
with every part on an instruction dialect, and ignored by the others.

```pycon
>>> d = instruction_dialect("instructions")
>>> [(p.text, p.options["instructions"]) for p in plan_directions(
...     "[curious] What gives? [emphatic] Never.", d, instruction="An essayist.")]
[('What gives?', 'An essayist. Delivery: curious, genuinely asking.'), ('Never.', 'An essayist. Delivery: emphatic, leaning on the key words.')]
```

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Directed`](#voxy.direction.Directed)]

### voxy.direction.register_dialect(backend, model_glob, dialect)

Declare that `backend`’s models matching `model_glob` speak `dialect`
(checked before the ones already registered).

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

### voxy.direction.strip_directions(text)

`text` with its direction tags removed and spacing tidied.

```pycon
>>> strip_directions("[wryly] I have met , [beat] some.")
'I have met, some.'
```

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

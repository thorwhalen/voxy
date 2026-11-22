# voxy.plan

The requests one model needs for a directed text: the text and options of each, and the
silence to splice after it, without synthesizing anything.

A caller that sends its own requests (its own client, cache key and billing) still needs
voxy’s knowledge of how a model takes a directed line: its markup dialect
([`voxy.direction`](voxy.direction.html.md#module-voxy.direction)) and the longest pause it renders itself ([`voxy.pauses`](voxy.pauses.html.md#module-voxy.pauses)).
[`plan_requests()`](#voxy.plan.plan_requests) combines the two, as `voxy.text_to_speech` does before speaking.

```pycon
>>> [(r.text, r.options, r.pause_s) for r in plan_requests('Hello. <break time="5s"/> [whispers] Bye.', "elevenlabs", "eleven_v3")]
[('Hello.', {}, 5.0), ('[whispers] Bye.', {}, 0.0)]
>>> [(r.text, r.pause_s) for r in plan_requests('A. <break time="1s"/> B.', "elevenlabs", "eleven_multilingual_v2")]
[('A. <break time="1s" /> B.', 0.0)]
```

### Functions

| [`plan_requests`](#voxy.plan.plan_requests)(text, backend[, model_id, ...])   | The requests that speak `text` on `backend`'s model `model_id`, in order.   |
|--------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------|

### Classes

| [`Request`](#voxy.plan.Request)(text[, options, pause_s])   | One request to send: its `text` (SSML breaks the model renders itself kept inline), the synthesis `options` carrying its direction, and the `pause_s` of silence to splice after its audio.   |
|--------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

### *class* voxy.plan.Request(text, options=dict(), pause_s=0.0)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

One request to send: its `text` (SSML breaks the model renders itself kept
inline), the synthesis `options` carrying its direction, and the `pause_s` of
silence to splice after its audio. A request with no text is a pause alone.

### voxy.plan.plan_requests(text, backend, model_id=None, , instruction=None, dialect=None, native_break_s=None)

The requests that speak `text` on `backend`’s model `model_id`, in order.

`text` may hold delivery directions (`[dryly]`, `[beat]`) and SSML
`<break>``s; ``instruction` is a direction for the whole text. The model’s
dialect ([`voxy.direction.dialect_for()`](voxy.direction.html.md#voxy.direction.dialect_for)) decides how directions are sent, and
the longest break it renders ([`voxy.pauses.native_break_for()`](voxy.pauses.html.md#voxy.pauses.native_break_for)) decides which
breaks stay in the text and which become silence between requests. `dialect` and
`native_break_s` override the two lookups (a model object passes its own). Pass
the model id the caller will send: an unknown model is planned conservatively (plain
text, every break cut into silence).

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Request`](#voxy.plan.Request)]

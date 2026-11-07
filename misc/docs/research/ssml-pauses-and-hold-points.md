# Standards for timed and user-gated pauses in narration (checked 2026-11-07)

1. Recommendation: author pauses as SSML `<break time="Ns"/>` (the W3C SSML 1.1 syntax, [1]) and use SSML `<mark name="..."/>` ([1]) for named hold points; do not invent new tag syntax. Optionally accept Speech Markdown `[5s]` ([13]) as sugar that compiles to `<break>`.
2. voxy (the text-to-speech facade) should split text at every break longer than the engine can render (ElevenLabs: 3 s per tag, and v3/v4 do not support break at all [2][3]) and insert real silence between the TTS calls; the spec itself sets no maximum, so splitting is the only portable path.
3. For "hold until the user continues, or for N seconds", the closest standard is SMIL 3.0 timing: an element with `end="continueBtn.activateEvent"` and `max="30s"` (or `dur`) [8]. Neither EPUB Media Overlays nor VoiceXML models this well [9][10].
4. For the rendered file, store pause points as chapter/marker metadata: Podcasting 2.0 JSON chapters or WebVTT chapters for web playback, ID3 CHAP for embedding; keep a `hold: true|false` flag in a sidecar since none of these has a wait semantic.
5. Unverified: Google's maximum break length, whether ElevenLabs v3/v4 read `<break>` aloud, and primary-source text for Podcasting 2.0 and OTIO markers (details below).

## 1. W3C SSML 1.1 `<break>`

The `break` element is empty and has two optional attributes, `strength` and `time` [1]. `strength` takes `none`, `x-weak`, `weak`, `medium` (default), `strong`, `x-strong`; "none" suppresses a break the processor would otherwise insert, and the others are non-decreasing in strength, "typically accompanied by pauses" [1]. `time` follows the CSS2 time format, examples `250ms` and `3s`; the closest explicit grammar is the spec's Time Designation (non-negative real number followed by `ms` or `s`) [1]. With `time`, the pause lasts that duration; with neither attribute, the break is stronger than the processor would otherwise produce [1].

The spec gives no numeric duration for each strength level (it only requires them to be ordered; engines choose, for example Azure maps x-weak..x-strong to 250..1250 ms [4]). The spec states no maximum for `time` [1]. Maxima are therefore engine-specific (section 2).

## 2. Per-engine break support and limits

Amazon Polly: `break` is supported on standard, neural, long-form and generative voices [5][6]. `time` accepts `Ns` (max `10s`) or `Nms` (max `10000ms`); strength maps to the comma, sentence and paragraph pause; `x-weak` equals none [6].

Azure Speech: `time` in seconds or milliseconds, valid 0 to 20000 ms, larger values are clamped to 20000 ms, and `strength` is ignored when `time` is set [4]. Strength durations are 250/500/750/1000/1250 ms [4]. Azure also offers `mstts:silence` (also capped at 20000 ms) but only at text boundaries [4]. HD voices, personal voices and embedded voices do not support all tags [4]; I did not open the HD voice page, so break support there is unverified.

Google Cloud TTS: `time` in seconds or milliseconds, `strength` values include `none` [7]. The page states no maximum, and says nothing about Chirp 3 HD or Journey voices, so both are unverified [7].

ElevenLabs: the best-practices page says `<break time="x.xs" />` gives natural pauses "up to 3 seconds", a hard maximum of 3 s per tag, and warns that many breaks in one generation can cause speed-ups, noise or artifacts [2]. It states "Eleven v4 and Eleven v3 do not support SSML break tags" and recommends audio tags such as `[Brief pause]`, punctuation (ellipses, dashes) and text structure instead, noting these are less consistent than breaks [2]. The v4 page says "SSML is not supported" and documents no pause tag [3]. The model page lists `eleven_v4`, `eleven_v4_turbo`, `eleven_multilingual_v2`, `eleven_flash_v2_5` as current, `eleven_v3` as previous generation (not deprecated) and `eleven_turbo_v2_5` as deprecated in favour of `eleven_flash_v2_5` with no removal date [11]. The docs pages do not say which older models support `break` [2][11]. An ElevenLabs help-center article, seen only through search-result snippets (the page returned 403), says all models except v3 support break tags via the API, up to 3 s [12]. That article predates v4 and conflicts in scope with [2], so I treat Multilingual v2, Flash v2.5 and Turbo v2.5 as "supported, 3 s cap" with the help-center source as secondary evidence.

| ElevenLabs model | `<break>` support | Max per tag | Source / confidence |
|---|---|---|---|
| eleven_multilingual_v2 | yes (API) | 3 s | help center snippet [12], docs [2] do not contradict; medium |
| eleven_flash_v2_5 | yes (API) | 3 s | same; medium |
| eleven_turbo_v2_5 | yes (API), model deprecated | 3 s | [12], deprecation from [11]; medium |
| eleven_v3 | no | n/a | [2] explicit; high |
| eleven_v4 (and v4_turbo) | no ("SSML is not supported") | n/a | [2][3] explicit; high |

No ElevenLabs model honours `<break>` beyond 3 s in anything I could find. Whether v3/v4 read `<break>` aloud or silently drop it is not documented in [2] or [3]; this could not be verified, and a one-call test against the API is the only way to know. Safe approach: strip `<break>` before sending text to v3/v4, which is what voxy does (it never sends a break a model does not render).

## 3. SSML `<mark>`

`mark` is an empty element with a required `name` attribute. It marks a position in the text/tag sequence; the processor must report the name and position to the host, or issue an event with the name when audio output reaches the mark, and it does not affect the speech [1]. It is for playback synchronisation and notification, not for pausing.

Engine support: Polly emits SSML marks as speech marks of type `ssml` (via the speech-mark output type); the Polly tag table lists `mark` as full on neural and long-form and partial on generative [5]. I did not open the speech-marks page, so the output format is from memory and unverified. Azure uses a `bookmark` element (`mark` attribute) and a `BookmarkReached` SDK event giving the audio offset; the element is not spoken [4]. Google returns timepoints only when `<mark name>` is used and `TimepointType` is set to `SSML_MARK`; consecutive marks may not generate events [7]. ElevenLabs: `mark` is not mentioned in the docs I read [2][3], so marks must be tracked by voxy itself from the split segment boundaries. VoiceXML 2.1 also uses `mark` to report where barge-in happened (`marktime`) [10].

## 4. Standards for audio that waits for a user action

SMIL 3.0 timing [8]: `dur="indefinite"` gives an element no fixed simple duration; an element with an `end` but no `dur` has an indefinite simple duration bounded by the `end` value. Event-based timing: `begin="show.activateEvent"` starts when the user activates element `show`; `end="activateEvent"` ends it on a click; `end="activateEvent; next.activateEvent"` combined with `dur` ends at the earlier of the time limit or a click. `max="10s"` caps the active duration (default indefinite), `min` sets a floor. If the triggering event never occurs, the element never ends. This is exactly "hold for up to N seconds or until Continue is activated".

EPUB 3 Media Overlays [9]: a subset of SMIL 3.0 (`smil`, `seq`, `par`, `text`, `audio` with `clipBegin`/`clipEnd`); the overlay section I read defines no `dur`, `begin` or `end` attributes and nothing about pauses or waiting (the schema in appendix G.3 is authoritative and I did not read it). Skippable/escapable structures are reading-system behaviours, not waits. So EPUB overlays do not support waits.

VoiceXML 2.x [10]: a form-based dialog model where the interpreter "waits for and collects an input or an event (such as a no input timeout)" [10]; this is the right behavioural model (play prompt, wait for input, timeout event) but it is a telephony dialog language, heavy, and I could not read the `timeout` property text (section 6.3 of VoiceXML 2.0) so the exact syntax is unverified.

Closest standard model: SMIL timing (`end` on `activateEvent` plus `max`/`dur`). Practical mapping: the renderer's JSON/sidecar uses a pause object `{at, duration, hold}` and the player implements it with the SMIL semantics.

## 5. Lightweight text syntaxes for pauses

Speech Markdown: shorthand `[3s]` and `[250ms]` compile to `<break time="3s"/>` (example `Sample [3s] speech [250ms] markdown`), and the "standard format" has break-by-time and break-by-strength variants [13]. A Rust grammar shows `[break:...]` taking named strengths (none..x-strong) [14]. The form `[break:"3s"]` could not be confirmed (the speechmarkdown.org site returned 403), so treat it as unverified.

SSMD (Speech Synthesis Markdown), a separate project: `...s` for a strong break and `...5s` for 5 seconds, with a 10 s maximum [15]; this is lower-adoption and unverified beyond the search snippet.

ElevenLabs v3/v4 style bracketed audio tags (`[Brief pause]`) are free-form prompts, not a standard [2]. Plain punctuation (ellipsis, dash) is the universal fallback but imprecise [2].

## 6. Chapter and marker interchange for the rendered file

WebVTT chapters: a chapter cue is a cue whose text is the chapter title, used as a navigation target; chapter cues are consecutive and non-overlapping, can nest, and the track kind is set by HTML `<track kind="chapters">` [16]. Fits web playback and is human-editable, but cues have no flag for "hold".

Podcasting 2.0 JSON chapters: a JSON file with a chapters array of objects (`startTime`, `title`, optional `img`, `url`, `toc`, `endTime`, `location`). Fits point-in-time pause markers with a title and is easy to extend; I could not fetch the primary document (404), so the field list is from memory and unverified. Note `toc: false` hides a chapter from the table of contents, which suits pure pause points.

ID3v2 CHAP/CTOC: CHAP frames carry an element ID, start and end times in ms, byte offsets (ignored if all 0xFF) and optional TIT2/TIT3 sub-frames; CTOC arranges them in a tree; chapters may overlap or leave gaps [17]. Best for embedding in an MP3 so any chapter-aware player sees the pause points; no wait semantics.

OpenTimelineIO markers: a Marker has a `marked_range`, `name`, `color` and free `metadata` dictionary (from memory; the tutorial page I fetched did not cover markers [18]). Fits an editing-oriented interchange where `metadata` could carry `{"hold": true}`, but it is an editorial format, heavier than a player needs.

Suggested fit: write Podcasting 2.0 JSON (or WebVTT) as the sidecar for the interactive player, optionally embed ID3 CHAP in the MP3, and put the hold semantics in an extension field.

## What voxy implements (2026-11-07)

`voxy.pauses` follows the recommendation. `parse_sections(text)` cuts a text at every SSML `<break>` (`time` in `s`/`ms`, or `strength` with Azure's durations [4]) and every `<mark name>`, giving `Section(text, pause_s, mark)`. `plan_sections(text, native_break_s=...)` is the linear plan: breaks the backend renders stay inline, marks are dropped, and the text is cut at every other break. `voxy.text_to_speech` uses it automatically: each backend says how long a break it renders (`SpeechModel.native_break_s`; ElevenLabs v2 family 3 s, v3/v4 none, local models none), and when a cut is needed the parts are spoken separately and joined as WAV with the exact silence. Text without markup is sent unchanged. Speech Markdown's `[3s]` shorthand is not accepted: it collides with ElevenLabs' bracketed audio tags.

The interactive case (hold at a point until the listener continues, or for N seconds) needs no change to this core: a player reads `parse_sections`, speaks section by section, and applies SMIL's `end="<id>.activateEvent"` with `max=` semantics [8] at each mark. Its pause records would go in a Podcasting 2.0 JSON or WebVTT chapters sidecar with a hold flag, since none of those formats has a wait.

## REFERENCES

[1] Speech Synthesis Markup Language (SSML) Version 1.1. W3C; 2010. [W3C SSML 1.1](https://www.w3.org/TR/speech-synthesis11/)
[2] Best practices, Text to Speech. ElevenLabs; 2026. [ElevenLabs best practices](https://elevenlabs.io/docs/overview/capabilities/text-to-speech/best-practices)
[3] Eleven v4. ElevenLabs; 2026. [ElevenLabs v4](https://elevenlabs.io/docs/overview/capabilities/text-to-speech/eleven-v4)
[4] SSML document structure and events. Microsoft; 2026. [Azure Speech SSML structure](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/speech-synthesis-markup-structure)
[5] Supported SSML tags. Amazon Web Services; 2026. [Polly supported tags](https://docs.aws.amazon.com/polly/latest/dg/supportedtags.html)
[6] Adding a pause (break tag). Amazon Web Services; 2026. [Polly break tag](https://docs.aws.amazon.com/polly/latest/dg/break-tag.md)
[7] SSML reference. Google Cloud; 2026. [Google Cloud TTS SSML](https://docs.cloud.google.com/text-to-speech/docs/ssml)
[8] Synchronized Multimedia Integration Language (SMIL 3.0), The timing and synchronization module. W3C; 2008. [SMIL 3.0 timing](https://www.w3.org/TR/SMIL3/smil-timing.html)
[9] EPUB 3.3, Media Overlays. W3C; 2023. [EPUB 3.3](https://www.w3.org/TR/epub-33/)
[10] Voice Extensible Markup Language (VoiceXML) 2.1 and 2.0. W3C; 2007, 2004. [VoiceXML 2.1](https://www.w3.org/TR/voicexml21/), [VoiceXML 2.0](https://www.w3.org/TR/voicexml20/)
[11] Models. ElevenLabs; 2026. [ElevenLabs models](https://elevenlabs.io/docs/overview/models)
[12] Do pauses and SSML phoneme tags work with the API? ElevenLabs Help Center; n.d. (page not fetchable, snippet seen via search). [help article](https://help.elevenlabs.io/hc/en-us/articles/24352686926609)
[13] speechmarkdown-js. Speech Markdown project; 2026. [GitHub](https://github.com/speechmarkdown/speechmarkdown-js)
[14] speechmarkdown-rust grammar. docs.rs; 2026. [grammar.pest](https://docs.rs/crate/speechmarkdown-rust/0.1.10/source/src/grammar.pest)
[15] SSMD specification. npm package ssmd 1.0.8; 2026. [SPECIFICATION.md](https://app.unpkg.com/ssmd@1.0.8/files/SPECIFICATION.md)
[16] WebVTT: The Web Video Text Tracks Format. W3C; 2019. [WebVTT](https://www.w3.org/TR/webvtt1/)
[17] ID3v2 Chapter Frame Addendum 1.0. id3.org; 2005. [ID3 chapters](https://id3.org/id3v2-chapters-1.0)
[18] OpenTimelineIO timeline structure. Academy Software Foundation; 2026. [OTIO docs](https://opentimelineio.readthedocs.io/en/latest/tutorials/otio-timeline-structure.html)

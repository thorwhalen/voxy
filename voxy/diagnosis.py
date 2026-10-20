"""
Diagnosis pages: render one text through a grid of (model x parameters), listen side by side.

    >>> from voxy import diagnosis                                         # doctest: +SKIP
    >>> variants = diagnosis.expand_grid([
    ...     {"backend": "f5tts", "params": {"nfe_step": [16, 32], "cfg_strength": 2}},
    ...     {"backend": "chatterbox", "params": {"exaggeration": [0.3, 0.6]}},
    ... ])                                                                  # doctest: +SKIP
    >>> results = diagnosis.render_grid(text, references, variants, "out/")  # doctest: +SKIP
    >>> diagnosis.build_page(results, "out/", title="First listen")         # doctest: +SKIP

Or from a spec file: ``python -m voxy diagnose spec.json --out DIR --publish``.

Every variant has a stable ID (``F5-7KQ2``) computed from its backend, model
options, parameters and reference clip, never from the text, so "use F5-7KQ2"
means the same configuration on every page and every chapter. The page shows,
per variant, the ID, an audio player, the parameters (those that differ from the
model's defaults in bold), generation time and real-time factor, and lets each
listener rate it (1-5) and write a note. Ratings stay in the listener's browser;
"Copy my notes" turns them into a text block to paste into a message.

For a page shared with other listeners: ``notes_sink`` saves every rating and
note as it is made (to an annoquest server, whose owner page then lists them per
listener), ``appendix`` folds the variants of some references into a collapsed
section, and ``imported`` adds clips rendered elsewhere (a cloud service).
"""

import base64
import hashlib
import html
import itertools
import json
import os
import shutil
import subprocess
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

RESULTS_FILE = "results.json"
AUDIO_DIR = "audio"
PAGE_FILE = "index.html"
DFLT_AUDIO_FORMAT = "mp3"
DFLT_MP3_BITRATE = "192k"
ID_CHARS = 4

#: Short, readable ID prefixes (other backends use their first two letters).
BACKEND_PREFIXES = {
    "f5tts": "F5",
    "chatterbox": "CB",
    "xtts": "XT",
    "voxcpm": "VX",
    "qwen3tts": "QW",
    "elevenlabs": "EL",
}

#: Display names for page sections and default variant labels.
BACKEND_NAMES = {
    "f5tts": "F5-TTS",
    "chatterbox": "Chatterbox",
    "xtts": "XTTS-v2",
    "voxcpm": "VoxCPM2",
    "qwen3tts": "Qwen3-TTS",
    "elevenlabs": "ElevenLabs",
}
ANNOQUEST_REQUEST_FILE = "annoquest-request.json"
DFLT_RATING_PROMPT = (
    "How much does this sound like the real voice, and how natural is it?"
)


@dataclass
class Variant:
    """One configuration to render: a backend, its model options and parameters."""

    backend: str
    params: dict = field(default_factory=dict)
    init: dict = field(default_factory=dict)
    reference: str = ""
    label: str = ""

    @property
    def id(self) -> str:
        """Stable ID from backend, init, params and reference (not the text).

        >>> Variant("f5tts", {"nfe_step": 32}, reference="r1").id
        'F5-...'
        >>> Variant("f5tts", {"nfe_step": 32}).id == Variant("f5tts", {"nfe_step": 32}).id
        True
        """
        content = json.dumps(
            [self.backend, self.init, self.params, self.reference], sort_keys=True
        )
        digest = hashlib.sha256(content.encode()).digest()
        code = base64.b32encode(digest).decode()[:ID_CHARS]
        prefix = BACKEND_PREFIXES.get(self.backend, self.backend[:2].upper())
        return f"{prefix}-{code}"


def _as_list(value) -> list:
    return value if isinstance(value, list) else [value]


def expand_grid(
    grid: Iterable[Mapping], *, references: Iterable[str] = ("",)
) -> list[Variant]:
    """Variants from grid rows: list values are swept (cartesian product), scalars fixed.

    A row: ``{"backend": ..., "params": {...}, "init": {...}, "label": ...}``.
    ``init`` values may be lists too (e.g. ``{"variant": ["original", "turbo"]}``).
    Every variant is rendered once per reference name.

    >>> vs = expand_grid([{"backend": "f5tts", "params": {"nfe_step": [16, 32], "speed": 1}}])
    >>> [(v.backend, v.params) for v in vs]
    [('f5tts', {'nfe_step': 16, 'speed': 1}), ('f5tts', {'nfe_step': 32, 'speed': 1})]
    """
    variants = []
    for row in grid:

        def combos(d):
            keys = list(d)
            for values in itertools.product(*(_as_list(d[k]) for k in keys)):
                yield dict(zip(keys, values))

        for init in combos(row.get("init", {})):
            for params in combos(row.get("params", {})):
                for ref in references:
                    variants.append(
                        Variant(row["backend"], params, init, ref, row.get("label", ""))
                    )
    return variants


def _encode_audio(wav_bytes: bytes, path: Path, audio_format: str) -> Path:
    """Write the audio as WAV, or transcode it with ffmpeg (mp3 for phones)."""
    if audio_format == "wav" or shutil.which("ffmpeg") is None:
        path = path.with_suffix(".wav")
        path.write_bytes(wav_bytes)
        return path
    path = path.with_suffix(f".{audio_format}")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", "pipe:0"]
    if audio_format == "mp3":
        cmd += ["-b:a", DFLT_MP3_BITRATE]
    subprocess.run(cmd + [str(path)], input=wav_bytes, check=True)
    return path


def _audio_seconds(wav_bytes: bytes) -> float:
    from voxy.wavutil import read_wav

    samples, sr = read_wav(wav_bytes)
    return len(samples) / sr


def render_grid(
    text: str,
    references: Mapping,
    variants: Iterable[Variant],
    out_dir: str | os.PathLike,
    *,
    model_factory: Callable | None = None,
    audio_format: str = DFLT_AUDIO_FORMAT,
    resume: bool = True,
    log=print,
) -> list[dict]:
    """Render every variant to ``out_dir/audio/<id>.<format>``; return (and save) the results.

    Models are built one backend configuration at a time and closed before the
    next, so only one model is in memory. With ``resume``, variants already in
    ``results.json`` (without error) are not rendered again.

    Args:
        text: What every variant says.
        references: Reference name -> voice (a ``VoiceProfile`` or ``{"ref_audio", "ref_text"}``).
        variants: From ``expand_grid`` (their ``reference`` names key ``references``).
        model_factory: ``(backend, **init) -> SpeechModel`` (default ``create_speech_model``).
        audio_format: 'mp3' (default, small) or 'wav'.
    """
    from voxy.base import create_speech_model

    model_factory = model_factory or create_speech_model
    out = Path(out_dir)
    (out / AUDIO_DIR).mkdir(parents=True, exist_ok=True)
    results_path = out / RESULTS_FILE
    done = {}
    if resume and results_path.exists():
        done = {
            r["id"]: r
            for r in json.loads(results_path.read_text())["results"]
            if not r.get("error")
        }
    results = []

    def save():
        results_path.write_text(
            json.dumps({"text": text, "results": results}, indent=2, default=str)
        )

    def config_key(v):
        return v.backend, json.dumps(v.init, sort_keys=True)

    variants = sorted(variants, key=config_key)
    for (backend, _), group in itertools.groupby(variants, key=config_key):
        group = list(group)
        todo = [v for v in group if v.id not in done]
        results += [done[v.id] for v in group if v.id in done]
        if not todo:
            continue
        model = None
        try:
            model = model_factory(backend, **group[0].init)
            for v in todo:
                entry = {**asdict(v), "id": v.id}
                log(f"voxy.diagnosis: {v.id} {backend} {v.init} {v.params}")
                try:
                    start = time.perf_counter()
                    speech = model.synthesize(text, references[v.reference], **v.params)
                    seconds = time.perf_counter() - start
                    audio_s = _audio_seconds(speech.audio)
                    path = _encode_audio(
                        speech.audio, out / AUDIO_DIR / v.id, audio_format
                    )
                    entry.update(
                        audio=f"{AUDIO_DIR}/{path.name}",
                        generation_seconds=round(seconds, 1),
                        audio_seconds=round(audio_s, 1),
                        rtf=round(seconds / audio_s, 2) if audio_s else None,
                    )
                except Exception as e:  # one bad variant must not sink the grid
                    entry["error"] = f"{type(e).__name__}: {e}"[:2000]
                results.append(entry)
                save()
        except Exception as e:
            for v in todo:
                results.append(
                    {
                        **asdict(v),
                        "id": v.id,
                        "error": f"{type(e).__name__}: {e}"[:2000],
                    }
                )
            save()
        finally:
            close = getattr(model, "close", None)
            if close:
                close()
    save()
    return results


# the page ---------------------------------------------------------------------

_CSS = """
:root{--bg:#fbfaf7;--fg:#1d1d1f;--muted:#6b6b70;--card:#fff;--line:#e3e1dc;--accent:#2f5d8a;--hi:#fff4c2}
@media (prefers-color-scheme:dark){:root{--bg:#161618;--fg:#ececee;--muted:#9a9aa0;--card:#202023;--line:#333336;--accent:#8db7e8;--hi:#4a4220}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:860px;margin:0 auto;padding:16px}h1{font-size:1.5rem;margin:.2em 0}h2{font-size:1.15rem;margin:1.6em 0 .5em;border-bottom:1px solid var(--line);padding-bottom:.2em}
.muted{color:var(--muted);font-size:.9rem}.excerpt{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;white-space:pre-wrap;font-family:Georgia,serif}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px;margin:10px 0}
.head{display:flex;flex-wrap:wrap;gap:8px;align-items:baseline;justify-content:space-between}
.name{font-weight:700;font-size:1.1rem}.vid{font:12px ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--muted);letter-spacing:.04em;cursor:pointer}
audio{width:100%;margin:8px 0}.params{font:13px ui-monospace,Menlo,monospace;color:var(--muted);word-break:break-word}.params b{color:var(--fg);background:var(--hi);padding:0 2px;border-radius:3px}
.stars button{font-size:1.3rem;background:none;border:none;color:var(--muted);cursor:pointer;padding:0 2px}.stars button.on{color:#e0a800}
textarea{width:100%;min-height:2.6em;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--fg);padding:6px;font:inherit}
.err{color:#c0392b;font-size:.9rem;white-space:pre-wrap}.bar{position:sticky;bottom:0;background:var(--bg);border-top:1px solid var(--line);padding:10px 0;display:flex;gap:8px;flex-wrap:wrap;align-items:center}
button.act{background:var(--accent);color:var(--bg);border:none;border-radius:6px;padding:8px 12px;font:inherit;cursor:pointer}input.who{border:1px solid var(--line);border-radius:6px;padding:7px;background:var(--card);color:var(--fg);font:inherit;width:9em}
details.appendix{border-top:1px solid var(--line);margin-top:2em;padding-top:.6em}details.appendix>summary{font-weight:700;font-size:1.1rem;cursor:pointer}
table{border-collapse:collapse;font-size:.9rem}td,th{padding:2px 8px 2px 0;text-align:left}details{margin:8px 0}
"""

_JS = """
const PAGE=document.body.dataset.page;const K=(id)=>`voxy-diag:${PAGE}:${id}`;
const get=(k,d)=>{try{return JSON.parse(localStorage.getItem(k))??d}catch(e){return d}};
const put=(k,v)=>{try{localStorage.setItem(k,JSON.stringify(v))}catch(e){}};
const label=(c)=>c.dataset.label||c.dataset.id;
const who=document.querySelector('input.who');who.value=get('voxy-diag:who','');
const sink=window.voxySink||{answer(){},name(){},play(){}};
who.oninput=()=>{put('voxy-diag:who',who.value);sink.name(who.value)};
document.querySelectorAll('.card[data-id]').forEach(card=>{
  const id=card.dataset.id;const st=get(K(id),{rating:0,note:'',rev:0});
  const stars=[...card.querySelectorAll('.stars button')];const ta=card.querySelector('textarea');
  const paint=()=>stars.forEach((b,i)=>b.classList.toggle('on',i<st.rating));
  const save=(now)=>{st.rev=(st.rev||0)+1;put(K(id),st);sink.answer(id,st,now)};
  stars.forEach((b,i)=>b.onclick=()=>{st.rating=(st.rating===i+1)?0:i+1;paint();save(true)});
  ta.value=st.note;ta.oninput=()=>{st.note=ta.value;save(false)};paint();
  card.querySelector('.vid').onclick=()=>copy(id);
  const a=card.querySelector('audio');if(a)a.addEventListener('play',()=>sink.play(id,label(card)));
});
function notes(){const lines=[`${document.title} (${PAGE})${who.value?' - '+who.value:''}`];
  document.querySelectorAll('.card[data-id]').forEach(c=>{const s=get(K(c.dataset.id),{rating:0,note:''});
    if(s.rating||s.note)lines.push(`${label(c)} [${c.dataset.id}]: ${s.rating?'★'.repeat(s.rating):'-'}${s.note?' - '+s.note.replace(/\\n/g,' '):''}`)});
  return lines.join('\\n')}
function copy(t){const out=document.getElementById('out');out.value=t;out.hidden=false;out.select();
  (navigator.clipboard?navigator.clipboard.writeText(t):Promise.reject()).then(()=>flash('Copied'),()=>{document.execCommand&&document.execCommand('copy');flash('Selected: copy it')})}
function flash(m){const f=document.getElementById('flash');f.textContent=m;setTimeout(()=>f.textContent='',2500)}
document.getElementById('copy').onclick=()=>copy(notes());
"""

#: Saves as you go to an annoquest server (its event-log protocol: ``annoquest/events``
#: batches; ratings and notes are ``answer`` events, clip plays are ``view`` events of
#: ``#<id>``, so the owner page counts plays per clip). Runs before ``_JS``.
_ANNOQUEST_JS = """
(()=>{const S=JSON.parse(document.body.dataset.sink);
const get=(k,d)=>{try{return JSON.parse(localStorage.getItem(k))??d}catch(e){return d}};
const put=(k,v)=>{try{localStorage.setItem(k,JSON.stringify(v))}catch(e){}};
const rid=()=>(crypto.randomUUID?crypto.randomUUID():Date.now().toString(36)+Math.random().toString(36).slice(2));
let device=get('voxy-diag:device','');if(!device){device=rid();put('voxy-diag:device',device)}
const reader=new URLSearchParams(location.search).get('reader')||device;
const session=rid();let seq=0,queue=[],timer=null,active=false;
const status=(m)=>{const el=document.getElementById('saved');if(el)el.textContent=m};
const client={userAgent:navigator.userAgent,language:navigator.language,timeZone:Intl.DateTimeFormat().resolvedOptions().timeZone,screen:`${screen.width}x${screen.height}`,referrer:document.referrer||undefined,page:document.body.dataset.page};
function flush(){clearTimeout(timer);timer=null;if(!queue.length)return;const events=queue;queue=[];
  const name=(document.querySelector('input.who')||{}).value;
  const body={schema:'annoquest/events',request:S.request,reader,device,person:name?{name}:undefined,client,events};
  fetch(`${S.url}/requests/${encodeURIComponent(S.request)}/events`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body),keepalive:true})
   .then(r=>{if(!r.ok)throw new Error(r.status);status('Saved ✓')})
   .catch(()=>{queue=events.concat(queue);status('Not saved yet (offline?), retrying…');timer=setTimeout(flush,8000)})}
function ev(type,extra,delay){queue.push({id:rid(),session,seq:seq++,at:new Date().toISOString(),type,...extra});
  if(delay===0)return flush();if(!timer)timer=setTimeout(flush,delay??1500)}
window.voxySink={
  answer(id,st,now){const answer={at:new Date().toISOString(),rev:st.rev||0};if(st.rating)answer.value=String(st.rating);if(st.note)answer.comment=st.note;
    queue=queue.filter(e=>!(e.type==='answer'&&e.item===id));ev('answer',{item:id,answer},now?300:1500);status('Saving…')},
  name(n){queue=queue.filter(e=>e.type!=='name');ev('name',{name:n},1500)},
  play(id,label){ev('view',{url:location.origin+location.pathname+'#'+id,title:'play '+label},3000)}};
ev('open',{viewport:{w:innerWidth,h:innerHeight}});ev('view',{url:location.origin+location.pathname,title:document.title});
['pointerdown','keydown','scroll'].forEach(t=>addEventListener(t,()=>active=true,{passive:true}));
setInterval(()=>{if(document.visibilityState==='visible'){ev('beat',{active},0);active=false}},15000);
document.addEventListener('visibilitychange',()=>ev('visibility',{state:document.visibilityState},document.visibilityState==='hidden'?0:1500));
})();
"""


def _fmt_params(params: Mapping, defaults: Mapping) -> str:
    """``k=v`` pairs, those differing from the defaults in bold.

    >>> _fmt_params({"a": 1, "b": 2}, {"a": 1, "b": 3})
    'a=1 <b>b=2</b>'
    """
    parts = []
    for k, v in params.items():
        item = html.escape(f"{k}={v}")
        parts.append(item if k in defaults and defaults[k] == v else f"<b>{item}</b>")
    return " ".join(parts)


def _defaults(backend: str) -> dict:
    from voxy.local_clone import LOCAL_CLONE_BACKENDS

    spec = LOCAL_CLONE_BACKENDS.get(backend)
    return {**spec.init, **spec.params} if spec else {}


def backend_name(backend: str) -> str:
    """The display name of a backend (its key when it has none)."""
    return BACKEND_NAMES.get(backend, backend)


def variant_labels(results: Iterable[Mapping]) -> dict[str, str]:
    """Readable names: the backend's name, then the values of the settings that vary
    between that backend's variants. A result's own ``label`` wins.

    >>> rs = [
    ...     {"id": "a", "backend": "chatterbox", "params": {"language": "fr", "exaggeration": 0.3}},
    ...     {"id": "b", "backend": "chatterbox", "params": {"language": "fr", "exaggeration": 0.5}},
    ...     {"id": "c", "backend": "voxcpm", "params": {"cfg_value": 2.0}},
    ...     {"id": "d", "backend": "elevenlabs", "label": "ElevenLabs v3"},
    ... ]
    >>> variant_labels(rs)
    {'a': 'Chatterbox 0.3', 'b': 'Chatterbox 0.5', 'c': 'VoxCPM2', 'd': 'ElevenLabs v3'}
    """
    results = list(results)
    settings = {r["id"]: {**r.get("init", {}), **r.get("params", {})} for r in results}
    varying: dict[str, list[str]] = {}
    for backend in {r["backend"] for r in results}:
        group = [settings[r["id"]] for r in results if r["backend"] == backend]
        keys = sorted({k for g in group for k in g})
        varying[backend] = [
            k
            for k in keys
            if len({json.dumps(g.get(k), sort_keys=True) for g in group}) > 1
        ]
    labels = {}
    for r in results:
        values = [
            str(settings[r["id"]][k])
            for k in varying[r["backend"]]
            if k in settings[r["id"]]
        ]
        labels[r["id"]] = r.get("label") or " ".join(
            [backend_name(r["backend"]), *values]
        )
    return labels


def page_id(results: Iterable[Mapping], title: str) -> str:
    """A short ID for the page (its notes are stored under it)."""
    content = json.dumps([title, sorted(r["id"] for r in results)])
    return base64.b32encode(hashlib.sha256(content.encode()).digest()).decode()[:6]


def _page_order(results: Iterable[Mapping], appendix: Mapping) -> list[dict]:
    """Main results first, then the appendix's, each grouped by backend (first-seen order)."""
    results = list(results)
    backends = list(dict.fromkeys(r["backend"] for r in results))
    main = [
        r
        for b in backends
        for r in results
        if r["backend"] == b and r.get("reference") not in appendix
    ]
    rest = [
        r
        for b in backends
        for r in results
        if r["backend"] == b and r.get("reference") in appendix
    ]
    return main + rest


def annoquest_request(
    results: Iterable[Mapping],
    *,
    request_id: str,
    page_url: str,
    title: str,
    intro: str = "",
    api_url: str = "/api/annoquest",
    appendix: Mapping[str, str] | None = None,
    prompt: str = DFLT_RATING_PROMPT,
) -> dict:
    """The annoquest request a ``notes_sink`` page reports to: one item per variant
    (a 1-5 rating and a note), so the owner page reads the saved notes per listener.
    Register it on the server with ``python -m annoquest_server put-request <file>``.

    >>> r = annoquest_request([{"id": "CB-AAAA", "backend": "chatterbox"}],
    ...     request_id="abcdefgh12", page_url="https://example.org/p/", title="T")
    >>> [(i["id"], i["title"], i["response"]) for i in r["items"]]
    [('CB-AAAA', 'Chatterbox', 'stars')]
    """
    appendix = dict(appendix or {})
    results = _page_order(results, appendix)
    labels = variant_labels(results)
    stars = [
        {"value": str(n), "label": "★" * n, "tone": "positive" if n >= 4 else "neutral"}
        for n in range(1, 6)
    ]
    return {
        "schema": "annoquest/request",
        "version": 1,
        "id": request_id,
        "title": title,
        "intro": intro,
        "documents": [
            {"id": "page", "title": title, "source": {"kind": "url", "url": page_url}}
        ],
        "responseTypes": {
            "stars": {"kind": "choice", "label": "Rating", "options": stars}
        },
        "items": [
            {
                "id": r["id"],
                "doc": "page",
                "title": labels[r["id"]]
                + (f" ({r['reference']})" if r.get("reference") in appendix else ""),
                "prompt": prompt,
                "response": "stars",
                "comment": "optional",
            }
            for r in results
            if not r.get("error")
        ],
        "sink": {"kind": "http", "url": api_url},
        "allowExtra": False,
        "track": True,
    }


def build_page(
    results: list[dict],
    out_dir: str | os.PathLike,
    *,
    title: str = "Voice diagnosis",
    text: str = "",
    references: Mapping[str, Mapping] | None = None,
    intro: str = "",
    appendix: Mapping[str, str] | None = None,
    backend_notes: Mapping[str, str] | None = None,
    notes_sink: Mapping | None = None,
    noindex: bool = False,
) -> Path:
    """Write ``out_dir/index.html``: one card per variant, grouped by backend.

    Args:
        references: name -> ``{"audio": relative path, "text": ..., "label"?, "processing"?, ...}``
            shown so listeners can compare with the real voice.
        appendix: reference name -> a note: the variants made from those references
            (and the references themselves) go in a collapsed section at the end,
            under the note.
        backend_notes: backend -> a note shown under its section heading (default:
            the local backend's notes and licence).
        notes_sink: ``{"kind": "annoquest", "url": "/api/annoquest", "request": id}``
            saves ratings, notes, the listener's name and clip plays as they happen
            (the page says so). Without it they stay in the listener's browser.
        noindex: Ask search engines not to index the page (for a page shared by link).
    """
    from voxy.local_clone import LOCAL_CLONE_BACKENDS

    appendix = dict(appendix or {})
    references = dict(references or {})
    backend_notes = dict(backend_notes or {})
    if notes_sink and notes_sink.get("kind") != "annoquest":
        raise ValueError(
            f"Unknown notes_sink kind {notes_sink.get('kind')!r} (have: 'annoquest')"
        )
    out = Path(out_dir)
    results = _page_order(results, appendix)
    labels = variant_labels(results)
    pid = page_id(results, title)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    e = html.escape

    def ref_card(name, ref):
        stats = " &middot; ".join(
            f"{k} {ref[k]}"
            for k in ("duration", "score", "snr_db", "asr_confidence")
            if ref.get(k) is not None
        )
        processing = (
            f"<div class='muted'>{e(ref['processing'])}</div>"
            if ref.get("processing")
            else ""
        )
        return (
            f"<div class='card'><div class='head'><span class='name'>{e(ref.get('label') or name)}</span>"
            f"<span class='muted'>{stats}</span></div>{processing}"
            f"<audio controls preload='none' src='{e(ref['audio'])}'></audio>"
            f"<div class='muted'>{e(ref.get('text', ''))}</div></div>"
        )

    def variant_card(r):
        stats = (
            f"{r['generation_seconds']} s to make {r['audio_seconds']} s"
            f" &middot; RTF {r['rtf']}"
            if not r.get("error") and r.get("rtf") is not None
            else ""
        )
        settings = _fmt_params(
            {**r.get("init", {}), **r.get("params", {})}, _defaults(r["backend"])
        )
        ref = r.get("reference")
        ref_name = (references.get(ref) or {}).get("label") or ref
        ref_note = f" &middot; from {e(ref_name)}" if ref else ""
        player = (
            f"<div class='err'>{e(r['error'])}</div>"
            if r.get("error")
            else f"<audio controls preload='none' src='{e(r['audio'])}'></audio>"
        )
        return (
            f"<div class='card' data-id='{e(r['id'])}' data-label='{e(labels[r['id']])}'><div class='head'>"
            f"<span class='name'>{e(labels[r['id']])}</span>"
            f"<span class='vid' title='tap to copy'>{e(r['id'])}</span></div>{player}"
            f"<div class='params'>{settings or '(defaults)'}</div>"
            f"<div class='muted'>{stats}{ref_note}</div>"
            "<div class='stars'>"
            + "".join(f"<button aria-label='{i} stars'>★</button>" for i in range(1, 6))
            + "</div><textarea placeholder='Notes on this one'></textarea></div>"
        )

    def sections(rows):
        parts = []
        for backend in dict.fromkeys(r["backend"] for r in rows):
            parts.append(f"<h2>{e(backend_name(backend))}</h2>")
            spec = LOCAL_CLONE_BACKENDS.get(backend)
            note = backend_notes.get(backend) or (
                f"{spec.notes} Licence: {spec.licence}." if spec else ""
            )
            if note:
                parts.append(f"<p class='muted'>{e(note)}</p>")
            parts += [variant_card(r) for r in rows if r["backend"] == backend]
        return parts

    saving = (
        "Your ratings, notes and name are saved automatically as you go, with basic "
        "browser details (browser, language, time zone, screen size) and which clips you play."
        if notes_sink
        else "Ratings and notes stay in this browser; use <em>Copy my notes</em> to send them."
    )
    body = [
        f"<h1>{e(title)}</h1>",
        f"<p class='muted'>Page {pid} &middot; {stamp} &middot; {len(results)} clips. {saving}</p>",
    ]
    if intro:
        body.append(f"<p>{e(intro)}</p>")
    if text:
        body.append(
            f"<details open><summary>Text</summary><div class='excerpt'>{e(text)}</div></details>"
        )
    main_refs = {n: r for n, r in references.items() if n not in appendix}
    if main_refs:
        body.append("<h2>Reference (the real voice)</h2>")
        body += [ref_card(n, r) for n, r in main_refs.items()]
    body += sections([r for r in results if r.get("reference") not in appendix])
    for name, note in appendix.items():
        rows = [r for r in results if r.get("reference") == name]
        ref_label = (references.get(name) or {}).get("label") or name
        body.append(
            f"<details class='appendix'><summary>Appendix: clones made from {e(ref_label)} "
            f"({len(rows)})</summary><p>{e(note)}</p>"
        )
        if name in references:
            body.append(ref_card(name, references[name]))
        body += sections(rows)
        body.append("</details>")
    body.append(
        "<div class='bar'><input class='who' placeholder='Your name'>"
        "<button class='act' id='copy'>Copy my notes</button><span id='saved' class='muted'></span>"
        "<span id='flash' class='muted'></span><textarea id='out' hidden rows='6'></textarea></div>"
    )
    sink_attr = f" data-sink='{e(json.dumps(dict(notes_sink)))}'" if notes_sink else ""
    scripts = (_ANNOQUEST_JS if notes_sink else "") + _JS
    robots = "<meta name='robots' content='noindex, nofollow'>" if noindex else ""
    page = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        f"<meta name='viewport' content='width=device-width,initial-scale=1'>{robots}"
        f"<title>{e(title)}</title><style>{_CSS}</style></head>"
        f"<body data-page='{pid}'{sink_attr}><main>{''.join(body)}</main><script>{scripts}</script></body></html>"
    )
    path = out / PAGE_FILE
    path.write_text(page)
    return path


# spec files -------------------------------------------------------------------


def references_from_spec(
    spec: Mapping, out_dir: Path, *, log=print
) -> tuple[dict, dict]:
    """``(voices, shown)``: reference voices for rendering, and their page entries.

    The spec names references directly (``"references": {name: {ref_audio, ref_text}}``)
    or picks the best clips of a prepared voice (``"voice": name, "n_references": k``).
    Reference audio is copied next to the page so listeners can play it.
    """
    from voxy.local_clone import reference_profile

    entries = dict(spec.get("references", {}))
    if "voice" in spec:
        from voxy import refprep
        from voxy.stores import samples_store

        samples = samples_store(spec["voice"])
        best = refprep.best_references(
            refprep.load_manifest(samples), n=spec.get("n_references", 1)
        )
        for i, clip in enumerate(best, 1):
            entries[f"ref{i}"] = {
                "ref_audio": clip["path"],
                "ref_text": clip["text"],
                **{
                    k: clip[k]
                    for k in ("duration", "score", "snr_db", "asr_confidence")
                },
            }
    if not entries:
        raise ValueError("The spec needs 'references' or 'voice'")
    (out_dir / "reference").mkdir(parents=True, exist_ok=True)
    voices, shown = {}, {}
    for name, ref in entries.items():
        voices[name] = reference_profile(ref["ref_audio"], ref["ref_text"])
        first = (
            ref["ref_audio"][0]
            if isinstance(ref["ref_audio"], list)
            else ref["ref_audio"]
        )
        copy = out_dir / "reference" / f"{name}{Path(first).suffix}"
        shutil.copyfile(first, copy)
        shown[name] = {
            "audio": f"reference/{copy.name}",
            "text": ref["ref_text"],
            **{k: v for k, v in ref.items() if k not in ("ref_audio", "ref_text")},
        }
    return voices, shown


def _media_seconds(path: str | os.PathLike) -> float:
    """Duration of any audio file ffmpeg reads."""
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(out.stdout.strip())


def import_results(
    entries: Iterable[Mapping], out_dir: str | os.PathLike
) -> list[dict]:
    """Results for clips rendered elsewhere (a cloud service, another machine).

    Each entry: ``backend``, ``audio_file``, and optionally ``params``, ``init``,
    ``reference``, ``label``, ``generation_seconds``. The audio is copied next to the
    page and the entry gets the same ID scheme as rendered variants.
    """
    out = Path(out_dir)
    (out / AUDIO_DIR).mkdir(parents=True, exist_ok=True)
    results = []
    for entry in entries:
        v = Variant(
            entry["backend"],
            dict(entry.get("params", {})),
            dict(entry.get("init", {})),
            entry.get("reference", ""),
            entry.get("label", ""),
        )
        src = Path(entry["audio_file"]).expanduser()
        dst = out / AUDIO_DIR / f"{v.id}{src.suffix}"
        shutil.copyfile(src, dst)
        audio_s = _media_seconds(dst)
        gen_s = entry.get("generation_seconds")
        results.append(
            {
                **asdict(v),
                "id": v.id,
                "audio": f"{AUDIO_DIR}/{dst.name}",
                "generation_seconds": gen_s,
                "audio_seconds": round(audio_s, 1),
                "rtf": round(gen_s / audio_s, 2) if gen_s and audio_s else None,
                "imported": True,
            }
        )
    return results


def run_spec(spec: Mapping, out_dir: str | os.PathLike, *, log=print) -> Path:
    """Render a spec's grid and build its page; return the page path.

    Spec keys: ``title``, ``text`` (or ``text_file``), ``references`` or ``voice``
    (+ ``n_references``), ``grid`` (rows for ``expand_grid``), optional ``intro``,
    ``audio_format``, ``imported`` (see ``import_results``), ``appendix``,
    ``backend_notes``, ``noindex`` and ``notes_sink`` (see ``build_page``). With an
    annoquest ``notes_sink`` and a ``page_url`` (where the page will be served), the
    request to register on that server is written next to the page.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    text = spec.get("text") or Path(spec["text_file"]).read_text()
    voices, shown = references_from_spec(spec, out, log=log)
    variants = expand_grid(spec["grid"], references=list(voices))
    results = render_grid(
        text,
        voices,
        variants,
        out,
        audio_format=spec.get("audio_format", DFLT_AUDIO_FORMAT),
        log=log,
    )
    results += import_results(spec.get("imported", []), out)
    (out / "spec.json").write_text(json.dumps(spec, indent=2, ensure_ascii=False))
    title = spec.get("title", "Voice diagnosis")
    sink = spec.get("notes_sink")
    if sink and sink.get("kind") == "annoquest" and spec.get("page_url"):
        request = annoquest_request(
            results,
            request_id=sink["request"],
            page_url=spec["page_url"],
            title=title,
            intro=spec.get("intro", ""),
            api_url=sink.get("url", "/api/annoquest"),
            appendix=spec.get("appendix"),
        )
        (out / ANNOQUEST_REQUEST_FILE).write_text(
            json.dumps(request, indent=1, ensure_ascii=False)
        )
    return build_page(
        results,
        out,
        title=title,
        text=text,
        references=shown,
        intro=spec.get("intro", ""),
        appendix=spec.get("appendix"),
        backend_notes=spec.get("backend_notes"),
        notes_sink=sink,
        noindex=spec.get("noindex", False),
    )

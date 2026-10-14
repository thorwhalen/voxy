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
    "elevenlabs": "EL",
}


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
.vid{font:700 1.15rem ui-monospace,SFMono-Regular,Menlo,monospace;letter-spacing:.04em;cursor:pointer}
audio{width:100%;margin:8px 0}.params{font:13px ui-monospace,Menlo,monospace;color:var(--muted);word-break:break-word}.params b{color:var(--fg);background:var(--hi);padding:0 2px;border-radius:3px}
.stars button{font-size:1.3rem;background:none;border:none;color:var(--muted);cursor:pointer;padding:0 2px}.stars button.on{color:#e0a800}
textarea{width:100%;min-height:2.6em;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--fg);padding:6px;font:inherit}
.err{color:#c0392b;font-size:.9rem;white-space:pre-wrap}.bar{position:sticky;bottom:0;background:var(--bg);border-top:1px solid var(--line);padding:10px 0;display:flex;gap:8px;flex-wrap:wrap;align-items:center}
button.act{background:var(--accent);color:var(--bg);border:none;border-radius:6px;padding:8px 12px;font:inherit;cursor:pointer}input.who{border:1px solid var(--line);border-radius:6px;padding:7px;background:var(--card);color:var(--fg);font:inherit;width:9em}
table{border-collapse:collapse;font-size:.9rem}td,th{padding:2px 8px 2px 0;text-align:left}details{margin:8px 0}
"""

_JS = """
const PAGE=document.body.dataset.page;const K=(id)=>`voxy-diag:${PAGE}:${id}`;
const get=(k,d)=>{try{return JSON.parse(localStorage.getItem(k))??d}catch(e){return d}};
const put=(k,v)=>{try{localStorage.setItem(k,JSON.stringify(v))}catch(e){}};
const who=document.querySelector('input.who');who.value=get('voxy-diag:who','');who.oninput=()=>put('voxy-diag:who',who.value);
document.querySelectorAll('.card[data-id]').forEach(card=>{
  const id=card.dataset.id;const st=get(K(id),{rating:0,note:''});
  const stars=[...card.querySelectorAll('.stars button')];const ta=card.querySelector('textarea');
  const paint=()=>stars.forEach((b,i)=>b.classList.toggle('on',i<st.rating));
  stars.forEach((b,i)=>b.onclick=()=>{st.rating=(st.rating===i+1)?0:i+1;paint();put(K(id),st)});
  ta.value=st.note;ta.oninput=()=>{st.note=ta.value;put(K(id),st)};paint();
  card.querySelector('.vid').onclick=()=>copy(id);
});
function notes(){const lines=[`${document.title} (${PAGE})${who.value?' - '+who.value:''}`];
  document.querySelectorAll('.card[data-id]').forEach(c=>{const s=get(K(c.dataset.id),{rating:0,note:''});
    if(s.rating||s.note)lines.push(`${c.dataset.id}: ${s.rating?'★'.repeat(s.rating):'-'}${s.note?' - '+s.note.replace(/\\n/g,' '):''}`)});
  return lines.join('\\n')}
function copy(t){const out=document.getElementById('out');out.value=t;out.hidden=false;out.select();
  (navigator.clipboard?navigator.clipboard.writeText(t):Promise.reject()).then(()=>flash('Copied'),()=>{document.execCommand&&document.execCommand('copy');flash('Selected: copy it')})}
function flash(m){const f=document.getElementById('flash');f.textContent=m;setTimeout(()=>f.textContent='',2500)}
document.getElementById('copy').onclick=()=>copy(notes());
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


def page_id(results: Iterable[Mapping], title: str) -> str:
    """A short ID for the page (its notes are stored under it)."""
    content = json.dumps([title, sorted(r["id"] for r in results)])
    return base64.b32encode(hashlib.sha256(content.encode()).digest()).decode()[:6]


def build_page(
    results: list[dict],
    out_dir: str | os.PathLike,
    *,
    title: str = "Voice diagnosis",
    text: str = "",
    references: Mapping[str, Mapping] | None = None,
    intro: str = "",
) -> Path:
    """Write ``out_dir/index.html``: one card per variant, grouped by backend.

    ``references``: name -> ``{"audio": relative path, "text": ..., ...}`` shown
    at the top so listeners can compare with the real voice.
    """
    from voxy.local_clone import LOCAL_CLONE_BACKENDS

    out = Path(out_dir)
    pid = page_id(results, title)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    e = html.escape
    body = [
        f"<h1>{e(title)}</h1>",
        f"<p class='muted'>Page {pid} &middot; {stamp} &middot; {len(results)} variants. "
        "Tap an ID to copy it. Ratings and notes stay in this browser; "
        "use <em>Copy my notes</em> to send them.</p>",
    ]
    if intro:
        body.append(f"<p>{e(intro)}</p>")
    if text:
        body.append(
            f"<details open><summary>Text</summary><div class='excerpt'>{e(text)}</div></details>"
        )
    if references:
        body.append("<h2>Reference (the real voice)</h2>")
        for name, ref in references.items():
            stats = " &middot; ".join(
                f"{k} {ref[k]}"
                for k in ("duration", "score", "snr_db", "asr_confidence")
                if ref.get(k) is not None
            )
            body.append(
                f"<div class='card'><div class='head'><span class='vid'>{e(name)}</span>"
                f"<span class='muted'>{stats}</span></div>"
                f"<audio controls preload='none' src='{e(ref['audio'])}'></audio>"
                f"<div class='muted'>{e(ref.get('text', ''))}</div></div>"
            )
    by_backend: dict[str, list] = {}
    for r in results:
        by_backend.setdefault(r["backend"], []).append(r)
    for backend, rows in by_backend.items():
        spec = LOCAL_CLONE_BACKENDS.get(backend)
        body.append(f"<h2>{e(backend)}</h2>")
        if spec:
            body.append(
                f"<p class='muted'>{e(spec.notes)} Licence: {e(spec.licence)}.</p>"
            )
        defaults = _defaults(backend)
        for r in rows:
            stats = (
                f"{r['generation_seconds']} s to make {r['audio_seconds']} s"
                f" &middot; RTF {r['rtf']}"
                if not r.get("error")
                else ""
            )
            settings = _fmt_params(
                {**r.get("init", {}), **r.get("params", {})}, defaults
            )
            ref = f" &middot; ref {e(r['reference'])}" if r.get("reference") else ""
            player = (
                f"<div class='err'>{e(r['error'])}</div>"
                if r.get("error")
                else f"<audio controls preload='none' src='{e(r['audio'])}'></audio>"
            )
            body.append(
                f"<div class='card' data-id='{e(r['id'])}'><div class='head'>"
                f"<span class='vid' title='tap to copy'>{e(r['id'])}</span>"
                f"<span class='muted'>{stats}{ref}</span></div>{player}"
                f"<div class='params'>{settings or '(defaults)'}</div>"
                "<div class='stars'>"
                + "".join(
                    f"<button aria-label='{i} stars'>★</button>" for i in range(1, 6)
                )
                + "</div><textarea placeholder='Notes on this variant'></textarea></div>"
            )
    body.append(
        "<div class='bar'><input class='who' placeholder='Your name'>"
        "<button class='act' id='copy'>Copy my notes</button><span id='flash' class='muted'></span>"
        "<textarea id='out' hidden rows='6'></textarea></div>"
    )
    page = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{e(title)}</title><style>{_CSS}</style></head>"
        f"<body data-page='{pid}'><main>{''.join(body)}</main><script>{_JS}</script></body></html>"
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


def run_spec(spec: Mapping, out_dir: str | os.PathLike, *, log=print) -> Path:
    """Render a spec's grid and build its page; return the page path.

    Spec keys: ``title``, ``text`` (or ``text_file``), ``references`` or ``voice``
    (+ ``n_references``), ``grid`` (rows for ``expand_grid``), optional ``intro``,
    ``audio_format``.
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
    (out / "spec.json").write_text(json.dumps(spec, indent=2))
    return build_page(
        results,
        out,
        title=spec.get("title", "Voice diagnosis"),
        text=text,
        references=shown,
        intro=spec.get("intro", ""),
    )

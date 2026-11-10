"""
A formant synthesiser that sings "la" on a list of notes: voices that belong to no one.

A zero-shot singing model learns a timbre from a sung prompt clip. When no consenting
singer is at hand (a default voice, a test, a demo), this makes one: glottal-like
harmonics shaped by /l/ then /a/ formants, a portamento between notes, a delayed
vibrato and a little breath noise. Because the clip is rendered from known notes,
its notes and words are exact by construction, which is what an SVS prompt needs.
It is buzzy, and words sung in it come out less clearly than with a real voice.

>>> from voxy.score import notes_from_pitches
>>> samples = sing_la(notes_from_pitches([(60, 1)], bpm=120, start=0.1), sample_rate=8000)
>>> len(samples), float(abs(samples).max()) <= 1.0
(8000, True)
"""

from collections.abc import Mapping, Sequence

import numpy as np

from voxy.score import Note

DFLT_SAMPLE_RATE = 24000
DFLT_TAIL_S = 0.4
DFLT_LEVEL = 0.8
#: ``(centre Hz, bandwidth Hz)`` per formant, for /a/ and /l/, per preset.
FORMANT_PRESETS: Mapping[str, Mapping[str, Sequence[tuple[float, float]]]] = {
    "high": {
        "a": ((850, 80), (1220, 90), (2810, 120), (3800, 150), (4950, 200)),
        "l": ((380, 60), (1100, 100), (2700, 150), (3700, 180), (4900, 220)),
    },
    "low": {
        "a": ((730, 80), (1090, 90), (2440, 120), (3400, 150), (4500, 200)),
        "l": ((360, 60), (1000, 100), (2500, 150), (3300, 180), (4400, 220)),
    },
}
#: Glide between notes (time constant of a one-pole smoother, run both ways).
PORTAMENTO_S = 0.035
VIBRATO_HZ, VIBRATO_SEMITONES, VIBRATO_DELAY_S = 5.6, 0.30, 0.25
#: /l/ turns into /a/ over this long at each onset.
CONSONANT_S = 0.08
#: Harmonics stop this far below Nyquist.
NYQUIST_MARGIN_HZ = 200
HARMONIC_ROLLOFF = 1.1
BREATH_LEVEL = 0.004
RAMP_S = 0.03


def sing_la(
    notes: Sequence[Note],
    *,
    sample_rate: int = DFLT_SAMPLE_RATE,
    preset: str = "high",
    tail_s: float = DFLT_TAIL_S,
    seed: int = 0,
) -> np.ndarray:
    """Mono float32 samples of "la" sung on ``notes`` (silence before the first note,
    ``tail_s`` after the last), peaking at 0.8."""
    if not notes:
        raise ValueError("No notes to sing")
    formants = FORMANT_PRESETS[preset]
    rng = np.random.default_rng(seed)
    n = int((max(note.end for note in notes) + tail_s) * sample_rate)
    t = np.arange(n) / sample_rate
    midi = np.full(n, np.nan)
    since_onset = np.full(n, 10.0)
    amp = np.zeros(n)
    for note in notes:
        a, b = int(note.start * sample_rate), int(note.end * sample_rate)
        midi[a:b] = note.pitch
        since_onset[a:b] = t[a:b] - note.start
        amp[a:b] = 1.0
    voiced = ~np.isnan(midi)
    midi = np.interp(t, t[voiced], midi[voiced])
    midi = _smooth(midi, PORTAMENTO_S * sample_rate)
    depth = VIBRATO_SEMITONES * np.clip((since_onset - VIBRATO_DELAY_S) / 0.3, 0, 1)
    midi = midi + depth * np.sin(2 * np.pi * VIBRATO_HZ * t)
    f0 = 440.0 * 2 ** ((midi - 69) / 12)
    phase = 2 * np.pi * np.cumsum(f0) / sample_rate
    mix = np.clip(since_onset / CONSONANT_S, 0, 1)[:, None, None]
    shape = (
        np.array(formants["l"])[None] * (1 - mix) + np.array(formants["a"])[None] * mix
    )
    out = np.zeros(n)
    top = sample_rate / 2 - NYQUIST_MARGIN_HZ
    for k in range(1, int(top / f0.min()) + 1):
        fk = k * f0
        gain = np.ones(n)
        for j in range(shape.shape[1]):
            fc, bw = shape[:, j, 0], shape[:, j, 1]
            gain /= np.sqrt((1 - (fk / fc) ** 2) ** 2 + (fk * bw / fc**2) ** 2)
        out += (fk < top) * gain * k**-HARMONIC_ROLLOFF * np.sin(k * phase)
    ramp = int(RAMP_S * sample_rate)
    kernel = np.hanning(2 * ramp + 1)
    level = np.convolve(amp, kernel / kernel.sum(), mode="same")
    level *= 0.55 + 0.45 * np.clip(since_onset / 0.06, 0, 1)
    breath = np.convolve(rng.standard_normal(n), [1, -0.9], mode="same") * BREATH_LEVEL
    y = (out / np.abs(out).max() * 0.7 + breath) * level
    return (y / np.abs(y).max() * DFLT_LEVEL).astype(np.float32)


def _smooth(x: np.ndarray, samples: float) -> np.ndarray:
    """A one-pole low-pass run forward then backward (no lag), ``samples`` time constant."""
    alpha = float(np.exp(-1 / samples))
    y = x.copy()
    for seq in (y, y[::-1]):
        for i in range(1, len(seq)):
            seq[i] = alpha * seq[i - 1] + (1 - alpha) * seq[i]
    return y

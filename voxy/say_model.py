"""
Local speech with the macOS ``say`` command: free, offline, no API key, no deps.

Good for previews and drafts before spending credits on a cloud voice. Voices are
the system's (``say -v '?'``): e.g. "Samantha", "Daniel", "Karen".

>>> SaySpeechModel(run=lambda *a, **k: None).dflt_voice
'Samantha'
"""

import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable

from voxy.base import Speech, SpeechModel, VoiceInfo, VoiceProfile, _resolve_text_input

SAY_MODEL_TYPE = "say"
DFLT_SAY_VOICE = os.environ.get("VOXY_SAY_VOICE", "Samantha")
DFLT_SAY_SAMPLE_RATE = 22050
_VOICE_LINE = re.compile(
    r"^(?P<name>.+?)\s+(?P<lang>[a-z]{2,3}_[A-Za-z0-9]+)\s+#\s?(?P<sample>.*)$"
)


def parse_say_voices(listing: str) -> list[VoiceInfo]:
    """Parse ``say -v '?'`` output.

    >>> [v.name for v in parse_say_voices("Albert              en_US    # Hello!\\n"
    ...                                     "Eddy (English (US)) en_US    # Hi!\\n")]
    ['Albert', 'Eddy (English (US))']
    """
    voices = []
    for line in listing.splitlines():
        if m := _VOICE_LINE.match(line.strip()):
            voices.append(
                VoiceInfo(
                    voice_id=m["name"],
                    name=m["name"],
                    backend=SAY_MODEL_TYPE,
                    description=m["sample"],
                    labels={"language": m["lang"]},
                )
            )
    return voices


class SaySpeechModel(SpeechModel):
    """macOS ``say`` as a voxy backend.

    Args:
        voice: Default system voice.
        sample_rate: Output WAV sample rate.
        run: ``subprocess.run``-like callable (tests inject a fake).
    """

    name = SAY_MODEL_TYPE

    def __init__(
        self,
        *,
        voice: str = DFLT_SAY_VOICE,
        sample_rate: int = DFLT_SAY_SAMPLE_RATE,
        run: Callable | None = None,
    ):
        super().__init__(device="cpu")
        self.dflt_voice = voice
        self.sample_rate = sample_rate
        self._run = run

    def _say(self, *args: str) -> subprocess.CompletedProcess:
        if self._run is not None:
            return self._run(["say", *args], capture_output=True, text=True, check=True)
        if shutil.which("say") is None:
            raise RuntimeError("The 'say' backend needs macOS's 'say' command.")
        return subprocess.run(
            ["say", *args], capture_output=True, text=True, check=True
        )

    def list_voices(self) -> list[VoiceInfo]:
        return parse_say_voices(self._say("-v", "?").stdout)

    def synthesize(
        self, text, voice: VoiceProfile | str | None = None, **kwargs
    ) -> Speech:
        """WAV speech from ``say`` (``voice``: a system voice name or profile)."""
        text = _resolve_text_input(text)
        voice = getattr(voice, "segment", voice) or self.dflt_voice
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "speech.wav")
            self._say(
                "-v",
                voice,
                "-o",
                out,
                "--file-format=WAVE",
                f"--data-format=LEI16@{self.sample_rate}",
                "--",
                text,
            )
            with open(out, "rb") as f:
                audio = f.read()
        return Speech(
            audio,
            format="wav",
            backend=self.name,
            voice=voice,
            sample_rate=self.sample_rate,
            text=text,
        )

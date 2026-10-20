"""The diagnosis page: labels, the appendix, imported clips and the notes sink."""

import json
import re
import shutil
import subprocess

import pytest

from voxy import diagnosis


def _results():
    return [
        {"id": f"CB-{ref[0].upper()}{x}", "backend": "chatterbox", "reference": ref,
         "params": {"language": "fr", "exaggeration": x / 10}, "audio": f"audio/{ref}{x}.mp3",
         "generation_seconds": 10.0, "audio_seconds": 5.0, "rtf": 2.0}
        for ref in ("raw", "denoised") for x in (3, 5)
    ]


def test_page_appendix_labels_and_sink(tmp_path):
    refs = {
        "raw": {"audio": "reference/raw.wav", "text": "t", "label": "the recording as is"},
        "denoised": {"audio": "reference/denoised.wav", "text": "t", "label": "the cleaned recording"},
    }
    sink = {"kind": "annoquest", "url": "/api/annoquest", "request": "abcdefgh1234"}
    path = diagnosis.build_page(
        _results(), tmp_path, title="T", references=refs,
        appendix={"raw": "Lesser quality: hiss."}, notes_sink=sink, noindex=True,
    )
    page = path.read_text()
    main, appendix = page.split("<details class='appendix'>")
    assert "Chatterbox 0.3" in main and "CB-D3" in main and "CB-R3" not in main
    assert "CB-R3" in appendix and "Lesser quality: hiss." in appendix
    assert "the recording as is" in appendix and "the cleaned recording" in main
    assert "noindex" in page and "saved automatically" in page
    assert json.loads(re.search(r"data-sink='([^']*)'", page).group(1).replace("&quot;", '"')) == sink


@pytest.mark.skipif(not shutil.which("node"), reason="node checks the page script")
def test_page_script_parses(tmp_path):
    sink = {"kind": "annoquest", "url": "/api/annoquest", "request": "abcdefgh1234"}
    page = diagnosis.build_page(_results(), tmp_path, notes_sink=sink).read_text()
    script = tmp_path / "page.js"
    script.write_text(re.search(r"<script>(.*)</script>", page, re.S).group(1))
    subprocess.run(["node", "--check", str(script)], check=True)


def test_annoquest_request_items_follow_page_order():
    req = diagnosis.annoquest_request(
        _results(), request_id="abcdefgh1234", page_url="https://example.org/p/",
        title="T", appendix={"raw": "note"},
    )
    assert [i["id"] for i in req["items"]] == ["CB-D3", "CB-D5", "CB-R3", "CB-R5"]
    assert req["items"][2]["title"] == "Chatterbox 0.3 (raw)"
    assert req["sink"] == {"kind": "http", "url": "/api/annoquest"}


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg to make a clip")
def test_import_results_copies_and_measures(tmp_path):
    clip = tmp_path / "el.mp3"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono",
                    "-t", "2", str(clip)], check=True)
    (r,) = diagnosis.import_results(
        [{"backend": "elevenlabs", "audio_file": str(clip), "label": "ElevenLabs v3",
          "params": {"model_id": "eleven_v3"}, "generation_seconds": 3.0}], tmp_path / "page")
    assert r["id"].startswith("EL-") and (tmp_path / "page" / r["audio"]).is_file()
    assert r["audio_seconds"] == pytest.approx(2.0, abs=0.1) and r["rtf"] == pytest.approx(1.5, abs=0.1)

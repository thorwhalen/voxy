"""
Command line for voxy: voices, speech, local model setup, reference prep, diagnosis pages.

    python -m voxy voices                       # our named voices
    python -m voxy voices --backend say         # a backend's voices
    python -m voxy speak "Hello" --voice ada -o hello.mp3
    python -m voxy setup f5tts chatterbox       # build local models' environments
    python -m voxy prep ada tape1.m4a tape2.mp3 # clean, cut, transcribe, score clips
    python -m voxy prep ada interview.m4a --speaker dominant  # drop the other voices
    python -m voxy diagnose spec.json --out DIR --publish
    python -m voxy benchmarks docs/benchmarks.md # regenerate the comparison tables/charts
"""

import argparse
import json
import subprocess
import sys

from voxy.facade import list_voices, text_to_speech


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="voxy", description=__doc__.splitlines()[1])
    commands = parser.add_subparsers(dest="command", required=True)
    voices = commands.add_parser("voices", help="list voices")
    voices.add_argument("--backend", help="a backend's voices instead of the library")
    speak = commands.add_parser("speak", help="synthesize speech to a file")
    speak.add_argument("text")
    speak.add_argument("--voice", help="library name/alias or the backend's voice")
    speak.add_argument("--backend", help="service to use")
    speak.add_argument("-o", "--output", required=True, help="audio file to write")
    setup = commands.add_parser("setup", help="build local backends' environments")
    setup.add_argument("backends", nargs="*", help="default: all local backends")
    prep = commands.add_parser("prep", help="prepare a voice's reference clips")
    prep.add_argument("voice", help="voice name (clips go to its samples store)")
    prep.add_argument("sources", nargs="+", help="recordings, any format")
    prep.add_argument("--language", help="force the ASR language (e.g. en)")
    prep.add_argument(
        "--denoise",
        default="afftdn",
        help="afftdn, anlmdn, deepfilternet (own env, needs uv) or none",
    )
    prep.add_argument(
        "--speaker",
        help="keep one voice: 'dominant' (whoever speaks most in each recording) "
        "or recordings of the target alone, comma-separated (own env, needs uv)",
    )
    diagnose = commands.add_parser("diagnose", help="render a grid, build its page")
    diagnose.add_argument("spec", help="JSON spec (see voxy.diagnosis.run_spec)")
    diagnose.add_argument("--out", required=True, help="output folder")
    diagnose.add_argument("--publish", action="store_true", help="annals publish it")
    diagnose.add_argument("--session", default=None, help="annals --session name")
    bench = commands.add_parser(
        "benchmarks", help="regenerate a benchmarks doc's tables and charts"
    )
    bench.add_argument("doc", help="Markdown doc with voxy-benchmarks blocks")
    bench.add_argument("--data", help="benchmarks JSON (default: the shipped data)")
    args = parser.parse_args(argv)

    if args.command == "benchmarks":
        from voxy.benchmarks import load_benchmarks, render_docs

        for name in render_docs(args.doc, data=load_benchmarks(args.data)):
            print(name)
        return 0

    if args.command == "setup":
        from voxy.envs import ensure_env
        from voxy.local_clone import LOCAL_CLONE_BACKENDS
        from voxy.refprep import ASR_ENV
        from voxy.speakers import SPEAKER_ENV

        envs = {name: b.env for name, b in LOCAL_CLONE_BACKENDS.items()}
        envs["asr"] = ASR_ENV
        envs["speaker"] = SPEAKER_ENV
        for name in args.backends or list(envs):
            print(f"{name}\t{ensure_env(envs[name])}")
        return 0
    if args.command == "prep":
        from voxy import refprep

        speaker = None
        if args.speaker:
            from voxy import speakers

            target = None
            if args.speaker != "dominant":
                target = speakers.voice_print(args.speaker.split(","))
            speaker = speakers.speaker_filter(target)
        manifest = refprep.prepare_reference(
            args.sources,
            voice=args.voice,
            language=args.language,
            denoise=None if args.denoise == "none" else args.denoise,
            speaker=speaker,
        )
        print(json.dumps(refprep.summarize(manifest)))
        for clip in refprep.best_references(manifest):
            print(f"{clip['key']}\t{clip['duration']}\t{clip['score']}\t{clip['text']}")
        return 0
    if args.command == "diagnose":
        from voxy.diagnosis import run_spec

        with open(args.spec) as f:
            spec = json.load(f)
        page = run_spec(spec, args.out)
        print(page)
        if args.publish:
            cmd = [
                "annals",
                "publish",
                str(page.parent),
                "--title",
                spec.get("title", "Voice diagnosis"),
            ]
            if args.session:
                cmd += ["--session", args.session]
            subprocess.run(cmd, check=True)
        return 0

    if args.command == "voices":
        for v in list_voices(args.backend):
            extra = v.labels or {}
            if args.backend is None:
                print(
                    f"{v.name}\t{', '.join(extra.get('aliases', []))}\t{', '.join(extra.get('backends', []))}"
                )
            else:
                print(f"{v.voice_id}\t{v.name}\t{v.description}")
    else:
        speech = text_to_speech(
            args.text, args.voice, backend=args.backend, output_path=args.output
        )
        print(f"{args.output}\t{speech.backend}\t{speech.voice}\t{speech.format}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

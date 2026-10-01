"""
Command line for voxy: ``python -m voxy voices`` / ``python -m voxy speak``.

    python -m voxy voices                       # our named voices
    python -m voxy voices --backend say         # a backend's voices
    python -m voxy speak "Hello" --voice cora -o hello.mp3
"""

import argparse
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
    args = parser.parse_args(argv)

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

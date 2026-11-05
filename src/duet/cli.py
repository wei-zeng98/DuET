"""Command-line entry point."""

import argparse
import json
import sys
from pathlib import Path
import torch
from .pipeline import DuET

DEFAULT_MODEL = "Wayne42/DuET"


def _default_output(input_path):
    """results/<input stem> in the current directory, numbered if it already exists."""
    base = Path("results") / Path(input_path).stem
    candidate, n = base, 2
    while candidate.exists():
        candidate = base.with_name(f"{base.name}_{n}")
        n += 1
    return candidate


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe piano audio or performance MIDI into MusicXML and score MIDI."
    )
    parser.add_argument("input", help="Input audio or performance MIDI")
    parser.add_argument(
        "--input-type", choices=["auto", "midi", "audio"], default="auto"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Hugging Face model repository or downloaded model directory "
        f"(default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="New output directory (default: results/<input name>, numbered if taken)",
    )
    parser.add_argument(
        "--device", default=None, help="Default: CUDA when available, otherwise CPU"
    )
    parser.add_argument("--revision", help="Hugging Face commit or tag")
    parser.add_argument(
        "--offline", action="store_true", help="Use only local/cached model files"
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--engraving-config", type=Path, help="JSON object of engraving Config parameters")
    for name in ('keys', 'spelling', 'runs'):
        parser.add_argument('--auto-' + name, action=argparse.BooleanOptionalAction,
                            default=None, help="Override automatic " + name + " (default: enabled)")
    parser.add_argument('--key-switch', type=float, help="Penalty for a key-signature change")
    parser.add_argument('--key-model-cost', type=float, help="Penalty per note for disagreeing with model keys")
    args = parser.parse_args()
    from .readable.config import Config
    try:
        engraving = json.loads(args.engraving_config.read_text(encoding='utf-8')) if args.engraving_config else {}
        if not isinstance(engraving, dict):
            raise ValueError("Engraving configuration must be a JSON object")
        for name in ('auto_keys', 'auto_spelling', 'auto_runs', 'key_switch', 'key_model_cost'):
            value = getattr(args, name)
            if value is not None:
                engraving[name] = value
        engraving = Config.from_dict(engraving)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if args.threads < 1:
        parser.error("--threads must be positive")
    output = Path(args.output) if args.output else _default_output(args.input)
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        model = DuET.from_pretrained(
            args.model,
            device=args.device,
            revision=args.revision,
            local_files_only=args.offline,
        )
        is_midi = args.input_type == "midi" or (
            args.input_type == "auto"
            and Path(args.input).suffix.lower() in {".mid", ".midi"}
        )
        transcribe = model.transcribe if is_midi else model.transcribe_audio
        report = transcribe(
            args.input, output, progress=lambda w: print(json.dumps(w), flush=True),
            engraving_config=engraving
        )
        print(f"Saved {report['input_notes']} note predictions to {output}")
    except Exception as exc:
        print(f"DuET: {exc}", file=sys.stderr)
        return 1
    return 0

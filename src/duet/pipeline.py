"""Public loading and full-piece transcription API."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from .assembly import assemble_records
from .config import BUNDLE_FORMAT, CORE_SECONDS, HISTORY_SECONDS, MAX_NOTES, DuETConfig
from .export import write_listening_midi
from .engraving import predict_engraving_frames, write_readable_musicxml
from .features import build_window
from .midi import read_performance_midi
from .model import DuETModel


def _sha256(path, *, normalize_newlines=False):
    if normalize_newlines:
        # Git may check text files out with CRLF on Windows. The released
        # vocabulary checksum is based on the repository's canonical LF form.
        data = Path(path).read_bytes().replace(b"\r\n", b"\n")
        return hashlib.sha256(data).hexdigest()
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


class DuET:
    """Load the fixed release and transcribe performance MIDI into a score."""

    def __init__(self, model, metadata, device):
        self.model = model.to(device).eval()
        self.metadata = metadata
        self.device = torch.device(device)

    @classmethod
    def from_pretrained(
        cls, model_id, *, device=None, revision=None, local_files_only=False
    ):
        """Load a local bundle or a Hugging Face repo, without executing remote code."""
        path = Path(model_id).expanduser()
        if not path.is_dir():
            from huggingface_hub import snapshot_download

            path = Path(
                snapshot_download(
                    repo_id=str(model_id),
                    revision=revision,
                    local_files_only=local_files_only,
                    allow_patterns=["config.json", "model.safetensors"],
                )
            )
        metadata = json.loads((path / "config.json").read_text(encoding="utf-8"))
        if (
            metadata.get("format") != BUNDLE_FORMAT
            or metadata.get("architecture") != DuETConfig().to_dict()
            or metadata.get("core_seconds") != CORE_SECONDS
            or metadata.get("history_seconds") != HISTORY_SECONDS
        ):
            raise ValueError(
                "Model bundle is incompatible with this DuET package version."
            )
        if metadata.get("vocabulary_sha256") != _sha256(
            Path(__file__).with_name("vocabulary.json"), normalize_newlines=True
        ):
            raise ValueError("Model vocabulary does not match the installed package.")
        weights = path / "model.safetensors"
        if _sha256(weights) != metadata.get("weights_sha256"):
            raise ValueError("Model weights checksum mismatch.")
        model = DuETModel()
        model.load_state_dict(load_file(str(weights), device="cpu"), strict=True)
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        return cls(model, metadata, device)

    def transcribe_audio(self, audio, output_dir, *, progress=None, audio_model=None):
        """Audio -> performance.mid -> score.mid/MusicXML, using optional Transkun v2.

        A failed stage leaves a report and any completed intermediate MIDI.
        """
        from .audio import audio_to_midi

        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=False)
        report_path = root / "report.json"
        report = {"status": "running", "input_type": "audio", "stage": "audio_to_midi"}
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        # The existing MIDI pipeline requires a new directory. Use a child and
        # publish its artifacts at the root after success or failure.
        score_dir = root / "score_stage"
        try:
            frontend = audio_to_midi(
                audio, root / "performance.mid", device=self.device, model=audio_model
            )
            report.update(frontend=frontend, stage="midi_to_score")
            report_path.write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
            score_report = self.transcribe(
                root / "performance.mid",
                score_dir,
                progress=progress,
                title=Path(audio).stem,
            )
            report.update(score_report, stage="complete")
            return report
        except Exception as exc:
            stage_report = score_dir / "report.json"
            if stage_report.is_file():
                report.update(json.loads(stage_report.read_text(encoding="utf-8")))
            report.update(status="failed", error=str(exc))
            raise
        finally:
            if score_dir.is_dir():
                for artifact in score_dir.iterdir():
                    if artifact.name != "report.json":
                        artifact.rename(root / artifact.name)
                (score_dir / "report.json").unlink(missing_ok=True)
                score_dir.rmdir()
            report_path.write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )

    @torch.inference_mode()
    def transcribe(self, midi, output_dir, *, progress=None, title=None):
        """Generate all notes; fail explicitly on incomplete coverage or oversized windows.

        No reference score is required. Output directories must not already exist.
        """
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=False)
        report = {
            "status": "running",
            "model_version": BUNDLE_FORMAT,
            "weights_sha256": self.metadata["weights_sha256"],
            "core_seconds": CORE_SECONDS,
            "history_seconds": HISTORY_SECONDS,
            "windows": [],
            "device": str(self.device),
        }

        def save_report():
            (root / "report.json").write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )

        save_report()
        try:
            data = read_performance_midi(Path(midi))
            report["midi_input_cleanup"] = data["midi_input_cleanup"]
            on = data["ref_onset"]
            pitch = data["ref_pitch"]
            report["notes_outside_roll_range"] = int(
                ((pitch < 21) | (pitch > 108)).sum()
            )
            records = []
            committed = {}
            for index, start in enumerate(
                np.arange(0.0, float(on.max()) + CORE_SECONDS, CORE_SECONDS)
            ):
                new_ids = np.flatnonzero((on >= start) & (on < start + CORE_SECONDS))
                if not len(new_ids):
                    continue
                batch, ids, left = build_window(data, float(start))
                history = [int(i) for i in ids if on[i] < start]
                if any(i not in committed for i in history):
                    raise ValueError("Missing generated history note.")
                anchor = (
                    min(committed[i]["bar"] for i in history)
                    if history
                    else max((r["bar"] for r in records), default=0)
                )
                owned = [int(i) for i in ids if on[i] >= start]
                if set(owned) != set(new_ids):
                    raise ValueError("Window lost input note ownership.")
                if len(ids) > MAX_NOTES:
                    raise ValueError(
                        f"Window {index} exceeds the {MAX_NOTES}-note limit; no notes were truncated."
                    )
                prefix = []
                for i in history:
                    row = committed[i]["row"].copy()
                    row[0] = committed[i]["bar"] - anchor + self.model.codec.bar_offset
                    if not 0 <= row[0] < self.model.codec.sizes[0]:
                        raise ValueError("History exceeds relative bar vocabulary.")
                    prefix.append(row)
                batch = {k: v.to(self.device) for k, v in batch.items()}
                prefix = torch.tensor(
                    np.asarray(prefix, dtype=np.int64).reshape(
                        -1, self.model.codec.width
                    ),
                    device=self.device,
                )
                pred = (
                    self.model.generate_rows(batch, prefix_rows=prefix, anchor=anchor)
                    .cpu()
                    .numpy()
                )
                if len(pred) - len(prefix) != len(owned):
                    raise ValueError("Generated note count mismatch.")
                for i, row in zip(owned, pred[len(prefix) :]):
                    if i in committed:
                        raise ValueError("Repeated input note ownership.")
                    r = {
                        "input_id": i,
                        "pitch": int(pitch[i]),
                        "bar": int(row[0]) - self.model.codec.bar_offset + anchor,
                        "row": row.tolist(),
                    }
                    records.append(r)
                    committed[i] = r
                detail = {
                    "index": index,
                    "start_seconds": float(start),
                    "history_start_seconds": left,
                    "anchor_bar": anchor,
                    "prefix_notes": len(prefix),
                    "new_notes": len(owned),
                }
                report["windows"].append(detail)
                save_report()
                if progress:
                    progress(detail)
            if set(committed) != set(range(len(on))):
                raise ValueError("Full-piece input coverage is incomplete.")
            score, stats = assemble_records(self.model.codec, records)
            report.update(stats, input_notes=len(on), input_coverage=1.0)
            (root / "notes.json").write_text(
                json.dumps(records) + "\n", encoding="utf-8"
            )
            report["midi"] = write_listening_midi(
                self.model.codec, score, root / "score.mid"
            )
            report["stage"] = "engraving"
            report["score_title"] = title if title is not None else Path(midi).stem
            save_report()
            frames = predict_engraving_frames(self.model, data, self.device)
            report["engraving"] = write_readable_musicxml(
                score,
                data,
                frames,
                root / "score.musicxml",
                title=report["score_title"],
            )
            report["stage"] = "complete"
            report["status"] = "complete"
            return report
        except Exception as exc:
            report.update(status="failed", error=str(exc))
            raise
        finally:
            save_report()

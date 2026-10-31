"""The repository demo renders readable XML without importing score MIDI."""

import importlib.util
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "demo_engraving", Path(__file__).resolve().parents[1] / "demo/engraving.py"
)
engraving = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engraving)


class DemoEngravingTests(unittest.TestCase):
    def test_renderer_preserves_readable_xml_and_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            xml = b'<score-partwise version="4.0"/>'
            (folder / "score.musicxml").write_bytes(xml)
            (folder / "report.json").write_text(
                json.dumps(
                    {
                        "engraving": {
                            "mode": "readable",
                            "version": "1.0.0",
                            "summary": {"bars": 2},
                        }
                    }
                )
            )

            def render(command, **kwargs):
                job = json.loads(Path(command[-1]).read_text())[0]
                self.assertEqual(Path(job["in"]), folder / "score.musicxml")
                self.assertFalse(any(p.endswith(".musicxml") for p in job["out"]))
                (folder / "score-1.svg").write_text("<svg/>")
                (folder / "score.pdf").write_bytes(b"%PDF-test")
                return types.SimpleNamespace(returncode=0, stdout="", stderr="")

            runtime = types.SimpleNamespace(
                ensure_musescore=lambda: "MuseScore", VERSION="test"
            )
            with (
                patch.dict("sys.modules", {"musescore_runtime": runtime}),
                patch.object(engraving.subprocess, "run", side_effect=render),
            ):
                engraving.engrave(folder)
            self.assertEqual((folder / "score.musicxml").read_bytes(), xml)
            metadata = json.loads((folder / "report.json").read_text())["engraving"]
            self.assertEqual(metadata["mode"], "readable")
            self.assertEqual(metadata["version"], "1.0.0")
            self.assertEqual(metadata["summary"], {"bars": 2})
            self.assertEqual(metadata["source"], "score.musicxml")

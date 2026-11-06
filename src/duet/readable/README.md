# Readable engraving

Engraving 1.1.0, the same version as the engraving project (`engrave`). The modules are copied from that project by its `tools/sync_duet.py`, which leaves out the code only that project uses (edits files and the diagnostic faithful mode); the two adapters are maintained here: `inputs.py` takes the inputs in memory from DuET’s validated MIDI reader and frame predictions (`Inputs.from_prediction`, which copies the raw score arrays and keeps exact input-note alignment, including FIFO overlap pairing and invalid-event cleanup), and `pipeline.run(inp, out_dir, title)` keeps the diagnostic log in memory and writes only `score.musicxml` and `score.review.json` (published by `duet.engraving` as `review.json`).

Since 1.1.0 the engraving takes key signatures from the notes (`keyfit`), spells notes outside the key from their harmony and line (`spellfit`), and in fast runs writes an isolated brushed note as a grace note and cuts one-place overlaps (`runs`); places it is not sure about are listed in `review.json` with the edits (`edits.py` format) for each answer (`review.py`). On the engraving project's 148-piece test set this raised correct spelling from 94.8% to 97.4% and key-signature agreement from 67% to 77%, with note rhythms unchanged.

For pieces with fewer than two integer-beat anchors, fractional model attacks establish the grid. A single attack uses the MIDI exporter’s 120 BPM default and records a warning in the engraving summary. No model weights, evaluation data, alternative engraving modes (the diagnostic faithful mode), or renderer binaries are included. MuseScore is only used by the separate demo PDF renderer.


## Runtime configuration

Automatic keys, spelling, and run cleanup are independent and all default to
enabled. Run cleanup can change ordinary/grace classification and durations;
disabling it does not disable key or spelling inference.

```bash
duet performance.mid --no-auto-runs
duet performance.mid --key-switch 20 --key-model-cost 0.025
duet performance.mid --engraving-config engraving.json --auto-runs
```

The JSON file is an object of `readable.config.Config` fields, for example:

```json
{
  "auto_keys": true,
  "auto_spelling": true,
  "auto_runs": true,
  "key_switch": 10.0,
  "key_per_fifth": 0.3,
  "key_model_cost": 0.05,
  "spelling_window": 2.0,
  "spelling_w_key": 1.0,
  "spelling_w_line": 3.0,
  "spelling_review_margin": 1.5
}
```

Explicit CLI switches override the JSON file. Unknown fields and invalid numeric
or boolean values are rejected before inference. The effective configuration is
stored in `report.json` under `engraving.config`. Python callers can pass a Config
or dictionary through `transcribe(..., engraving_config=...)` and
`transcribe_audio(..., engraving_config=...)`.

## Glissandi

The model writes a glissando as a crowded run of 32nds that overruns the bar. By default DuET finds
fast one-direction runs on one scale in the performance and writes each as a wavy line from its first note to its last,
at their performed positions. Notes struck during the line go back to where they were played, in the second voice.

Each detected run is a `glissando` item in `review.json`, because a wrong detection takes notes off the page.
`--no-auto-glissandi` turns this off.

All score times remain in quarter notes, including 3/8 (3/2 quarters), 4/8 (2),
and 9/8 (9/2). Simple eighth meters use an eighth-note grid unit; compound eighth
meters use dotted-quarter pulses. Bar durations, not integer casts, control the
MusicXML clock. Default beams are grouped 3 in 3/8, 2+2 in 4/8, and 3+3+3 in 9/8.

Bar-line snapping and frame count changes are proposals. A proposal that merges
or reverses bar boundaries is rejected and the whole rhythm stage restarts from
the original input with both corrections disabled. This preserves short pickups;
the reason and action are recorded in `engraving.summary.grid_fallback`.

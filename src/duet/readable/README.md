# Readable engraving

Engraving 1.3.1, the same version as the engraving project (`engrave`). The modules originate from the engraving project (`engrave`), copied by its `tools/sync_duet.py`, which leaves out the code only that project uses (edits files and the diagnostic faithful mode); the two adapters are maintained here: `inputs.py` takes the inputs in memory from DuET’s validated MIDI reader and frame predictions (`Inputs.from_prediction`, which copies the raw score arrays and keeps exact input-note alignment, including FIFO overlap pairing and invalid-event cleanup), and `pipeline.run(inp, out_dir, title)` keeps the diagnostic log in memory and writes only `score.musicxml` and `score.review.json` (published by `duet.engraving` as `review.json`).

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

The model writes a glissando as a crowded run of 32nds that overruns the bar. By default DuET finds glissandi in the
performance and writes each as a wavy line from its first note to its last: at least 8 notes on one key colour (white
or black keys) in one direction, spanning an octave or more, about 22 notes a second or faster. The first note keeps
its position and lasts until the last one, which goes where it was played; each stays on its own staff, so a line can
cross from one hand to the other. Only the run's own notes are removed; notes of the other hand struck during the line
go back to where they were played, in the second voice. White-key glissandi are spelled with white keys.

Each glissando is a `glissando` item in `review.json`, because a wrong detection takes notes off the page. Fast runs on
a major scale are only listed (`scale-run` items): fingered scales that fast are far more common than scale glissandi.
`--no-auto-gliss` turns this off; the thresholds are the `gliss_*` settings of the engraving configuration.
Glissandi were contributed by @hibikiwtnb (#8, #9).

All score times remain in quarter notes, including 3/8 (3/2 quarters), 4/8 (2),
and 9/8 (9/2). Simple eighth meters use an eighth-note grid unit; compound eighth
meters use dotted-quarter pulses. Bar durations, not integer casts, control the
MusicXML clock. Default beams are grouped 3 in 3/8, 2+2 in 4/8, and 3+3+3 in 9/8.

Bar-line snapping and frame count changes are proposals. A proposal that merges
or reverses bar boundaries is rejected and the whole rhythm stage restarts from
the original input with both corrections disabled. This preserves short pickups;
the reason and action are recorded in `engraving.summary.grid_fallback`.

Since 1.2.0 notes played together (within 15 ms) but written apart become one chord. `review.json` also asks about beats written as triplets or sextuplets that were played as 16ths, bars with the same notes as other bars but a different rhythm, and lone right-hand notes below middle C that complete a left-hand chord.

Since 1.2.1 a slow-down that returns to tempo within a bar (a phrase end breathing) gets no *rit.* / *a tempo* pair; playback still follows it.

Since 1.2.2 16th triplets that fill only half a beat get their own small 3:2 bracket (the other half is written plainly), and `review.json` offers 16ths or 32nds for sextuplet figures, one item per figure.

Since 1.2.5 a left-hand figure too high for the bass clef, in a bar whose left hand also has a bass, is written on the right-hand staff instead of switching the bass staff to treble clef in the middle of the bar.

Since 1.2.6 `review.json` also lists beats with 5 or 7 notes played about evenly, offering one 5:4 / 7:4 tuplet per beat.

Since 1.2.8 glissandi are written as described under Glissandi above (#9).


Engraving 1.3.0 fixes notation safety issues:
- A glissando is compressed only when its two endpoints fit before the next attack.
  If no grid position is available, all notes remain written out and a
  `glissando-unplaced` review item explains why. Stop-note lengths are capped again
  after voice cleanup. A later unsafe endpoint restarts engraving from the original
  inputs with gliss compression disabled, recorded in `summary.gliss_fallback`.
- Crossed LH figures reserve their own voice over actual occupied intervals.
  A real second-voice conflict keeps the contiguous figure on its original staff
  and emits `cross-staff-conflict`; a disjoint voice elsewhere in the bar does not
  force the figure onto RH stems. `original_staff` preserves the source staff.
- `summary.half_bar_shift` records eligibility, frame vote and the applied offset.
  The half-bar share and gliss coverage controls accept only values in [0, 1].

The keyfit configuration in `docs/engraving-keyfit-val-candidate.json` can be
selected with `--engraving-config`.

Since 1.3.1 a glissando the transcription broke in two (a few keys missed in the middle, so the key-by-key chain stops) is joined back into one line: two runs on the same key colour and in the same direction, the second starting within `gliss_gap` s of the end of the first and at most `gliss_join_keys` (3) keys further on.

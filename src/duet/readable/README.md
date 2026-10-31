# Readable engraving

Vendored from the online DuET demo: production engraving 1.0.0, derived from engraving 0.9.1 default readable. Only readable runtime modules are included. The normal piano notation pipeline is retained; inputs are supplied in memory from DuET’s validated MIDI reader and frame predictions instead of rereading MIDI/NPZ/CSV files.

The adapter copies raw score arrays and preserves exact input-note alignment, including FIFO overlap pairing and invalid-event cleanup. No model weights, evaluation data, alternative engraving modes, or renderer binaries are included. MuseScore is only used by the separate demo PDF renderer.

For pieces with fewer than two integer-beat anchors, fractional model attacks establish the grid. A single attack uses the MIDI exporter’s 120 BPM default and records a warning in the engraving summary. Diagnostic logs are collected in memory; the output directory receives only the final MusicXML.

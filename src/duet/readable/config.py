"""Runtime engraving controls, including independent automatic decisions and
key/spelling costs. Units: "beat" means one
notated beat (a quarter note in n/4 meters); "s" means seconds of performance."""
from dataclasses import dataclass, fields
import math


@dataclass
class Config:
    # ---- mode -----------------------------------------------------------------
    # ---- beat grid (frames.csv) -------------------------------------------
    beat_peak_height: float = 0.3      # min p_any_beat for a frame peak to count as a beat
    beat_peak_distance: int = 15       # min frames (50 fps) between beat peaks
    beat_collapse_ratio: float = 0.65  # drop a peak whose interval < ratio * previous interval
    beat_refine_window: float = 0.06   # s; snap a beat time to the median of onsets this close
    beat_phase_tol: float = 0.25       # beats; a frame beat further than this from a model beat is off-beat
    phase_window_bars: int = 8         # window (bars) for downbeat evidence after a gap / diagnostics
    snap_bar_lines: bool = True       # propose half-quarter bar-line snapping, roll back if topology changes
    grid_count_fix: bool = True        # let frame evidence change the model's beat count in gaps
    rewind_min: float = 1.5            # beats the model numbering must jump back to count as a rewind
    rewind_fit_s: float = 3.0          # seconds of earlier notes used to extend the earlier numbering
    rewind_check_s: float = 2.0        # seconds after the jump that must follow the new numbering
    rewind_persist: float = 0.75       # share of those notes closer to the new numbering than the old
    rewind_bar_tol: float = 1.5        # beats; a shift this close to whole bars is taken as whole bars
    rewind_frame_conf: float = 0.8     # frame beat peaks this confident are used for that phase check
    rewind_phase_margin: float = 0.2   # a half-beat phase from the frame beats must fit this much better
    grid_keep_agreed: bool = True      # frames and model counting the same beats in a gap: keep that count
    agreed_override_margin: float = 1.0  # ... unless the later downbeats favour another count by this much
    free_zone_frames: bool = True      # a gap whose count the frames corrected is placed by timing
    grid_count_max: int = 1            # largest change of a gap's beat count (evaluation: larger changes hurt)
    gap_support_margin: float = 0.3    # downbeat-score gain needed to change the beat count of a gap
    implausible_min_margin: float = 0.0   # an impossible model count is changed only if downbeats do not prefer it
    gap_tempo_tol: float = 0.3         # a gap's beat count may differ from the local-tempo estimate by this fraction

    # ---- onset fusion -----------------------------------------------------
    retime_outliers: bool = False      # re-snap notes whose timing disagrees with the model (evaluation: net harmful)
    model_onset_tol: float = 0.06      # beats; keep model rhythm if timing agrees this well
    drift_window: float = 1.0          # beats; neighbourhood for the local timing drift
    drift_tol: float = 0.08            # beats; deviation that matches the neighbours' drift is rubato
    drift_max: float = 0.30            # beats; but never trust the model beyond this deviation
    grid_onset_tol: float = 0.10       # beats; else snap timing to {0,1/4,1/3,1/2,2/3,3/4}
    triplet_context_tol: float = 0.07  # beats; snap 16th-grid notes to triplets in triplet half-bars
    run_min_events: int = 6            # a regular figure needs this many evenly spaced attacks
    run_max_step: float = 0.5          # beats; only figures of 8ths or faster
    run_max_ratio: float = 2.6         # neighbouring inter-onset times may differ by at most this factor
    run_chord_spread: float = 0.08     # s; notes sharing a model position must be this close to join a figure
    roll_from_timing: bool = False     # timing-placed rolls outside free zones too (evaluation: off is better)
    triplet_context: bool = True       # 16th positions next to triplets snap to the triplet grid by timing
    compound_snap: bool = True         # 6/8 / 12/8: 32nd positions rounded to 16ths (model triplets kept)
    min_note_dur: float = 0.25         # beats; shorter model durations are written this long
    run_extend_max: int = 3            # attacks a run may absorb backwards when the model placed them off its grid
    run_extend_tol: float = 0.15       # beats; how far off the run's grid such an attack may be

    # ---- durations --------------------------------------------------------------
    extend_by_hold: bool = True        # lengthen notes to how long the key was held (avoids rests)
    # ---- second voice ------------------------------------------------------------
    voice_b_over: float = 0.5          # beats a note must outlast the next attack to move to layer B
    voice_b_min_dur: float = 1.0       # minimum written length of such a note
    voice_b_hold: float = 0.0          # beats the key must stay down past the next attack
    rest_fill_max: float = 0.25        # beats; rests up to this long before the next attack are absorbed
    rest_fill_gap: float = 1.0         # ... only when the note-to-next-attack span is at most this
    rest_fill_ratio: float = 1.0       # ... and the written note is at least this times the rest

    # ---- swing ----------------------------------------------------------------
    swing_min: float = 0.57            # median position of off-beat 8ths (in the beat) that counts as swing
    swing_max: float = 0.72
    swing_iqr_max: float = 0.12        # and they must be consistent
    swing_min_notes: int = 20

    # ---- rolled chords ----------------------------------------------------
    roll_gap: float = 0.32             # beats; max gap between successive notes of a roll
    roll_max_spread: float = 0.7       # beats; max total spread (model says same onset)
    roll_spread_free: float = 0.45     # beats; max spread when inferred without the model
    roll_min_spread: float = 0.12      # beats; below this it is just a chord

    # ---- playability --------------------------------------------------------
    hand_span: int = 14                # semitones one hand may stretch (dyads: +2)
    hand_max_notes: int = 5
    cross_over_min: int = 60           # LH attacks at/above this lying above all sounding RH notes go to the RH staff
    cross_high: bool = True            # LH attacks too high for the bass clef (local_clef_min_ledgers) in a bar with a LH bass go to the RH staff
    hand_move_near: int = 9            # a note may change hands only if within this of that hand's context
    roll_lh_max_span: int = 19         # wider LH chords become bass-grace + chord instead of a roll

    # ---- layout -------------------------------------------------------------
    clef_tie_penalty: float = 3.0      # per ledger line of a note tied across a bar-level clef change
    clef_switch_penalty: float = 6.0   # LH (evaluation: 10 -> 6 lowers ledger lines, clef changes near target)
    clef_switch_penalty_rh: float = 14.0  # RH changes clef less readily
    clef_rh_bass_bias: float = 2.0     # per-bar cost of a bass clef in the RH staff
    grace_clef_weight: float = 6.0
    local_clef_min_ledgers: int = 5
    ottava_rh: tuple = (72, 94)        # (min pitch of every event, top pitch needed) for RH 8va
    ottava_lh: tuple = (76, 89)        # same for LH while in treble clef
    ottava_bassa: tuple = (43, 31, 6, 4)  # LH 8vb: >= 6 events all <= G2, >= 4 reaching G1
    report_ledgers: int = 5            # report notes needing at least this many ledger lines

    # ---- decisions from the notes (engrave/review.py); unsure spots go to <title>.review.json ----
    auto_keys: bool = True             # key signatures from the notes (keyfit) instead of the model's
    auto_spelling: bool = True         # notes outside the key spelled from harmony and line (spellfit)
    auto_runs: bool = True             # brushed notes in runs as grace notes, one-place overlaps cut
    auto_gliss: bool = True            # glissandi written as a glissando line between their two ends
    auto_chords: bool = True           # notes played together but written apart joined into one chord
    gliss_min_notes: int = 8           # notes of one key colour in a row, one direction
    gliss_min_span: int = 12           # semitones (an octave)
    gliss_gap: float = 0.07            # s between two notes of the glissando at most (one missed key)
    gliss_mean_gap: float = 0.045      # s between notes on average at most
    gliss_disorder: float = 0.02       # s a note of the glissando may be transcribed before the previous one
    gliss_coverage: float = 0.7        # share of the keys in the range a glissando / scale run must play
    scale_run_min_notes: int = 8       # review only: fast runs on a major scale (glissando or fingered?)
    scale_run_min_span: int = 12
    scale_run_mean_gap: float = 0.045     # as fast as a glissando: fingered scales run at 50-70 ms a note
    chord_split_ms: float = 15.0       # notes played this close are one chord
    duple_tol: float = 0.06            # beat: largest playing error for a 16th / 32nd proposal (review)
    duple_bias: float = 0.02           # beat: how much worse than the tuplet a simpler writing may fit
    odd_beat_tol: float = 0.2          # beat: largest deviation from even spacing for a 5:4 / 7:4 tuplet proposal (review)
    key_switch: float = 10.0
    key_per_fifth: float = 0.3
    key_model_cost: float = 0.05
    key_borrowed_share: float = 0.3
    key_borrowed_max_bars: int = 24
    spelling_window: float = 2.0
    spelling_w_key: float = 1.0
    spelling_w_line: float = 3.0
    spelling_passes: int = 2
    spelling_w_sound: float = 1.0
    spelling_w_odd: float = 0.0
    spelling_line_prev: float = 0.0
    spelling_review_margin: float = 1.5
    write_review: bool = True          # write <title>.review.json

    # ---- expression ---------------------------------------------------------
    dynamics_block_bars: int = 8
    tempo_section_bars: int = 4        # a tempo change lasting this many bars gets a new metronome mark
    tempo_change: float = 0.08         # relative change that counts as a new tempo section
    tempo_play_window: int = 5         # bars in the rolling median the playback tempo follows
    tempo_play_change: float = 0.06    # playback tempo changes only when that curve moves this much
    closing_bars: int = 4              # slow-downs reaching into the last bars stay a "rit."
    rit_ratio: float = 0.9             # bar tempo below ratio * current mark -> "rit."
    a_tempo_ratio: float = 0.96
    rit_min_bars: int = 2              # a slow-down shorter than this many bars gets no rit. / a tempo


    def __post_init__(self):
        for f in fields(self):
            value = getattr(self, f.name)
            if f.type is bool:
                if type(value) is not bool:
                    raise ValueError(f"{f.name} must be a boolean")
            elif f.type in (int, float):
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"{f.name} must be numeric")
                if f.type is int and not isinstance(value, int):
                    raise ValueError(f"{f.name} must be an integer")
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"{f.name} must be finite and nonnegative")
            elif f.type is tuple:
                if not isinstance(value, (tuple, list)) or len(value) != len(f.default):
                    raise ValueError(f"{f.name} must contain {len(f.default)} numbers")
                if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value):
                    raise ValueError(f"{f.name} must contain finite numbers")
        for name in ('spelling_window', 'spelling_passes', 'key_borrowed_max_bars'):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.key_borrowed_share > 1:
            raise ValueError("key_borrowed_share must be between zero and one")

    @classmethod
    def from_dict(cls, values):
        if not isinstance(values, dict):
            raise ValueError("Engraving configuration must be a JSON object")
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown engraving parameters: {sorted(unknown)}")
        return cls(**values)


    @classmethod
    def resolve(cls, values=None):
        if values is None:
            return cls()
        if isinstance(values, cls):
            values.__post_init__()
            return values
        return cls.from_dict(values)

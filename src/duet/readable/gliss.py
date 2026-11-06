"""Glissandi written as one wavy line from the first note to the last.

The model gives every input note its own score position, which turns a glissando into a crowded run of 32nds, and it
has no way to say "glissando". Runs are found in the performance (find; Config.auto_glissandi). Other notes change only
when struck during the line: they go back to where they were played and become the second voice of the staff.

Detection cannot use note length: the pedal carries a run and the chords it crosses to the same end. A wrong detection
removes notes from the page, so every detected run is a review item.
"""
from fractions import Fraction as F
import math
import statistics

import numpy as np

from .spelling import STEP_PC

MATCH_SECONDS = 0.005
POSITION_GRID = F(1, 4)     # the ends of the line go on sixteenths: its inner rhythm is not notated
END_NOTE = F(1, 4)

# ---- detection ---------------------------------------------------------------
WHITE_KEYS = (0, 2, 4, 5, 7, 9, 11)
BLACK_KEYS = (1, 3, 6, 8, 10)
NOTE_NAMES = ('C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B')
MIN_DEGREES = 8
MAX_SECONDS_PER_DEGREE = 0.075      # slower than this is a played scale
MIN_SEED_SECONDS = 0.08
MAX_SEED_SECONDS = 3.0
MIN_COVERAGE = 0.7                  # share of the scale degrees in the span that must be present
MIN_CENTER_PITCH = 60               # below these the lines found were octave-doubled chords
MIN_TOP_PITCH = 72
SEED_MARGIN_SECONDS = 0.03
# In degrees from the line: a note on it and a note extending the run past its end.
LINE_TOLERANCE = 1.0
GROW_TOLERANCE = 2.0
# A note may fill a degree that would otherwise be empty, so it cannot displace a note on the line. Late is wide because
# onsets near the end of a fast run are transcribed late and out of order. Early is narrow because a note struck before
# the run reaches it is not part of the run.
CLAIM_LATE_TOLERANCE = 5.0
CLAIM_EARLY_TOLERANCE = 2.0
# Up to this many notes past a gap of this many median gaps do not belong to the run. Inside a run such gaps stay,
# because they are the notes the transcription missed.
MAX_END_GAP_RATIO = 2.0
MAX_END_NOTES = 2
# An end note struck with this many notes off the line, within this time, belongs to their chord.
END_CHORD_NOTES = 2
END_CHORD_SECONDS = 0.015
# The same chord within this many beats means a note on the line can be a chord note.
REPEAT_BEATS = 2.0
REPEAT_GROUP_SECONDS = 0.03
REPEAT_MIN_PITCHES = 2
MAX_REPEATED_SHARE = 0.3            # a run with this share of repeated chord notes is a repeated chord pattern


def _major_scale(root):
    return tuple(sorted((root + step) % 12 for step in WHITE_KEYS))


def _scales():
    """In the order they win a tie: a run that fits white keys equally well is a white-key run."""
    return [('white', WHITE_KEYS), ('black', BLACK_KEYS)] + [
        (NOTE_NAMES[root] + ' major', _major_scale(root)) for root in range(1, 12)]


def _degree(pitch, scale):
    return (pitch // 12) * len(scale) + scale.index(pitch % 12)


def _seed_lines(notes, scale):
    """Straight lines in time and scale degree, best first and not overlapping in one direction."""
    members = [note for note in notes if note['pitch'] % 12 in scale]
    if len(members) < MIN_DEGREES * MIN_COVERAGE:
        return []
    times = np.array([note['t'] for note in members])
    degrees = np.array([_degree(note['pitch'], scale) for note in members])
    lines = []
    for first in range(len(members)):
        low = np.searchsorted(times, times[first] + MIN_SEED_SECONDS)
        high = np.searchsorted(times, times[first] + MAX_SEED_SECONDS)
        for last in range(low, high):
            rise = degrees[last] - degrees[first]
            if abs(rise) < MIN_DEGREES - 1:
                continue
            duration = times[last] - times[first]
            if duration / abs(rise) > MAX_SECONDS_PER_DEGREE:
                continue
            window = slice(first, last + 1)
            predicted = degrees[first] + rise * (times[window] - times[first]) / duration
            on_line = degrees[window][np.abs(degrees[window] - predicted) <= LINE_TOLERANCE]
            lowest, highest = sorted((degrees[first], degrees[last]))
            covered = len(set(on_line[(on_line >= lowest) & (on_line <= highest)].tolist()))
            if covered / (abs(rise) + 1) >= MIN_COVERAGE:
                lines.append((covered, times[first], times[last], int(degrees[first]), int(degrees[last])))
    kept = []
    for line in sorted(lines, key=lambda item: (-item[0], item[1])):
        direction = np.sign(line[4] - line[3])
        if all(line[2] < other[1] or line[1] > other[2] or direction != np.sign(other[4] - other[3]) for other in kept):
            kept.append(line)
    return kept


def _closest_on_line(candidates, line, early, late, skip):
    """The note closest to the line for each scale degree within the tolerances, as {degree: (distance, note)}."""
    origin, seconds_per_degree, start_degree = line
    chosen = {}
    for degree, note in candidates:
        if id(note) in skip:
            continue
        lateness = (note['t'] - (origin + (degree - start_degree) * seconds_per_degree)) / abs(seconds_per_degree)
        if -early <= lateness <= late and (degree not in chosen or abs(lateness) < chosen[degree][0]):
            chosen[degree] = (abs(lateness), note)
    return chosen


def _fit_line(members):
    """Least squares through {degree: note}: (time at the first degree, seconds per degree signed, that degree)."""
    degrees = sorted(members)
    slope, intercept = np.polyfit(degrees, [members[degree]['t'] for degree in degrees], 1)
    return intercept + slope * degrees[0], slope, degrees[0]


def _collect_members(notes, scale, seed, taken):
    _, first_time, last_time, first_degree, last_degree = seed
    candidates = [(_degree(note['pitch'], scale), note) for note in notes if note['pitch'] % 12 in scale]
    lowest, highest = sorted((first_degree, last_degree))
    line = (first_time, (last_time - first_time) / (last_degree - first_degree), first_degree)
    inside = [item for item in candidates if lowest <= item[0] <= highest
              and first_time - SEED_MARGIN_SECONDS <= item[1]['t'] <= last_time + SEED_MARGIN_SECONDS]
    chosen = _closest_on_line(inside, line, LINE_TOLERANCE, LINE_TOLERANCE, taken)
    if len(chosen) < 2:
        return None
    line = _fit_line({degree: item[1] for degree, item in chosen.items()})

    # Grow past both ends, degree by degree, allowing one missing degree.
    for step in (-1, 1):
        growing = True
        while growing:
            growing = False
            edge = max(chosen) if step > 0 else min(chosen)
            for degree in (edge + step, edge + 2 * step):
                predicted = line[0] + (degree - line[2]) * line[1]
                near = [(abs(note['t'] - predicted), note) for value, note in candidates
                        if value == degree and id(note) not in taken
                        and abs(note['t'] - predicted) <= GROW_TOLERANCE * abs(line[1])]
                if near:
                    chosen[degree] = min(near, key=lambda item: item[0])
                    growing = True
                    break

    lowest, highest = min(chosen), max(chosen)
    inside = [item for item in candidates if lowest <= item[0] <= highest]
    for degree, item in _closest_on_line(inside, line, CLAIM_EARLY_TOLERANCE, CLAIM_LATE_TOLERANCE, taken).items():
        chosen.setdefault(degree, item)
    return {degree: item[1] for degree, item in chosen.items()}


def _trim_gaps(members):
    """Drop a few notes at either end that sit past a gap the rest of the run does not have."""
    while len(members) > MAX_END_NOTES + 1:
        ordered = sorted(members, key=lambda degree: members[degree]['t'])
        gaps = [members[b]['t'] - members[a]['t'] for a, b in zip(ordered, ordered[1:])]
        limit = MAX_END_GAP_RATIO * statistics.median(gaps)
        for count in range(1, MAX_END_NOTES + 1):
            if gaps[count - 1] > limit:
                for degree in ordered[:count]:
                    del members[degree]
                break
            if gaps[-count] > limit:
                for degree in ordered[-count:]:
                    del members[degree]
                break
        else:
            return


def _repeated_chord_note(note, notes, member_ids, seconds_per_beat):
    """Is this note struck with notes off the line, in a chord also struck shortly before or after?"""
    start = note['t']
    chord = {other['pitch'] for other in notes
             if abs(other['t'] - start) <= REPEAT_GROUP_SECONDS and id(other) not in member_ids}
    if not chord:
        return False
    chord.add(note['pitch'])
    for other in notes:
        distance = abs(other['t'] - start)
        if other['pitch'] != note['pitch'] or id(other) in member_ids \
                or not REPEAT_GROUP_SECONDS < distance <= REPEAT_BEATS * seconds_per_beat:
            continue
        group = {third['pitch'] for third in notes if abs(third['t'] - other['t']) <= REPEAT_GROUP_SECONDS}
        if len(group & chord) >= REPEAT_MIN_PITCHES:
            return True
    return False


def _drop_chord_ends(members, notes, seconds_per_beat):
    """An end note that belongs to a chord leaves no hole in the run, so it can go back to the chord."""
    while len(members) > 2:
        member_ids = {id(note) for note in members.values()}
        ordered = sorted(members, key=lambda degree: members[degree]['t'])
        for degree in (ordered[0], ordered[-1]):
            note = members[degree]
            struck_with = sum(1 for other in notes if id(other) not in member_ids
                              and abs(other['t'] - note['t']) <= END_CHORD_SECONDS)
            if struck_with >= END_CHORD_NOTES or _repeated_chord_note(note, notes, member_ids, seconds_per_beat):
                del members[degree]
                break
        else:
            return


def find(perf, seconds_per_beat):
    """Runs in a performance, as {notes: [[seconds, pitch], ...], scale}, and the number rejected as repeated chords.

    perf holds (onset, offset, pitch, velocity). A run is a straight line in time and scale degree, on the scale that
    covers the most degrees of it, so a major scale run is not read as white keys with stray notes.
    """
    notes = sorted((dict(t=float(row[0]), pitch=int(row[2])) for row in perf), key=lambda note: (note['t'], note['pitch']))
    seeds = [(seed, order, name, scale) for order, (name, scale) in enumerate(_scales())
             for seed in _seed_lines(notes, scale)]
    runs, occupied, taken = [], [], set()
    rejected = 0
    for seed, _, name, scale in sorted(seeds, key=lambda item: (-item[0][0], item[1], item[0][1])):
        direction = 1 if seed[4] > seed[3] else -1
        if any(seed[1] <= end and seed[2] >= start and direction == other for start, end, other in occupied):
            continue
        members = _collect_members(notes, scale, seed, taken)
        if not members:
            continue
        _trim_gaps(members)
        _drop_chord_ends(members, notes, seconds_per_beat)
        pitches = [note['pitch'] for note in members.values()]
        ordered = sorted(members.values(), key=lambda note: note['t'])
        span = max(members) - min(members) + 1
        duration = ordered[-1]['t'] - ordered[0]['t']
        if span < MIN_DEGREES or len(members) / span < MIN_COVERAGE or duration <= 0 \
                or duration / (span - 1) > MAX_SECONDS_PER_DEGREE \
                or (min(pitches) + max(pitches)) / 2 < MIN_CENTER_PITCH or max(pitches) < MIN_TOP_PITCH:
            continue
        member_ids = {id(note) for note in ordered}
        occupied.append((ordered[0]['t'], ordered[-1]['t'], direction))
        repeated = sum(1 for note in ordered if _repeated_chord_note(note, notes, member_ids, seconds_per_beat))
        if repeated / len(ordered) >= MAX_REPEATED_SHARE:
            rejected += 1
            continue
        taken |= member_ids
        runs.append(dict(notes=[[note['t'], note['pitch']] for note in ordered], scale=name))
    runs.sort(key=lambda run: run['notes'][0][0])
    return runs, rejected


def snap(value):
    """The nearest sixteenth, half up so that a tie does not depend on the beat."""
    return F(math.floor(value / float(POSITION_GRID) + 0.5)) * POSITION_GRID


def natural_spelling(pitch):
    """A white key written as itself: in E major a white-key glissando has F natural, not E sharp."""
    return next(((step, 0) for step, pc in STEP_PC.items() if pc == pitch % 12), None)


def apply(notes, runs, ctx):
    """Keep the first and last note of each run with the line between them and drop the rest. Sets ctx.gliss_spans."""
    ctx.gliss_spans = []
    ctx.gliss_detected = []
    last_end = None
    for number, run in enumerate(runs, 1):
        wanted = sorted(run['notes'])
        found = []
        for seconds, pitch in wanted:
            match = [n for n in notes if n['pitch'] == pitch and abs(n['t'] - seconds) <= MATCH_SECONDS]
            found.append(match[0] if match else None)
        # An inner note may be gone (a duplicate removed earlier); the line only needs its two ends.
        if len(found) < 2 or found[0] is None or found[-1] is None:
            ctx.log(f"WARNING gliss: run {number} did not match its first or last note; left as it was")
            continue
        found = [n for n in found if n is not None]
        first, last = found[0], found[-1]
        # Performed positions: the model stretches a glissando onto its 32nd grid, so its own positions overrun.
        start, end = snap(first['x']), snap(last['x'])
        if last_end is not None and start <= last_end:
            start = last_end + POSITION_GRID
        if end <= start:
            end = start + POSITION_GRID
        last_end = end
        staff = first['staff']
        for note, position, duration, kind in ((first, start, end - start, 'start'), (last, end, END_NOTE, 'stop')):
            note.update(on=position, dur=duration, staff=staff, grace=False, bass_grace=False, roll=False,
                        gliss=kind, gliss_staff=staff, gliss_scale=run['scale'])
        inner = {id(n) for n in found[1:-1]}
        notes[:] = [n for n in notes if id(n) not in inner]
        ctx.gliss_spans.append((staff, start, end))
        ctx.gliss_detected.append((first, last, len(found), run['scale'], start))
        ctx.log(f"gliss: run {number} {first['pitch']} to {last['pitch']}, {len(found)} notes written as a line "
                f"({ctx.where(start)})")
    replace_crossing_notes(notes, ctx)
    return notes


def replace_crossing_notes(notes, ctx):
    """Notes struck during a line go back where they were played.

    The model's stretched glissando carried them along, by up to half a beat in Dream Believers m136.
    """
    moved = 0
    for _, start, end in ctx.gliss_spans:
        for note in notes:
            if 'gliss' in note or note['grace'] or not float(start - POSITION_GRID) <= note['x'] < float(end):
                continue
            position = snap(note['x'])
            if abs(float(note['on']) - note['x']) > float(POSITION_GRID) and note['on'] != position:
                note['on'] = position
                moved += 1
    if moved:
        ctx.log(f"gliss: {moved} crossing notes moved to where they were played")


def review_items(rv, ctx):
    """Every detected run is a question: a wrong one has taken notes off the page."""
    for first, last, count, scale, start in ctx.gliss_detected:
        bar = ctx.bar_of(start)
        rv.add('glissando', 'medium', rv.bars([bar]),
               "A fast run written as a glissando line: right, or real notes to write out?",
               f"{count} notes on {scale} keys from {first['pitch']} to {last['pitch']}, "
               f"played in {last['t'] - first['t']:.2f} s",
               f"{first['t']:.3f}:{first['pitch']} to {last['t']:.3f}:{last['pitch']} written as a glissando",
               [dict(label='keep', edit=None), dict(label='write the notes: run with --no-auto-glissandi', edit=None)])


def finish(notes):
    """After the passes that change staves and spelling: a line stays on one staff, and white keys stay natural."""
    for note in notes:
        if 'gliss' in note:
            note['staff'] = note['gliss_staff']
            if note['gliss_scale'] == 'white':
                note['spell'] = natural_spelling(note['pitch'])


def split_layers(pool, staff, spans):
    """The line is the main voice of its staff; other notes struck during it are the second voice."""
    for span_staff, start, end in spans:
        if span_staff != staff:
            continue
        for note in pool:
            if 'gliss' in note:
                note['layer'] = 'A'
            elif start <= note['on'] < end:
                note['layer'] = 'B'


def restore_lengths(pool, spans):
    """Duration cleanup must not end the first note before the line reaches the last one."""
    for note in pool:
        if note.get('gliss') == 'start':
            for _, start, end in spans:
                if start == note['on']:
                    note['dur'] = end - start


def notation(note, is_attack, is_last_piece):
    """The <notations> content of one written piece of a note: the line leaves the last piece of its first note."""
    kind = note.get('gliss')
    if kind == 'stop' and is_attack:
        return '<glissando type="stop" line-type="wavy" number="1"/>'
    if kind == 'start' and is_last_piece:
        return '<glissando type="start" line-type="wavy" number="1">gliss.</glissando>'
    return ''

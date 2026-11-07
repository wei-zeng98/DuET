"""Glissandi written as one wavy line from the first note to the last.

The model gives every input note its own score position, which turns a glissando into a crowded run of 32nds, and it
has no way to say "glissando". Runs on one key colour are found in the performance (find; Config.auto_gliss). Other
notes change only when struck during the line: they go back to where they were played and become the second voice of
the staff.

Detection cannot use note length: the pedal carries a run and the chords it crosses to the same end. A wrong detection
removes notes from the page, so every detected run is a review item. Fast runs on a major scale are review items only.
"""
from fractions import Fraction as F
from bisect import bisect_left, bisect_right
import math

from .spelling import STEP_PC

MATCH_SECONDS = 0.005
POSITION_GRID = F(1, 4)     # the ends of the line go on sixteenths: its inner rhythm is not notated
END_NOTE = F(1, 4)

# ---- detection ---------------------------------------------------------------
WHITE_KEYS = (0, 2, 4, 5, 7, 9, 11)
BLACK_KEYS = (1, 3, 6, 8, 10)
NOTE_NAMES = ('C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B')


def _chains(notes, scale_of, min_notes, min_span, gap, mean_gap, disorder, coverage):
    """Runs played one key after another on a scale: at least min_notes notes, each the next or next-but-one key of
    the scale (scale_of(first note) -> pitch classes), within gap s after the previous one (or up to disorder s before
    it: fast runs come out of the transcription slightly out of order), in one direction, spanning at least min_span
    semitones, on average within mean_gap s a note, playing at least coverage of the scale's keys in that range."""
    ns = sorted(notes, key=lambda note: (note['t'], note['pitch']))
    times = [note['t'] for note in ns]
    used, out = set(), []
    for first in ns:
        if id(first) in used:
            continue
        scale = scale_of(first)
        if first['pitch'] % 12 not in scale:
            continue
        for direction in (1, -1):
            chain = [first]
            while True:
                last = chain[-1]
                keys, pitch = [], last['pitch']
                while len(keys) < 2:
                    pitch += direction
                    if pitch % 12 in scale:
                        keys.append(pitch)
                low, high = bisect_left(times, last['t'] - disorder), bisect_right(times, last['t'] + gap)
                inside = {id(note) for note in chain}
                candidates = [note for note in ns[low:high]
                              if id(note) not in inside and id(note) not in used and note['pitch'] in keys]
                if not candidates:
                    break
                chain.append(min(candidates, key=lambda note: (keys.index(note['pitch']), note['t'])))
            start, end = min(n['t'] for n in chain), max(n['t'] for n in chain)
            lowest, highest = sorted((first['pitch'], chain[-1]['pitch']))
            keys_in_range = sum(1 for p in range(lowest, highest + 1) if p % 12 in scale)
            if len(chain) >= min_notes and highest - lowest >= min_span and len(chain) >= coverage * keys_in_range \
                    and (end - start) / (len(chain) - 1) <= mean_gap:
                out.append(chain)
                used.update(id(note) for note in chain)
                break
    return out


def find(perf, cfg):
    """Glissandi and fast major-scale runs in a performance, each as {notes: [[seconds, pitch], ...], scale}.

    perf holds (onset, offset, pitch, velocity). A glissando stays on one key colour (white or black keys). A fast run on
    a major scale is only reported (scale_runs): fingered scales that fast are far more common than scale glissandi.
    """
    notes = [dict(t=float(row[0]), pitch=int(row[2])) for row in perf]
    colour = lambda note: WHITE_KEYS if note['pitch'] % 12 in WHITE_KEYS else BLACK_KEYS
    runs, taken = [], set()
    for chain in _chains(notes, colour, cfg.gliss_min_notes, cfg.gliss_min_span, cfg.gliss_gap, cfg.gliss_mean_gap,
                         cfg.gliss_disorder, cfg.gliss_coverage):
        taken.update(id(note) for note in chain)
        runs.append(dict(notes=[[n['t'], n['pitch']] for n in chain],
                         scale='white' if chain[0]['pitch'] % 12 in WHITE_KEYS else 'black'))
    found = []                                      # runs on every major scale; the scale playing most notes wins
    for root in range(1, 12):                       # C major is the white keys
        scale = tuple(sorted((root + step) % 12 for step in WHITE_KEYS))
        for chain in _chains([n for n in notes if id(n) not in taken], lambda note: scale, cfg.scale_run_min_notes,
                             cfg.scale_run_min_span, cfg.gliss_gap, cfg.scale_run_mean_gap, cfg.gliss_disorder,
                             cfg.gliss_coverage):
            if not (all(n['pitch'] % 12 in WHITE_KEYS for n in chain) or all(n['pitch'] % 12 in BLACK_KEYS for n in chain)):
                found.append((root, chain))
    scale_runs = []
    for root, chain in sorted(found, key=lambda item: -len(item[1])):
        if any(id(note) in taken for note in chain):
            continue
        taken.update(id(note) for note in chain)
        scale_runs.append(dict(notes=[[n['t'], n['pitch']] for n in chain], scale=NOTE_NAMES[root] + ' major'))
    runs.sort(key=lambda run: run['notes'][0][0])
    scale_runs.sort(key=lambda run: run['notes'][0][0])
    return runs, scale_runs


def snap(value):
    """The nearest sixteenth, half up so that a tie does not depend on the beat."""
    return F(math.floor(value / float(POSITION_GRID) + 0.5)) * POSITION_GRID


def natural_spelling(pitch):
    """A white key written as itself: in E major a white-key glissando has F natural, not E sharp."""
    return next(((step, 0) for step, pc in STEP_PC.items() if pc == pitch % 12), None)


def _match(notes, run):
    found = []
    for seconds, pitch in sorted(run['notes']):
        match = [n for n in notes if n['pitch'] == pitch and abs(n['t'] - seconds) <= MATCH_SECONDS]
        found.append(match[0] if match else None)
    return found


def apply(notes, runs, ctx, scale_runs=()):
    """Keep the first and last note of each run with the line between them and drop the rest. Sets ctx.gliss_spans.

    The first note keeps the model's position (on the nearest 16th when off that grid) and lasts until the last note,
    which goes on the 16th nearest to where it was played. Each end stays on its own staff, so the line can cross."""
    ctx.gliss_spans = []
    ctx.gliss_detected = []
    ctx.gliss_scale_runs = []
    last_end = None
    for number, run in enumerate(runs, 1):
        found = _match(notes, run)
        # An inner note may be gone (a duplicate removed earlier); the line only needs its two ends.
        if len(found) < 2 or found[0] is None or found[-1] is None:
            ctx.log(f"WARNING gliss: run {number} did not match its first or last note; left as it was")
            continue
        found = [n for n in found if n is not None]
        first, last = found[0], found[-1]
        start = first['on'] if (first['on'] / POSITION_GRID).denominator == 1 else snap(first['on'])
        end = snap(last['x'])
        if last_end is not None and start <= last_end:
            start = last_end + POSITION_GRID
        if end <= start:
            end = start + POSITION_GRID
        last_end = end
        line = (number - 1) % 6 + 1
        first.update(on=start, dur=end - start)
        last.update(on=end, dur=max(END_NOTE, snap(last['dur'])))
        for note, kind in ((first, 'start'), (last, 'stop')):
            note.update(grace=False, bass_grace=False, roll=False, gliss=kind, gliss_staff=note['staff'],
                        gliss_scale=run['scale'], gliss_number=line)
        inner = {id(n) for n in found[1:-1]}
        notes[:] = [n for n in notes if id(n) not in inner]
        ctx.gliss_spans.append((first['staff'], start, end))
        ctx.gliss_detected.append((first, last, len(found), run['scale'], start))
        ctx.log(f"gliss: run {number} {first['pitch']} to {last['pitch']}, {len(found)} notes written as a line "
                f"({ctx.where(start)})")
    for run in scale_runs:
        found = [n for n in _match(notes, run) if n is not None]
        if len(found) >= 2:
            ctx.gliss_scale_runs.append((found[0], found[-1], len(found), run['scale'], run['notes']))
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
    """Every detected run is a question: a wrong one has taken notes off the page. A fast run on a major scale is asked
    about too, but left written out."""
    for first, last, count, scale, start in ctx.gliss_detected:
        bar = ctx.bar_of(start)
        rv.add('glissando', 'medium', rv.bars([bar]),
               "A fast run written as a glissando line: right, or real notes to write out?",
               f"{count} notes on {scale} keys from {first['pitch']} to {last['pitch']}, "
               f"played in {last['t'] - first['t']:.2f} s",
               f"{first['t']:.3f}:{first['pitch']} to {last['t']:.3f}:{last['pitch']} written as a glissando",
               [dict(label='keep', edit=None), dict(label='write the notes: run with --no-auto-gliss', edit=None)])
    for first, last, count, scale, played in getattr(ctx, 'gliss_scale_runs', []):
        bars = sorted({ctx.bar_of(first['on']), ctx.bar_of(last['on'])})
        mean = (played[-1][0] - played[0][0]) / (len(played) - 1) * 1000
        rv.add('scale-run', 'low', rv.bars(bars),
               f"{count} notes of {scale} played in a row ({mean:.0f} ms a note): a fingered scale written out (keep), "
               f"or a glissando through the scale?",
               f"from {first['pitch']} to {last['pitch']}", 'written out',
               [dict(label='keep the notes', edit=None),
                dict(label='a glissando', edit={'gliss': [{'from': f"{first['t']:.3f}:{first['pitch']}",
                                                           'to': f"{last['t']:.3f}:{last['pitch']}"}]})])


def finish(notes):
    """After the passes that change staves and spelling: each end stays on its staff, and white keys stay natural."""
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
    kind, number = note.get('gliss'), note.get('gliss_number', 1)
    if kind == 'stop' and is_attack:
        return f'<glissando type="stop" line-type="wavy" number="{number}"/>'
    if kind == 'start' and is_last_piece:
        return f'<glissando type="start" line-type="wavy" number="{number}">gliss.</glissando>'
    return ''

"""Decisions taken from the notes, and the places left for a reviewer.

The score model's key signatures, spelling and fast runs are often wrong in ways
the notes themselves show. Where a rule was measured to help on the test set it
is applied here (config switches auto_keys, auto_spelling, auto_runs):

  keys      engrave.keyfit: key signatures from the notes (key agreement with the
            target scores 67% -> 77%)
  spelling  engrave.spellfit: notes outside the key spelled from their harmony and
            line (right 57% -> 73%)
  runs      engrave.runs, the safe part only: an isolated brushed note played
            first becomes a grace note; a note overlapping the next note of a run
            by at most one grid place is cut there

Everything a rule cannot settle goes to <title>.review.json: each item says what
the pipeline did, why it is unsure, and gives edits (the engrave/edits.py format)
for each answer, so a reviewer (the engrave-polish skill) only decides. Items:

  key           a key change taken from the notes (undo: the model's key)
  key-borrowed  a passage that leaves a key and comes back while the old tonic
                stays in the bass: borrowed chords or a modulation?
  spelling      spellings chosen with a small margin, grouped by harmony
  grace         a brushed note written as a grace note (undo)
  respace       a run whose beat mixes subdivisions or stacks notes played apart
  free-run      more notes in a beat than the run's subdivision holds (tuplet?)
  displaced     a note written away from where it was played while its
                neighbours are on time
"""
import json
from collections import defaultdict
from fractions import Fraction as F
from statistics import median

from . import keyfit, spellfit, runs as runs_mod
from .spelling import Speller, ACC_TXT

KEY_NAME = {-7: 'Cb', -6: 'Gb', -5: 'Db', -4: 'Ab', -3: 'Eb', -2: 'Bb', -1: 'F', 0: 'C',
            1: 'G', 2: 'D', 3: 'A', 4: 'E', 5: 'B', 6: 'F#', 7: 'C#'}
SPELL_MARGIN = 1.5          # spelling decisions closer than this (line-of-fifths units) are listed


def note_id(n):
    """a note's id in review and edits files: performance onset (s, 3 decimals) and MIDI pitch"""
    return f"{n['t']:.3f}:{n['pitch']}"


class Review:
    def __init__(self, ctx):
        self.ctx, self.items = ctx, []
        self.in_runs = set()          # notes of fast runs: their questions come from auto_runs

    def num(self, m):
        return m + (0 if self.ctx.pickup else 1)

    def bars(self, ms):
        ms = sorted(set(ms))
        out, a = [], 0
        for i in range(1, len(ms) + 1):
            if i == len(ms) or ms[i] != ms[i - 1] + 1:
                lo, hi = self.num(ms[a]), self.num(ms[i - 1])
                out.append(str(lo) if lo == hi else f"{lo}-{hi}"); a = i
        return ','.join(out)

    def add(self, kind, priority, bars, question, evidence, auto, options):
        key = f"{kind}:{bars}"
        n = sum(1 for it in self.items if it['id'].startswith(key))
        self.items.append(dict(id=key + (f"#{n + 1}" if n else ''), kind=kind, priority=priority, bars=bars,
                               question=question, evidence=evidence, auto=auto, options=options))

    def write(self, path, title):
        order = {'high': 0, 'medium': 1, 'low': 2}
        items = sorted(self.items, key=lambda it: (order[it['priority']], it['kind'], it['bars']))
        counts = defaultdict(int)
        for it in items:
            counts[it['kind']] += 1
        json.dump(dict(title=title, about="places the pipeline was not sure about; each option is an edits entry "
                                          "(engrave/edits.py) to merge into <title>.edits.json",
                       counts=counts, items=items), open(path, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)


def key_changes(keys, ctx):
    """keys per bar -> the edits "keys" list"""
    out = [[0 if ctx.pickup else 1, keys[0]]]
    for m in range(1, len(keys)):
        if keys[m] != keys[m - 1]:
            out.append([m + (0 if ctx.pickup else 1), keys[m]])
    return out


def regions(keys):
    out, a = [], 0
    for m in range(1, len(keys) + 1):
        if m == len(keys) or keys[m] != keys[a]:
            out.append((a, m - 1, keys[a])); a = m
    return out


def auto_keys(notes, ctx, rv, log):
    bp = [[] for _ in range(ctx.nbars)]
    bass = [None] * ctx.nbars
    for n in notes:
        m = ctx.bar_of(n['on'])
        bp[m].append((n['pitch'], 0.5 if n['grace'] else 1.0))
        if not n['grace'] and n['on'] - ctx.bars.start(m) < 1 and (bass[m] is None or n['pitch'] < bass[m]):
            bass[m] = n['pitch']
    model = list(ctx.keys)
    prop, _ = keyfit.propose(bp, model)
    out_m, out_p = keyfit.outside(bp, model), keyfit.outside(bp, prop)
    for lo, hi, k in regions(prop):
        olds = sorted(set(model[lo:hi + 1]))
        if olds == [k]:
            continue
        undo = list(prop); undo[lo:hi + 1] = model[lo:hi + 1]
        rv.add('key', 'high', rv.bars(range(lo, hi + 1)),
               f"Key signature {KEY_NAME[k]} ({k:+d}) here, where the model wrote "
               f"{'/'.join(f'{KEY_NAME[o]} ({o:+d})' for o in olds)}: is the change of key right, and does it start "
               f"and end on these bars (phrase starts)?",
               f"notes outside the scale: {sum(out_p[lo:hi + 1]):.0f} with {KEY_NAME[k]}, "
               f"{sum(out_m[lo:hi + 1]):.0f} with the model's key",
               f"key signature set to {KEY_NAME[k]}",
               [dict(label='keep', edit=None), dict(label="the model's key here", edit={'keys': key_changes(undo, ctx)})])
        log(f"auto keys: bars {rv.bars(range(lo, hi + 1))} {olds} -> {k}")
    _, kept = keyfit.guard(prop, bass, share=0.3)
    for lo, hi, k1, k0 in kept:
        alt = list(prop); alt[lo:hi + 1] = [k0] * (hi - lo + 1)
        rv.add('key-borrowed', 'high', rv.bars(range(lo, hi + 1)),
               f"These bars go to {KEY_NAME[k1]} and back to {KEY_NAME[k0]}, but {KEY_NAME[k0]}'s tonic is still often "
               f"in the bass: a real modulation, or borrowed chords around the same tonic (then keep {KEY_NAME[k0]} "
               f"and use accidentals)?",
               "bass on the old tonic in at least 30% of the bars",
               f"key signature set to {KEY_NAME[k1]}",
               [dict(label='modulation (keep)', edit=None),
                dict(label=f'borrowed chords: stay in {KEY_NAME[k0]}', edit={'keys': key_changes(alt, ctx)})])
    ctx.keys = prop


def acc(step, alt):
    return step + ACC_TXT[alt]


def harmony(ns, i, name):
    """(bass, names sounding with ns[i], line) - ns sorted by onset"""
    n = ns[i]
    window = ns[max(0, i - 200):i + 200]
    sounding = sorted((m for m in window if m['on'] <= n['on'] < m['on'] + m['dur']), key=lambda m: m['pitch'])
    names = list(dict.fromkeys(name(m) for m in sounding))
    same = [m for m in window if m['staff'] == n['staff'] and m is not n]
    prev = [name(m) for m in same if n['on'] - 1 <= m['on'] < n['on']][-2:]
    nxt = [name(m) for m in same if n['on'] < m['on'] <= n['on'] + 1][:2]
    return (name(sounding[0]) if sounding else '-', ' '.join(names),
            f"{' '.join(prev) or '-'} > {name(n)} > {' '.join(nxt) or '-'}")


def auto_spelling(notes, ctx, rv, log):
    key_of = lambda n: ctx.keys[ctx.bar_of(n['on'])]
    margins = {}
    prop = spellfit.propose(notes, key_of, margins=margins)
    for n in notes:
        if id(n) in prop:
            n['spell'], n['spell_auto'] = prop[id(n)], True
    log(f"auto spelling: {len(prop)} notes outside the key respelled from their harmony and line")
    ns = sorted((n for n in notes if not n['grace']), key=lambda n: (n['on'], n['pitch']))
    name = lambda m: acc(*Speller(key_of(m)).spell(m)[:2])
    groups = defaultdict(list)
    for i, n in enumerate(ns):
        gap, other = margins.get(id(n), (99.0, None))
        if gap >= SPELL_MARGIN or other is None:
            continue
        bass, snd, line = harmony(ns, i, name)
        groups[(name(n), acc(*other), bass, snd)].append((ctx.bar_of(n['on']), line, gap))
    for (now, alt, bass, snd), occ in groups.items():
        ms = [m for m, _, _ in occ]
        bars_list = ','.join(str(rv.num(m)) for m in sorted(set(ms)))
        lines = sorted({l for _, l, _ in occ})[:3]
        rv.add('spelling', 'medium', rv.bars(ms),
               f"{now} or {alt}? (bass {bass}, sounding {snd})",
               f"line {'; '.join(lines)}; margin {min(g for _, _, g in occ):.1f} (small = unsure)",
               f"written {now}",
               [dict(label=f'keep {now}', edit=None),
                dict(label=f'write {alt}', edit={'spell': [{'bars': bars_list, 'from': now, 'to': alt}]})])


def auto_runs(notes, ctx, perf, rv, log):
    info = {(round(p[0], 3), p[2]): (p[3], p[1] - p[0]) for p in perf}
    for n in notes:
        n['vel'], n['plen'] = info.get((round(n['t'], 3), n['pitch']), (0, 0))
    beat_of = lambda on: ctx.bars.start(ctx.bar_of(on)) + int(on - ctx.bars.start(ctx.bar_of(on)))
    origin = lambda m: ctx.bars.start(m) + (ctx.pickup if m == 0 else 0)
    graces = cuts = 0
    for st, run in runs_mod.find_runs(notes):
        rv.in_runs.update(id(n) for ev in run for n in ev)
        k, ch = runs_mod.propose(run, beat_of, [n for n in notes if n['staff'] == st])
        respace = []
        for n, d, why in ch.values():
            if d.get('grace'):
                n['grace'], n['dur'] = True, F(0)
                graces += 1
                m = ctx.bar_of(n['on'])
                rv.add('grace', 'low', rv.bars([m]),
                       "A brushed note written as a grace note: right, or a real note of the line?",
                       why, f"{note_id(n)} written as a grace note",
                       [dict(label='keep', edit=None), dict(label='a normal note', edit={'notes': [{'id': note_id(n), 'grace': False}]})])
            elif 'on' in d:
                respace.append((n, d, why))
            elif 'dur' in d:
                n['dur'] = d['dur']
                cuts += 1
        if respace:
            ms = [ctx.bar_of(n['on']) for n, _, _ in respace]
            edits = []
            for n, d, _ in respace:
                e = dict(id=note_id(n))
                m = ctx.bar_of(d['on'])
                e.update(bar=rv.num(m), beat=str(d['on'] - origin(m)))
                if 'dur' in d:
                    e['dur'] = str(d['dur'])
                edits.append(e)
            rv.add('respace', 'medium', rv.bars(ms),
                   f"In this run, notes are written off its {k}-per-beat grid or stacked though played one after "
                   f"another. Place them in playing order on the grid (as proposed), write a free tuplet, or keep?",
                   '; '.join(sorted({w for _, _, w in respace})),
                   'unchanged', [dict(label='keep', edit=None), dict(label='place in playing order', edit={'notes': edits})])
        for b, span, cnt in runs_mod.free_tuplets(run, k, beat_of):
            m = ctx.bar_of(b)
            rv.add('free-run', 'medium', rv.bars([m]),
                   f"{cnt} notes played in {span} beat(s), more than the {k}-per-beat grid holds: write them as one "
                   f"{cnt}-note tuplet in playing order?",
                   f"{'RH' if st == 0 else 'LH'} from beat {b - origin(m)}", 'unchanged',
                   [dict(label='keep', edit=None),
                    dict(label=f'{cnt}-note tuplet', edit={'tuplets': [{'bar': rv.num(m), 'beat': str(b - origin(m)),
                                                                        'span': int(span) if span == int(span) else str(span),
                                                                        'hand': 'RH' if st == 0 else 'LH'}]})])
    log(f"auto runs: {graces} brushed notes as grace notes, {cuts} overlaps cut")
    return notes


def displaced(notes, ctx, rv):
    """notes written away from where they were played while their neighbours are on time"""
    ns = sorted((n for n in notes if not n['grace']), key=lambda n: n['on'])
    p = {id(n): n['x'] - float(n['on']) for n in ns}
    spread = defaultdict(list)
    for n in ns:
        spread[(n['staff'], n['on'])].append(n['t'])
    rolled = {k for k, ts in spread.items() if len(ts) > 1 and max(ts) - min(ts) > 0.03}
    by_on = defaultdict(list)
    for i, n in enumerate(ns):
        if abs(p[id(n)]) < 0.2 or (n['staff'], n['on']) in rolled or id(n) in rv.in_runs:
            continue
        near = [p[id(m)] for m in ns[max(0, i - 40):i + 40]
                if m is not n and abs(float(m['on'] - n['on'])) <= 1 and abs(m['t'] - n['t']) > 0.03]
        if abs(median(near) if near else 0.0) < 0.08:
            by_on[n['on']].append(n)
    origin = lambda m: ctx.bars.start(m) + (ctx.pickup if m == 0 else 0)
    for on, g in sorted(by_on.items()):
        m = ctx.bar_of(on)
        off = median(p[id(n)] for n in g)
        b = ctx.bars.start(m) + int(on - ctx.bars.start(m))
        grid = next(F(1, d) for d in (4, 6, 8, 12) if ((on - b) * d).denominator == 1)   # the note's own grid
        to = b + F(round((float(on - b) + off) / float(grid))) * grid
        if not ctx.bars.start(0) <= to < ctx.END:
            continue
        m2 = ctx.bar_of(to)
        names = ' '.join(f"{acc(*Speller(ctx.keys[m]).spell(n)[:2])}{n['pitch'] // 12 - 1}{'' if n['staff'] == 0 else '(LH)'}" for n in g)
        rv.add('displaced', 'medium', rv.bars([m]),
               f"{names} at beat {on - origin(m)} was played {off:+.2f} beat from there while the notes around it are "
               f"on time: written in the wrong place (move to bar {rv.num(m2)} beat {to - origin(m2)}), or a deliberate "
               f"anticipation / ornament (keep)?",
               f"ids {' '.join(note_id(n) for n in g)}", 'unchanged',
               [dict(label='keep', edit=None),
                dict(label='move to where it was played', edit={'notes': [dict(id=note_id(n), bar=rv.num(m2), beat=str(to - origin(m2)))
                                                                          for n in g]})])

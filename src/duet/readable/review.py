"""Decisions taken from the notes, and the places left for a reviewer.

The score model's key signatures, spelling and fast runs are often wrong in ways
the notes themselves show. Where a rule was measured to help on the test set it
is applied here (config switches auto_keys, auto_spelling, auto_runs, auto_chords;
glissandi: engrave.gliss, auto_gliss):

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
  glissando     a glissando written as a glissando line (engrave.gliss)
  chord         notes played together but written apart, joined (undo)
  duple         a figure written as triplets / sextuplets that reads better as
                16ths or 32nds and was played about as close to that (per figure)
  parallel      a bar with the same notes as other bars but another rhythm
  inner-voice   a lone low RH note that completes an LH chord of two notes or more
  odd-beat      beats with 5 or 7 notes played about evenly: one 5:4 / 7:4 tuplet each
  scale-run     a fast run on a major scale: written out, or a glissando? (engrave.gliss)

Since 1.2.0 notes of one staff played within 15 ms but written apart become one
chord (auto_chords). Glissandi are found and written by engrave.gliss (from
wei-zeng98/DuET#9 by @hibikiwtnb).
"""
import json
from collections import defaultdict
from fractions import Fraction as F
from statistics import median

from . import keyfit, spellfit, runs as runs_mod
from .spelling import Speller, ACC_TXT

KEY_NAME = {-7: 'Cb', -6: 'Gb', -5: 'Db', -4: 'Ab', -3: 'Eb', -2: 'Bb', -1: 'F', 0: 'C',
            1: 'G', 2: 'D', 3: 'A', 4: 'E', 5: 'B', 6: 'F#', 7: 'C#'}


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
        with open(path, 'w', encoding='utf-8') as handle:
            json.dump(dict(title=title, about="places the pipeline was not sure about; each option is an edits entry "
                                          "(engrave/edits.py) to merge into <title>.edits.json",
                       counts=counts, items=items), handle, indent=1, ensure_ascii=False)


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
    prop, _ = keyfit.propose(bp, model, switch=ctx.cfg.key_switch,
                             per_fifth=ctx.cfg.key_per_fifth, model_cost=ctx.cfg.key_model_cost)
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
    _, kept = keyfit.guard(prop, bass, share=ctx.cfg.key_borrowed_share,
                            max_len=ctx.cfg.key_borrowed_max_bars)
    for lo, hi, k1, k0 in kept:
        alt = list(prop); alt[lo:hi + 1] = [k0] * (hi - lo + 1)
        rv.add('key-borrowed', 'high', rv.bars(range(lo, hi + 1)),
               f"These bars go to {KEY_NAME[k1]} and back to {KEY_NAME[k0]}, but {KEY_NAME[k0]}'s tonic is still often "
               f"in the bass: a real modulation, or borrowed chords around the same tonic (then keep {KEY_NAME[k0]} "
               f"and use accidentals)?",
               f"bass on the old tonic in at least {100 * ctx.cfg.key_borrowed_share:g}% of the bars",
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
    prop = spellfit.propose(notes, key_of, margins=margins,
                           window=ctx.cfg.spelling_window, w_key=ctx.cfg.spelling_w_key,
                           w_line=ctx.cfg.spelling_w_line, passes=ctx.cfg.spelling_passes,
                           w_sound=ctx.cfg.spelling_w_sound, w_odd=ctx.cfg.spelling_w_odd,
                           line_prev=ctx.cfg.spelling_line_prev)
    for n in notes:
        if id(n) in prop and not n.get('spell_fixed'):
            n['spell'], n['spell_auto'] = prop[id(n)], True
    log(f"auto spelling: {len(prop)} notes outside the key respelled from their harmony and line")
    ns = sorted((n for n in notes if not n['grace']), key=lambda n: (n['on'], n['pitch']))
    name = lambda m: acc(*Speller(key_of(m)).spell(m)[:2])
    groups = defaultdict(list)
    for i, n in enumerate(ns):
        gap, other = margins.get(id(n), (99.0, None))
        if gap >= ctx.cfg.spelling_review_margin or other is None:
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


# ---- notes played together but written apart ------------------------------------
def auto_chords(notes, ctx, rv, log):
    """notes of one staff played within chord_split_ms of each other (and at least a
    minor third apart: closer ones are brushed notes, see runs) but written at
    different positions less than half a beat apart are one chord: they move to the
    written position nearest to where they were played (the coarser grid on a tie)"""
    cfg = ctx.cfg
    origin = lambda m: ctx.bars.start(m) + (ctx.pickup if m == 0 else 0)
    moved = 0
    for st in (0, 1):
        ns = sorted((n for n in notes if n['staff'] == st and not n['grace'] and id(n) not in rv.in_runs
                     and not n.get('gliss')), key=lambda n: n['t'])
        groups, cur = [], []
        for n in ns:
            if cur and n['t'] - cur[0]['t'] > cfg.chord_split_ms / 1000:
                groups.append(cur); cur = []
            cur.append(n)
        groups.append(cur)
        for g in groups:
            ons = sorted({n['on'] for n in g})
            if len(ons) < 2 or ons[-1] - ons[0] >= F(1, 2):
                continue
            ps = sorted(n['pitch'] for n in g)
            if min(b - a for a, b in zip(ps, ps[1:])) < 3:
                continue
            # part of a rolled chord or a fast figure (more notes just before or after): left alone
            if any(n not in g and -0.03 <= n['t'] - g[0]['t'] <= g[-1]['t'] - g[0]['t'] + 0.03 for n in ns):
                continue
            x = median(n['x'] for n in g)
            to = min(ons, key=lambda o: (round(abs(float(o) - x), 2), (o - ctx.bars.start(ctx.bar_of(o))).denominator, o))
            mv = [n for n in g if n['on'] != to]
            taken = {n['pitch'] for n in notes if n['staff'] == st and n['on'] == to and not n['grace'] and n not in g}
            if any(n['pitch'] in taken for n in mv):
                continue
            m = ctx.bar_of(to)
            undo = [dict(id=note_id(n), bar=rv.num(ctx.bar_of(n['on'])), beat=str(n['on'] - origin(ctx.bar_of(n['on']))))
                    for n in mv]
            for n in mv:
                n['on'] = to
            moved += len(mv)
            rv.add('chord', 'low', rv.bars([m]),
                   f"Notes played together ({(g[-1]['t'] - g[0]['t']) * 1000:.0f} ms apart) but written at "
                   f"{', '.join(str(o - origin(ctx.bar_of(o))) for o in ons)} are written as one chord at beat "
                   f"{to - origin(m)}: right?",
                   f"{'RH' if st == 0 else 'LH'} ids {' '.join(note_id(n) for n in g)}", 'moved',
                   [dict(label='keep the chord', edit=None), dict(label='as the model wrote them', edit={'notes': undo})])
    if moved:
        log(f"auto chords: {moved} notes joined to the chord they were played with")
    return notes


# ---- rhythm questions (after quantization) ---------------------------------------
def _origin(ctx, m):
    return ctx.bars.start(m) + (ctx.pickup if m == 0 else 0)


def _pos(ctx, rv, on):
    m = ctx.bar_of(on)
    return rv.num(m), str(on - _origin(ctx, m))


def _run32(ons):
    """sextuplet runs (onsets 1/6 apart) as 32nds from the 32nd nearest to the run's
    start; None when another onset is off the 32nd grid or the run does not fit"""
    out, i = list(ons), 0
    while i < len(ons):
        j = i
        while j + 1 < len(ons) and ons[j + 1] - ons[j] == F(1, 6):
            j += 1
        if j > i:
            s0 = F(round(ons[i] * 8), 8)
            out[i:j + 1] = [s0 + F(k, 8) for k in range(j - i + 1)]
        i = j + 1
    if any((o * 8).denominator != 1 for o in out) or any(b <= a for a, b in zip(out, out[1:])) or out[-1] >= 1 \
            or out == list(ons):
        return None
    return out


def duple_fit(notes, ctx, rv):
    """beats written in triplet / sextuplet positions that read better as 16ths, or
    whose fast sextuplet runs (three notes in half a beat, a common fill) read better
    as 32nds, when the playing fits that writing about as well (16ths: up to
    duple_bias beat worse, readability first; 32nds: at least as well) and within
    duple_tol. One item per figure (same
    staff, same written and proposed positions), listing all its bars, so a figure
    is decided once and written the same way everywhere (the score model often
    writes fast 16th / 32nd figures as sextuplets)"""
    if ctx.compound:
        return
    cfg = ctx.cfg
    beats = defaultdict(list)
    for n in notes:
        if n['grace'] or n.get('tuplet_member'):
            continue
        m = ctx.bar_of(n['on'])
        if ctx.bars.compound(m):
            continue
        b = ctx.bars.start(m) + int(n['on'] - ctx.bars.start(m))
        beats[(n['staff'], b)].append(n)
    from itertools import combinations
    fams = defaultdict(list)
    for (st, b), ns in sorted(beats.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        ons = sorted({n['on'] - b for n in ns})
        if not any((o * 4).denominator != 1 and (o * 12).denominator == 1 for o in ons):
            continue
        xs = [min(n['x'] for n in ns if n['on'] - b == o) - float(b) for o in ons]

        def fit(pos):
            off = sum(x - float(p) for x, p in zip(xs, pos)) / len(pos)
            return max(abs(x - off - float(p)) for x, p in zip(xs, pos))
        e_w = fit(ons)
        cands = []
        if len(ons) <= 4:
            e16, c16 = min((fit(c), c) for c in combinations([F(i, 4) for i in range(4)], len(ons)))
            cands.append((e16, '16ths', list(c16)))
        c32 = _run32(ons)
        if c32:
            cands.append((fit(c32) + cfg.duple_bias, '32nds', c32))        # 16ths read easier
        # 16ths may fit up to duple_bias worse than the tuplet; 32nds must fit at least as well
        # (an even 16th triplet in half a beat is written with its own small bracket)
        ok = [c for c in cands if c[0] <= cfg.duple_tol + (cfg.duple_bias if c[1] == '32nds' else 0)
              and c[0] <= e_w + cfg.duple_bias]
        if not ok:
            continue
        e_d, grid, best = min(ok, key=lambda c: c[0])
        if grid == '32nds':
            e_d -= cfg.duple_bias
        fams[(st, tuple(ons), tuple(best), grid)].append((b, ns, e_w, e_d))
    for (st, ons, best, grid), beats_ in sorted(fams.items(), key=lambda kv: kv[1][0][0]):
        edits, ev, ms = [], [], []
        new = dict(zip(ons, best))
        for b, ns, e_w, e_d in beats_:
            m = ctx.bar_of(b)
            ms.append(m)
            for n in ns:
                o = n['on'] - b
                i = ons.index(o)
                end = o + n['dur'] - new[o]                  # the last note keeps its end, on the 32nd grid
                dur = (best[i + 1] - new[o]) if i + 1 < len(best) else max(F(1, 8), F(-(-end * 8 // 1), 8))
                edits.append(dict(id=note_id(n), bar=rv.num(m), beat=str(b + new[o] - _origin(ctx, m)), dur=str(dur)))
            ev.append(f"bar {rv.num(m)} beat {b - _origin(ctx, m)}: error {e_w:.2f} -> {e_d:.2f}")
        rv.add('duple', 'medium', rv.bars(ms),
               f"{'RH' if st == 0 else 'LH'}: a figure written at {' '.join(map(str, ons))} of the beat (triplet / "
               f"sextuplet positions) in {len(beats_)} beat(s) reads more easily as {grid} at "
               f"{' '.join(map(str, best))}, and was played about as close to that. Write it as {grid} everywhere it "
               f"occurs, or keep the tuplet (a real triplet feel, or a figure that should match triplets nearby)?",
               '; '.join(ev) + ' (beats, played vs written)', 'unchanged',
               [dict(label='keep', edit=None), dict(label=f'write as {grid}', edit={'notes': edits})])


def parallel_bars(notes, ctx, rv):
    """bars of one staff with the same notes as other bars but a different rhythm:
    the most common rhythm (the one played closest to its writing on a tie) is
    offered for the others"""
    bars_ = defaultdict(list)
    for n in notes:
        if not n['grace'] and not n.get('tuplet_member') and not n.get('gliss'):
            bars_[(n['staff'], ctx.bar_of(n['on']))].append(n)
    groups = defaultdict(list)
    for (st, m), ns in bars_.items():
        if len(ns) >= 4 and not ctx.bars.compound(m):
            groups[(st, ctx.bars.lens[m], tuple(sorted(n['pitch'] for n in ns)))].append(m)
    for (st, _, _), ms in groups.items():
        if len(ms) < 2:
            continue
        sig = {m: tuple(sorted((n['on'] - ctx.bars.start(m), n['pitch']) for n in bars_[(st, m)])) for m in ms}
        variants = defaultdict(list)
        for m in ms:
            variants[sig[m]].append(m)
        if not 2 <= len(variants) <= 3:
            continue
        def spread(m):              # played minus written, around the bar's own offset
            d = [n['x'] - float(n['on']) for n in bars_[(st, m)]]
            c = median(d)
            return sum(abs(v - c) for v in d) / len(d)
        err = lambda v: sum(spread(m) for m in variants[v]) / len(variants[v])
        ref = max(variants, key=lambda v: (len(variants[v]), -err(v)))
        rm = variants[ref][0]
        ref_notes = sorted(bars_[(st, rm)], key=lambda n: (n['pitch'], n['t']))
        for v, vms in variants.items():
            if v == ref:
                continue
            for m in vms:
                mine = sorted(bars_[(st, m)], key=lambda n: (n['pitch'], n['t']))
                edits = []
                for n, r in zip(mine, ref_notes):
                    pos = r['on'] - ctx.bars.start(rm)                  # from the bar line
                    if n['on'] - ctx.bars.start(m) != pos or n['dur'] != r['dur']:
                        edits.append(dict(id=note_id(n), beat=str(pos + ctx.bars.start(m) - _origin(ctx, m)),
                                          dur=str(r['dur'])))
                if not edits:
                    continue
                rv.add('parallel', 'medium', rv.bars([m]),
                       f"{'RH' if st == 0 else 'LH'} of bar {rv.num(m)} has the same notes as bar(s) "
                       f"{rv.bars(variants[ref])} but a different rhythm: write it like bar {rv.num(rm)}, or keep?",
                       f"{len(edits)} notes differ; mean played-vs-written deviation {err(v):.2f} beat here, {err(ref):.2f} there",
                       'unchanged', [dict(label='keep', edit=None),
                                     dict(label=f'rhythm of bar {rv.num(rm)}', edit={'notes': edits})])


def inner_voice(notes, ctx, rv):
    """a lone RH note below middle C struck together with an LH chord (two notes or
    more) it completes (above the LH's top, within an octave of its lowest note):
    probably part of the LH chord. Left out when the RH plays below middle C
    within a beat before or after it (other than such notes): then the RH itself
    has a figure in that register"""
    at = defaultdict(lambda: ([], []))
    for n in notes:
        if not n['grace'] and not n.get('tuplet_member'):
            at[n['on']][n['staff']].append(n)
    cand = []
    for on, (rh, lh) in at.items():
        if len(rh) == 1 and rh[0]['pitch'] < 60 and len(lh) >= 2 and rh[0]['pitch'] - min(n['pitch'] for n in lh) <= 12 \
                and rh[0]['pitch'] > max(n['pitch'] for n in lh):
            cand.append(rh[0])
    ids = {id(n) for n in cand}
    low_rh = [n for on, (rh, _) in at.items() for n in rh if n['pitch'] < 60 and id(n) not in ids]
    by_bar = defaultdict(list)
    for n in cand:
        if not any(0 < abs(m['on'] - n['on']) <= 1 for m in low_rh):
            by_bar[ctx.bar_of(n['on'])].append(n)
    for m, ns in sorted(by_bar.items()):
        names = ' '.join(f"{acc(*Speller(ctx.keys[m]).spell(n)[:2])}{n['pitch'] // 12 - 1}" for n in ns)
        rv.add('inner-voice', 'low', rv.bars([m]),
               f"RH {names} sound(s) alone below middle C, together with an LH chord they complete: move them "
               f"to the LH staff, or keep them in the RH?",
               f"ids {' '.join(note_id(n) for n in ns)}", 'unchanged',
               [dict(label='keep', edit=None),
                dict(label='move to the LH', edit={'notes': [dict(id=note_id(n), staff='LH') for n in ns]})])


def odd_beats(notes, ctx, rv):
    """a beat of one staff holding 5 or 7 attacks (no plain grid holds them, so the
    model mixes tuplets, rests and ties) played about evenly: offered as one 5:4 /
    7:4 tuplet per beat (edits "tuplets"; a one-way run in pitch order). Neighbouring
    beats of one staff are one item. Beats with a note of that staff held into them
    from before are left out (the tuplet would not fit)"""
    cfg = ctx.cfg
    beats = defaultdict(list)
    for n in notes:
        if n['grace'] or n.get('tuplet_member'):
            continue
        m = ctx.bar_of(n['on'])
        if ctx.bars.compound(m):
            continue
        beats[(n['staff'], ctx.bars.start(m) + int(n['on'] - ctx.bars.start(m)))].append(n)
    found = []
    for (st, b), ns in sorted(beats.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        ons = sorted({n['on'] for n in ns})
        k = len(ons)
        if k not in (5, 7):
            continue
        if any(n['staff'] == st and not n['grace'] and n['on'] < b < n['on'] + n['dur'] for n in notes):
            continue
        xs = [min(n['x'] for n in ns if n['on'] == o) - float(b) for o in ons]
        off = sum(x - i / k for i, x in enumerate(xs)) / k
        err = max(abs(x - off - i / k) for i, x in enumerate(xs))
        if err > cfg.odd_beat_tol:
            continue
        tops = [max(n['pitch'] for n in ns if n['on'] == o) for o in ons]
        order = 'up' if all(q > p for p, q in zip(tops, tops[1:])) else \
            'down' if all(q < p for p, q in zip(tops, tops[1:])) else None
        found.append((st, b, k, err, order))
    groups = []
    for f in found:
        if groups and groups[-1][-1][0] == f[0] and f[1] - groups[-1][-1][1] == 1:
            groups[-1].append(f)
        else:
            groups.append([f])
    for g in groups:
        st = g[0][0]
        ms = [ctx.bar_of(b) for _, b, _, _, _ in g]
        tup = []
        for _, b, k, err, order in g:
            m = ctx.bar_of(b)
            e = {'bar': rv.num(m), 'beat': str(b - _origin(ctx, m)), 'span': 1, 'hand': 'RH' if st == 0 else 'LH'}
            if order:
                e['order'] = order
            tup.append(e)
        ev = '; '.join(f"bar {rv.num(ctx.bar_of(b))} beat {b - _origin(ctx, ctx.bar_of(b))}: {k} notes, "
                       f"{'a run ' + order if order else 'not one way'}, largest deviation from even {err:.2f} beat"
                       for _, b, k, err, order in g)
        rv.add('odd-beat', 'medium', rv.bars(ms),
               f"{'RH' if st == 0 else 'LH'}: {len(g)} beat(s) with 5 or 7 notes, played about evenly, are written "
               f"with mixed tuplets, rests or ties. Write each beat as one 5:4 / 7:4 tuplet, or keep?",
               ev, 'unchanged', [dict(label='keep', edit=None), dict(label='one tuplet per beat', edit={'tuplets': tup})])


def rhythm_questions(notes, ctx, rv):
    odd_beats(notes, ctx, rv)
    duple_fit(notes, ctx, rv)
    parallel_bars(notes, ctx, rv)
    inner_voice(notes, ctx, rv)

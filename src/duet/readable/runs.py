"""Fast runs written untidily: proposals for the engrave-polish skill.

A run is a stretch of one staff where single notes (or pairs) follow each
other at most a third of a beat apart, at least eight in a row. Inside a run
three things make the page dirty, and each is decided from the performance:

  stuck pair   two notes a step apart on one onset, played within 30 ms: a
               finger brushing the next key. The softer (then shorter) one is
               written as a grace note before the other; its pitch is kept.
               Only when it was played first (or within 5 ms): a grace note
               means "before", and a note played after is part of the line.
  re-spacing   a beat whose onsets leave the run's subdivision (16ths among
               sextuplets) or that holds two notes played apart on one onset:
               its notes go onto the run's subdivision in playing order, each
               as near as possible to where it was played (performed position
               corrected for the beat's local drift).
  overlap      a note lasting past the next note of the run is cut there, so
               the run stays one voice (otherwise it splits into two voices
               with rests). A note lasting two or more grid places past it is a
               held voice over the run (a melody over an arpeggio) and is kept.

propose() returns changes; nothing is applied here (tools/polish.py runs writes
them as edits for review). free_tuplets() points at beats holding more played
notes than the run's subdivision has places: such a run was played freely and
reads best as one n:m tuplet (an edits "tuplets" entry).
"""
from fractions import Fraction as F
from statistics import median

GRIDS = (2, 3, 4, 6, 8, 12)


def find_runs(notes, max_gap=F(1, 3), min_events=8, max_size=2):
    """[(staff, [event])]; an event is the list of non-grace notes on one onset"""
    runs = []
    for st in (0, 1):
        groups = {}
        for n in notes:
            if n['staff'] == st and not n['grace']:
                groups.setdefault(n['on'], []).append(n)
        cur = []
        for on in sorted(groups):
            g = groups[on]
            if len(g) <= max_size and (not cur or on - cur[-1][0]['on'] <= max_gap):
                cur.append(g)
                continue
            if len(cur) >= min_events:
                runs.append((st, cur))
            cur = [g] if len(g) <= max_size else []
        if len(cur) >= min_events:
            runs.append((st, cur))
    return runs


def run_grid(run):
    """smallest subdivision of the beat holding 80% of the run's onsets"""
    offs = [ev[0]['on'] - int(ev[0]['on']) for ev in run]
    for k in GRIDS:
        if sum((o * k).denominator == 1 for o in offs) >= 0.8 * len(offs):
            return k
    return None


def stuck(ev):
    """the stray note of a stuck pair, or None"""
    if len(ev) != 2:
        return None
    a, b = ev
    if abs(a['pitch'] - b['pitch']) > 2 or abs(a['t'] - b['t']) >= 0.03:
        return None
    slip = min(ev, key=lambda n: (n.get('vel', 0), n.get('plen', 0)))
    keep = b if slip is a else a
    return slip if slip['t'] <= keep['t'] + 0.005 else None


def place(targets, k):
    """strictly increasing slots 0..k-1 nearest to the targets (in beats from the beat start)"""
    n = len(targets)
    INF = float('inf')
    cost = [[INF] * k for _ in range(n)]
    back = [[0] * k for _ in range(n)]
    for j in range(k):
        cost[0][j] = abs(j / k - targets[0])
    for i in range(1, n):
        best, arg = INF, 0
        for j in range(k):
            if j >= 1 and cost[i - 1][j - 1] < best:
                best, arg = cost[i - 1][j - 1], j - 1
            if best < INF:
                cost[i][j] = best + abs(j / k - targets[i])
                back[i][j] = arg
    j = min(range(k), key=lambda j: cost[n - 1][j])
    if cost[n - 1][j] == INF:
        return None
    out = [j]
    for i in range(n - 1, 0, -1):
        j = back[i][j]
        out.append(j)
    return [F(j, k) for j in reversed(out)]


def propose(run, beat_of=lambda on: F(int(on)), context=()):
    """(subdivision, changes {id(note): (note, dict(on=, dur=, grace=), reason)}) for one run;
    context: the staff's other notes (they anchor placement and are never changed)"""
    k = run_grid(run)
    changes = {}

    def change(n, reason, **kv):
        old = changes.get(id(n))
        d = dict(old[1]) if old else {}
        d.update(kv)
        changes[id(n)] = (n, d, old[2] + '; ' + reason if old else reason)

    on_grid = lambda o: k is not None and ((o - beat_of(o)) * k).denominator == 1
    steps = [(b[0]['t'] - a[0]['t']) / float(b[0]['on'] - a[0]['on'])
             for a, b in zip(run, run[1:]) if b[0]['on'] > a[0]['on']]
    slot_s = median(steps) / k if steps and k else None          # seconds per grid slot
    # stuck pairs: isolated ones are slips; several within a beat are a texture (e.g. a run in seconds)
    cand = {i: stuck(ev) for i, ev in enumerate(run) if stuck(ev) is not None}
    texture = {i for i in cand for j in cand if i != j and abs(run[i][0]['on'] - run[j][0]['on']) <= 1}
    events, loose = [], []          # events: [onset, [notes], t]
    for i, ev in enumerate(run):
        o = ev[0]['on']
        if i in cand and i not in texture:
            slip = cand[i]
            keep = [n for n in ev if n is not slip][0]
            change(slip, f"stuck to {keep['pitch']} (played {1000 * abs(slip['t'] - keep['t']):.0f} ms apart, "
                         f"velocity {slip.get('vel')} vs {keep.get('vel')})", grace=True)
            ev = [keep]
        if len(ev) == 2 and slot_s and abs(ev[0]['t'] - ev[1]['t']) >= 0.5 * slot_s:
            a, b = sorted(ev, key=lambda n: n['t'])
            (events if on_grid(o) else loose).append([o, [a], a['t']])
            loose.append([o, [b], b['t']])
        elif on_grid(o):
            events.append([o, list(ev), min(n['t'] for n in ev)])
        else:
            loose.append([o, list(ev), min(n['t'] for n in ev)])
    if k is None:
        loose = []
    mine = {id(n) for ev in run for n in ev}
    groups = {}
    for n in context:
        if id(n) not in mine and not n['grace']:
            groups.setdefault(n['on'], []).append(n)
    events += [[o, ns, min(n['t'] for n in ns)] for o, ns in groups.items()]
    for ev in sorted(loose, key=lambda e: e[2]):
        o, ns, t = ev
        near = [e for e in events if abs(e[2] - t) < 0.03]
        if near:                          # played with another note: one chord
            new = near[0][0]
            near[0][1].extend(ns)
            reason = 'played with the note there'
        else:
            prev = max((e[0] for e in events if e[2] < t), default=None)
            nxt = min((e[0] for e in events if e[2] > t), default=None)
            ref = [n['x'] - float(n['on']) for e in events for n in e[1] if abs(e[0] - o) <= 1]
            target = sum(n['x'] for n in ns) / len(ns) - (median(ref) if ref else 0.0)
            new = None
            for kk in (k, 2 * k):
                if kk > 12:
                    break
                lo = prev if prev is not None else beat_of(o) - 1
                hi = nxt if nxt is not None else beat_of(o) + 2
                slots = [lo + F(j, kk) for j in range(1, int((hi - lo) * kk) + 1)
                         if lo + F(j, kk) < hi and ((lo + F(j, kk) - beat_of(lo + F(j, kk))) * kk).denominator == 1]
                slots = [v for v in slots if v not in {e[0] for e in events}]
                if slots:
                    new = min(slots, key=lambda v: abs(float(v) - target))
                    reason = f"placed in playing order on the {kk}-per-beat grid"
                    break
            if new is None:
                continue
            events.append([new, list(ns), t])
        for n in ns:
            if new != n['on']:
                change(n, reason, on=new)
    # durations: the run is one voice
    seq = sorted({e[0] for e in events})
    nxt = {a: b for a, b in zip(seq, seq[1:])}
    for o, ns, _ in events:
        if o not in nxt:
            continue
        gap = nxt[o] - o
        for n in ns:
            if id(n) not in mine:
                continue
            moved = id(n) in changes and 'on' in changes[id(n)][1]
            if not moved and k and n['dur'] - gap >= F(2, k):
                continue                  # a held voice, not an overlap
            if n['dur'] > gap or (moved and n['dur'] != gap):
                change(n, 'cut at the next note of the run' if n['dur'] > gap else 'fills to the next note', dur=gap)
    return k, changes


def free_tuplets(run, k, beat_of=lambda on: F(int(on)), chord_s=0.010):
    """[(beat start, span in beats, played events)] where one beat, or two, hold
    more played events than the grid has places. Notes played within chord_s are
    one chord; so are two notes written on one onset when they are a third or
    more apart and played within 3/4 of a grid place (a slightly broken interval)
    or are a stuck pair. Notes written apart are always notes of the line."""
    if not k:
        return []
    steps = [(b[0]['t'] - a[0]['t']) / float(b[0]['on'] - a[0]['on'])
             for a, b in zip(run, run[1:]) if b[0]['on'] > a[0]['on']]
    slot = median(steps) / k if steps else 0.0
    notes = sorted((n for ev in run for n in ev), key=lambda n: n['t'])
    beats = {}
    for n in notes:
        beats.setdefault(beat_of(n['on']), []).append(n)

    def one_chord(a, n):
        dt = n['t'] - a['t']
        if dt <= chord_s:
            return True
        if a['on'] != n['on']:                    # written apart: notes of the line
            return False
        return (dt <= 0.75 * slot and abs(n['pitch'] - a['pitch']) >= 3) or stuck([a, n]) is not None

    def count(ns):
        c, last = 0, None
        for n in sorted(ns, key=lambda n: n['t']):
            if last is None or not one_chord(last, n):
                c += 1
            last = n
        return c
    out = []
    for b in sorted(beats):
        n1 = count(beats[b])
        if n1 > k:
            out.append((b, F(1), n1))
            for b2 in (b - 1, b):
                if b2 in beats and b2 + 1 in beats:
                    n2 = count(beats[b2] + beats[b2 + 1])
                    if n2 > 2 * k and (b2, F(2), n2) not in out:
                        out.append((b2, F(2), n2))
    return out

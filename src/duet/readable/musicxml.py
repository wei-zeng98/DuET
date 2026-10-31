"""MusicXML 4.0 writer (one piano part, two staves, up to two voices per staff)."""

from fractions import Fraction as F
from xml.sax.saxutils import escape
import unicodedata
import numpy as np
from .notation import layout_voice, beam, type_of
from .spelling import Speller

DIV = 24
ACC_NAME = {-2: "flat-flat", -1: "flat", 0: "natural", 1: "sharp", 2: "double-sharp"}


def bar_items(evs, lo, hi):
    items = []
    for ev in evs:
        if ev["end"] <= lo or ev["on"] >= hi:
            continue
        s, e = (max(ev["on"], lo), min(ev["end"], hi))
        items.append(
            (s - lo, e - lo, dict(ev=ev, tie_in=ev["on"] < lo, tie_out=ev["end"] > hi))
        )
    return items


def piece_ties(pc):
    """(tie_stop, is_attack) of a laid-out note piece"""
    pl = pc["pl"]
    tie_stop = pc["cont_prev"] or pl["tie_in"]
    return (tie_stop, not tie_stop)


def grace_owners(laid, grace_by, st):
    """onset -> (voice index, piece index) of the one attack that carries the grace
    notes before it: when both voices attack together the graces are written once"""
    own = {}
    for vi, pieces in enumerate(laid):
        for k, pc in enumerate(pieces):
            if pc["pl"] is not None and piece_ties(pc)[1]:
                on = pc["pl"]["ev"]["on"]
                if (st, on) in grace_by and on not in own:
                    own[on] = (vi, k)
    return own


def accidentals(laid, graces_in, grace_by, st, sp):
    """which notes print an accidental: the staff's notes in time order across both
    voices (graces before their main note); an accidental holds for that step and
    octave until the bar line; a note tied over from before shows none and sets none"""
    ev = []
    for vi, pieces in enumerate(laid):
        for k, pc in enumerate(pieces):
            if pc["pl"] is None:
                continue
            tie_stop, attack = piece_ties(pc)
            e = pc["pl"]["ev"]
            if attack and graces_in.get(e["on"]) == (vi, k):
                for j, gn in enumerate(grace_by[st, e["on"]]):
                    ev.append((pc["s"], 0, ("g", e["on"], j), gn["pitch"], False))
            for j, n in enumerate(e["notes"]):
                tied = tie_stop
                ev.append((pc["s"], 1, ("n", vi, k, j), n["pitch"], tied))
    state, show = ({}, {})
    for s, _, key, p, tied in sorted(ev, key=lambda x: (x[0], x[1])):
        if tied:
            continue
        step, alt, octv = sp.spell(p)
        if alt != state.get((step, octv), sp.key_alt[step]):
            show[key] = True
            state[step, octv] = alt
    return show


def clef_xml(c, staff=2):
    return f'''<attributes><clef number="{staff}"><sign>{c}</sign><line>{(4 if c == "F" else 2)}</line></clef></attributes>'''


def write(plan, ctx, path):
    NB, END, bars = (ctx.nbars, ctx.END, ctx.bars)
    DIV = getattr(plan, "divisions", 24)
    out = []
    w = out.append
    w('<?xml version="1.0" encoding="UTF-8" standalone="no"?>')
    w(
        '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">'
    )
    w('<score-partwise version="4.0">')
    lines, line, width = [], "", 0
    for char in ctx.title:
        size = 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1
        if width + size > 52:
            lines.append(line)
            line, width = "", 0
        line += char
        width += size
    lines.append(line)
    display_title = "\n".join(lines)
    w(f"<work><work-title>{escape(display_title)}</work-title></work>")
    w(
        '<identification><creator type="arranger">score model + Transkun, fused &amp; engraved</creator><encoding><software>engrave</software></encoding></identification>'
    )
    w(
        '<part-list><score-part id="P1"><part-name>Piano</part-name><score-instrument id="P1-I1"><instrument-name>Piano</instrument-name></score-instrument><midi-instrument id="P1-I1"><midi-channel>1</midi-channel><midi-program>1</midi-program></midi-instrument></score-part></part-list>'
    )
    w('<part id="P1">')

    def dir_xml(td, staff, rel_from):
        rel, order, dtype, attrs, sound = td
        off = int((rel - rel_from) * DIV)
        return (
            f"<direction{attrs}><direction-type>{dtype}</direction-type>"
            + (f"<offset>{off}</offset>" if off else "")
            + f"<staff>{staff}</staff>{sound}</direction>"
        )

    def pitch_el(sp, p, show):
        step, alt, octv = sp.spell(p)
        acc = f"<accidental>{ACC_NAME[alt]}</accidental>" if show else ""
        return (
            f"<pitch><step>{step}</step>{(f'<alter>{alt}</alter>' if alt else '')}<octave>{octv}</octave></pitch>",
            acc,
        )

    prev_key = None
    for m in range(NB):
        pick = ctx.pickup if m == 0 and ctx.pickup else None
        B, lo, hi = (bars.length(m), bars.start(m), bars.end(m))
        cmp = bars.compound(m)
        mlen = B - pick if pick else B
        num = m + (0 if ctx.pickup else 1)
        nominal = F(bars.sig(m)[0] * 4, bars.sig(m)[1])
        w(
            f'<measure number="{num}"'
            + (' implicit="yes"' if pick or B != nominal else "")
            + ">"
        )
        sp = Speller(ctx.keys[m])
        attrs = []
        if m == 0:
            attrs.append(f"<divisions>{DIV}</divisions>")
        if ctx.keys[m] != prev_key:
            attrs.append(f"<key><fifths>{ctx.keys[m]}</fifths></key>")
            prev_key = ctx.keys[m]
        sig = bars.sig(m)
        if m == 0 or sig != bars.sig(m - 1):
            attrs.append(
                f"<time><beats>{sig[0]}</beats><beat-type>{sig[1]}</beat-type></time>"
            )
        if m == 0:
            attrs.append("<staves>2</staves>")
            for st in (0, 1):
                c0 = plan.clefs[st][0]
                attrs.append(
                    f'''<clef number="{st + 1}"><sign>{c0}</sign><line>{(4 if c0 == "F" else 2)}</line></clef>'''
                )
        if attrs:
            w("<attributes>" + "".join(attrs) + "</attributes>")
        written = 0
        for st in (0, 1):
            A = bar_items(plan.EV[st, "A"], lo, hi)
            Bv = bar_items(plan.EV[st, "B"], lo, hi)
            streams = [("A", A)]
            if Bv:
                ma = (
                    np.mean([n["pitch"] for _, _, pl in A for n in pl["ev"]["notes"]])
                    if A
                    else 0
                )
                mb = np.mean([n["pitch"] for _, _, pl in Bv for n in pl["ev"]["notes"]])
                streams = [("B", Bv), ("A", A)] if mb > ma else [("A", A), ("B", Bv)]
            two = len(streams) >= 2
            laid = []
            for L, items in streams:
                grace_cuts = ()
                pieces = layout_voice(
                    items, B, cut=pick, compound=cmp, extra_cuts=grace_cuts
                )
                if pick:
                    pieces = [p for p in pieces if p["s"] >= pick]
                beam(pieces, B, cmp)
                laid.append(pieces)
            graces_in = grace_owners(laid, plan.grace_by, st)
            show_acc = accidentals(laid, graces_in, plan.grace_by, st, sp)
            for vi, ((L, items), pieces) in enumerate(zip(streams, laid)):
                voice = (1 if st == 0 else 5) + vi
                if written:
                    w(f"<backup><duration>{int(mlen * DIV)}</duration></backup>")
                written += 1
                timed = (
                    sorted(
                        plan.dirs.td.get((m, st + 1), []), key=lambda t: (t[0], t[1])
                    )
                    if vi == 0
                    else []
                )
                clefs_here = (
                    [(t - lo, c) for t, c in plan.clef_tls[st] if lo <= t < hi]
                    if vi == 0
                    else []
                )
                hide = ' print-object="no"' if two and L == "B" else ""
                if (
                    len(pieces) == 1 and pieces[0]["pl"] is None or not pieces
                ) and mlen == nominal:
                    for td in [t for t in timed if t[0] < B]:
                        w(dir_xml(td, st + 1, pick or F(0)))
                    for rel, c in clefs_here:
                        w(clef_xml(c, st + 1))
                    w(
                        f'<note{hide}><rest measure="yes"/><duration>{int(mlen * DIV)}</duration><voice>{voice}</voice><staff>{st + 1}</staff></note>'
                    )
                    for td in [t for t in timed if t[0] >= B]:
                        w(dir_xml(td, st + 1, B))
                    continue
                for k, pc in enumerate(pieces):
                    here = [t for t in timed if pc["s"] <= t[0] < pc["s"] + pc["d"]]
                    for td in here:
                        w(dir_xml(td, st + 1, pc["s"]))
                    timed = [t for t in timed if t not in here]
                    for rel, c in [x for x in clefs_here if x[0] < pc["s"] + pc["d"]]:
                        w(clef_xml(c, st + 1))
                    clefs_here = [x for x in clefs_here if x[0] >= pc["s"] + pc["d"]]
                    tname, dots = type_of(pc["d"], pc["reg"] is not None)
                    dur = int(pc["d"] * DIV)
                    tup_xml, tm_xml = ("", "")
                    if pc["reg"]:
                        sext = pc["reg"][2] == F(1, 3) and any(
                            (
                                p2["d"] == F(1, 6)
                                for p2 in pieces
                                if p2["reg"] == pc["reg"]
                            )
                        )
                        an, nn = (6, 4) if sext else (3, 2)
                        tm_xml = f"<time-modification><actual-notes>{an}</actual-notes><normal-notes>{nn}</normal-notes></time-modification>"
                        r = pc["reg"]
                        first = k == 0 or pieces[k - 1]["reg"] != r
                        lastp = k == len(pieces) - 1 or pieces[k + 1]["reg"] != r
                        if first and (not lastp):
                            tup_xml = '<tuplet type="start" bracket="yes"/>'
                        elif lastp and (not first):
                            tup_xml = '<tuplet type="stop"/>'
                    stem = (
                        ("<stem>up</stem>" if vi == 0 else "<stem>down</stem>")
                        if two
                        else ""
                    )
                    beams = "".join(
                        (
                            f'<beam number="{lv}">{tp}</beam>'
                            for lv, tp in sorted(pc["beam"].items())
                        )
                    )
                    if pc["pl"] is None:
                        nots = f"<notations>{tup_xml}</notations>" if tup_xml else ""
                        w(
                            f"<note{hide}><rest/><duration>{dur}</duration><voice>{voice}</voice><type>{tname}</type>{'<dot/>' * dots}{tm_xml}<staff>{st + 1}</staff>{nots}</note>"
                        )
                        continue
                    pl = pc["pl"]
                    ev = pl["ev"]
                    first_piece = not pc["cont_prev"]
                    last_piece = not pc["cont_next"]
                    tie_stop = pc["cont_prev"] or (pl["tie_in"] and first_piece)
                    tie_start = pc["cont_next"] or (pl["tie_out"] and last_piece)
                    is_attack = first_piece and (not pl["tie_in"])
                    if is_attack and graces_in.get(ev["on"]) == (vi, k):
                        gr = plan.grace_by.get((st, ev["on"]), [])
                        for j, gn in enumerate(gr):
                            pel, acc = pitch_el(
                                sp, gn["pitch"], show_acc.get(("g", ev["on"], j))
                            )
                            chord = (
                                "<chord/>"
                                if j > 0
                                and gn["bass_grace"]
                                and gr[j - 1]["bass_grace"]
                                else ""
                            )
                            w(
                                f"""<note><grace slash="yes"/>{chord}{pel}<voice>{voice}</voice><type>{("eighth" if gn["bass_grace"] else "16th")}</type>{acc}{stem}<staff>{st + 1}</staff></note>"""
                            )
                    fermata = (
                        m == NB - 1 and last_piece and (ev["end"] >= END) and (st == 0)
                    )
                    for j, n in enumerate(ev["notes"]):
                        pel, acc = pitch_el(
                            sp, n["pitch"], show_acc.get(("n", vi, k, j))
                        )
                        note_stop = tie_stop
                        note_start = tie_start
                        ties = ('<tie type="stop"/>' if note_stop else "") + (
                            '<tie type="start"/>' if note_start else ""
                        )
                        nt = ('<tied type="stop"/>' if note_stop else "") + (
                            '<tied type="start"/>' if note_start else ""
                        )
                        if j == 0:
                            nt += tup_xml
                        if ev["roll"] and is_attack:
                            nt += "<arpeggiate/>"
                        if fermata and j == len(ev["notes"]) - 1:
                            nt += '<fermata type="upright"/>'
                        nots = f"<notations>{nt}</notations>" if nt else ""
                        chord = "<chord/>" if j > 0 else ""
                        bm = beams if j == 0 else ""
                        identity = ""
                        w(
                            f"<note{identity}>{chord}{pel}<duration>{dur}</duration>{ties}<voice>{voice}</voice><type>{tname}</type>{'<dot/>' * dots}{acc}{tm_xml}{stem}<staff>{st + 1}</staff>{bm}{nots}</note>"
                        )
                for td in timed:
                    w(dir_xml(td, st + 1, B if td[0] >= B else td[0]))
        if m == NB - 1:
            w('<barline location="right"><bar-style>light-heavy</bar-style></barline>')
        w("</measure>")
    w("</part></score-partwise>")
    with open(path, "w") as f:
        f.write("\n".join(out))

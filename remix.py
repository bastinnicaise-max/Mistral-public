#!/usr/bin/env python3
"""Super remix de sample1 (8 pages, 100 BPM) : réarrangement en 12 pages à 130 BPM.

Structure :
  P0   intro            basse triangle rebond + grosse caisse + charley clairsemé
  P1   thème (entrée)   mélodie p25 + groove d'origine
  P2-P3 thème complet    mélodie p25/p12 alternées + groove enrichi
  P4   montée           transition + roulement de charley + tom fill
  P5-P6 drop             mélodie p50, basse rebond +12, grosse caisse 4 temps, charley 1/8
  P7   break             mélodie en triangle, épuré, roulement final
  P8-P9 final            thème transposé +2 demi-tons, double couche p25/p12, gros groove
  P10  outro             matériel de fin d'origine + accord final
"""
import sb3_music_codec as C

BPM = 130
KICK, HAT = 21, 15          # timbres de bruit du morceau d'origine

lines = open('original.txt').read().rstrip('\n').split('\n')
_, _, _, src_pages = C.verify(lines)
mel, bass, noi = {}, {}, {}                       # par page source : colonnes absolues
for pi, pg in enumerate(src_pages):
    g = C.page_to_grid(pg)
    for i, code in enumerate(g):
        if not code:
            continue
        r, c = divmod(i, C.COLS)
        m = C.CODE2MASK[code]
        if m[3] == '1':
            bass[(pi, c)] = C.TRI_LO + 23 - r
        for b in range(3):
            if m[b] == '1':
                mel[(pi, c)] = C.PULSE_LO + 23 - r
        if m[4] == '1':
            noi[(pi, c)] = 24 - r

cells = {}                                        # (page, idx) -> code
PRI = {'p50': 3, 'p25': 2, 'p12': 1}

def place(col, row, mask):
    pg, c = divmod(col, C.COLS)
    key = (pg, row * C.COLS + c)
    cur = list(C.CODE2MASK[cells.get(key, 0)])
    for b, ch in enumerate(mask):
        if ch == '1':
            if b < 3 and cur[b] == '1':
                continue                          # une seule pulse par cellule
            cur[b] = '1'
    cells[key] = C.MASK2CODE[''.join(cur)]

def place_pulse(col, pitch, tb):
    if tb == 'tri':
        place(col, C.TRI_LO + 23 - C.fold(pitch, C.TRI_LO), '00010')
        return
    prio, old = PRI[tb], None
    pg, c = divmod(col, C.COLS)
    key = (pg, (C.PULSE_LO + 23 - C.fold(pitch, C.PULSE_LO)) * C.COLS + c)
    cur = list(C.CODE2MASK[cells.get(key, 0)])
    for b in range(3):
        if cur[b] == '1':
            old = 'p%d' % [12, 25, 50][b]
    if old and PRI[old] >= prio:
        return
    for b in range(3):
        cur[b] = '1' if 'p%d' % [12, 25, 50][b] == tb else '0'
    cells[key] = C.MASK2CODE[''.join(cur)]

def melody_page(src, dst, tb=None, alt=None, shift=0):
    """Rejoue la mélodie d'une page source ; alt=(tb_a, tb_b) alterne par groupe de 8."""
    for (p, c), pitch in mel.items():
        if p != src:
            continue
        if tb:
            t = tb
        elif alt:
            t = alt[0] if (c // 8) % 2 == 0 else alt[1]
        else:
            t = 'p25'
        place_pulse(dst * C.COLS + c, pitch + shift, t)

def bass_page(src, dst, shift=0):
    for (p, c), pitch in bass.items():
        if p != src:
            continue
        place(dst * C.COLS + c, C.TRI_LO + 23 - C.fold(pitch + shift, C.TRI_LO), '00010')

def bounce_bass(src, dst, shift=0):
    """Basse triangle rebond : noires alternées grave/aiguë à partir de la tenue d'origine."""
    seen = set()
    for (p, c), pitch in sorted(bass.items()):
        if p != src or c in seen:
            continue
        p2 = C.fold(pitch + shift, C.TRI_LO)
        p3 = C.fold(p2 + 12, C.TRI_LO)
        for cc in range(c, c + 8):
            if (p, cc) in bass:
                seen.add(cc)
        for i in range(8):
            pl = p2 if i % 2 == 0 else p3
            place(dst * C.COLS + c + i, C.TRI_LO + 23 - pl, '00010')

def groove(src, dst, extra_hats=False, four=False):
    for (p, c), r in noi.items():
        if p != src:
            continue
        place(dst * C.COLS + c, r, '00001')
    if four:
        for c in range(0, 32, 4):
            place(dst * C.COLS + c, 24 - KICK, '00001')
    if extra_hats:
        for c in range(0, 32, 2):
            place(dst * C.COLS + c, 24 - HAT, '00001')

def roll(dst, start=16, end=32, tb=HAT):
    for i, c in enumerate(range(start, end)):
        step = 4 if c < start + 4 else (2 if c < start + 8 else 1)
        if (c - start) % step == 0:
            place(dst * C.COLS + c, 24 - tb, '00001')

def tom_fill(dst):
    for i, k in enumerate((15, 17, 19, 21)):
        place(dst * C.COLS + 28 + i, 24 - k, '00001')

# ---------------------------------------------------------------- arrangement
bounce_bass(0, 0)                                             # intro épurée
for c in range(0, 32, 8):
    place(c, 24 - KICK, '00001')
for c in range(4, 32, 8):
    place(c, 24 - HAT, '00001')
melody_page(0, 1, tb='p25'); bounce_bass(0, 1); groove(0, 1)
melody_page(0, 2, alt=('p25', 'p12')); bass_page(0, 2); groove(0, 2)
melody_page(1, 3, alt=('p25', 'p12')); bass_page(1, 3); groove(1, 3, extra_hats=True)
melody_page(3, 4, tb='p12'); bass_page(3, 4); groove(3, 4)
roll(4); tom_fill(4)
melody_page(4, 5, tb='p50'); bounce_bass(4, 5); groove(4, 5, four=True)
melody_page(5, 6, tb='p50'); bounce_bass(5, 6); groove(5, 6, four=True, extra_hats=True)
melody_page(6, 7, tb='tri'); bass_page(6, 7); groove(6, 7)
roll(7, start=24, end=32, tb=HAT)
melody_page(0, 8, tb='p25', shift=2); bass_page(0, 8, shift=2); groove(0, 8, four=True)
melody_page(1, 9, alt=('p25', 'p12'), shift=2); bass_page(1, 9, shift=2); groove(1, 9, four=True, extra_hats=True)
melody_page(7, 10); bass_page(7, 10); groove(7, 10, four=True)
tom_fill(10)
melody_page(2, 11, tb='p25', shift=2); bass_page(2, 11, shift=2); groove(2, 11, extra_hats=True)
place(11 * C.COLS + 31, C.TRI_LO + 23 - 48, '00010')          # accord final : C3
place(11 * C.COLS + 31, C.TRI_LO + 23 - 60, '00010')          # + C4
place_pulse(11 * C.COLS + 31, 72, 'p50')                      # + C5
place(11 * C.COLS + 31, 24 - KICK, '00001')

n_pages = max(k[0] for k in cells) + 1
grids = []
for p in range(n_pages):
    g = [0] * C.PAGE_CELLS
    for (pg, i), code in cells.items():
        if pg == p:
            g[i] = code
    grids.append(C.grid_to_page(g))
out = C.build_code(BPM, 1, grids)
ok, v_bpm, loop, pages = C.verify(out)
print('remix : vérif', ok, '| bpm', v_bpm, '| pages', len(pages))
open('super_remix.txt', 'w').write('\n'.join(out) + '\n')
open('super_remix_compact.txt', 'w').write(C.compact_encode(v_bpm, loop, pages) + '\n')
print('compact :', open('super_remix_compact.txt').read().strip())
data, dur, warn = C.render_code(out, 'test_sounds', 1, 'wav')
open('super_remix_preview.wav', 'wb').write(data)
print('aperçu rendu : %.1f s (%d pages)' % (dur, len(pages)))

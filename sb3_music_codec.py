#!/usr/bin/env python3
"""
Convertisseur pour "8-Bit Music Maker" (projet Scratch .sb3)

    python3 sb3_music_codec.py midi2code  song.mid  -o song.txt  [options]
    python3 sb3_music_codec.py code2midi  song.txt  -o song.mid
    python3 sb3_music_codec.py check      song.txt

FORMAT DU CODE (un élément de liste Scratch par ligne) :
    ligne 1 : BPM du projet (entier 100..400)
    ligne 2 : loop (0 ou 1)
    ligne 3 : checksum (voir checksum())
    ligne 4+: une ligne par page = 768 cellules "NN" séparées par des espaces (2303 car.)

UNE PAGE = grille 24 lignes x 32 colonnes, stockée ligne par ligne :
    index = ligne*32 + colonne   (ligne 0 = en haut = note la plus aiguë)
    1 colonne = 1/4 de temps (double-croche), donc 1 page = 8 temps = 2 mesures 4/4

VALEUR D'UNE CELLULE (décimal sur 2 chiffres) = index dans la liste "noteinfo" :
    0 rien | 1=12.5% 2=25% 3=50% 4=triangle 5=bruit
    6=12.5%+tri 7=25%+tri 8=50%+tri 9=12.5%+bruit 10=25%+bruit 11=50%+bruit
    12=12.5%+tri+bruit 13=25%+tri+bruit 14=50%+tri+bruit 15=tri+bruit

HAUTEUR (ligne i, 0..23) :
    pulse 12.5/25/50 : MIDI 83-i  (C4=60 .. B5=83)
    triangle         : MIDI 71-i  (une octave plus bas : 48..71)
    bruit            : 16 timbres seulement, lignes 8..23 (23 = grave, 8 = aigu)
"""
import argparse
import sys

COLS, ROWS = 32, 24
PAGE_CELLS = COLS * ROWS            # 768
PAGE_LEN = PAGE_CELLS * 3 - 1       # 2303
MAX_PAGES = 99

# bits : [12.5%, 25%, 50%, tri, noise]  -> code (= index noteinfo - 1)
NOTEINFO = ['00000', '10000', '01000', '00100', '00010', '00001', '10010', '01010',
            '00110', '10001', '01001', '00101', '10011', '01011', '00111', '00011']
MASK2CODE = {m: i for i, m in enumerate(NOTEINFO)}
CODE2MASK = {i: m for i, m in enumerate(NOTEINFO)}

PULSE_NAMES = ['p12', 'p25', 'p50']
PULSE_LO, TRI_LO = 60, 48


def checksum(text):
    """Copie exacte du calcul Scratch (blocs 'get code' / 'verify')."""
    chk = 0
    for i, ch in enumerate(text, 1):
        w = (i % 11) + 1
        chk += 43 * w if ch == ' ' else (int(ch) + 1) * w
        chk %= 1000000
    return chk


def song_checksum(bpm, loop, pages):
    # @Code = items joints par des espaces (le 1er item "150" fait plus d'1 car.)
    return checksum(' '.join([str(bpm), str(loop)] + pages))


def build_code(bpm, loop, pages):
    return [str(bpm), str(loop), str(song_checksum(bpm, loop, pages))] + pages


def grid_to_page(grid):
    return ' '.join('%02d' % grid[i] for i in range(PAGE_CELLS))


def page_to_grid(page):
    return [int(x) for x in page.split(' ')]



# ----------------------------------------------------------------- format COMPACT
# 1 seule ligne, alphabet de 64 caractères (chiffres, lettres minuscules, 28 symboles).
# IMPORTANT : jamais de , ; | tabulation " ni espace : le sélecteur de fichier de Scratch lit les listes comme du CSV
# et devine le séparateur parmi , ; | tab -> il demanderait « quelle colonne utiliser ».
# Scratch compare les textes sans tenir compte de la casse : les majuscules sont donc acceptées à la lecture.
#   '~' | 2 car. (bpm-100)*2+loop | 2 car. nombre de pages | charge utile LZ | 2 car. checksum
# Charge utile : flux de jetons 0..63 (voir stage1/lz), compressé par LZ77 glouton.
#   jeton 0-23 = ligne r (pose la valeur en attente) | 24-38 = valeur 1..15 | 39-54 = saut de 1..16 colonnes
#   55-56 = copie proche  (longueur 3..4, décalage 1..63 sur 1 car.)
#   57-62 = copie courte  (longueur 4..9, décalage sur 2 car.)
#   63    = copie longue  (longueur-10 sur 2 car., décalage sur 2 car.)
COMPACT_ALPHA = "0123456789abcdefghijklmnopqrstuvwxyz" + "!#$%&()*+-./:<=>?@[]^_{}~`'\\"
COMPACT_MARKER = '~'


def _cp_stage1(pages):
    toks, cur = [], 0
    for p, pg in enumerate(pages):
        g = page_to_grid(pg)
        for c in range(COLS):
            col = [(r, g[r * COLS + c]) for r in range(ROWS) if g[r * COLS + c]]
            if not col:
                continue
            gc = p * COLS + c
            gap = gc - cur
            while gap > 16:
                toks.append(54)
                gap -= 16
            if gap > 0:
                toks.append(38 + gap)
            cur = gc
            for r, v in col:
                toks += [23 + v, r]
    return toks


def _cp_lz(T):
    n = len(T)
    out, head, prev, i = [], {}, [0] * (n + 2), 1
    h = lambda i: T[i - 1] * 64 + T[i] + 1
    while i <= n:
        bl = bo = 0
        if i < n:
            j = head.get(h(i), 0)
            while j != 0 and i - j <= 4095:
                L = 0
                while i + L <= n and T[j + L - 1] == T[i + L - 1] and L < 4105:
                    L += 1
                if L > bl:
                    bl, bo = L, i - j
                j = prev[j]
        adv = 1
        if bl > 2 and bo < 64 and bl < 5:
            out += [52 + bl, bo]; adv = bl
        elif 3 < bl < 10:
            out += [53 + bl, bo // 64, bo % 64]; adv = bl
        elif bl > 9:
            out += [63, (bl - 10) // 64, (bl - 10) % 64, bo // 64, bo % 64]; adv = bl
        else:
            out.append(T[i - 1])
        for _ in range(adv):
            if i < n:
                hh = h(i)
                prev[i] = head.get(hh, 0)
                head[hh] = i
            i += 1
    return out


def _cp_chk(toks):
    chk = 0
    for t in toks:
        chk = (chk * 37 + t + 1) % 4093
    return chk


def compact_encode(bpm, loop, pages):
    a = (bpm - 100) * 2 + loop
    body = [a // 64, a % 64, len(pages) // 64, len(pages) % 64] + _cp_lz(_cp_stage1(pages))
    chk = _cp_chk(body)
    return COMPACT_MARKER + ''.join(COMPACT_ALPHA[t] for t in body + [chk // 64, chk % 64])


def compact_decode(text):
    """Retourne (ok, bpm, loop, pages[str]). Même validation que les blocs Scratch."""
    s = ''.join(text.split())
    if len(s) < 8 or s[0] != COMPACT_MARKER:
        return False, None, None, []
    out = []
    for ch in s[1:]:
        k = COMPACT_ALPHA.find(ch.lower())
        if k < 0:
            return False, None, None, []
        out.append(k)
    m = len(out)
    if _cp_chk(out[:m - 2]) != out[m - 2] * 64 + out[m - 1]:
        return False, None, None, []
    a, np_ = out[0] * 64 + out[1], out[2] * 64 + out[3]
    loop = a % 2
    bpm = 100 + (a - loop) // 2
    if not (100 <= bpm <= 400 and 1 <= np_ <= 99):
        return False, None, None, []
    tok, k, e = [], 4, m - 3          # indices 0-based : charge utile = out[4 : m-2]
    while k <= e:
        t = out[k]
        if t < 55:
            tok.append(t); k += 1
            continue
        if t < 57:
            L, o, k = t - 52, out[k + 1] if k + 1 < m else 0, k + 2
        elif t < 63:
            L, o, k = t - 53, (out[k + 1] * 64 + out[k + 2]) if k + 2 < m else 0, k + 3
        else:
            L = (out[k + 1] * 64 + out[k + 2] + 10) if k + 4 < m else 10
            o = (out[k + 3] * 64 + out[k + 4]) if k + 4 < m else 0
            k += 5
        if o < 1 or o > len(tok) or k > e + 1:
            return False, None, None, []
        for _ in range(L):
            tok.append(tok[len(tok) - o])
    grids, pg, c, pv = [], 1, 0, 0
    grid = [0] * PAGE_CELLS
    for t in tok:
        if t < 24:
            if pv == 0:
                return False, None, None, []
            cellp = c // COLS + 1
            if cellp > np_:
                return False, None, None, []
            while pg < cellp:
                grids.append(grid); grid = [0] * PAGE_CELLS; pg += 1
            grid[t * COLS + c % COLS] = pv
            pv = 0
        elif t < 39:
            pv = t - 23
        else:
            c += t - 38
    while pg <= np_:
        grids.append(grid); grid = [0] * PAGE_CELLS; pg += 1
    return True, bpm, loop, [grid_to_page(g) for g in grids]


def compact_from_lines(lines):
    ok, bpm, loop, pages = verify(lines)
    if not pages:
        raise ValueError('Code invalide')
    return compact_encode(bpm, loop, pages)


def verify(lines):
    """Port fidèle du script 'verify' du sprite import. Retourne (ok, bpm, loop, pages)."""
    joined = ''.join(l.strip() for l in lines)
    if joined.startswith(COMPACT_MARKER):
        return compact_decode(joined)
    items = [l.strip('\r\n') for l in lines]
    items = [x for x in items if x != '']
    while items and not (items[0].lstrip('-').isdigit() and 100 <= int(items[0]) <= 400
                         and len(items) > 2 and items[1] in ('0', '1')
                         and items[2].lstrip('-').isdigit() and 0 <= int(items[2]) < 1000000):
        items.pop(0)
    if not items:
        return False, None, None, []
    while items and len(items[-1]) != PAGE_LEN:
        items.pop()
    bpm, loop, tchk, pages = items[0], items[1], int(items[2]), items[3:]
    ok = all(len(p) == PAGE_LEN for p in pages) and len(pages) > 0
    ok &= tchk == song_checksum(bpm, loop, pages)
    return ok, int(bpm), int(loop), pages


# ----------------------------------------------------------------- MIDI -> code
# Notes de percussion GM -> ligne de bruit (23 = grave ... 8 = aigu)
DRUM_ROW = {35: 23, 36: 22, 41: 21, 43: 20, 45: 19, 47: 18, 48: 17, 50: 16,
            38: 15, 40: 15, 37: 14, 39: 14, 56: 13, 54: 12, 70: 12, 69: 12,
            42: 10, 44: 10, 46: 9, 49: 8, 51: 8, 52: 8, 53: 8, 55: 8, 57: 8, 59: 8}


def drum_row(n):
    if n in DRUM_ROW:
        return DRUM_ROW[n]
    return max(8, min(23, 23 - (n - 35) // 2))  # repli : répartition grossière


# ligne de bruit -> note de percussion GM représentative (inverse de drum_row)
NOISE_REP = {23: 35, 22: 36, 21: 41, 20: 43, 19: 45, 18: 47, 17: 48, 16: 50,
             15: 38, 14: 37, 13: 56, 12: 54, 11: 60, 10: 42, 9: 46, 8: 49}
assert all(drum_row(n) == r for r, n in NOISE_REP.items())


def read_midi(path):
    import mido
    mid = mido.MidiFile(path)
    tpq = mid.ticks_per_beat
    tempo_us = 500000
    parts = {}   # (track, channel) -> {'notes': [(start_tick, end_tick, pitch)], 'prog': int}
    for ti, tr in enumerate(mid.tracks):
        t = 0
        active = {}
        got_tempo = False
        for msg in tr:
            t += msg.time
            if msg.type == 'set_tempo' and not got_tempo and t == 0:
                tempo_us = msg.tempo
                got_tempo = True
            elif msg.type == 'set_tempo' and tempo_us == 500000 and not got_tempo:
                tempo_us = msg.tempo
                got_tempo = True
            elif msg.type == 'program_change':
                parts.setdefault((ti, msg.channel), {'notes': [], 'prog': 0})['prog'] = msg.program
            elif msg.type == 'note_on' and msg.velocity > 0:
                active.setdefault((msg.channel, msg.note), []).append(t)
            elif msg.type in ('note_off', 'note_on'):
                k = (msg.channel, msg.note)
                if active.get(k):
                    s = active[k].pop(0)
                    parts.setdefault((ti, msg.channel), {'notes': [], 'prog': 0})['notes'].append((s, t, msg.note))
    return tpq, 60000000.0 / tempo_us, {k: v for k, v in parts.items() if v['notes']}


def fold(pitch, lo):
    while pitch < lo:
        pitch += 12
    while pitch > lo + 23:
        pitch -= 12
    return pitch


def best_shift(pitches, lo):
    """Décalage d'octaves (multiple de 12) qui garde le plus de notes dans [lo, lo+23]."""
    best, bs = None, 0
    for s in range(-48, 49, 12):
        out = sum(1 for p in pitches if not (lo <= p + s <= lo + 23))
        key = (out, abs(s))
        if best is None or key < best:
            best, bs = key, s
    return bs


def assign_channels(parts, mapping):
    """Décide pour chaque (piste,canal) : 'noise' | 'tri' | 'p12' | 'p25' | 'p50'."""
    assign, pulses = {}, 0
    order = sorted((k for k in parts if k[1] != 9),
                   key=lambda k: -sum(n[2] for n in parts[k]['notes']) / len(parts[k]['notes']))
    for k in parts:
        if k[1] == 9:
            assign[k] = 'noise'
    for k in order:
        avg = sum(n[2] for n in parts[k]['notes']) / len(parts[k]['notes'])
        if 32 <= parts[k]['prog'] <= 39 or avg < 52:
            assign[k] = 'tri'
        else:
            assign[k] = ['p25', 'p12', 'p50'][pulses % 3]
            pulses += 1
    for k, v in mapping.items():
        if k in parts:
            assign[k] = v
    return assign


def analyze_midi(path):
    """Liste les pistes/canaux d'un MIDI avec l'affectation automatique proposée."""
    tpq, mbpm, parts = read_midi(path)
    assign = assign_channels(parts, {})
    out = []
    for k, p in sorted(parts.items()):
        pitches = [n[2] for n in p['notes']]
        out.append({'track': k[0], 'channel': k[1], 'program': p['prog'], 'notes': len(pitches),
                    'lo': min(pitches), 'hi': max(pitches), 'assign': assign[k]})
    return {'bpm': mbpm, 'parts': out}


def midi2code(path, loop=0, cpq=None, bpm=None, sustain='hold', mapping=None, transpose=0,
              verbose=True, log_out=None):
    tpq, mbpm, parts = read_midi(path)
    if bpm is None:
        if cpq is None:
            cpq = next(c for c in (4, 8, 2, 16, 1) if 100 <= round(mbpm * c / 4) <= 400)
        bpm = round(mbpm * cpq / 4)
    else:
        cpq = 4 * bpm / mbpm
    bpm = max(100, min(400, bpm))
    ticks_per_col = tpq / cpq
    assign = assign_channels(parts, mapping or {})
    cells = {}          # (col,row) -> set(bits)
    dropped = 0
    total_cols = 0
    log = []
    for k, part in sorted(parts.items()):
        ch = assign[k]
        notes = part['notes']
        if ch == 'skip':
            log.append('piste %d canal %d ignorée' % (k[0], k[1] + 1))
            continue
        if ch == 'noise':
            for s, e, n in notes:
                c = int(round(s / ticks_per_col))
                cells.setdefault((c, drum_row(n)), set()).add('noise')
                total_cols = max(total_cols, c + 1)
            log.append('piste %d canal %d -> bruit (%d notes)' % (k[0], k[1] + 1, len(notes)))
            continue
        lo = TRI_LO if ch == 'tri' else PULSE_LO
        sh = best_shift([n[2] + transpose for n in notes], lo)
        for s, e, n in notes:
            p = fold(n + transpose + sh, lo)
            row = (lo + 23) - p
            c0 = int(round(s / ticks_per_col))
            c1 = max(c0 + 1, int(round(e / ticks_per_col)))
            cols = range(c0, c1) if sustain == 'hold' else [c0]
            for c in cols:
                cells.setdefault((c, row), set()).add(ch)
            total_cols = max(total_cols, c1)
        log.append('piste %d canal %d -> %s (%d notes, décalage %+d demi-tons)'
                   % (k[0], k[1] + 1, ch, len(notes), sh + transpose))
    n_pages = max(1, -(-total_cols // COLS))
    if n_pages > MAX_PAGES:
        log.append('ATTENTION : %d pages > %d, coupé' % (n_pages, MAX_PAGES))
        n_pages = MAX_PAGES
    grids = [[0] * PAGE_CELLS for _ in range(n_pages)]
    for (c, row), chans in cells.items():
        pg = c // COLS
        if pg >= n_pages:
            continue
        pulse = [x for x in PULSE_NAMES if x in chans]
        if len(pulse) > 1:                       # 2 formes pulse sur la même cellule : impossible
            dropped += len(pulse) - 1
        bits = ['0'] * 5
        if pulse:
            bits[PULSE_NAMES.index(pulse[0])] = '1'
        if 'tri' in chans:
            bits[3] = '1'
        if 'noise' in chans:
            bits[4] = '1'
        grids[pg][row * COLS + (c % COLS)] = MASK2CODE[''.join(bits)]
    pages = [grid_to_page(g) for g in grids]
    if verbose:
        log.append('tempo MIDI %.1f -> BPM projet %d (%s colonnes par noire), %d page(s)'
                   % (mbpm, bpm, cpq if isinstance(cpq, int) else '%.2f' % cpq, n_pages))
        if dropped:
            log.append('%d note(s) pulse perdue(s) (deux formes pulse sur la même cellule)' % dropped)
        if log_out is not None:
            log_out.extend(log)
        else:
            print('\n'.join(log), file=sys.stderr)
    return build_code(bpm, loop, pages)


# ----------------------------------------------------------------- code -> MIDI
def code2midi(lines, out):
    mid, bpm, loop, n = code_to_midifile(lines)
    mid.save(out)
    print('%d page(s), BPM %d, loop %d -> %s' % (n, bpm, loop, out), file=sys.stderr)


def code_to_midifile(lines):
    import mido
    ok, bpm, loop, pages = verify(lines)
    if not pages:
        raise SystemExit('Code invalide.')
    if not ok:
        print('Attention : checksum / format incorrect, lecture quand même.', file=sys.stderr)
    mid = mido.MidiFile(ticks_per_beat=480)
    tracks = {n: mido.MidiTrack() for n in ('p12', 'p25', 'p50', 'tri', 'noise')}
    ch_of = {'p12': 0, 'p25': 1, 'p50': 2, 'tri': 3, 'noise': 9}
    events = {n: [] for n in tracks}
    col_ticks = 480 // 4
    for pi, pg in enumerate(pages):
        g = page_to_grid(pg)
        for row in range(ROWS):
            for c in range(COLS):
                code = g[row * COLS + c]
                if not code:
                    continue
                t = (pi * COLS + c) * col_ticks
                for b, name in enumerate(['p12', 'p25', 'p50', 'tri', 'noise']):
                    if CODE2MASK[code][b] == '1':
                        if name == 'noise':
                            note = NOISE_REP[row] if row >= 8 else 35
                        elif name == 'tri':
                            note = TRI_LO + 23 - row
                        else:
                            note = PULSE_LO + 23 - row
                        events[name].append((t, note))
    tempo = mido.bpm2tempo(bpm)
    tracks['p12'].append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))
    for name, tr in tracks.items():
        evs = []
        for t, note in events[name]:
            evs.append((t, 1, note))                       # note_on
            evs.append((t + col_ticks - 1, 0, note))       # note_off (0 < 1 : les off passent avant les on)
        evs.sort()
        last = 0
        for t, on, note in evs:
            note = min(note, 127)
            if on:
                tr.append(mido.Message('note_on', channel=ch_of[name], note=note, velocity=100, time=t - last))
            else:
                tr.append(mido.Message('note_off', channel=ch_of[name], note=note, velocity=0, time=t - last))
            last = t
        mid.tracks.append(tr)
    return mid, bpm, loop, len(pages)


# ----------------------------------------------------------------- code -> audio
import os
import re
import shutil
import subprocess
import wave

CHANNELS = ['p12', 'p25', 'p50', 'tri', 'noise']
# préfixes de fichiers par canal. "s3" est accepté comme synonyme de "s2" (50 %).
SAMPLE_PREFIX = {'p12': ('s0',), 'p25': ('s1',), 'p50': ('s2', 's3'), 'tri': ('tri',), 'noise': ('noise',)}
SAMPLE_RE = re.compile(r'^(s[0-3]|tri|noise)[-_ ]?(\d+)\.wav$', re.I)
SCRATCH_VOLUME = 0.5        # le projet fait "set volume to 50"


def expand_dir(d):
    return os.path.expanduser(os.path.expandvars(d.strip().strip('"').strip("'")))


def scan_sounds(folder):
    """Retourne ({(canal, k): chemin}, [manquants])."""
    folder = expand_dir(folder)
    if not os.path.isdir(folder):
        raise FileNotFoundError('Dossier introuvable : %s' % folder)
    found = {}
    for fn in os.listdir(folder):
        m = SAMPLE_RE.match(fn)
        if not m:
            continue
        pre, k = m.group(1).lower(), int(m.group(2))
        for ch, pres in SAMPLE_PREFIX.items():
            if pre in pres and (ch, k) not in found:
                found[(ch, k)] = os.path.join(folder, fn)
    need = [('p12', k) for k in range(1, 25)] + [('p25', k) for k in range(1, 25)] + \
           [('p50', k) for k in range(1, 25)] + [('tri', k) for k in range(1, 25)] + \
           [('noise', k) for k in range(1, 17)]
    return found, [n for n in need if n not in found]


def read_wav(path):
    import numpy as np
    with wave.open(path) as w:
        n, sr, ch, sw = w.getnframes(), w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(n)
    if sw == 1:
        d = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128
    elif sw == 2:
        d = np.frombuffer(raw, dtype='<i2').astype(np.float32) / 32768
    elif sw == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        d = ((b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16)) << 8 >> 8).astype(np.float32) / 8388608
    else:
        d = np.frombuffer(raw, dtype='<i4').astype(np.float32) / 2147483648
    if ch > 1:
        d = d.reshape(-1, ch).mean(axis=1)
    return d, sr


def synth_triangle(midi, sr, dur=0.112):
    """Repli si les fichiers triX.wav manquent."""
    import numpy as np
    f = 440 * 2 ** ((midi - 69) / 12)
    t = np.arange(int(sr * dur)) / sr
    return (2 / np.pi * np.arcsin(np.sin(2 * np.pi * f * t)) * np.exp(-t * 6)).astype(np.float32)


def load_samples(folder):
    """Charge tous les échantillons au même taux. Retourne (dict, sr, avertissements)."""
    import numpy as np
    found, missing = scan_sounds(folder)
    if not found:
        raise FileNotFoundError('Aucun fichier s0-1.wav / s1-1.wav / s2-1.wav / noise1.wav trouvé dans %s' % folder)
    raw = {k: read_wav(p) for k, p in found.items()}
    sr = max({v[1] for v in raw.values()}, key=lambda r: sum(1 for v in raw.values() if v[1] == r))
    out = {}
    for k, (d, r) in raw.items():
        if r != sr:
            d = np.interp(np.linspace(0, len(d) - 1, int(len(d) * sr / r)), np.arange(len(d)), d).astype(np.float32)
        out[k] = d
    warn = []
    for ch, k in missing:
        if ch == 'tri':
            out[(ch, k)] = synth_triangle(TRI_LO + k - 1, sr)
        elif ch == 'noise':
            out[(ch, k)] = np.zeros(1, np.float32)
    miss_tri = sum(1 for c, _ in missing if c == 'tri')
    other = [(c, k) for c, k in missing if c not in ('tri', 'noise')]
    if miss_tri:
        warn.append('%d fichier(s) triangle manquant(s) : remplacés par un triangle synthétique' % miss_tri)
    if any(c == 'noise' for c, _ in missing):
        warn.append('fichiers noiseX manquants : silence à la place')
    if other:
        warn.append('fichiers manquants : ' + ', '.join('%s-%d' % o for o in other[:6]) + ('…' if len(other) > 6 else ''))
    return out, sr, warn


def render_pages(bpm, pages, samples, sr, repeats=1):
    """Mixe les pages en un tableau float32 mono. Reproduit le comportement Scratch :
    une colonne = 0,25 temps ; rejouer un même son le coupe et le relance."""
    import numpy as np
    col_s = 0.25 * 60.0 / bpm
    plan = {}                                   # clé échantillon -> [indices de départ]
    n_col = 0
    for rep in range(repeats):
        for pg in pages:
            g = page_to_grid(pg) if isinstance(pg, str) else pg
            for row in range(ROWS):
                for c in range(COLS):
                    code = g[row * COLS + c]
                    if code:
                        start = int(round((n_col + c) * col_s * sr))
                        mask = CODE2MASK[code]
                        for b, ch in enumerate(CHANNELS):
                            if mask[b] == '1':
                                if ch == 'noise':
                                    if row < 8:
                                        continue
                                    key = (ch, 24 - row)
                                else:
                                    key = (ch, 24 - row)
                                plan.setdefault(key, []).append(start)
            n_col += COLS
    longest = max((len(v) for v in samples.values()), default=0)
    out = np.zeros(int(round(n_col * col_s * sr)) + longest + int(0.25 * sr), np.float32)
    for key, starts in plan.items():
        smp = samples.get(key)
        if smp is None or len(smp) < 2:
            continue
        starts = sorted(set(starts))
        for i, s in enumerate(starts):
            length = len(smp) if i + 1 == len(starts) else min(len(smp), starts[i + 1] - s)
            if length > 0:
                out[s:s + length] += smp[:length]
    out *= SCRATCH_VOLUME
    peak = float(np.abs(out).max())
    if peak > 0.98:
        out *= 0.98 / peak
    return out


def to_wav_bytes(audio, sr):
    import io
    import numpy as np
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype('<i2').tobytes())
    return buf.getvalue()


def to_mp3_bytes(audio, sr, quality=2):
    """MP3 via ffmpeg (libmp3lame) ou, à défaut, le paquet pip 'lameenc'."""
    import numpy as np
    pcm = (np.clip(audio, -1, 1) * 32767).astype('<i2').tobytes()
    ff = shutil.which('ffmpeg')
    if ff:
        r = subprocess.run([ff, '-hide_banner', '-loglevel', 'error', '-f', 's16le', '-ar', str(sr), '-ac', '1',
                            '-i', '-', '-codec:a', 'libmp3lame', '-q:a', str(quality), '-f', 'mp3', '-'],
                           input=pcm, capture_output=True)
        if r.returncode == 0 and r.stdout:
            return r.stdout
    try:
        import lameenc
    except ImportError:
        raise RuntimeError("Export MP3 impossible : installez ffmpeg (sudo apt install ffmpeg) "
                           "ou lancez : pip install lameenc")
    enc = lameenc.Encoder()
    enc.set_bit_rate(192); enc.set_in_sample_rate(sr); enc.set_channels(1); enc.set_quality(2)
    return bytes(enc.encode(pcm) + enc.flush())


def render_code(lines, folder, repeats=1, fmt='wav'):
    ok, bpm, loop, pages = verify(lines)
    if not pages:
        raise ValueError('Code invalide')
    samples, sr, warn = load_samples(folder)
    audio = render_pages(bpm, pages, samples, sr, repeats)
    data = to_mp3_bytes(audio, sr) if fmt == 'mp3' else to_wav_bytes(audio, sr)
    return data, len(audio) / sr, warn


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('midi2code'); a.add_argument('midi'); a.add_argument('-o', '--out', required=True)
    a.add_argument('--loop', type=int, default=0, choices=[0, 1])
    a.add_argument('--cols-per-quarter', type=int, default=None, help='colonnes par noire (4 = double-croche)')
    a.add_argument('--bpm', type=int, default=None, help='forcer le BPM du projet (100..400)')
    a.add_argument('--sustain', choices=['hold', 'none'], default='hold',
                   help='hold : répète la note à chaque colonne tant qu\'elle dure (convention du projet)')
    a.add_argument('--transpose', type=int, default=0)
    a.add_argument('--compact', action='store_true', help='écrire le code au format compact (1 ligne)')
    a.add_argument('--map', default='', help='forcer un canal, ex: "0.0=p50,1.1=tri" (piste.canal=forme, canal 0-15)')
    b = sub.add_parser('code2midi'); b.add_argument('code'); b.add_argument('-o', '--out', required=True)
    c = sub.add_parser('check'); c.add_argument('code')
    z = sub.add_parser('compress', help='ancien code .txt -> code compact (1 ligne, ~20 a 50 fois plus petit)')
    z.add_argument('code'); z.add_argument('-o', '--out', required=True)
    y = sub.add_parser('decompress', help='code compact -> ancien format (lignes)')
    y.add_argument('code'); y.add_argument('-o', '--out', required=True)
    r = sub.add_parser('render', help='.mid ou code .txt -> .mp3 / .wav avec vos échantillons')
    r.add_argument('input'); r.add_argument('-o', '--out', required=True, help='.mp3 ou .wav')
    r.add_argument('--sounds', required=True, help='dossier contenant s0-1.wav … noise1.wav …')
    r.add_argument('--repeats', type=int, default=1, help='nombre de passages')
    r.add_argument('--cols-per-quarter', type=int, default=None); r.add_argument('--bpm', type=int, default=None)
    r.add_argument('--sustain', choices=['hold', 'none'], default='hold'); r.add_argument('--transpose', type=int, default=0)
    r.add_argument('--map', default='')
    args = ap.parse_args()
    if args.cmd in ('compress', 'decompress'):
        lines = open(args.code).read().split('\n')
        ok, bpm, loop, pages = verify(lines)
        if not pages:
            raise SystemExit('Code invalide.')
        if not ok:
            print('Attention : checksum / format incorrect.', file=sys.stderr)
        text = compact_encode(bpm, loop, pages) + '\n' if args.cmd == 'compress' else '\n'.join(build_code(bpm, loop, pages)) + '\n'
        open(args.out, 'w', newline='\n').write(text)
        print('%d -> %d caracteres : %s' % (sum(len(l) + 1 for l in lines), len(text), args.out), file=sys.stderr)
        return
    if args.cmd == 'render':
        if args.input.lower().endswith(('.mid', '.midi')):
            mapping = {}
            for kv in filter(None, args.map.split(',')):
                k, v = kv.split('=')
                t, ch = k.split('.')
                mapping[(int(t), int(ch))] = v
            lines = midi2code(args.input, 0, args.cols_per_quarter, args.bpm, args.sustain, mapping, args.transpose)
        else:
            lines = open(args.input).read().split('\n')
        data, dur, warn = render_code(lines, args.sounds, args.repeats, 'mp3' if args.out.lower().endswith('.mp3') else 'wav')
        for w in warn:
            print('Attention :', w, file=sys.stderr)
        with open(args.out, 'wb') as f:
            f.write(data)
        print('écrit : %s (%.1f s)' % (args.out, dur), file=sys.stderr)
        return
    if args.cmd == 'midi2code':
        mapping = {}
        for kv in filter(None, args.map.split(',')):
            k, v = kv.split('=')
            t, ch = k.split('.')
            mapping[(int(t), int(ch))] = v
        code = midi2code(args.midi, args.loop, args.cols_per_quarter, args.bpm, args.sustain, mapping, args.transpose)
        with open(args.out, 'w', newline='\n') as f:
            f.write((compact_from_lines(code) if args.compact else '\n'.join(code)) + '\n')
        print('écrit : %s (%d lignes)' % (args.out, len(code)), file=sys.stderr)
    elif args.cmd == 'code2midi':
        code2midi(open(args.code).read().split('\n'), args.out)
    else:
        ok, bpm, loop, pages = verify(open(args.code).read().split('\n'))
        print('OK' if ok else 'INVALIDE', 'bpm=%s loop=%s pages=%d' % (bpm, loop, len(pages)))
        sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
audio2code pour "8-Bit Music Maker" (projet Scratch .sb3)

Transcrit un fichier audio (WAV ou MP3) en code jouable par le projet, en
réutilisant sb3_music_codec.py : même format, mêmes checksums, mêmes contraintes.

    pip install numpy          (ffmpeg conseillé pour le MP3, sinon : pip install miniaudio)
    python3 sb3_music_audio2code.py chanson.mp3 -o code.txt
    python3 sb3_music_audio2code.py chanson.wav -o code.txt \
        --sounds "~/Sons musique scratch backup" --preview apercu.wav
    python3 sb3_music_audio2code.py voix.wav -o code.txt --approx   # n'importe quel son (voix…)

Ce que fait le script :
  1. décode l'audio en mono 22 050 Hz (WAV natif ; MP3 via ffmpeg, sinon miniaudio) ;
  2. détecte les attaques (flux spectral) et le tempo (autocorrélation de l'enveloppe) ;
  3. pour chaque colonne de la grille (1 colonne = 1/4 de temps), cherche jusqu'à 3 hauteurs
     par somme harmonique, puis devine la forme d'onde (12,5 % / 25 % / 50 % / triangle)
     en comparant le spectre mesuré aux profils harmoniques de VOS échantillons
     s0/s1/s2/tri (option --sounds), à défaut de profils théoriques ;
  4. les percussions (spectre large, peu tonal) vont sur les lignes de bruit : la ligne
     (16 timbres, lignes 8..23) est choisie par ressemblance spectrale avec vos
     noise1..16.wav, sinon par centre de gravité spectral ;
  5. --approx : pour un son quelconque (voix « Bonjour », effets…), suit le contour
     colonne par colonne — 1 à 2 notes tonales par colonne, une percussion par colonne
     large-bande — quitte à spammer des notes pour ressembler au plus près ;
  6. quantifie sur la grille, respecte les contraintes du système (BPM 100..400, 24 lignes,
     une seule forme pulse par cellule, bruit lignes 8..23, 99 pages) et écrit le code
     (format compact par défaut ; ancien format en plus avec --lines).

Options : --bpm pour forcer le tempo du projet, --cpq pour la résolution (4 = double-croche),
--sensitivity (0.5..2, défaut 1 : plus grand = plus de notes), --transpose, --loop,
--preview apercu.wav (rendu WAV de contrôle avec vos échantillons), --midi export.mid.
"""
import argparse
import math
import os
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sb3_music_codec as C  # noqa: E402

ANALYSIS_SR = 22050
HOP = 256
ONSET_WIN = 1024
NFFT = 16384
MIDI_MIN, MIDI_MAX = 36, 96          # C2..C7
HARM_N = 8
FMAX_HZ = 8000.0
WBAND = 0.035                        # demi-largeur (relative) de recherche de chaque harmonique


def log(msg):
    print(msg, file=sys.stderr)


def _clip(v, lo, hi):
    return max(lo, min(hi, v))


# ----------------------------------------------------------------- décodage audio
def resample(x, sr, target=ANALYSIS_SR):
    if sr == target:
        return x.astype(np.float32)
    n = int(len(x) * target / sr)
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def decode_audio(path):
    """-> (float32 mono, taux). WAV natif, sinon ffmpeg, sinon miniaudio."""
    if path.lower().endswith('.wav'):
        try:
            return C.read_wav(path)
        except Exception:
            pass
    ff = shutil.which('ffmpeg')
    if ff:
        r = subprocess.run([ff, '-hide_banner', '-loglevel', 'error', '-i', path,
                            '-ac', '1', '-ar', str(ANALYSIS_SR), '-f', 'f32le', '-'],
                           capture_output=True)
        if r.returncode == 0 and r.stdout:
            return np.frombuffer(r.stdout, np.float32).copy(), ANALYSIS_SR
    try:
        import miniaudio
    except ImportError:
        raise SystemExit("Lecture de %s impossible : installez ffmpeg (sudo apt install ffmpeg) "
                         "ou lancez : pip install miniaudio" % path)
    d = miniaudio.decode_file(path, miniaudio.SampleFormat.SIGNED16, 1, ANALYSIS_SR)
    x = np.asarray(d.samples, dtype=np.float32) / 32768
    if d.nchannels > 1:
        x = x.reshape(-1, d.nchannels).mean(axis=1)
    return x, d.sample_rate


# ----------------------------------------------------------------- attaques et tempo
def onset_envelope(x):
    """Flux spectral (demi-onde redressé) par trame de ONSET_WIN / HOP échantillons."""
    w = np.hanning(ONSET_WIN).astype(np.float32)
    n_frames = max(1, (len(x) - ONSET_WIN) // HOP + 1)
    prev = None
    flux = np.zeros(n_frames, np.float32)
    for i in range(n_frames):
        seg = x[i * HOP:i * HOP + ONSET_WIN]
        if len(seg) < ONSET_WIN:
            seg = np.pad(seg, (0, ONSET_WIN - len(seg)))
        mag = np.abs(np.fft.rfft(seg * w))
        lm = np.log1p(mag * 10)
        if prev is not None:
            flux[i] = float(np.maximum(0, lm - prev).sum())
        prev = lm
    return flux


def onset_peaks(flux, hop_s, sens=1.0):
    if len(flux) < 3:
        return []
    base = float(np.median(flux))
    spread = float(np.percentile(flux, 95)) - base + 1e-9
    thr = base + (0.12 / sens) * spread
    peaks = []
    min_gap = max(1, int(0.08 / hop_s))
    for i in range(1, len(flux) - 1):
        if flux[i] >= flux[i - 1] and flux[i] > flux[i + 1] and flux[i] > thr:
            if peaks and i - peaks[-1] < min_gap:
                if flux[i] > flux[peaks[-1]]:
                    peaks[-1] = i
            else:
                peaks.append(i)
    return peaks


def refine_onsets(x, sr, onsets, radius_s=0.06):
    """Recentre chaque attaque sur la plus forte montée d'énergie locale (±radius_s)."""
    out = []
    for t in onsets:
        c = int(t * sr)
        a = max(0, c - int(radius_s * sr))
        b = min(len(x) - 1, c + int(radius_s * sr))
        if b - a < 128:
            out.append(t)
            continue
        e = np.convolve(x[a:b] ** 2, np.ones(64, np.float32) / 64, 'same')
        rise = np.maximum(0, np.diff(e))
        out.append((a + int(np.argmax(rise)) + 32) / sr)
    out.sort()
    merged = []
    for t in out:                                 # fusionne les attaques quasi confondues
        if merged and t - merged[-1] < 0.06:
            continue
        merged.append(t)
    return merged


def estimate_bpm(flux, hop_s):
    e = flux - flux.mean()
    n = len(e)
    hi_lag = min(n - 1, int(1.5 / hop_s))
    lo_lag = max(2, int(0.3 / hop_s))
    if hi_lag <= lo_lag:
        return 120.0
    ac = np.correlate(e, e, 'full')[n - 1:n + hi_lag + 1]
    ac = ac / (ac[0] + 1e-9)
    lags = np.arange(lo_lag, hi_lag + 1)
    bpms = 60.0 / (lags * hop_s)
    w = np.exp(-0.5 * (np.log2(bpms / 120.0)) ** 2)      # aroundit vers ~120 BPM
    lag = int(lags[np.argmax(ac[lo_lag:hi_lag + 1] * w)])
    return 60.0 / (lag * hop_s)


def grid_fit(onsets, pb, n_phase=16):
    """(distance moyenne en secondes, phase) de la grille 15/pb la plus proche des
    attaques, phase optimisée. 0 = attaques parfaitement sur la grille."""
    if not onsets:
        return 1.0, 0.0
    g = 15.0 / pb
    t = np.asarray(onsets)
    best, phi0 = 1.0, 0.0
    for k in range(n_phase):
        phi = k * g / n_phase
        d = np.abs((t - phi + g / 2) % g - g / 2)
        m = float(np.mean(d))
        if m < best:
            best, phi0 = m, phi
    return best, phi0


def bpm_candidates(est, onsets, user_bpm, user_cpq, prefer_fine=False):
    if user_bpm:
        return [user_bpm]
    est = float(est)
    while est < 40.0:
        est *= 2
    while est > 800.0:
        est /= 2
    bases = [est / 2, est * 2 / 3, est * 3 / 4, est, est * 4 / 3, est * 3 / 2, est * 2]
    cands = set()
    for base in bases:
        for cpq in ((user_cpq,) if user_cpq else (4, 2, 8, 16, 1)):
            pb = int(round(base * cpq / 4))
            for d in (-2, -1, 0, 1, 2):          # le lag entier de l'autocorrélation
                if 100 <= pb + d <= 400:         # peut être à 1-2 BPM de la vraie grille
                    cands.add(pb + d)
    if not cands:
        cands = {_clip(int(round(est)), 100, 400)}
    # distance moyenne en secondes : à qualité égale (±30 %, ±5 ms), préfère la grille
    # la moins fine (une grille fine fit toujours mieux, même à tort) — sauf en mode
    # approximation, où une grille fine suit mieux le contour du son
    scored = sorted((grid_fit(onsets, pb)[0], pb) for pb in cands)
    best_m = scored[0][0]
    ok = [pb for m, pb in scored if m <= best_m * 1.3 + 0.005]
    if prefer_fine:
        return sorted(ok, reverse=True) + [pb for _, pb in scored if pb not in ok]
    return sorted(ok) + [pb for _, pb in scored if pb not in ok]


# ----------------------------------------------------------------- profils de timbre
def _normalize(v):
    v = np.asarray(v, np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def analytic_profiles():
    """Amplitudes H1..H8 (H1 = 1) pour pulse 12,5/25/50 % et triangle."""
    out = {}
    for name, d in (('p12', 0.125), ('p25', 0.25), ('p50', 0.5)):
        out[name] = np.array([abs(math.sin(math.pi * n * d)) / n for n in range(1, HARM_N + 1)],
                             np.float32)
    out['tri'] = np.array([(1.0 / (n * n)) if n % 2 else 0.0 for n in range(1, HARM_N + 1)], np.float32)
    return out


def harmonic_bands(mag, bin_hz, f0, n=HARM_N):
    """Amplitude captée autour de chaque harmonique h*f0."""
    v = np.zeros(n, np.float32)
    for h in range(1, n + 1):
        fh = f0 * h
        if fh > FMAX_HZ:
            break
        b0 = int(fh * (1 - WBAND) / bin_hz)
        b1 = int(fh * (1 + WBAND) / bin_hz) + 1
        if b1 > b0 and b1 <= len(mag):
            v[h - 1] = mag[b0:b1].max()
    return v


def measure_profile(mag, bin_hz, f0):
    """Amplitudes harmoniques relatives (H1 = 1) mesurées autour de f0."""
    return harmonic_bands(mag, bin_hz, f0)


def sample_profile(d, sr, f0):
    a = int(0.1 * len(d))
    b = len(d)
    if b - a < 256:
        a, b = 0, len(d)
    seg = d[a:b] * np.hanning(b - a)
    mag = np.abs(np.fft.rfft(seg, 8192))
    return measure_profile(mag, sr / 8192, f0)


LOGF = None


def _logspec(mag, bin_hz):
    """Spectre interpolé sur 96 bandes log 150..8000 Hz (axe commun pour les comparaisons)."""
    global LOGF
    if LOGF is None:
        LOGF = np.logspace(math.log10(150), math.log10(FMAX_HZ), 96)
    freqs = np.arange(len(mag)) * bin_hz
    v = np.interp(LOGF, freqs, mag, left=0, right=0)
    return _normalize(np.log1p(v * 50))


def noise_spectrum(d, sr):
    n = min(len(d), int(0.09 * sr))
    if n < 128:
        n = len(d)
    seg = d[:n] * np.hanning(n)
    mag = np.abs(np.fft.rfft(seg, 4096))
    return _logspec(mag, sr / 4096)


class Templates:
    def __init__(self, sounds=None):
        analytic = analytic_profiles()
        self.rel = dict(analytic)            # H1..H8 relatifs, pour la soustraction
        self.harm = {t: _normalize(v[1:]) for t, v in analytic.items()}   # pour la ressemblance
        self.noise = {}
        self.wave = {}                       # (timbre, k) -> forme d'onde normalisée
        if not sounds:
            return
        try:
            samples, sr, warn = C.load_samples(sounds)
        except Exception as e:
            log('Attention : échantillons non chargés (%s)' % e)
            return
        for w in warn:
            log('Attention : ' + w)
        for tb in ('p12', 'p25', 'p50', 'tri'):
            lo = C.TRI_LO if tb == 'tri' else C.PULSE_LO
            profs = []
            for k in (8, 12, 16, 20, 24):
                d = samples.get((tb, k))
                if d is None or len(d) < 256:
                    continue
                f0 = 440 * 2 ** ((lo + k - 1 - 69) / 12)
                profs.append(sample_profile(d, sr, f0))
            if profs:
                rel = np.mean(profs, axis=0)
                self.rel[tb] = rel / (rel[0] + 1e-9)
                self.harm[tb] = _normalize(self.rel[tb][1:])
        for k in range(1, 17):
            d = samples.get(('noise', k))
            if d is not None and len(d) > 256:
                self.noise[k] = noise_spectrum(d, sr)
        for (tb, k), d in samples.items():
            if tb == 'noise' or len(d) < 256:
                continue
            if sr != ANALYSIS_SR:
                d = resample(d, sr)
            d = d - float(d.mean())
            self.wave[(tb, k)] = (d / (np.linalg.norm(d) + 1e-9)).astype(np.float32)


# ----------------------------------------------------------------- analyse par colonne
def spectrum_at(x, start, win, nfft=NFFT):
    seg = x[start:start + win]
    if len(seg) < win:
        seg = np.pad(seg, (0, win - len(seg)))
    return np.abs(np.fft.rfft(seg * np.hanning(win), nfft))


def pitch_scores(mag, bin_hz):
    mids = np.arange(MIDI_MIN, MIDI_MAX + 1)
    scores = np.zeros(len(mids), np.float32)
    for i, m in enumerate(mids):
        f0 = 440 * 2 ** ((m - 69) / 12)
        v = harmonic_bands(mag, bin_hz, f0)
        w = np.array([1.0 / h for h in range(1, HARM_N + 1)], np.float32)
        scores[i] = float((v * w).sum() / w.sum())
    return mids, scores


def pick_pitches(scores, sens):
    if not len(scores) or scores.max() <= 0:
        return []
    rel = _clip(0.35 / sens, 0.12, 0.7)
    order = np.argsort(scores)[::-1]
    top = scores[order[0]]
    picked = []
    for i in order:
        s = scores[i]
        if s < rel * top or len(picked) == 3:
            break
        if any(abs(int(i) - j) < 2 for j in picked):
            continue
        picked.append(int(i))
    picked.sort()
    picked_m = [MIDI_MIN + i for i in picked]         # évite les fantômes d'octave
    out = []
    for m in picked_m:
        s = scores[m - MIDI_MIN]
        if m - 12 in picked_m and s < scores[m - 12 - MIDI_MIN] * 0.7:
            continue
        if m + 12 in picked_m and s < scores[m + 12 - MIDI_MIN] * 0.7:
            continue
        out.append(m)
    return out


def xcorr_best(res, tpl, max_lag):
    """Corrélation croisée normalisée ; -> (score, décalage, échelle) ou None."""
    n, m = len(res), len(tpl)
    if m > n:
        tpl = tpl[:n]
        m = n
    N = 1 << (n + m).bit_length()
    cc = np.fft.irfft(np.fft.rfft(res, N) * np.conj(np.fft.rfft(tpl, N)), N)
    lags = list(range(0, max_lag + 1)) + list(range(N - max_lag, N))
    j = int(np.argmax(np.abs(cc[lags])))
    idx = lags[j]
    lag = idx if idx <= N // 2 else idx - N
    peak = float(cc[idx])
    denom = float(np.linalg.norm(res) * np.linalg.norm(tpl)) + 1e-9
    score = abs(peak) / denom
    scale = peak / (float(np.linalg.norm(tpl)) ** 2 + 1e-9)
    return score, lag, scale


def _subtract(residual_mag, bin_hz, f0, h1, rel):
    for h in range(1, HARM_N + 1):
        fh = f0 * h
        if fh > FMAX_HZ:
            break
        b0 = int(fh * (1 - WBAND) / bin_hz)
        b1 = int(fh * (1 + WBAND) / bin_hz) + 1
        if b1 > b0 and b1 <= len(residual_mag):
            residual_mag[b0:b1] = np.maximum(residual_mag[b0:b1] - h1 * rel[h - 1], 0)


def analyze_column_voices(seg, hann, bin_hz, templates, sens, col_s, sr):
    """Cherche jusqu'à 3 voix par poursuite : la voix la mieux corrélée aux
    échantillons réels du projet est classée puis soustraite du segment.
    Sans échantillons, repli sur la comparaison de profils harmoniques."""
    if templates.wave:
        return _voices_wave(seg, hann, bin_hz, templates, sens, col_s, sr)
    mag = np.abs(np.fft.rfft(seg * hann, NFFT))
    return _voices_spectral(mag / (len(seg) / 4), bin_hz, templates, sens)


def _voices_wave(seg, hann, bin_hz, templates, sens, col_s, sr):
    """-> (voix, résidu temporel après soustraction des voix trouvées)."""
    residual = seg.copy()
    voices, taken = [], set()
    max_lag = max(8, int(0.3 * col_s * sr))
    for _ in range(3):
        if len(voices) >= 3:
            break
        mag = np.abs(np.fft.rfft(residual * hann, NFFT))
        mids, scores = pitch_scores(mag / (len(seg) / 4), bin_hz)
        picked = [m for m in pick_pitches(scores, sens)
                  if m not in taken and all(abs(m - t) >= 2 for t in taken)]
        if not picked:
            break
        cands = sorted({mm for m in picked for mm in (m - 1, m, m + 1) if MIDI_MIN <= mm <= MIDI_MAX})
        best = None
        for m in cands:
            for tb in ('p12', 'p25', 'p50', 'tri'):
                lo = C.TRI_LO if tb == 'tri' else C.PULSE_LO
                tpl = templates.wave.get((tb, m - lo + 1))
                if tpl is None:
                    continue
                sc, lag, scale = xcorr_best(residual, tpl, max_lag)
                if best is None or sc > best[0]:
                    best = (sc, m, tb, lag, scale, tpl)
        if best is None or best[0] < 0.25:
            break
        sc, m, tb, lag, scale, tpl = best
        voices.append((m, tb, min(1.0, sc * 1.3)))
        taken.add(m)
        i0 = max(0, lag)
        i1 = min(len(residual), lag + len(tpl))
        t0 = max(0, -lag)
        if i1 > i0:
            residual[i0:i1] -= scale * tpl[t0:t0 + (i1 - i0)]
    return voices, residual


def _voices_spectral(amp, bin_hz, templates, sens):
    """-> (voix, None) : pas de soustraction temporelle possible sans échantillons."""
    residual = amp.copy()
    voices, taken = [], set()
    for _ in range(3):
        if len(voices) >= 3:
            break
        mids, scores = pitch_scores(residual, bin_hz)
        picked = [m for m in pick_pitches(scores, sens)
                  if m not in taken and all(abs(m - t) >= 2 for t in taken)]
        if not picked:
            break
        found = False
        for m in sorted(picked, key=lambda m: -scores[m - MIDI_MIN]):
            f0 = 440 * 2 ** ((m - 69) / 12)
            v = measure_profile(residual, bin_hz, f0)
            sims = {t: float(np.dot(_normalize(v[1:]), tv)) for t, tv in templates.harm.items()}
            tb, sim = max(sims.items(), key=lambda kv: kv[1])
            if sim < 0.55 or v[0] < 0.008:
                continue
            voices.append((m, tb, sim))
            taken.add(m)
            _subtract(residual, bin_hz, f0, v[0], templates.rel[tb])
            found = True
        if not found:
            break
    return voices, None


def analyze_columns(x, sr, col_s, templates, onsets, sens=1.0):
    """-> (notes, bruits) en secondes.
    notes : (t0, t1, midi, timbre, score) ; bruits : (t, k_bruit, score)."""
    bin_hz = sr / NFFT
    n_cols = max(1, int(math.ceil(len(x) / sr / col_s)))
    win = 1 << (_clip(int(1.2 * col_s * sr), 2048, 4096).bit_length() - 1)
    hann = np.hanning(win).astype(np.float32)
    onset_cols = set(int(round(t / col_s)) for t in onsets)
    col_events = {}
    noise_events = []
    for c in range(n_cols):
        start = int(round(c * col_s * sr))
        mag = spectrum_at(x, start, win)
        amp = mag / (win / 4)
        band = amp[int(200 / bin_hz):int(FMAX_HZ / bin_hz)]
        if float(band.max() if len(band) else 0) < 0.004:
            continue
        mids, scores = pitch_scores(amp, bin_hz)
        top = float(scores.max()) if len(scores) else 0.0
        mean = float(band.mean()) + 1e-9
        ratio = top / mean                       # tonal ~30, percussif ~2..6
        tonal_thr = 12.0 / _clip(sens, 0.5, 2.0)
        if ratio < tonal_thr:
            if c in onset_cols or ratio < 3.0:
                noise_events.append(_classify_noise(amp, bin_hz, c * col_s, templates))
            continue
        if top < 0.01:
            continue
        seg = x[start:start + win]
        if len(seg) < win:
            seg = np.pad(seg, (0, win - len(seg)))
        voices, residual = analyze_column_voices(seg, hann, bin_hz, templates, sens, col_s, sr)
        for m, tb, sim in voices:
            col_events.setdefault(c, []).append((m, tb, sim))
        if voices and residual is not None and c in onset_cols:
            mag_r = np.abs(np.fft.rfft(residual * hann, NFFT)) / (win / 4)
            band_r = mag_r[int(200 / bin_hz):int(FMAX_HZ / bin_hz)]
            if float(band_r.max() if len(band_r) else 0) >= 0.006:
                noise_events.append(_classify_noise(mag_r, bin_hz, c * col_s, templates))
    return _group_voices(col_events, col_s), noise_events


def _group_voices(col_events, col_s):
    """Regroupe les voix par colonnes consécutives -> notes (t0, t1, midi, timbre, score)."""
    notes = []
    voices = {}
    for c in sorted(col_events):
        for m, tb, s in col_events[c]:
            voices.setdefault((m, tb), []).append((c, s))
    for (m, tb), cols in voices.items():
        run = None
        for c, s in cols:
            if run and c - run[1] <= 2:
                run[1] = c
                run[2] = max(run[2], s)
            else:
                if run:
                    notes.append((run[0] * col_s, (run[1] + 1) * col_s, m, tb, run[2]))
                run = [c, c, s]
        if run:
            notes.append((run[0] * col_s, (run[1] + 1) * col_s, m, tb, run[2]))
    return notes


def _best_timbre(seg, hann, bin_hz, m, templates):
    """Timbre qui ressemble le plus au segment à la hauteur m : corrélation avec les
    vrais échantillons du projet si disponibles, sinon profil harmonique.
    -> (timbre, score) ; toujours un timbre (le mode approximation spamme des notes)."""
    if templates.wave:
        best = None
        for tb in ('p12', 'p25', 'p50', 'tri'):
            lo = C.TRI_LO if tb == 'tri' else C.PULSE_LO
            tpl = templates.wave.get((tb, C.fold(m, lo) - lo + 1))
            if tpl is None:
                continue
            sc, _, _ = xcorr_best(seg, tpl, 8)
            if best is None or sc > best[1]:
                best = (tb, sc)
        if best is None:
            return 'p25', 0.0
        return best[0], min(1.0, best[1] * 1.3)
    mag = np.abs(np.fft.rfft(seg * hann, NFFT)) / (len(seg) / 4)
    f0 = 440 * 2 ** ((m - 69) / 12)
    v = measure_profile(mag, bin_hz, f0)
    sims = {t: float(np.dot(_normalize(v[1:]), tv)) for t, tv in templates.harm.items()}
    tb, sim = max(sims.items(), key=lambda kv: kv[1])
    return tb, max(0.0, sim)


def analyze_columns_approx(x, sr, col_s, templates, sens=1.0):
    """Mode approximation : suit n'importe quel son (voix, effets…) colonne par
    colonne en spammant des notes — 1 à 2 hauteurs suivies par colonne tonale,
    une percussion par colonne large-bande. -> (notes, bruits) en secondes."""
    bin_hz = sr / NFFT
    n_cols = max(1, int(math.ceil(len(x) / sr / col_s)))
    win = 1 << (_clip(int(1.2 * col_s * sr), 2048, 4096).bit_length() - 1)
    hann = np.hanning(win).astype(np.float32)
    gate = 0.0018 / _clip(sens, 0.5, 2.0)
    tonal_thr = 8.0 / _clip(sens, 0.5, 2.0)
    col_events = {}
    noise_events = []
    picked = []
    for c in range(n_cols):
        start = int(round(c * col_s * sr))
        mag = spectrum_at(x, start, win)
        amp = mag / (win / 4)
        band = amp[int(200 / bin_hz):int(FMAX_HZ / bin_hz)]
        if float(band.max() if len(band) else 0) < gate:
            continue
        mids, scores = pitch_scores(amp, bin_hz)
        top = float(scores.max()) if len(scores) else 0.0
        mean = float(band.mean()) + 1e-9
        ratio = top / mean
        seg = x[start:start + win]
        if len(seg) < win:
            seg = np.pad(seg, (0, win - len(seg)))
        picked = pick_pitches(scores, sens * 1.5)[:2] if (ratio >= tonal_thr and top >= 0.006) else []
        for m in picked:
            tb, sc = _best_timbre(seg, hann, bin_hz, m, templates)
            col_events.setdefault(c, []).append((m, tb, sc))
        if not picked:
            noise_events.append(_classify_noise(amp, bin_hz, c * col_s, templates))
    return _group_voices(col_events, col_s), noise_events


def _classify_noise(amp, bin_hz, t, templates):
    spec = _logspec(amp, bin_hz)
    best_k, best_s = None, -1.0
    for k, ref in templates.noise.items():
        s = float(np.dot(spec, ref))
        if s > best_s:
            best_k, best_s = k, s
    if best_k is None:
        fc = float((np.arange(len(amp)) * bin_hz * amp).sum() / (amp.sum() + 1e-9))
        fc = _clip(fc, 150.0, FMAX_HZ)
        best_k = 1 + int(round(15 * (1 - math.log(fc / 150) / math.log(FMAX_HZ / 150))))
        best_k = _clip(best_k, 1, 16)
    return (t, best_k, max(best_s, 0.0))


# ----------------------------------------------------------------- assemblage du code
def assemble(notes, noises, bpm, loop, transpose=0):
    col_s = 0.25 * 60.0 / bpm
    cells = {}
    dropped = 0
    for kind, lo in (('pulse', C.PULSE_LO), ('tri', C.TRI_LO)):
        sel = [n for n in notes if (n[3] == 'tri') == (kind == 'tri')]
        if not sel:
            continue
        smax = max(n[4] for n in sel)
        pits = [n[2] + transpose for n in sel if n[4] >= 0.9 * smax]
        shift = C.best_shift(pits or [n[2] + transpose for n in sel], lo)
        for t0, t1, m, tb, s in notes:
            if (tb == 'tri') != (kind == 'tri'):
                continue
            p = C.fold(m + transpose + shift, lo)
            row = (lo + 23) - p
            c0 = int(round(t0 / col_s))
            c1 = max(c0 + 1, int(round(t1 / col_s)))
            for c in range(c0, c1):
                key = (c, row)
                if tb == 'tri':
                    cells.setdefault(key, {'tri': None, 'pulse': None, 'score': 0})['tri'] = s
                else:
                    cell = cells.setdefault(key, {'tri': None, 'pulse': None, 'score': 0})
                    if cell['pulse'] is not None and cell['pulse'] != tb:
                        dropped += 1
                        if s > cell['score']:
                            cell['pulse'] = tb
                    else:
                        cell['pulse'] = tb
                    cell['score'] = max(cell['score'], s)
    for t, k, s in noises:
        cells.setdefault((int(round(t / col_s)), 24 - k), {'tri': None, 'pulse': None, 'score': 0})['noise'] = True
    max_col = max((c for c, _ in cells), default=-1)
    n_pages = max(1, -(- (max_col + 1) // C.COLS)) if max_col >= 0 else 1
    truncated = False
    if n_pages > C.MAX_PAGES:
        n_pages = C.MAX_PAGES
        truncated = True
    grids = [[0] * C.PAGE_CELLS for _ in range(n_pages)]
    for (c, row), cell in cells.items():
        pg = c // C.COLS
        if pg >= n_pages:
            continue
        bits = ['0'] * 5
        if cell.get('pulse'):
            bits[C.PULSE_NAMES.index(cell['pulse'])] = '1'
        if cell.get('tri') is not None:
            bits[3] = '1'
        if cell.get('noise'):
            bits[4] = '1'
        grids[pg][row * C.COLS + (c % C.COLS)] = C.MASK2CODE[''.join(bits)]
    return C.build_code(bpm, loop, [C.grid_to_page(g) for g in grids]), dropped, truncated


def transcribe(path, out, sounds=None, user_bpm=None, user_cpq=None, loop=0,
               sens=1.0, transpose=0, write_lines=False, preview=None, midi_out=None,
               approx=False):
    x, sr = decode_audio(path)
    x = x.astype(np.float32)
    x = x - float(x.mean())
    peak = float(np.abs(x).max()) if len(x) else 0.0
    if peak > 0:
        x *= 0.95 / peak
    x = resample(x, sr)
    log('audio : %.1f s, %d Hz' % (len(x) / ANALYSIS_SR, ANALYSIS_SR))

    hop_s = HOP / ANALYSIS_SR
    flux = onset_envelope(x)
    onsets = refine_onsets(x, ANALYSIS_SR,
                           [(i * HOP + ONSET_WIN / 2) / ANALYSIS_SR
                            for i in onset_peaks(flux, hop_s, sens)])
    est = estimate_bpm(flux, hop_s)
    log('attaques détectées : %d · tempo estimé : %.0f BPM' % (len(onsets), est))

    templates = Templates(sounds)
    cands = bpm_candidates(est, onsets, user_bpm, user_cpq, prefer_fine=approx)
    if approx:
        log('mode approximation : suivi du son colonne par colonne (notes spammées)')
    log('grille retenue : BPM projet %d (candidats %s)' % (cands[0], cands[:4]))

    _, phi = grid_fit(onsets, cands[0])         # délai global (débuts d'enregistrement,
    k = int(round(phi * ANALYSIS_SR))           # délai d'encodeur MP3…) : on réaligne
    if k:
        x = x[k:]
        onsets = [t - k / ANALYSIS_SR for t in onsets]
        log('réalignement de la grille : %d ms' % round(1000 * phi))

    col_s = 0.25 * 60.0 / cands[0]
    if approx:
        notes, noises = analyze_columns_approx(x, ANALYSIS_SR, col_s, templates, sens)
    else:
        notes, noises = analyze_columns(x, ANALYSIS_SR, col_s, templates, onsets, sens)
    log('notes : %d · percussions : %d' % (len(notes), len(noises)))

    lines, dropped, truncated = [], 0, False
    for pb in cands:
        lines, dropped, truncated = assemble(notes, noises, pb, loop, transpose)
        if not truncated:
            break
    ok, v_bpm, v_loop, pages = C.verify(lines)
    compact = C.compact_encode(v_bpm, v_loop, pages)
    with open(out, 'w', newline='\n') as f:
        f.write(compact + '\n')
    log('BPM projet %d · %d page(s) · checksum %s' % (v_bpm, len(pages), 'OK' if ok else 'INCORRECT'))
    if dropped:
        log('%d note(s) pulse perdue(s) (deux formes pulse sur la même cellule)' % dropped)
    if truncated:
        log('ATTENTION : plus de %d pages, morceau coupé (essayez --bpm plus bas)' % C.MAX_PAGES)
    if write_lines:
        lp = os.path.splitext(out)[0] + '_lignes.txt'
        with open(lp, 'w', newline='\n') as f:
            f.write('\n'.join(lines) + '\n')
        log('ancien format : %s' % lp)
    if preview:
        if not sounds:
            log('Attention : --preview nécessite --sounds, aperçu ignoré')
        else:
            data, dur, warn = C.render_code(lines, sounds, 1, 'wav' if preview.lower().endswith('.wav') else 'mp3')
            with open(preview, 'wb') as f:
                f.write(data)
            log('aperçu : %s (%.1f s)' % (preview, dur))
    if midi_out:
        mid, _, _, _ = C.code_to_midifile(lines)
        mid.save(midi_out)
        log('midi : %s' % midi_out)
    log('écrit : %s (%d caractères, format compact)' % (out, len(compact) + 1))
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('audio', help='fichier .wav ou .mp3')
    ap.add_argument('-o', '--out', required=True, help='code .txt (format compact)')
    ap.add_argument('--sounds', default=None, help='dossier des échantillons s0-1.wav … noise16.wav')
    ap.add_argument('--bpm', type=int, default=None, help='forcer le BPM du projet (100..400)')
    ap.add_argument('--cpq', type=int, default=None, choices=[1, 2, 4, 8, 16],
                    help='colonnes par noire (4 = double-croche)')
    ap.add_argument('--loop', type=int, default=0, choices=[0, 1])
    ap.add_argument('--sensitivity', type=float, default=1.0, help='0.5..2 (défaut 1)')
    ap.add_argument('--transpose', type=int, default=0, help='demi-tons')
    ap.add_argument('--lines', action='store_true', help='écrire aussi le code au format ancien')
    ap.add_argument('--preview', default=None, help='rendu WAV/MP3 de contrôle (nécessite --sounds)')
    ap.add_argument('--midi', default=None, help='export MIDI de contrôle')
    ap.add_argument('--approx', action='store_true',
                    help="mode approximation : suit n'importe quel son (voix…) en "
                         'spammant des notes colonne par colonne')
    args = ap.parse_args()
    transcribe(args.audio, args.out, args.sounds, args.bpm, args.cpq, args.loop,
               _clip(args.sensitivity, 0.3, 3.0), args.transpose, args.lines,
               args.preview, args.midi, args.approx)


if __name__ == '__main__':
    main()

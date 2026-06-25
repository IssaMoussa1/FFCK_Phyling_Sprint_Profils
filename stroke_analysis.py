"""Signal loading, stroke detection and analysis helpers."""

import hashlib
import os
import pickle

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, find_peaks

from config import CACHE_DIR, DATA_DIR
from phyling_client import fetch_csv_from_api


FC_SMOOTH = 3.0
MIN_DIST_S = 0.25
MIN_PEAK_H = 0.2
N_NORM = 200
FS = 100


def cache_path(fname, fc, min_d, min_h, cache_dir=CACHE_DIR):
    """Chemin du fichier cache .pkl pour une combinaison fichier+paramètres."""
    key = '{}|{:.3f}|{:.3f}|{:.3f}'.format(fname, fc, min_d, min_h)
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    return os.path.join(cache_dir, f"{os.path.splitext(fname)[0]}_{h}.pkl")


def load_with_cache(fname, fc, min_d, min_h, flip=False, api_key="",
                    data_dir=DATA_DIR, cache_dir=CACHE_DIR):
    """
    Charge et détecte les coups pour fname.
    - Si un cache valide existe, le relit.
    - Sinon, appelle load_and_detect et sauvegarde le résultat.
    """
    flip_suffix = '_flip' if flip else ''
    cp = cache_path(fname + flip_suffix, fc, min_d, min_h, cache_dir)

    if os.path.exists(cp):
        try:
            with open(cp, 'rb') as f:
                data = pickle.load(f)
            return data['strokes'], data['raw']
        except Exception:
            pass

    strokes, raw = load_and_detect(
        fname,
        fc=fc,
        min_d=min_d,
        min_h=min_h,
        flip=flip,
        api_key=api_key,
        data_dir=data_dir,
    )

    try:
        with open(cp, 'wb') as f:
            pickle.dump({'strokes': strokes, 'raw': raw,
                         'fc': fc, 'min_d': min_d, 'min_h': min_h}, f)
    except Exception:
        pass

    return strokes, raw


def load_and_detect(fname, fc=FC_SMOOTH, min_d=MIN_DIST_S, min_h=MIN_PEAK_H,
                    flip=False, api_key="", data_dir=DATA_DIR):
    """Load one Phyling CSV/API record and detect paddle strokes."""
    if ':' in str(fname):
        rec_id, sel_id = str(fname).split(':', 1)
        df = fetch_csv_from_api(api_key=api_key, rec_id=int(rec_id), sel_id=sel_id)
        if df.empty:
            return [], {}
    else:
        local_path = os.path.join(data_dir, fname)
        if not os.path.exists(local_path):
            return [], {}
        df = pd.read_csv(local_path)

    df = df.sort_values('T').copy()

    if flip:
        for col in ['acc_x', 'acc_y', 'gyro_x', 'gyro_y', 'roll', 'pitch']:
            if col in df.columns:
                df[col] = -df[col]
        for c1, c2 in [('pic_acc', 'pic_down'), ('t_acc', 't_down')]:
            if c1 in df.columns and c2 in df.columns:
                df[c1], df[c2] = df[c2].copy(), df[c1].copy()
        if 'speed_gps' in df.columns and 'speed_i' in df.columns:
            df['speed_i'] = df['speed_gps']

    acc = df['acc_x'].values
    t = df['T'].values
    d = df['D'].values
    spd = df['speed'].values if 'speed' in df.columns else np.full(len(t), np.nan)

    b, a = butter(2, fc / (FS / 2), btype='low')
    sm = filtfilt(b, a, acc)
    peaks, _ = find_peaks(sm, height=min_h, distance=int(min_d * FS))
    troughs = [peaks[i] + np.argmin(sm[peaks[i]:peaks[i + 1]])
               for i in range(len(peaks) - 1)]

    strokes = []
    for i in range(len(troughs) - 1):
        i0, i1 = troughs[i], troughs[i + 1]
        sa, st_, s_d = acc[i0:i1], t[i0:i1], d[i0:i1]
        if len(sa) < 8:
            continue
        an = interp1d(np.linspace(0, 1, len(sa)), sa)(np.linspace(0, 1, N_NORM))
        auc_pos = float(np.trapezoid(np.clip(sa, 0, None), st_))
        auc_neg = float(np.trapezoid(np.clip(sa, None, 0), st_))
        auc_abs = float(np.trapezoid(np.abs(sa), st_))
        idx_pk = int(np.argmax(sa))
        rfd = float(sa[idx_pk] / max(st_[idx_pk] - st_[0], 1e-6)) if idx_pk > 0 else np.nan
        jerk = float(np.sqrt(np.mean((np.diff(sa) * FS) ** 2)))
        pos_pic = float(idx_pk / len(sa) * 100)
        above = np.where(sa >= sa[idx_pk] / 2)[0]
        fwhm = float(st_[above[-1]] - st_[above[0]]) if len(above) > 1 else np.nan
        sym = float(abs(auc_neg) / auc_pos) if auc_pos > 0 else np.nan
        strokes.append({
            'D_start': float(s_d[0]),
            'D_end': float(s_d[-1]),
            'duration': float(st_[-1] - st_[0]),
            'pic_acc': float(np.max(sa)),
            'pic_down': float(np.min(sa)),
            't_acc_frac': float(np.sum(sa > 0)) / len(sa),
            'd_stroke': float(s_d[-1] - s_d[0]),
            'speed_moy': float(np.nanmean(spd[i0:i1])),
            'auc_pos': auc_pos,
            'auc_neg': auc_neg,
            'auc_abs': auc_abs,
            'rfd': rfd,
            'jerk_rms': jerk,
            'pos_pic_pct': pos_pic,
            'fwhm_s': fwhm,
            'sym_ratio': sym,
            'acc_norm': an.tolist(),
        })

    raw = {'T': t.tolist(), 'acc_x': acc.tolist(), 'D': d.tolist(), 'speed': spd.tolist()}
    return strokes, raw


def to_df(strokes):
    return pd.DataFrame([{k: v for k, v in s.items() if k != 'acc_norm'} for s in strokes])


def apply_filters(strokes, d_range, s_lo, s_hi):
    if not strokes:
        return []
    d0 = min(s['D_start'] for s in strokes)
    out = [s for s in strokes
           if (d0 + d_range[0]) <= s['D_start'] <= (d0 + d_range[1])]
    lo, hi = max(0, s_lo - 1), min(len(out), s_hi)
    return out[lo:hi]


def get_mat(strokes):
    mat = np.vstack([s['acc_norm'] for s in strokes])
    return mat[np.argsort([s['D_start'] for s in strokes])]


def mean_sd(strokes):
    mat = get_mat(strokes)
    return mat.mean(0), mat.std(0)


def get_quarters(strokes):
    d0, d1 = min(s['D_start'] for s in strokes), max(s['D_end'] for s in strokes)
    step = (d1 - d0) / 4
    return [[s for s in strokes if d0 + q * step <= s['D_start'] < d0 + (q + 1) * step]
            for q in range(4)]


def rolling_mean(x, y, w=15):
    idx = np.argsort(x)
    return x[idx], pd.Series(y[idx]).rolling(w, center=True, min_periods=1).mean().values


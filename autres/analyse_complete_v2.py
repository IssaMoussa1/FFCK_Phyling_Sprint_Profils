"""
=============================================================================
ANALYSE COMPLÈTE DES COUPS DE PAGAIE – MÉTHODE DES CREUX LOCAUX
=============================================================================
Standard retenu : détection par creux locaux entre pics positifs de acc_x
Martin Noé : EXCLU (capteur inversé)

FIGURES PRODUITES :
  Fig 01 – Heatmap des coups normalisés par athlète
  Fig 02 – Enveloppe complète (small multiples)
  Fig 03 – Enveloppe complète (tous athlètes superposés)
  Fig 04 – Comparaison des profils moyens inter-athlètes
  Fig 05 – Profils par quart de course (small multiples)
  Fig 06 – Pattern moyen avec AUC colorées (vert/rouge)
  Fig 07 – Tous les coups individuels + profil moyen (small multiples)
  Fig 08 – Profil moyen par quart de course (small multiples)
  Fig 09 – Scatter AUC+ vs |AUC-| avec centroïdes et ellipses
  Fig 10 – Scatter AUC+ vs Ratio symétrie
  Fig 11 – Dendrogrammes de similarité (4 métriques)
  Fig 12 – Matrices de similarité (corrélation + métriques scalaires)
  Fig 13 – Tableau métriques de base
  Fig 14 – Tableau métriques avancées

  CSV    – recap_metriques_complet.csv

MÉTRIQUES :
  Base    : AUC+, AUC-, AUC_abs, pic_acc, pic_down, durée, dist/coup, %propulsion
  Avancées: RFD (Rate of Force Development), Jerk RMS, Position du pic (%),
            FWHM (largeur pic), Ratio symétrie, CV AUC+ (consistance)
=============================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.transforms as transforms
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Ellipse, Patch
from scipy.signal import butter, filtfilt, find_peaks
from scipy.interpolate import interp1d
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION  ← modifier uniquement ici
# =============================================================================
DATA_DIR = r'C:\Users\imoussa\OneDrive - FFCK\Documents\Application\FFCK_Sprint_Profil'
OUT_DIR  = r'C:\Users\imoussa\OneDrive - FFCK\Documents\Application\FFCK_Sprint_Profil\outputs'

ATHLETES = {
    'Bavenkoff Viktor':   'bavenkoff_viktor-20260218_024153-sel_250.csv',
    'Gilhard Tom':        'gilhard_tom-20260215_101526-sel_250.csv',
    # 'Martin Noe':       EXCLU – capteur inversé
    'Polet Theophile':    'polet_theophile-20260215_103845-sel_250.csv',
    'Siabas Simon':       'siabas_simon_anatole-20260215_101514-sel_250.csv',
    'Zappaterra Clement': 'zappaterra_clement-20260215_111129-sel_250.csv',
    'Zoualegh Nathan':    'zoualegh_nathan-20260215_084430-sel_250.csv',
    'Zwiller Tao':        'zwiller_tao-20260215_102746-sel_250.csv',
}

# Paramètres détection creux (ne pas modifier sauf si résultats aberrants)
FC_SMOOTH  = 3.0   # Hz  – lissage pour éviter faux pics
MIN_DIST_S = 0.25  # s   – distance minimale entre deux pics
MIN_PEAK_H = 0.2   # m/s² – hauteur minimale d'un pic valide

N_NORM  = 200      # points de normalisation par coup
FS      = 100      # Hz

AFFICHER = True    # True = ouvre les fenêtres interactives
SAUVER   = True    # True = enregistre les PNG dans OUT_DIR


# =============================================================================
# SETUP
# =============================================================================
os.makedirs(OUT_DIR, exist_ok=True)

COLORS = {n: plt.cm.tab10(i / len(ATHLETES)) for i, n in enumerate(ATHLETES)}
xn     = np.linspace(0, 100, N_NORM)
names  = list(ATHLETES.keys())

# Palette visuelle cohérente
C_POS  = '#2E7D32'   # vert  – propulsion
C_NEG  = '#C62828'   # rouge – freinage
C_MEAN = '#1565C0'   # bleu  – profil moyen
C_BG   = '#FAFAFA'   # fond

QUART_COLORS = ['#2196F3', '#66BB6A', '#FF9800', '#EF5350']
QUART_LABELS = ['1er quart (0–25%)', '2ème quart (25–50%)',
                '3ème quart (50–75%)', '4ème quart (75–100%)']


def savefig(fig, fname):
    if SAUVER:
        fig.savefig(os.path.join(OUT_DIR, fname), dpi=150, bbox_inches='tight')
        print(f'    → Sauvé : {fname}')
    if AFFICHER:
        plt.show()
    plt.close(fig)


# =============================================================================
# DÉTECTION & CALCUL DES MÉTRIQUES
# =============================================================================

def load(fname):
    return pd.read_csv(os.path.join(DATA_DIR, fname))


def detect_trough(df, fc=FC_SMOOTH, min_d=MIN_DIST_S, min_h=MIN_PEAK_H):
    """
    Détecte les coups en plaçant les frontières aux CREUX LOCAUX
    entre les pics positifs de acc_x.

    Pour chaque coup [creux_i → creux_{i+1}] calcule :
      Métriques de base :
        auc_pos     Impulsion de propulsion           (m/s)
        auc_neg     Impulsion de freinage             (m/s)
        auc_abs     Effort mécanique total            (m/s)
        pic_acc     Pic d'accélération                (m/s²)
        pic_down    Pic de décélération               (m/s²)
        t_acc_frac  Part de propulsion dans le cycle  (0–1)
        d_stroke    Distance par coup                 (m)
        duration    Durée du coup                     (s)

      Métriques avancées :
        rfd         Rate of Force Development         (m/s³)
                    → explosivité du catch
        jerk_rms    Jerk RMS (dérivée de acc_x)       (m/s³)
                    → régularité / fluidité du coup
        pos_pic_pct Position relative du pic (%)
                    → timing de la propulsion dans le cycle
        fwhm_s      Largeur du pic à mi-hauteur       (s)
                    → durée de la phase de propulsion intense
        sym_ratio   |AUC-| / AUC+
                    → équilibre propulsion/freinage (idéal < 0.7)
    """
    df   = df.sort_values('T').copy()
    acc  = df['acc_x'].values
    t    = df['T'].values
    D    = df['D'].values
    spd  = df['speed'].values if 'speed' in df.columns else np.full(len(t), np.nan)

    # Lissage léger
    b, a = butter(2, fc / (FS / 2), btype='low')
    sm   = filtfilt(b, a, acc)

    # Détection des pics positifs
    peaks, _ = find_peaks(sm, height=min_h, distance=int(min_d * FS))

    # Creux entre chaque paire de pics consécutifs
    troughs = []
    for i in range(len(peaks) - 1):
        seg = sm[peaks[i]:peaks[i+1]]
        troughs.append(peaks[i] + np.argmin(seg))

    strokes = []
    for i in range(len(troughs) - 1):
        i0, i1 = troughs[i], troughs[i+1]
        sa, st, sD = acc[i0:i1], t[i0:i1], D[i0:i1]
        if len(sa) < 8:
            continue

        # Signal normalisé
        acc_norm = interp1d(np.linspace(0, 1, len(sa)), sa)(np.linspace(0, 1, N_NORM))

        # ── Métriques de base ─────────────────────────────────────────────
        auc_pos = float(np.trapezoid(np.clip(sa, 0, None), st))
        auc_neg = float(np.trapezoid(np.clip(sa, None, 0), st))
        auc_abs = float(np.trapezoid(np.abs(sa), st))

        # ── Métriques avancées ────────────────────────────────────────────
        idx_pk  = int(np.argmax(sa))

        # RFD : pente de montée jusqu'au pic
        if idx_pk > 0:
            rfd = float(sa[idx_pk] / max(st[idx_pk] - st[0], 1e-6))
        else:
            rfd = np.nan

        # Jerk RMS (dérivée de acc_x × FS)
        jerk_rms = float(np.sqrt(np.mean((np.diff(sa) * FS) ** 2)))

        # Position relative du pic positif dans le cycle
        pos_pic_pct = float(idx_pk / len(sa) * 100)

        # FWHM : largeur du pic à mi-hauteur
        half  = sa[idx_pk] / 2
        above = np.where(sa >= half)[0]
        fwhm_s = float(st[above[-1]] - st[above[0]]) if len(above) > 1 else np.nan

        # Ratio symétrie freinage/propulsion
        sym_ratio = float(abs(auc_neg) / auc_pos) if auc_pos > 0 else np.nan

        strokes.append({
            # Position
            'D_start':      float(sD[0]),
            'D_end':        float(sD[-1]),
            # Temporel
            'duration':     float(st[-1] - st[0]),
            # Base
            'pic_acc':      float(np.max(sa)),
            'pic_down':     float(np.min(sa)),
            't_acc_frac':   float(np.sum(sa > 0)) / len(sa),
            'd_stroke':     float(sD[-1] - sD[0]),
            'speed_moy':    float(np.nanmean(spd[i0:i1])),
            'auc_pos':      auc_pos,
            'auc_neg':      auc_neg,
            'auc_abs':      auc_abs,
            # Avancées
            'rfd':          rfd,
            'jerk_rms':     jerk_rms,
            'pos_pic_pct':  pos_pic_pct,
            'fwhm_s':       fwhm_s,
            'sym_ratio':    sym_ratio,
            # Signal
            'acc_norm':     acc_norm,
        })
    return strokes


def to_df(strokes):
    return pd.DataFrame([{k: v for k, v in s.items() if k != 'acc_norm'}
                          for s in strokes])


def get_mat(strokes):
    mat = np.vstack([s['acc_norm'] for s in strokes])
    idx = np.argsort([s['D_start'] for s in strokes])
    return mat[idx]


def mean_sd(strokes):
    mat = get_mat(strokes)
    return mat.mean(axis=0), mat.std(axis=0)


def get_quarters(strokes):
    d0   = min(s['D_start'] for s in strokes)
    d1   = max(s['D_end']   for s in strokes)
    step = (d1 - d0) / 4
    return [[s for s in strokes if d0 + q*step <= s['D_start'] < d0 + (q+1)*step]
            for q in range(4)]


def rma(x, y, w=10):
    idx = np.argsort(x)
    xs, ys = x[idx], y[idx]
    return xs, pd.Series(ys).rolling(w, center=True, min_periods=1).mean().values


def confidence_ellipse(x, y, ax, n_std=1.5, **kwargs):
    """Ellipse de confiance à n_std écarts-types."""
    if len(x) < 3:
        return
    cov     = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    rx, ry  = np.sqrt(1 + pearson), np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=rx*2, height=ry*2, **kwargs)
    sx, sy  = np.sqrt(cov[0, 0]) * n_std, np.sqrt(cov[1, 1]) * n_std
    transf  = (transforms.Affine2D()
               .rotate_deg(45)
               .scale(sx, sy)
               .translate(np.mean(x), np.mean(y)))
    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)


# =============================================================================
# CHARGEMENT
# =============================================================================
print('=' * 65)
print('ANALYSE COUPS DE PAGAIE – Méthode des creux locaux')
print('Martin Noé : EXCLU (capteur inversé)')
print('=' * 65)

data = {}
for name, fname in ATHLETES.items():
    df      = load(fname)
    strokes = detect_trough(df)
    data[name] = strokes
    df_s    = to_df(strokes)
    print(f'  {name:25s}: {len(strokes):3d} coups | '
          f'AUC+={df_s["auc_pos"].mean():.4f} m/s | '
          f'RFD={df_s["rfd"].mean():.2f} | '
          f'Jerk={df_s["jerk_rms"].mean():.1f}')
print()

# Vecteur de métriques scalaires normalisé pour similarité
def feat_vector(name):
    df_s = to_df(data[name])
    return [
        df_s['auc_pos'].mean(),
        df_s['auc_neg'].abs().mean(),
        df_s['rfd'].mean() / 100,
        df_s['jerk_rms'].mean() / 1000,
        df_s['pos_pic_pct'].mean() / 100,
        df_s['sym_ratio'].mean(),
        df_s['duration'].mean(),
    ]

means_dict = {n: np.vstack([s['acc_norm'] for s in data[n]]).mean(0) for n in names}


# =============================================================================
# FIG 01 – Heatmap des coups normalisés
# =============================================================================
print('Fig 01 – Heatmap...')
fig, axes = plt.subplots(2, 4, figsize=(24, 10))
fig.patch.set_facecolor(C_BG)
fig.suptitle('Heatmap des coups de pagaie normalisés – méthode creux (250m)\n'
             'Chaque ligne = un coup trié du début vers la fin de course\n'
             'Bleu = décélération  |  Rouge = accélération',
             fontsize=13, fontweight='bold')

vmax = max(np.abs(get_mat(data[n])).max() for n in names)
for idx, name in enumerate(names):
    ax  = axes.flatten()[idx]
    mat = get_mat(data[name])
    im  = ax.imshow(mat, aspect='auto', origin='upper',
                    cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, label='acc_x (m/s²)', shrink=0.85)
    ax.set_title(f'{name}\n(n={len(mat)} coups)', fontsize=10,
                 fontweight='bold', color=COLORS[name])
    ax.set_xlabel('Cycle normalisé (%)', fontsize=8)
    ax.set_ylabel('Coup # (ordre distance)', fontsize=8)
    ax.set_xticks(np.linspace(0, N_NORM-1, 6))
    ax.set_xticklabels([f'{int(v)}%' for v in np.linspace(0, 100, 6)], fontsize=7)

plt.tight_layout()
savefig(fig, 'fig01_heatmap.png')


# =============================================================================
# FIG 02 – Enveloppe complète (small multiples)
# =============================================================================
print('Fig 02 – Enveloppe small multiples...')
fig, axes = plt.subplots(2, 4, figsize=(24, 11), sharey=False)
fig.patch.set_facecolor(C_BG)
fig.suptitle('Enveloppe complète des coups par athlète – méthode creux (250m)\n'
             'Trait fin = coup individuel (rouge→vert selon distance)  '
             '|  Trait épais = profil moyen  |  Ombrage = ±1σ',
             fontsize=13, fontweight='bold')

for idx, name in enumerate(names):
    ax      = axes.flatten()[idx]
    strokes = data[name]
    mat     = get_mat(strokes)
    mu, sd  = mat.mean(0), mat.std(0)
    c       = COLORS[name]

    d_starts = np.sort([s['D_start'] for s in strokes])
    d_norm   = (d_starts - d_starts.min()) / (d_starts.max() - d_starts.min() + 1e-9)
    for acc_i, dr in zip(mat, d_norm):
        ax.plot(xn, acc_i, color=plt.cm.RdYlGn(dr), lw=0.5, alpha=0.12, zorder=1)

    ax.fill_between(xn, mat.min(0), mat.max(0), alpha=0.04, color=c, zorder=1)
    ax.fill_between(xn, mu - sd, mu + sd, alpha=0.22, color=c, zorder=2)
    ax.plot(xn, mu, color=c, lw=3, zorder=3, label=f'Moy. (n={len(strokes)})')
    ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.4)
    ax.scatter(xn[np.argmax(mu)], mu.max(), color=C_POS, s=80, zorder=5, marker='^')
    ax.scatter(xn[np.argmin(mu)], mu.min(), color=C_NEG, s=80, zorder=5, marker='v')
    ax.set_title(name, fontsize=10, fontweight='bold', color=c)
    ax.set_xlabel('Cycle normalisé (%)', fontsize=8)
    ax.set_ylabel('acc_x (m/s²)', fontsize=8)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.2)

sm = ScalarMappable(cmap='RdYlGn', norm=Normalize(0, 250))
sm.set_array([])
fig.colorbar(sm, ax=axes, label='Distance relative (m)', shrink=0.4, pad=0.02)
plt.tight_layout()
savefig(fig, 'fig02_enveloppe_small_multiples.png')


# =============================================================================
# FIG 03 – Enveloppe complète (tous superposés)
# =============================================================================
print('Fig 03 – Enveloppe superposée...')
fig, ax = plt.subplots(figsize=(14, 7))
fig.patch.set_facecolor(C_BG)
for name in names:
    mat    = get_mat(data[name])
    mu, sd = mat.mean(0), mat.std(0)
    c      = COLORS[name]
    df_s   = to_df(data[name])
    ax.fill_between(xn, mat.min(0), mat.max(0), alpha=0.03, color=c)
    ax.fill_between(xn, mu - sd, mu + sd,       alpha=0.10, color=c)
    ax.plot(xn, mu, color=c, lw=2.5,
            label=f'{name.split()[-1]}  (AUC+={df_s["auc_pos"].mean():.4f})')

ax.axhline(0, color='k', lw=1, ls='--', alpha=0.4)
ax.set_xlabel('Cycle normalisé (%)  [0% = entrée pagaie]', fontsize=11)
ax.set_ylabel('acc_x (m/s²)', fontsize=11)
ax.set_title('Enveloppe complète – tous athlètes superposés (250m)\n'
             'Ombrage clair = étendue min/max  |  Trait = profil moyen  |  Ombrage sombre = ±1σ',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9, ncol=2, loc='lower right')
ax.grid(True, alpha=0.25)
plt.tight_layout()
savefig(fig, 'fig03_enveloppe_superposee.png')


# =============================================================================
# FIG 04 – Comparaison des profils moyens inter-athlètes
# =============================================================================
print('Fig 04 – Comparaison profils moyens...')
fig, ax = plt.subplots(figsize=(14, 7))
fig.patch.set_facecolor(C_BG)
for name in names:
    mu, sd = mean_sd(data[name])
    c      = COLORS[name]
    df_s   = to_df(data[name])
    ax.fill_between(xn, mu - sd, mu + sd, alpha=0.10, color=c)
    ax.plot(xn, mu, color=c, lw=2.5,
            label=f'{name.split()[-1]}  (AUC+={df_s["auc_pos"].mean():.4f} | '
                  f'RFD={df_s["rfd"].mean():.1f})')

ax.axhline(0, color='k', lw=1, ls='--', alpha=0.4)
ax.set_xlabel('Cycle normalisé (%)  [0% = entrée pagaie]', fontsize=11)
ax.set_ylabel('acc_x (m/s²)', fontsize=11)
ax.set_title('Comparaison des profils moyens de coup – méthode creux (250m)\n'
             'Ombrage = ±1σ', fontsize=12, fontweight='bold')
ax.legend(fontsize=9, ncol=2, loc='lower right')
ax.grid(True, alpha=0.25)
plt.tight_layout()
savefig(fig, 'fig04_comparaison_profils_moyens.png')


# =============================================================================
# FIG 05 – Profils par quart de course (small multiples)
# =============================================================================
print('Fig 05 – Profils par quart (small multiples)...')
fig, axes = plt.subplots(2, 4, figsize=(24, 11))
fig.patch.set_facecolor(C_BG)
fig.suptitle('Évolution du profil de coup par quart de course – méthode creux (250m)\n'
             'Bleu=début → Rouge=fin  |  Ombrage = ±1σ  |  AUC+ = impulsion propulsion',
             fontsize=13, fontweight='bold')

for idx, name in enumerate(names):
    ax       = axes.flatten()[idx]
    quarters = get_quarters(data[name])
    df_s     = to_df(data[name])
    d0       = df_s['D_start'].min()
    d_range  = df_s['D_end'].max() - d0

    for q_idx, (q_str, qc, ql) in enumerate(zip(quarters, QUART_COLORS, QUART_LABELS)):
        if not q_str:
            continue
        mat_q = np.vstack([s['acc_norm'] for s in q_str])
        mu_q  = mat_q.mean(0)
        sd_q  = mat_q.std(0)
        auc_q = np.mean([s['auc_pos'] for s in q_str])
        ax.fill_between(xn, mu_q - sd_q, mu_q + sd_q, alpha=0.12, color=qc)
        ax.plot(xn, mu_q, color=qc, lw=2.5,
                label=f'{ql}\nn={len(q_str)}, AUC+={auc_q:.4f}')

    ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.4)
    ax.set_title(name, fontsize=10, fontweight='bold', color=COLORS[name])
    ax.set_xlabel('Cycle normalisé (%)', fontsize=8)
    ax.set_ylabel('acc_x (m/s²)', fontsize=8)
    ax.legend(fontsize=6.5, loc='lower right')
    ax.grid(True, alpha=0.2)

plt.tight_layout()
savefig(fig, 'fig05_profils_quarts.png')


# =============================================================================
# FIG 06 – Pattern moyen avec AUC colorées
# =============================================================================
print('Fig 06 – Pattern AUC...')
fig, axes = plt.subplots(2, 4, figsize=(24, 11))
fig.patch.set_facecolor(C_BG)
fig.suptitle('Pattern moyen du coup avec AUC – méthode creux (250m)\n'
             'Vert = propulsion (AUC+)  |  Rouge = freinage (AUC-)',
             fontsize=13, fontweight='bold')

for idx, name in enumerate(names):
    ax      = axes.flatten()[idx]
    strokes = data[name]
    mu, sd  = mean_sd(strokes)
    df_s    = to_df(strokes)
    c       = COLORS[name]

    ax.fill_between(xn, mu - 2*sd, mu + 2*sd, alpha=0.06, color=c, zorder=1)
    ax.fill_between(xn, mu - sd,   mu + sd,   alpha=0.18, color=c, zorder=2)
    ax.fill_between(xn, mu, 0, where=(mu > 0), alpha=0.40, color=C_POS, zorder=3,
                    label=f'AUC+ = {df_s["auc_pos"].mean():.4f} m/s')
    ax.fill_between(xn, mu, 0, where=(mu < 0), alpha=0.40, color=C_NEG, zorder=3,
                    label=f'|AUC-| = {df_s["auc_neg"].abs().mean():.4f} m/s')
    ax.plot(xn, mu, color=c, lw=3, zorder=4)
    ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.4)
    ax.scatter(xn[np.argmax(mu)], mu.max(), color=C_POS, s=100, zorder=6, marker='^',
               label=f'Pic: {mu.max():.2f} m/s²')
    ax.scatter(xn[np.argmin(mu)], mu.min(), color=C_NEG, s=100, zorder=6, marker='v',
               label=f'Pic déc: {mu.min():.2f} m/s²')

    ratio = df_s['auc_pos'].mean() / df_s['auc_abs'].mean() * 100
    sym   = df_s['sym_ratio'].mean()
    ax.text(0.02, 0.98,
            f'AUC_abs = {df_s["auc_abs"].mean():.4f} m/s\n'
            f'Ratio prop. = {ratio:.1f}%\n'
            f'Symétrie = {sym:.3f}',
            transform=ax.transAxes, fontsize=7.5, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.75))

    ax.set_title(f'{name}\n(n={len(strokes)} coups)', fontsize=10,
                 fontweight='bold', color=c)
    ax.set_xlabel('Cycle normalisé (%)', fontsize=8)
    ax.set_ylabel('acc_x (m/s²)', fontsize=8)
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, alpha=0.2)

plt.tight_layout()
savefig(fig, 'fig06_pattern_auc.png')


# =============================================================================
# FIG 07 – Tous les coups individuels + profil moyen (small multiples)
# =============================================================================
print('Fig 07 – Coups individuels + moyen...')
fig, axes = plt.subplots(2, 4, figsize=(24, 11))
fig.patch.set_facecolor(C_BG)
fig.suptitle('Tous les coups individuels superposés + profil moyen – méthode creux (250m)\n'
             'Coups colorés rouge→vert selon la progression dans la course',
             fontsize=13, fontweight='bold')

for idx, name in enumerate(names):
    ax      = axes.flatten()[idx]
    strokes = data[name]
    mat     = get_mat(strokes)
    mu, sd  = mat.mean(0), mat.std(0)
    c       = COLORS[name]

    d_starts = np.sort([s['D_start'] for s in strokes])
    d_norm   = (d_starts - d_starts.min()) / (d_starts.max() - d_starts.min() + 1e-9)
    for acc_i, dr in zip(mat, d_norm):
        ax.plot(xn, acc_i, color=plt.cm.RdYlGn(dr), lw=0.6, alpha=0.15, zorder=1)

    ax.fill_between(xn, mu - sd, mu + sd, alpha=0.25, color=C_MEAN, zorder=2)
    ax.plot(xn, mu, color=C_MEAN, lw=3.5, zorder=3,
            label=f'Profil moyen (n={len(strokes)})')
    ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.4)
    ax.scatter(xn[np.argmax(mu)], mu.max(), color=C_POS, s=80, zorder=5, marker='^')
    ax.scatter(xn[np.argmin(mu)], mu.min(), color=C_NEG, s=80, zorder=5, marker='v')
    ax.set_title(name, fontsize=10, fontweight='bold', color=c)
    ax.set_xlabel('Cycle normalisé (%)', fontsize=8)
    ax.set_ylabel('acc_x (m/s²)', fontsize=8)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.2)

sm = ScalarMappable(cmap='RdYlGn', norm=Normalize(0, 250))
sm.set_array([])
fig.colorbar(sm, ax=axes, label='Distance relative (m)', shrink=0.4, pad=0.02)
plt.tight_layout()
savefig(fig, 'fig07_coups_individuels_moyen.png')


# =============================================================================
# FIG 08 – Profil moyen par quart de course (vue par individu)
# =============================================================================
print('Fig 08 – Quarts par individu...')
fig, axes = plt.subplots(2, 4, figsize=(24, 11))
fig.patch.set_facecolor(C_BG)
fig.suptitle('Profil moyen par quart de course par athlète – méthode creux (250m)\n'
             'Bleu=1er quart  |  Vert=2ème  |  Orange=3ème  |  Rouge=4ème',
             fontsize=13, fontweight='bold')

for idx, name in enumerate(names):
    ax       = axes.flatten()[idx]
    quarters = get_quarters(data[name])
    df_s     = to_df(data[name])
    d_range  = df_s['D_end'].max() - df_s['D_start'].min()

    for q_idx, (q_str, qc, ql) in enumerate(zip(quarters, QUART_COLORS, QUART_LABELS)):
        if not q_str:
            continue
        mat_q = np.vstack([s['acc_norm'] for s in q_str])
        mu_q  = mat_q.mean(0)
        sd_q  = mat_q.std(0)
        auc_q = np.mean([s['auc_pos'] for s in q_str])
        rfd_q = np.mean([s['rfd']     for s in q_str])
        ax.fill_between(xn, mu_q - sd_q, mu_q + sd_q, alpha=0.12, color=qc)
        ax.plot(xn, mu_q, color=qc, lw=2.5,
                label=f'{ql}\nn={len(q_str)}\nAUC+={auc_q:.4f}\nRFD={rfd_q:.1f}')

    ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.4)
    ax.set_title(name, fontsize=10, fontweight='bold', color=COLORS[name])
    ax.set_xlabel('Cycle normalisé (%)', fontsize=8)
    ax.set_ylabel('acc_x (m/s²)', fontsize=8)
    ax.legend(fontsize=5.5, loc='lower right')
    ax.grid(True, alpha=0.2)

plt.tight_layout()
savefig(fig, 'fig08_quarts_par_individu.png')


# =============================================================================
# FIG 09 – Scatter AUC+ vs |AUC-| avec centroïdes et ellipses
# =============================================================================
print('Fig 09 – Scatter AUC+ vs |AUC-|...')
fig, ax = plt.subplots(figsize=(13, 9))
fig.patch.set_facecolor(C_BG)

for name in names:
    df_s = to_df(data[name])
    c    = COLORS[name]
    x    = df_s['auc_pos'].values
    y    = df_s['auc_neg'].abs().values

    ax.scatter(x, y, alpha=0.18, s=20, color=c, zorder=2)
    confidence_ellipse(x, y, ax, n_std=1.5,
                       facecolor=c, alpha=0.07,
                       edgecolor=c, linewidth=1.5, linestyle='--', zorder=3)
    cx, cy = x.mean(), y.mean()
    ax.scatter(cx, cy, color=c, s=220, zorder=5,
               edgecolors='black', linewidths=1.5, marker='D',
               label=f'{name.split()[-1]}  (+{cx:.3f} / -{cy:.3f})')
    ax.annotate(name.split()[-1], (cx, cy),
                textcoords='offset points', xytext=(7, 5),
                fontsize=9, color=c, fontweight='bold')

lim = max(to_df(data[n])['auc_pos'].max() for n in names) * 1.05
ax.plot([0, lim], [0, lim], 'k--', lw=1, alpha=0.3, label='AUC+ = |AUC-|')
ax.set_xlabel('AUC+ (m/s) – Impulsion de propulsion par coup', fontsize=11)
ax.set_ylabel('|AUC-| (m/s) – Impulsion de freinage par coup', fontsize=11)
ax.set_title('Relation propulsion / freinage par coup de pagaie – méthode creux (250m)\n'
             'Chaque point = un coup  |  ♦ = centroïde athlète  |  Ellipse = ±1.5σ',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9, loc='upper left', ncol=2)
ax.grid(True, alpha=0.25)
plt.tight_layout()
savefig(fig, 'fig09_scatter_auc_pos_neg.png')


# =============================================================================
# FIG 10 – Scatter AUC+ vs Ratio symétrie
# =============================================================================
print('Fig 10 – Scatter AUC+ vs sym_ratio...')
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.patch.set_facecolor(C_BG)
fig.suptitle('Profil propulsif par athlète – méthode creux (250m)\n'
             'Chaque point = un coup  |  ♦ = centroïde  |  Ellipse = ±1.5σ',
             fontsize=13, fontweight='bold')

for ax, (metric_y, ylabel, title) in zip(axes, [
    ('sym_ratio', 'Ratio symétrie |AUC-|/AUC+\n(freinage relatif)',
     'AUC+ vs Ratio de symétrie\n(haut = beaucoup de freinage relatif)'),
    ('rfd',       'RFD (m/s³) – Rate of Force Development\n(explosivité du catch)',
     'AUC+ vs RFD\n(haut-droite = puissant ET explosif)'),
]):
    for name in names:
        df_s = to_df(data[name])
        c    = COLORS[name]
        x    = df_s['auc_pos'].values
        y    = df_s[metric_y].values
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]

        ax.scatter(x, y, alpha=0.18, s=20, color=c, zorder=2)
        confidence_ellipse(x, y, ax, n_std=1.5,
                           facecolor=c, alpha=0.07,
                           edgecolor=c, linewidth=1.5, linestyle='--', zorder=3)
        cx, cy = x.mean(), y.mean()
        ax.scatter(cx, cy, color=c, s=220, zorder=5,
                   edgecolors='black', linewidths=1.5, marker='D',
                   label=f'{name.split()[-1]}')
        ax.annotate(name.split()[-1], (cx, cy),
                    textcoords='offset points', xytext=(7, 5),
                    fontsize=9, color=c, fontweight='bold')

    ax.set_xlabel('AUC+ (m/s) – Impulsion de propulsion', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='best', ncol=2)
    ax.grid(True, alpha=0.25)

plt.tight_layout()
savefig(fig, 'fig10_scatter_auc_sym_rfd.png')


# =============================================================================
# FIG 11 – Dendrogrammes de similarité (4 métriques)
# =============================================================================
print('Fig 11 – Dendrogrammes...')
fig = plt.figure(figsize=(22, 16))
fig.patch.set_facecolor(C_BG)
fig.suptitle('Similarité entre athlètes – 4 métriques complémentaires\n'
             'Plus les branches se rejoignent bas, plus les athlètes sont similaires',
             fontsize=14, fontweight='bold')
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.55, wspace=0.35)

metric_configs = [
    ('Forme du coup (profil moyen normalisé)\nDistance euclidienne',
     lambda: pdist([means_dict[n] for n in names], metric='euclidean')),
    ('Métriques scalaires\n(AUC+, AUC-, RFD, jerk, pos_pic, sym, durée)',
     lambda: pdist([feat_vector(n) for n in names], metric='euclidean')),
    ('Corrélation des formes\n(1 − Pearson)',
     lambda: pdist([means_dict[n] for n in names], metric='correlation')),
    ('Timing & symétrie\n(pos_pic, %propulsion, sym_ratio)',
     lambda: pdist([[to_df(data[n])['pos_pic_pct'].mean(),
                     to_df(data[n])['t_acc_frac'].mean() * 100,
                     to_df(data[n])['sym_ratio'].mean()]
                    for n in names], metric='euclidean')),
]

for i, (title, dist_fn) in enumerate(metric_configs):
    ax   = fig.add_subplot(gs[i//2, i%2])
    dist = dist_fn()
    Z    = linkage(dist, method='ward')
    dendrogram(Z, labels=[n.split()[-1] for n in names], ax=ax,
               leaf_rotation=35, leaf_font_size=11,
               color_threshold=0.6 * max(Z[:, 2]))
    for lbl in ax.get_xticklabels():
        txt = lbl.get_text()
        for name in names:
            if name.split()[-1] == txt:
                lbl.set_color(COLORS[name])
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_ylabel('Distance (Ward)', fontsize=9)
    ax.grid(axis='y', alpha=0.25)

savefig(fig, 'fig11_dendrogrammes.png')


# =============================================================================
# FIG 12 – Matrices de similarité
# =============================================================================
print('Fig 12 – Matrices de similarité...')
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.patch.set_facecolor(C_BG)
fig.suptitle('Matrices de similarité inter-athlètes – méthode creux (250m)',
             fontsize=13, fontweight='bold')

# Matrice de corrélation des profils
n_ath   = len(names)
corr_mat = np.zeros((n_ath, n_ath))
for i, n1 in enumerate(names):
    for j, n2 in enumerate(names):
        r, _ = pearsonr(means_dict[n1], means_dict[n2])
        corr_mat[i, j] = r

im1 = axes[0].imshow(corr_mat, cmap='RdYlGn', vmin=-1, vmax=1)
short = [n.split()[-1] for n in names]
axes[0].set_xticks(range(n_ath)); axes[0].set_xticklabels(short, rotation=40, ha='right', fontsize=9)
axes[0].set_yticks(range(n_ath)); axes[0].set_yticklabels(short, fontsize=9)
for i in range(n_ath):
    for j in range(n_ath):
        axes[0].text(j, i, f'{corr_mat[i,j]:.2f}', ha='center', va='center',
                     fontsize=9, fontweight='bold',
                     color='white' if abs(corr_mat[i,j]) > 0.75 else 'black')
plt.colorbar(im1, ax=axes[0], label='r de Pearson', shrink=0.85)
axes[0].set_title('Corrélation des profils moyens\n(1 = forme identique)',
                  fontsize=11, fontweight='bold')

# Matrice similarité scalaires
feat_mat = np.array([feat_vector(n) for n in names])
dist_mat = squareform(pdist(feat_mat, metric='euclidean'))
sim_mat  = 1 - dist_mat / (dist_mat.max() + 1e-9)

im2 = axes[1].imshow(sim_mat, cmap='RdYlGn', vmin=0, vmax=1)
axes[1].set_xticks(range(n_ath)); axes[1].set_xticklabels(short, rotation=40, ha='right', fontsize=9)
axes[1].set_yticks(range(n_ath)); axes[1].set_yticklabels(short, fontsize=9)
for i in range(n_ath):
    for j in range(n_ath):
        axes[1].text(j, i, f'{sim_mat[i,j]:.2f}', ha='center', va='center',
                     fontsize=9, fontweight='bold',
                     color='white' if sim_mat[i,j] > 0.75 else 'black')
plt.colorbar(im2, ax=axes[1], label='Similarité (0=différent, 1=identique)', shrink=0.85)
axes[1].set_title('Similarité par métriques scalaires\n(AUC+, AUC-, RFD, jerk, pos_pic, sym, durée)',
                  fontsize=11, fontweight='bold')

plt.tight_layout()
savefig(fig, 'fig12_matrices_similarite.png')


# =============================================================================
# FIG 13 – Tableau métriques de base
# =============================================================================
print('Fig 13 – Tableau métriques de base...')
rows_base = []
for name in names:
    df_s = to_df(data[name])
    rows_base.append([
        name, len(df_s),
        f"{df_s['pic_acc'].mean():.2f} ± {df_s['pic_acc'].std():.2f}",
        f"{df_s['pic_down'].abs().mean():.2f} ± {df_s['pic_down'].abs().std():.2f}",
        f"{df_s['duration'].mean():.3f} ± {df_s['duration'].std():.3f}",
        f"{df_s['t_acc_frac'].mean()*100:.1f}%",
        f"{df_s['d_stroke'].mean():.2f} ± {df_s['d_stroke'].std():.2f}",
        f"{df_s['auc_pos'].mean():.4f} ± {df_s['auc_pos'].std():.4f}",
        f"{df_s['auc_neg'].abs().mean():.4f} ± {df_s['auc_neg'].abs().std():.4f}",
        f"{df_s['auc_pos'].mean() / df_s['auc_abs'].mean() * 100:.1f}%",
    ])

cols_base = ['Athlète', 'N coups', 'Pic acc\n(m/s²)', '|Pic déc|\n(m/s²)',
             'Durée\n(s)', '% Propulsion\n(cycle)', 'Dist/coup\n(m)',
             'AUC+\n(m/s)', '|AUC-|\n(m/s)', 'Ratio\nprop.']

fig, ax = plt.subplots(figsize=(22, 4.5))
fig.patch.set_facecolor(C_BG)
ax.axis('off')
tbl = ax.table(cellText=rows_base, colLabels=cols_base, cellLoc='center', loc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 2.1)
for j in range(len(cols_base)):
    tbl[(0,j)].set_facecolor('#1a4a6b')
    tbl[(0,j)].set_text_props(color='white', fontweight='bold')
for i in range(1, len(rows_base)+1):
    bg = '#f0f4f8' if i%2==0 else 'white'
    for j in range(len(cols_base)): tbl[(i,j)].set_facecolor(bg)
ax.set_title('Métriques de base – méthode creux locaux (250m)  |  Martin Noé exclu\n'
             'AUC = intégrale de acc_x (impulsion en m/s)  |  ± = écart-type',
             fontsize=11, fontweight='bold', pad=15)
plt.tight_layout()
savefig(fig, 'fig13_tableau_base.png')


# =============================================================================
# FIG 14 – Tableau métriques avancées
# =============================================================================
print('Fig 14 – Tableau métriques avancées...')
rows_adv = []
for name in names:
    df_s = to_df(data[name])
    rows_adv.append([
        name,
        f"{df_s['rfd'].mean():.2f} ± {df_s['rfd'].std():.2f}",
        f"{df_s['jerk_rms'].mean():.1f} ± {df_s['jerk_rms'].std():.1f}",
        f"{df_s['pos_pic_pct'].mean():.1f}% ± {df_s['pos_pic_pct'].std():.1f}%",
        f"{df_s['fwhm_s'].mean()*1000:.0f} ± {df_s['fwhm_s'].std()*1000:.0f} ms",
        f"{df_s['sym_ratio'].mean():.3f} ± {df_s['sym_ratio'].std():.3f}",
        f"{df_s['auc_pos'].std() / df_s['auc_pos'].mean() * 100:.1f}%",
        f"{df_s['jerk_rms'].std() / df_s['jerk_rms'].mean() * 100:.1f}%",
    ])

cols_adv = ['Athlète',
            'RFD (m/s³)\nExplosivité catch',
            'Jerk RMS (m/s³)\nFluidité coup',
            'Position pic (%)\nTiming propulsion',
            'FWHM (ms)\nDurée propulsion intense',
            'Ratio symétrie\n|AUC-|/AUC+',
            'CV AUC+ (%)\nConsistance impulsion',
            'CV Jerk (%)\nConsistance fluidité']

fig, ax = plt.subplots(figsize=(24, 4.5))
fig.patch.set_facecolor(C_BG)
ax.axis('off')
tbl = ax.table(cellText=rows_adv, colLabels=cols_adv, cellLoc='center', loc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1, 2.1)
for j in range(len(cols_adv)):
    tbl[(0,j)].set_facecolor('#1a4a6b')
    tbl[(0,j)].set_text_props(color='white', fontweight='bold')
for i in range(1, len(rows_adv)+1):
    bg = '#f0f4f8' if i%2==0 else 'white'
    for j in range(len(cols_adv)): tbl[(i,j)].set_facecolor(bg)
ax.set_title('Métriques avancées – méthode creux locaux (250m)  |  Martin Noé exclu\n'
             'RFD = explosivité du catch  |  Jerk = régularité  |  '
             'FWHM = largeur du pic  |  CV = consistance coup-à-coup  |  ± = écart-type',
             fontsize=11, fontweight='bold', pad=15)
plt.tight_layout()
savefig(fig, 'fig14_tableau_avancees.png')


# =============================================================================
# EXPORT CSV
# =============================================================================
all_rows = []
for name in names:
    df_s = to_df(data[name])
    df_s['athlete'] = name
    all_rows.append(df_s)

df_export = pd.concat(all_rows, ignore_index=True)
df_export.to_csv(os.path.join(OUT_DIR, 'recap_metriques_complet.csv'), index=False)
print('    → CSV exporté : recap_metriques_complet.csv')


# =============================================================================
# RÉSUMÉ FINAL
# =============================================================================
print()
print('=' * 65)
print(f'TERMINÉ – 14 figures + 1 CSV dans :\n  {OUT_DIR}')
print('=' * 65)
print()
print(f'{"Athlète":<25} {"AUC+":>8} {"RFD":>8} {"Jerk":>8} '
      f'{"Pos_pic":>9} {"Sym":>7} {"CV_AUC+":>8}')
print('  ' + '-' * 60)
for name in names:
    df_s  = to_df(data[name])
    cv    = df_s['auc_pos'].std() / df_s['auc_pos'].mean() * 100
    print(f'  {name:<23} '
          f'{df_s["auc_pos"].mean():>8.4f} '
          f'{df_s["rfd"].mean():>8.2f} '
          f'{df_s["jerk_rms"].mean():>8.1f} '
          f'{df_s["pos_pic_pct"].mean():>8.1f}% '
          f'{df_s["sym_ratio"].mean():>7.3f} '
          f'{cv:>7.1f}%')
print()
print('Légende :')
print('  AUC+    = impulsion de propulsion par coup (m/s)')
print('  RFD     = rate of force development – explosivité (m/s³)')
print('  Jerk    = régularité du coup – fluidité (m/s³)')
print('  Pos_pic = timing du pic de propulsion dans le cycle (%)')
print('  Sym     = ratio |AUC-|/AUC+ – équilibre propulsion/freinage')
print('  CV_AUC+ = consistance coup-à-coup (%)')
print()
print('NOTE : ces fonctions sont prêtes à être intégrées dans Streamlit.')

"""
=============================================================================
ANALYSE COUPS DE PAGAIE – MÉTHODE DES CREUX LOCAUX (standard)
=============================================================================
Martin Noé exclu (capteur inversé).
Toutes les métriques recalculées à partir de la détection par creux locaux.

FIGURES :
  Fig 1  – Heatmap des coups normalisés par athlète
  Fig 2  – Enveloppe complète (small multiples, un par athlète)
  Fig 3  – Enveloppe complète (tous athlètes superposés)
  Fig 4  – Comparaison des profils moyens inter-athlètes
  Fig 5  – Profil par quart de course (par individu, small multiples)
  Fig 6  – Pattern moyen avec AUC (vert=propulsion, rouge=freinage)
  Fig 7  – Profil normalisé : tous les coups + moyen (small multiples)
  Fig 8  – Profil moyen par quart de course (small multiples)
  Fig 9  – Tableau récapitulatif des métriques
=============================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch
from scipy.signal import butter, filtfilt, find_peaks
from scipy.interpolate import interp1d
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

# Paramètres détection creux
FC_SMOOTH   = 3.0   # Hz – lissage pour éviter faux pics
MIN_DIST_S  = 0.25  # s  – distance minimale entre pics
MIN_PEAK_H  = 0.2   # m/s² – hauteur minimale d'un pic

N_NORM = 200        # points de normalisation
FS     = 100        # Hz

AFFICHER = True     # fenêtres interactives
SAUVER   = True     # enregistre PNG

# =============================================================================
# SETUP
# =============================================================================
os.makedirs(OUT_DIR, exist_ok=True)
COLORS = {n: plt.cm.tab10(i / len(ATHLETES)) for i, n in enumerate(ATHLETES)}
xn     = np.linspace(0, 100, N_NORM)

# Palettes style Phyling
C_POS  = '#2E7D32'
C_NEG  = '#C62828'
C_MEAN = '#1565C0'
C_BG   = '#FAFAFA'

QUART_COLORS  = ['#2196F3', '#66BB6A', '#FF9800', '#EF5350']
QUART_LABELS  = ['1er quart', '2ème quart', '3ème quart', '4ème quart']

def savefig(fig, fname):
    if SAUVER:
        path = os.path.join(OUT_DIR, fname)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f'  Sauvé : {fname}')
    if AFFICHER:
        plt.show()
    plt.close(fig)

# =============================================================================
# FONCTIONS CORE
# =============================================================================

def load(fname):
    return pd.read_csv(os.path.join(DATA_DIR, fname))

def detect_trough(df, fc=FC_SMOOTH, min_d=MIN_DIST_S, min_h=MIN_PEAK_H):
    """
    Détecte les coups de pagaie en plaçant les frontières aux CREUX LOCAUX
    entre les pics positifs de acc_x.
    Retourne une liste de dicts avec signal brut + normalisé + métriques.
    """
    df   = df.sort_values('T').copy()
    acc  = df['acc_x'].values
    t    = df['T'].values
    D    = df['D'].values
    spd  = df['speed'].values if 'speed' in df.columns else np.full(len(t), np.nan)

    # Lissage léger
    b, a = butter(2, fc / (FS / 2), btype='low')
    sm   = filtfilt(b, a, acc)

    # Pics positifs
    peaks, _ = find_peaks(sm, height=min_h, distance=int(min_d * FS))

    # Creux entre chaque paire de pics consécutifs
    troughs = []
    for i in range(len(peaks) - 1):
        seg = sm[peaks[i]:peaks[i+1]]
        troughs.append(peaks[i] + np.argmin(seg))

    # Construire les coups [creux_i → creux_{i+1}]
    strokes = []
    for i in range(len(troughs) - 1):
        i0, i1 = troughs[i], troughs[i+1]
        sa, st, sD, ss = acc[i0:i1], t[i0:i1], D[i0:i1], spd[i0:i1]
        if len(sa) < 5:
            continue

        acc_norm = interp1d(np.linspace(0, 1, len(sa)), sa)(np.linspace(0, 1, N_NORM))

        strokes.append({
            # Position
            'D_start':    float(sD[0]),
            'D_end':      float(sD[-1]),
            # Temporel
            'duration':   float(st[-1] - st[0]),
            # Cinématique
            'pic_acc':    float(np.max(sa)),
            'pic_down':   float(np.min(sa)),
            't_acc_frac': float(np.sum(sa > 0)) / len(sa),
            'd_stroke':   float(sD[-1] - sD[0]),
            'speed_moy':  float(np.nanmean(ss)),
            # AUC (impulsion en m/s)
            'auc_pos':    float(np.trapezoid(np.clip(sa, 0, None), st)),
            'auc_neg':    float(np.trapezoid(np.clip(sa, None, 0), st)),
            'auc_abs':    float(np.trapezoid(np.abs(sa), st)),
            # Signal normalisé
            'acc_norm':   acc_norm,
        })
    return strokes

def to_df(strokes):
    return pd.DataFrame([{k: v for k, v in s.items() if k != 'acc_norm'} for s in strokes])

def get_mat(strokes):
    """Matrice (n_coups × N_NORM) triée par distance."""
    mat = np.vstack([s['acc_norm'] for s in strokes])
    idx = np.argsort([s['D_start'] for s in strokes])
    return mat[idx]

def mean_sd(strokes):
    mat = get_mat(strokes)
    return mat.mean(axis=0), mat.std(axis=0)

def get_quarters(strokes):
    """Découpe les coups en 4 quarts égaux selon la distance."""
    d0 = min(s['D_start'] for s in strokes)
    d1 = max(s['D_end']   for s in strokes)
    step = (d1 - d0) / 4
    quarters = []
    for q in range(4):
        lo, hi = d0 + q * step, d0 + (q + 1) * step
        seg = [s for s in strokes if lo <= s['D_start'] < hi]
        quarters.append(seg)
    return quarters

def rma(x, y, w=10):
    idx = np.argsort(x)
    xs, ys = x[idx], y[idx]
    return xs, pd.Series(ys).rolling(w, center=True, min_periods=1).mean().values

# =============================================================================
# CHARGEMENT
# =============================================================================
print('=' * 65)
print('Chargement – méthode des CREUX LOCAUX')
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
          f'Dur={df_s["duration"].mean():.3f}s')

N_ATH = len(ATHLETES)
names = list(ATHLETES.keys())
print()

# =============================================================================
# FIG 1 – Heatmap des coups normalisés par athlète
# =============================================================================
print('Fig 1 – Heatmap des coups normalisés...')
fig, axes = plt.subplots(2, 4, figsize=(24, 10))
fig.patch.set_facecolor(C_BG)
fig.suptitle('Heatmap des coups de pagaie normalisés par athlète – méthode creux (250m)\n'
             'Chaque ligne = un coup  |  Trié du début vers la fin de course\n'
             'Bleu = décélération  |  Rouge = accélération',
             fontsize=13, fontweight='bold')

vmax = max(np.abs(get_mat(data[n])).max() for n in names)

for idx, name in enumerate(names):
    ax  = axes.flatten()[idx]
    mat = get_mat(data[name])
    im  = ax.imshow(mat, aspect='auto', origin='upper',
                    cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, label='acc_x (m/s²)', shrink=0.85)
    ax.set_title(f'{name}\n(n={len(mat)} coups)', fontsize=10, fontweight='bold',
                 color=COLORS[name])
    ax.set_xlabel('Cycle normalisé (%)', fontsize=8)
    ax.set_ylabel('Coup # (ordre distance)', fontsize=8)
    ax.set_xticks(np.linspace(0, N_NORM-1, 6))
    ax.set_xticklabels([f'{int(v)}%' for v in np.linspace(0, 100, 6)], fontsize=7)

plt.tight_layout()
savefig(fig, 'fig1_heatmap_creux.png')

# =============================================================================
# FIG 2 – Enveloppe complète (small multiples, 1 par athlète)
# =============================================================================
print('Fig 2 – Enveloppe complète (small multiples)...')
fig, axes = plt.subplots(2, 4, figsize=(24, 11), sharey=False)
fig.patch.set_facecolor(C_BG)
fig.suptitle('Enveloppe complète des coups de pagaie par athlète – méthode creux (250m)\n'
             'Trait fin = coup individuel (coloré par distance)  |  Trait épais = profil moyen  |  Ombrage = ±1σ',
             fontsize=13, fontweight='bold')

for idx, name in enumerate(names):
    ax      = axes.flatten()[idx]
    strokes = data[name]
    mat     = get_mat(strokes)
    mu, sd  = mat.mean(axis=0), mat.std(axis=0)
    c       = COLORS[name]

    # Coups individuels colorés par distance
    d_starts = np.array([s['D_start'] for s in strokes])
    d_norm   = (d_starts - d_starts.min()) / (d_starts.max() - d_starts.min() + 1e-9)
    for acc_i, dr in zip(mat, d_norm):
        ax.plot(xn, acc_i, color=plt.cm.RdYlGn(dr), lw=0.5, alpha=0.12, zorder=1)

    # Enveloppe min/max
    ax.fill_between(xn, mat.min(axis=0), mat.max(axis=0), alpha=0.05, color=c, zorder=1)
    # ±1σ
    ax.fill_between(xn, mu - sd, mu + sd, alpha=0.22, color=c, zorder=2)
    # Profil moyen
    ax.plot(xn, mu, color=c, lw=3, zorder=3,
            label=f'Moy. (n={len(strokes)})')
    ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.4, zorder=2)
    ax.scatter(xn[np.argmax(mu)], mu.max(), color=C_POS, s=80, zorder=5, marker='^')
    ax.scatter(xn[np.argmin(mu)], mu.min(), color=C_NEG, s=80, zorder=5, marker='v')

    ax.set_title(name, fontsize=10, fontweight='bold', color=c)
    ax.set_xlabel('Cycle normalisé (%)', fontsize=8)
    ax.set_ylabel('acc_x (m/s²)', fontsize=8)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.2)

sm = ScalarMappable(cmap='RdYlGn', norm=Normalize(0, 250))
sm.set_array([])
fig.colorbar(sm, ax=axes, label='Distance relative (m)', shrink=0.4, pad=0.02, orientation='vertical')
plt.tight_layout()
savefig(fig, 'fig2_enveloppe_small_multiples.png')

# =============================================================================
# FIG 3 – Enveloppe complète (tous sur le même graphique)
# =============================================================================
print('Fig 3 – Enveloppe complète (superposée)...')
fig, ax = plt.subplots(figsize=(14, 7))
fig.patch.set_facecolor(C_BG)

for name in names:
    strokes  = data[name]
    mat      = get_mat(strokes)
    mu, sd   = mat.mean(axis=0), mat.std(axis=0)
    env_min, env_max = mat.min(axis=0), mat.max(axis=0)
    c        = COLORS[name]
    short    = name.split()[-1]

    ax.fill_between(xn, env_min, env_max, alpha=0.04, color=c)
    ax.fill_between(xn, mu - sd, mu + sd, alpha=0.10, color=c)
    ax.plot(xn, mu, color=c, lw=2.5, label=short)

ax.axhline(0, color='k', lw=1, ls='--', alpha=0.4)
ax.set_xlabel('Cycle normalisé (%)  [0% = entrée pagaie → 100% = entrée pagaie suivante]',
              fontsize=11)
ax.set_ylabel('acc_x (m/s²)', fontsize=11)
ax.set_title('Enveloppe complète des coups – tous athlètes superposés (250m)\n'
             'Ombrage léger = étendue min/max  |  Trait = profil moyen',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10, ncol=2, loc='lower right')
ax.grid(True, alpha=0.25)
plt.tight_layout()
savefig(fig, 'fig3_enveloppe_superposee.png')

# =============================================================================
# FIG 4 – Comparaison profils moyens inter-athlètes
# =============================================================================
print('Fig 4 – Comparaison profils moyens...')
fig, ax = plt.subplots(figsize=(14, 7))
fig.patch.set_facecolor(C_BG)

for name in names:
    mu, sd = mean_sd(data[name])
    c      = COLORS[name]
    short  = name.split()[-1]
    df_s   = to_df(data[name])

    ax.fill_between(xn, mu - sd, mu + sd, alpha=0.10, color=c)
    ax.plot(xn, mu, color=c, lw=2.5,
            label=f'{short}  (AUC+={df_s["auc_pos"].mean():.4f} m/s)')

ax.axhline(0, color='k', lw=1, ls='--', alpha=0.4)
ax.set_xlabel('Cycle normalisé (%)  [0% = entrée pagaie]', fontsize=11)
ax.set_ylabel('acc_x (m/s²)', fontsize=11)
ax.set_title('Comparaison des profils moyens de coup entre athlètes – méthode creux (250m)\n'
             'Ombrage = ±1σ  |  AUC+ = impulsion de propulsion par coup',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9, ncol=2, loc='lower right')
ax.grid(True, alpha=0.25)
plt.tight_layout()
savefig(fig, 'fig4_comparaison_profils_moyens.png')

# =============================================================================
# FIG 5 – Profil par quart de course (small multiples, 1 par athlète)
# =============================================================================
print('Fig 5 – Profils par quart de course (small multiples)...')
fig, axes = plt.subplots(2, 4, figsize=(24, 11), sharey=False)
fig.patch.set_facecolor(C_BG)
fig.suptitle('Évolution du profil de coup par quart de course – méthode creux (250m)\n'
             'Bleu=1er quart → Rouge=4ème quart  |  Ombrage = ±1σ',
             fontsize=13, fontweight='bold')

for idx, name in enumerate(names):
    ax      = axes.flatten()[idx]
    strokes = data[name]
    quarters = get_quarters(strokes)
    df_s    = to_df(strokes)
    d0      = df_s['D_start'].min()
    d_max   = df_s['D_end'].max() - d0

    for q_idx, (q_strokes, qc, ql) in enumerate(zip(quarters, QUART_COLORS, QUART_LABELS)):
        if not q_strokes:
            continue
        mat_q    = np.vstack([s['acc_norm'] for s in q_strokes])
        mu_q     = mat_q.mean(axis=0)
        sd_q     = mat_q.std(axis=0)
        auc_q    = np.mean([s['auc_pos'] for s in q_strokes])
        d_lo_q   = int(q_idx * d_max / 4)
        d_hi_q   = int((q_idx + 1) * d_max / 4)
        ax.fill_between(xn, mu_q - sd_q, mu_q + sd_q, alpha=0.12, color=qc)
        ax.plot(xn, mu_q, color=qc, lw=2.5,
                label=f'{ql} ({d_lo_q}–{d_hi_q}m)\nn={len(q_strokes)}, AUC+={auc_q:.4f}')

    ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.4)
    ax.set_title(name, fontsize=10, fontweight='bold', color=COLORS[name])
    ax.set_xlabel('Cycle normalisé (%)', fontsize=8)
    ax.set_ylabel('acc_x (m/s²)', fontsize=8)
    ax.legend(fontsize=6.5, loc='lower right')
    ax.grid(True, alpha=0.2)

plt.tight_layout()
savefig(fig, 'fig5_profils_quarts_small_multiples.png')

# =============================================================================
# FIG 6 – Pattern moyen avec AUC (vert=propulsion, rouge=freinage)
# =============================================================================
print('Fig 6 – Pattern moyen avec AUC colorées...')
fig, axes = plt.subplots(2, 4, figsize=(24, 11), sharey=False)
fig.patch.set_facecolor(C_BG)
fig.suptitle('Pattern moyen du coup de pagaie avec AUC – méthode creux (250m)\n'
             'Vert = phase de propulsion (AUC+)  |  Rouge = phase de freinage (AUC-)',
             fontsize=13, fontweight='bold')

for idx, name in enumerate(names):
    ax       = axes.flatten()[idx]
    strokes  = data[name]
    mu, sd   = mean_sd(strokes)
    df_s     = to_df(strokes)
    c        = COLORS[name]

    # Bandes de confiance
    ax.fill_between(xn, mu - 2*sd, mu + 2*sd, alpha=0.06, color=c, zorder=1)
    ax.fill_between(xn, mu - sd,   mu + sd,   alpha=0.18, color=c, zorder=2)

    # AUC colorées
    auc_pos_moy = df_s['auc_pos'].mean()
    auc_neg_moy = df_s['auc_neg'].abs().mean()
    auc_abs_moy = df_s['auc_abs'].mean()
    ratio       = auc_pos_moy / auc_abs_moy * 100

    ax.fill_between(xn, mu, 0, where=(mu > 0), alpha=0.40, color=C_POS, zorder=3,
                    label=f'AUC+ = {auc_pos_moy:.4f} m/s')
    ax.fill_between(xn, mu, 0, where=(mu < 0), alpha=0.40, color=C_NEG, zorder=3,
                    label=f'|AUC-| = {auc_neg_moy:.4f} m/s')

    # Profil moyen
    ax.plot(xn, mu, color=c, lw=3, zorder=4)
    ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.4, zorder=3)

    # Pics
    ax.scatter(xn[np.argmax(mu)], mu.max(), color=C_POS, s=100, zorder=6, marker='^',
               label=f'Pic acc: {mu.max():.2f} m/s²')
    ax.scatter(xn[np.argmin(mu)], mu.min(), color=C_NEG, s=100, zorder=6, marker='v',
               label=f'Pic déc: {mu.min():.2f} m/s²')

    # Annotation ratio
    ax.text(0.02, 0.98,
            f'AUC_abs = {auc_abs_moy:.4f} m/s\nRatio prop. = {ratio:.1f}%',
            transform=ax.transAxes, fontsize=7.5, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    ax.set_title(f'{name}\n(n={len(strokes)} coups)', fontsize=10, fontweight='bold', color=c)
    ax.set_xlabel('Cycle normalisé (%)  [0% = entrée pagaie]', fontsize=8)
    ax.set_ylabel('acc_x (m/s²)', fontsize=8)
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, alpha=0.2)

plt.tight_layout()
savefig(fig, 'fig6_pattern_auc_creux.png')

# =============================================================================
# FIG 7 – Profil normalisé : tous les coups + moyen (small multiples)
# =============================================================================
print('Fig 7 – Tous les coups + moyen (small multiples)...')
fig, axes = plt.subplots(2, 4, figsize=(24, 11), sharey=False)
fig.patch.set_facecolor(C_BG)
fig.suptitle('Tous les coups individuels superposés + profil moyen – méthode creux (250m)\n'
             'Coups colorés rouge→vert selon la progression dans la course',
             fontsize=13, fontweight='bold')

for idx, name in enumerate(names):
    ax       = axes.flatten()[idx]
    strokes  = data[name]
    mat      = get_mat(strokes)
    mu, sd   = mat.mean(axis=0), mat.std(axis=0)
    c        = COLORS[name]
    n        = len(strokes)

    # Coups individuels
    d_starts = np.sort([s['D_start'] for s in strokes])
    d_norm   = (d_starts - d_starts.min()) / (d_starts.max() - d_starts.min() + 1e-9)
    for acc_i, dr in zip(mat, d_norm):
        ax.plot(xn, acc_i, color=plt.cm.RdYlGn(dr), lw=0.6, alpha=0.15, zorder=1)

    # ±1σ + profil moyen
    ax.fill_between(xn, mu - sd, mu + sd, alpha=0.25, color=C_MEAN, zorder=2)
    ax.plot(xn, mu, color=C_MEAN, lw=3.5, zorder=3, label=f'Profil moyen (n={n})')
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
savefig(fig, 'fig7_tous_coups_moyen_small_multiples.png')

# =============================================================================
# FIG 8 – Profil moyen par quart de course (small multiples, axes unifiés)
# =============================================================================
print('Fig 8 – Profil moyen par quart (vue comparée par individu)...')
fig = plt.figure(figsize=(24, 14))
fig.patch.set_facecolor(C_BG)
fig.suptitle('Évolution du profil de coup par quart de course – méthode creux (250m)\n'
             'Panneau gauche = profils superposés  |  Panneau droit = différence vs 1er quart',
             fontsize=13, fontweight='bold')

gs_outer = gridspec.GridSpec(2, 4, figure=fig, hspace=0.5, wspace=0.35)

for idx, name in enumerate(names):
    ax = fig.add_subplot(gs_outer[idx // 4, idx % 4])
    strokes  = data[name]
    quarters = get_quarters(strokes)
    profiles_q = []

    for q_idx, (q_strokes, qc, ql) in enumerate(zip(quarters, QUART_COLORS, QUART_LABELS)):
        if not q_strokes:
            profiles_q.append(None)
            continue
        mat_q  = np.vstack([s['acc_norm'] for s in q_strokes])
        mu_q   = mat_q.mean(axis=0)
        sd_q   = mat_q.std(axis=0)
        auc_q  = np.mean([s['auc_pos'] for s in q_strokes])
        profiles_q.append(mu_q)

        ax.fill_between(xn, mu_q - sd_q, mu_q + sd_q, alpha=0.12, color=qc)
        ax.plot(xn, mu_q, color=qc, lw=2.5,
                label=f'{ql}\nn={len(q_strokes)}\nAUC+={auc_q:.4f}')

    ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.4)
    ax.set_title(name, fontsize=10, fontweight='bold', color=COLORS[name])
    ax.set_xlabel('Cycle normalisé (%)', fontsize=8)
    ax.set_ylabel('acc_x (m/s²)', fontsize=8)
    ax.legend(fontsize=6, loc='lower right')
    ax.grid(True, alpha=0.2)

savefig(fig, 'fig8_quarts_par_individu.png')

# =============================================================================
# FIG 9 – Tableau récapitulatif des métriques
# =============================================================================
print('Fig 9 – Tableau récapitulatif...')
rows = []
for name in names:
    df_s = to_df(data[name])
    rows.append([
        name,
        len(df_s),
        f"{df_s['pic_acc'].mean():.2f} ± {df_s['pic_acc'].std():.2f}",
        f"{df_s['pic_down'].abs().mean():.2f} ± {df_s['pic_down'].abs().std():.2f}",
        f"{df_s['duration'].mean():.3f} ± {df_s['duration'].std():.3f}",
        f"{df_s['t_acc_frac'].mean()*100:.1f}%",
        f"{df_s['d_stroke'].mean():.2f} ± {df_s['d_stroke'].std():.2f}",
        f"{df_s['auc_pos'].mean():.4f} ± {df_s['auc_pos'].std():.4f}",
        f"{df_s['auc_neg'].abs().mean():.4f} ± {df_s['auc_neg'].abs().std():.4f}",
        f"{df_s['auc_pos'].mean() / df_s['auc_abs'].mean() * 100:.1f}%",
    ])

cols = ['Athlète', 'N coups', 'Pic acc (m/s²)', '|Pic déc| (m/s²)',
        'Durée (s)', '% phase acc', 'Dist/coup (m)',
        'AUC+ (m/s)', '|AUC-| (m/s)', 'Ratio prop.']

fig, ax = plt.subplots(figsize=(20, 4.5))
fig.patch.set_facecolor(C_BG)
ax.axis('off')
tbl = ax.table(cellText=rows, colLabels=cols, cellLoc='center', loc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 2.0)
for j in range(len(cols)):
    tbl[(0, j)].set_facecolor('#1a4a6b')
    tbl[(0, j)].set_text_props(color='white', fontweight='bold')
for i in range(1, len(rows) + 1):
    bg = '#f0f4f8' if i % 2 == 0 else 'white'
    for j in range(len(cols)):
        tbl[(i, j)].set_facecolor(bg)
ax.set_title('Tableau récapitulatif – méthode creux locaux (250m)  |  Martin Noé exclu',
             fontsize=12, fontweight='bold', pad=20)
plt.tight_layout()
savefig(fig, 'fig9_tableau_recap_creux.png')

# Export CSV
df_recap = pd.DataFrame(rows, columns=cols)
df_recap.to_csv(os.path.join(OUT_DIR, 'recap_metriques_creux.csv'), index=False)
print('  CSV exporté : recap_metriques_creux.csv')

# =============================================================================
# RÉSUMÉ
# =============================================================================
print()
print('=' * 65)
print('TERMINÉ – 9 figures + 1 CSV produits dans :')
print(f'  {OUT_DIR}')
print('=' * 65)
print()
print('Métriques clés (AUC+ = impulsion de propulsion par coup) :')
print(f'  {"Athlète":<25} {"AUC+ moy":>12} {"Ratio":>8}')
print('  ' + '-' * 48)
for name in names:
    df_s = to_df(data[name])
    ratio = df_s['auc_pos'].mean() / df_s['auc_abs'].mean() * 100
    print(f'  {name:<25} {df_s["auc_pos"].mean():>10.4f}   {ratio:>6.1f}%')
print()
print('NOTE : Ces fonctions (detect_trough, to_df, get_quarters…)')
print('       sont réutilisables directement dans une app Streamlit.')

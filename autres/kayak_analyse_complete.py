"""
=============================================================================
ANALYSE COMPLÈTE DES COUPS DE PAGAIE – Maxi-Phyling Canoë-Kayak Sprint
=============================================================================
Auteur : généré avec Claude / Anthropic
Données : exports CSV Maxi-Phyling (100 Hz)

FIGURES PRODUITES :
─── SECTION 1 : Détection & profils individuels ────────────────────────────
  Fig 01 – Signal acc_x brut avec détection des coups (motif_id)        [par athlète]
  Fig 02 – Signal acc_x brut avec détection des coups (zero-crossing)   [par athlète]
  Fig 03 – Profils moyens individuels – méthode motif_id                [grille 2x4]
  Fig 04 – Profils moyens individuels – méthode zero-crossing           [grille 2x4]

─── SECTION 2 : Comparaison inter-athlètes ─────────────────────────────────
  Fig 05 – Superposition des profils moyens (2 méthodes)
  Fig 06 – Barplots des métriques comparées (pic acc, durée, dist/coup…)
  Fig 07 – Évolution des métriques sur la distance (250m)
  Fig 08 – Tableau récapitulatif des métriques

─── SECTION 3 : Analyse AUC (aire sous la courbe = impulsion) ──────────────
  Fig 09 – Profils moyens avec aires AUC+ / AUC- colorées               [grille 3x3]
  Fig 10 – Superposition profils + barplots AUC comparés
  Fig 11 – Scatter AUC+ vs |AUC-| par athlète
  Fig 12 – Tableau récapitulatif AUC

─── SECTION 4 : Analyse longitudinale 2000m (Zwiller Tao) ──────────────────
  Fig 13 – Heatmap des 794 coups normalisés au fil de la distance
  Fig 14 – Évolution des métriques sur 2000m
  Fig 15 – Profils moyens par quart de course (0-500 / 500-1000 / …)
=============================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
# !! Modifiez uniquement cette section !!

DATA_DIR = r'C:\Users\imoussa\OneDrive - FFCK\Documents\Application\FFCK_Sprint_Profil'
OUT_DIR  = r'C:\Users\imoussa\OneDrive - FFCK\Documents\Application\FFCK_Sprint_Profil\outputs'

ATHLETES_250 = {
    'Bavenkoff Viktor':   'bavenkoff_viktor-20260218_024153-sel_250.csv',
    'Gilhard Tom':        'gilhard_tom-20260215_101526-sel_250.csv',
    'Martin Noe':         'martin_noe-20260215_103255-sel_250.csv',
    'Polet Theophile':    'polet_theophile-20260215_103845-sel_250.csv',
    'Siabas Simon':       'siabas_simon_anatole-20260215_101514-sel_250.csv',
    'Zappaterra Clement': 'zappaterra_clement-20260215_111129-sel_250.csv',
    'Zoualegh Nathan':    'zoualegh_nathan-20260215_084430-sel_250.csv',
    'Zwiller Tao':        'zwiller_tao-20260215_102746-sel_250.csv',
}

ATHLETE_2000 = ('Zwiller Tao', 'zwiller_tao-20260215_102746-sel_2000.csv')

N_NORM  = 200   # points pour normalisation d'un coup
FS      = 100   # fréquence d'échantillonnage (Hz)

AFFICHER_FIGURES = True   # True = fenêtres interactives, False = uniquement sauvegarder
SAUVEGARDER      = True   # True = enregistre les PNG dans OUT_DIR

# ─────────────────────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)
COLORS  = {n: plt.cm.tab10(i / len(ATHLETES_250)) for i, n in enumerate(ATHLETES_250)}
x_norm  = np.linspace(0, 100, N_NORM)

def savefig(fig, fname):
    if SAUVEGARDER:
        fig.savefig(os.path.join(OUT_DIR, fname), dpi=150, bbox_inches='tight')
    if AFFICHER_FIGURES:
        plt.show()
    plt.close(fig)


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def load(fname):
    return pd.read_csv(os.path.join(DATA_DIR, fname))

def extract_strokes_motif(df, n_norm=N_NORM):
    """Segmentation via motif_id Phyling."""
    strokes = []
    for mid, grp in df.groupby('motif_id'):
        grp = grp.sort_values('T')
        acc = grp['acc_x'].values
        if len(acc) < 5:
            continue
        t = grp['T'].values
        acc_norm = interp1d(np.linspace(0,1,len(acc)), acc)(np.linspace(0,1,n_norm))
        d0, d1   = float(grp['D'].iloc[0]), float(grp['D'].iloc[-1])
        strokes.append({
            'motif_id':   mid,
            'D_start':    d0,   'D_end':      d1,
            'duration':   t[-1] - t[0],
            'pic_acc':    float(np.max(acc)),
            'pic_down':   float(np.min(acc)),
            't_acc_frac': float(np.sum(acc > 0)) / len(acc),
            'd_stroke':   d1 - d0,
            'roll_amp':   float(grp['amp_roll'].mean()),
            'pitch_amp':  float(grp['amp_pitch'].mean()),
            'cadence':    float(grp['cadence'].mean()),
            'speed':      float(grp['speed'].mean()),
            # AUC
            'auc_pos':    float(np.trapezoid(np.clip(acc, 0, None), t)),
            'auc_neg':    float(np.trapezoid(np.clip(acc, None, 0), t)),
            'auc_net':    float(np.trapezoid(acc, t)),
            'auc_abs':    float(np.trapezoid(np.abs(acc), t)),
            # timing des pics
            't_pic_pos':  (t[np.argmax(acc)] - t[0]) / max(t[-1]-t[0], 1e-6),
            't_pic_neg':  (t[np.argmin(acc)] - t[0]) / max(t[-1]-t[0], 1e-6),
            'acc_norm':   acc_norm,
        })
    return strokes

def extract_strokes_accx(df, n_norm=N_NORM, min_period_s=0.3):
    """Segmentation via zero-crossing ascendant de acc_x filtré."""
    b, a  = butter(2, 2.0 / (FS / 2), btype='low')
    df    = df.sort_values('T').copy()
    acc   = filtfilt(b, a, df['acc_x'].values)
    t     = df['T'].values
    D     = df['D'].values
    sign  = np.sign(acc)
    cross = np.where((sign[:-1] < 0) & (sign[1:] >= 0))[0]
    mg    = int(min_period_s * FS)
    filt  = [cross[0]] if len(cross) > 0 else []
    for c in cross[1:]:
        if c - filt[-1] >= mg:
            filt.append(c)
    cross = np.array(filt)
    strokes = []
    for i in range(len(cross) - 1):
        i0, i1   = cross[i], cross[i+1]
        sa, st, sD = acc[i0:i1], t[i0:i1], D[i0:i1]
        if len(sa) < 5:
            continue
        acc_norm = interp1d(np.linspace(0,1,len(sa)), sa)(np.linspace(0,1,n_norm))
        strokes.append({
            'D_start':    sD[0],  'D_end':      sD[-1],
            'duration':   st[-1] - st[0],
            'pic_acc':    float(np.max(sa)),
            'pic_down':   float(np.min(sa)),
            't_acc_frac': float(np.sum(sa > 0)) / len(sa),
            'd_stroke':   sD[-1] - sD[0],
            'auc_pos':    float(np.trapezoid(np.clip(sa, 0, None), st)),
            'auc_neg':    float(np.trapezoid(np.clip(sa, None, 0), st)),
            'auc_net':    float(np.trapezoid(sa, st)),
            'auc_abs':    float(np.trapezoid(np.abs(sa), st)),
            'acc_norm':   acc_norm,
        })
    return strokes

def mean_profile(strokes):
    mat = np.vstack([s['acc_norm'] for s in strokes])
    return mat.mean(axis=0), mat.std(axis=0), mat

def to_df(strokes):
    return pd.DataFrame([{k: v for k, v in s.items() if k != 'acc_norm'} for s in strokes])

def rma(x, y, w=15):
    idx = np.argsort(x)
    xs, ys = x[idx], y[idx]
    return xs, pd.Series(ys).rolling(w, center=True, min_periods=1).mean().values


# =============================================================================
# CHARGEMENT
# =============================================================================
print("=" * 60)
print("Chargement et extraction des coups...")
print("=" * 60)

data_motif = {}
data_accx  = {}

for name, fname in ATHLETES_250.items():
    df = load(fname)
    data_motif[name] = extract_strokes_motif(df)
    data_accx[name]  = extract_strokes_accx(df)
    print(f"  {name:25s}: {len(data_motif[name])} coups (motif) | "
          f"{len(data_accx[name])} coups (accx)")

print()


# =============================================================================
# SECTION 1 – DÉTECTION & PROFILS INDIVIDUELS
# =============================================================================

# ── Fig 01 : Signal brut + détection motif_id (un graphe par athlète) ────────
print("Fig 01 – Signal brut + détection motif_id...")
fig, axes = plt.subplots(4, 2, figsize=(20, 20), sharey=False)
fig.suptitle("Signal acc_x et détection des coups – méthode motif_id Phyling (250m)",
             fontsize=14, fontweight='bold')

for idx, (name, fname) in enumerate(ATHLETES_250.items()):
    ax  = axes[idx // 2, idx % 2]
    df  = load(fname)
    df  = df.sort_values('T').copy()
    df['D_rel'] = df['D'] - df['D'].min()
    strokes = data_motif[name]
    palette = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(strokes)))

    ax.plot(df['D_rel'], df['acc_x'], color='#dddddd', lw=0.7, zorder=1)
    for i, s in enumerate(strokes):
        mask = (df['D'] >= s['D_start']) & (df['D'] <= s['D_end'])
        ax.plot(df.loc[mask, 'D_rel'], df.loc[mask, 'acc_x'],
                color=palette[i], lw=1.5, zorder=2)
    ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.5)
    ax.set_title(f"{name} | {len(strokes)} coups", fontsize=10, fontweight='bold')
    ax.set_xlabel("Distance relative (m)", fontsize=8)
    ax.set_ylabel("acc_x (m/s²)", fontsize=8)
    ax.grid(True, alpha=0.3)

sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=plt.Normalize(0, 250))
sm.set_array([])
fig.colorbar(sm, ax=axes, label='Distance relative (m)', shrink=0.5, pad=0.02)
savefig(fig, 'fig01_signal_detection_motif.png')

# ── Fig 02 : Signal brut + détection zero-crossing ───────────────────────────
print("Fig 02 – Signal brut + détection zero-crossing...")
fig, axes = plt.subplots(4, 2, figsize=(20, 20), sharey=False)
fig.suptitle("Signal acc_x et détection des coups – méthode zero-crossing (250m)",
             fontsize=14, fontweight='bold')

for idx, (name, fname) in enumerate(ATHLETES_250.items()):
    ax  = axes[idx // 2, idx % 2]
    df  = load(fname)
    b, a  = butter(2, 2.0/(FS/2), btype='low')
    df    = df.sort_values('T').copy()
    df['D_rel'] = df['D'] - df['D'].min()
    acc_f = filtfilt(b, a, df['acc_x'].values)
    strokes = data_accx[name]
    palette = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(strokes)))

    ax.plot(df['D_rel'], acc_f, color='#dddddd', lw=0.7, zorder=1)
    for i, s in enumerate(strokes):
        mask = (df['D'] >= s['D_start']) & (df['D'] <= s['D_end'])
        ax.plot(df.loc[mask, 'D_rel'], acc_f[mask.values],
                color=palette[i], lw=1.5, zorder=2)
    ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.5)
    ax.set_title(f"{name} | {len(strokes)} coups", fontsize=10, fontweight='bold')
    ax.set_xlabel("Distance relative (m)", fontsize=8)
    ax.set_ylabel("acc_x filtré (m/s²)", fontsize=8)
    ax.grid(True, alpha=0.3)

sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=plt.Normalize(0, 250))
sm.set_array([])
fig.colorbar(sm, ax=axes, label='Distance relative (m)', shrink=0.5, pad=0.02)
savefig(fig, 'fig02_signal_detection_accx.png')

# ── Fig 03 : Profils moyens individuels – motif_id ───────────────────────────
print("Fig 03 – Profils moyens individuels (motif_id)...")
fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharey=True)
fig.suptitle("Profils moyens du coup de pagaie par athlète – méthode motif_id (250m)",
             fontsize=14, fontweight='bold')

for idx, (name, strokes) in enumerate(data_motif.items()):
    ax      = axes.flatten()[idx]
    mu, sd, _ = mean_profile(strokes)
    c       = COLORS[name]
    ax.fill_between(x_norm, mu-2*sd, mu+2*sd, alpha=0.07, color=c)
    ax.fill_between(x_norm, mu-sd,   mu+sd,   alpha=0.20, color=c, label='±1 σ')
    ax.plot(x_norm, mu, color=c, lw=2.5, label='Profil moyen')
    ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.5)
    ax.scatter(x_norm[np.argmax(mu)], mu.max(), color='#4CAF50', s=60, zorder=5)
    ax.scatter(x_norm[np.argmin(mu)], mu.min(), color='#f44336', s=60, zorder=5)
    ax.set_title(name, fontsize=10, fontweight='bold')
    ax.set_xlabel("Cycle normalisé (%)", fontsize=8)
    if idx % 4 == 0: ax.set_ylabel("acc_x (m/s²)", fontsize=9)
    ax.text(0.98, 0.02, f'n={len(strokes)}', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=8, color='gray')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

plt.tight_layout()
savefig(fig, 'fig03_profils_individuels_motif.png')

# ── Fig 04 : Profils moyens individuels – zero-crossing ──────────────────────
print("Fig 04 – Profils moyens individuels (zero-crossing)...")
fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharey=True)
fig.suptitle("Profils moyens du coup de pagaie par athlète – méthode zero-crossing (250m)",
             fontsize=14, fontweight='bold')

for idx, (name, strokes) in enumerate(data_accx.items()):
    ax        = axes.flatten()[idx]
    mu, sd, _ = mean_profile(strokes)
    c         = COLORS[name]
    ax.fill_between(x_norm, mu-2*sd, mu+2*sd, alpha=0.07, color=c)
    ax.fill_between(x_norm, mu-sd,   mu+sd,   alpha=0.20, color=c, label='±1 σ')
    ax.plot(x_norm, mu, color=c, lw=2.5, label='Profil moyen')
    ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.5)
    ax.scatter(x_norm[np.argmax(mu)], mu.max(), color='#4CAF50', s=60, zorder=5)
    ax.scatter(x_norm[np.argmin(mu)], mu.min(), color='#f44336', s=60, zorder=5)
    ax.set_title(name, fontsize=10, fontweight='bold')
    ax.set_xlabel("Cycle normalisé (%)", fontsize=8)
    if idx % 4 == 0: ax.set_ylabel("acc_x (m/s²)", fontsize=9)
    ax.text(0.98, 0.02, f'n={len(strokes)}', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=8, color='gray')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

plt.tight_layout()
savefig(fig, 'fig04_profils_individuels_accx.png')


# =============================================================================
# SECTION 2 – COMPARAISON INTER-ATHLÈTES
# =============================================================================

# ── Fig 05 : Superposition des profils moyens ─────────────────────────────────
print("Fig 05 – Superposition comparative des profils...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Comparaison inter-athlètes – Profil moyen du coup de pagaie (250m)",
             fontsize=14, fontweight='bold')

for name in ATHLETES_250:
    c    = COLORS[name]
    short = name.split()[-1]
    mu1, sd1, _ = mean_profile(data_motif[name])
    mu2, sd2, _ = mean_profile(data_accx[name])
    ax1.fill_between(x_norm, mu1-sd1, mu1+sd1, alpha=0.07, color=c)
    ax1.plot(x_norm, mu1, color=c, lw=2, label=short)
    ax2.fill_between(x_norm, mu2-sd2, mu2+sd2, alpha=0.07, color=c)
    ax2.plot(x_norm, mu2, color=c, lw=2, label=short)

for ax, title in [(ax1, 'Méthode motif_id'), (ax2, 'Méthode zero-crossing acc_x')]:
    ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.5)
    ax.set_xlabel("Cycle normalisé (%)", fontsize=11)
    ax.set_ylabel("acc_x (m/s²)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
savefig(fig, 'fig05_superposition_profils.png')

# ── Fig 06 : Barplots des métriques ──────────────────────────────────────────
print("Fig 06 – Barplots métriques comparées...")
metrics_rows = []
for name in ATHLETES_250:
    df_s = to_df(data_motif[name])
    metrics_rows.append({
        'name':     name.split()[-1],
        'full':     name,
        'pic_acc':  df_s['pic_acc'].mean(),
        'pic_down': df_s['pic_down'].abs().mean(),
        'duration': df_s['duration'].mean(),
        'pct_acc':  df_s['t_acc_frac'].mean() * 100,
        'd_stroke': df_s['d_stroke'].mean(),
    })
df_met = pd.DataFrame(metrics_rows)

fig, axes = plt.subplots(1, 5, figsize=(22, 5))
fig.suptitle("Comparaison des métriques par athlète – 250m (motif_id)",
             fontsize=13, fontweight='bold')
specs = [('pic_acc','Pic acc (m/s²)'), ('pic_down','|Pic déc| (m/s²)'),
         ('duration','Durée coup (s)'), ('pct_acc','% phase acc'),
         ('d_stroke','Distance/coup (m)')]
for ax, (col, label) in zip(axes, specs):
    colors_bar = [COLORS[r['full']] for _, r in df_met.iterrows()]
    df_s_std   = [to_df(data_motif[r['full']])[col].std() if col in to_df(data_motif[r['full']]) else 0
                  for _, r in df_met.iterrows()]
    ax.bar(df_met['name'], df_met[col], color=colors_bar,
           yerr=df_s_std, capsize=4, edgecolor='white')
    ax.axhline(df_met[col].mean(), color='red', ls='--', lw=1.5, alpha=0.8, label='Moy.')
    ax.set_title(label, fontsize=10, fontweight='bold')
    ax.set_xticks(range(len(df_met)))
    ax.set_xticklabels(df_met['name'], rotation=40, ha='right', fontsize=8)
    ax.legend(fontsize=7); ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
savefig(fig, 'fig06_barplots_metriques.png')

# ── Fig 07 : Évolution des métriques sur la distance (250m) ──────────────────
print("Fig 07 – Évolution métriques sur distance 250m...")
fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
fig.suptitle("Évolution des coups sur la distance – épreuve 250m (motif_id)",
             fontsize=14, fontweight='bold')

metrics_ev = [('pic_acc','Pic acc (m/s²)',False),
              ('pic_down','|Pic déc| (m/s²)',True),
              ('d_stroke','Distance/coup (m)',False)]

for name in ATHLETES_250:
    df_s       = to_df(data_motif[name])
    df_s['D_rel'] = df_s['D_start'] - df_s['D_start'].min()
    c, short   = COLORS[name], name.split()[-1]
    for ax, (metric, ylabel, absval) in zip(axes, metrics_ev):
        y = df_s[metric].abs() if absval else df_s[metric]
        xs, ys = rma(df_s['D_rel'].values, y.values, w=5)
        ax.plot(xs, ys, color=c, lw=1.8, alpha=0.85, label=short)

for ax, (_, ylabel, _) in zip(axes, metrics_ev):
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=4, loc='upper right')
axes[-1].set_xlabel("Distance relative (m)", fontsize=11)
plt.tight_layout()
savefig(fig, 'fig07_evolution_distance_250m.png')

# ── Fig 08 : Tableau récapitulatif ────────────────────────────────────────────
print("Fig 08 – Tableau récapitulatif...")
recap_rows = []
for name in ATHLETES_250:
    df_s = to_df(data_motif[name])
    recap_rows.append([
        name, len(df_s),
        f"{df_s['pic_acc'].mean():.2f} ± {df_s['pic_acc'].std():.2f}",
        f"{df_s['pic_down'].abs().mean():.2f} ± {df_s['pic_down'].abs().std():.2f}",
        f"{df_s['duration'].mean():.3f}",
        f"{df_s['t_acc_frac'].mean()*100:.1f}%",
        f"{df_s['d_stroke'].mean():.2f}",
        f"{df_s['speed'].mean():.2f}",
    ])

cols = ['Athlète', 'N coups', 'Pic acc (m/s²)', '|Pic déc| (m/s²)',
        'Durée moy (s)', '% phase acc', 'Dist/coup (m)', 'Vit. moy (km/h)']

fig, ax = plt.subplots(figsize=(16, 4))
ax.axis('off')
tbl = ax.table(cellText=recap_rows, colLabels=cols, cellLoc='center', loc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.9)
for j in range(len(cols)):
    tbl[(0,j)].set_facecolor('#1a4a6b')
    tbl[(0,j)].set_text_props(color='white', fontweight='bold')
for i in range(1, len(recap_rows)+1):
    bg = '#f0f4f8' if i % 2 == 0 else 'white'
    for j in range(len(cols)): tbl[(i,j)].set_facecolor(bg)
ax.set_title("Tableau récapitulatif – métriques coups de pagaie – 250m (motif_id)",
             fontsize=12, fontweight='bold', pad=20)
plt.tight_layout()
savefig(fig, 'fig08_tableau_recap.png')


# =============================================================================
# SECTION 3 – ANALYSE AUC (AIRE SOUS LA COURBE = IMPULSION)
# =============================================================================
# AUC+ = ∫ acc_x⁺ dt  → impulsion de propulsion (m/s)
# AUC- = ∫ acc_x⁻ dt  → impulsion de freinage   (m/s)
# AUC_net = AUC+ + AUC- → variation nette de vitesse par cycle (m/s)
# AUC_abs = ∫|acc_x| dt → effort mécanique total (m/s)
# Ratio   = AUC+/AUC_abs → % du cycle utile à la propulsion

# ── Fig 09 : Profils AUC avec aires colorées ─────────────────────────────────
print("Fig 09 – Profils moyens avec aires AUC colorées...")
fig = plt.figure(figsize=(22, 18))
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.35)
fig.suptitle("Pattern moyen du coup avec aires AUC\n"
             "Vert = propulsion (AUC+)  |  Rouge = freinage (AUC-)",
             fontsize=14, fontweight='bold')

for idx, (name, strokes) in enumerate(data_motif.items()):
    ax        = fig.add_subplot(gs[idx // 3, idx % 3])
    mu, sd, _ = mean_profile(strokes)
    df_s      = to_df(strokes)
    c         = COLORS[name]

    ax.fill_between(x_norm, mu-2*sd, mu+2*sd, alpha=0.07, color=c)
    ax.fill_between(x_norm, mu-sd,   mu+sd,   alpha=0.18, color=c)
    ax.fill_between(x_norm, mu, 0, where=(mu > 0), alpha=0.35, color='#4CAF50',
                    label=f"AUC+ = {df_s['auc_pos'].mean():.4f} m/s")
    ax.fill_between(x_norm, mu, 0, where=(mu < 0), alpha=0.35, color='#f44336',
                    label=f"|AUC-|= {df_s['auc_neg'].abs().mean():.4f} m/s")
    ax.plot(x_norm, mu, color=c, lw=2.5, zorder=5)
    ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.5)
    ax.scatter(x_norm[np.argmax(mu)], mu.max(), color='#4CAF50', s=60, zorder=6)
    ax.scatter(x_norm[np.argmin(mu)], mu.min(), color='#f44336', s=60, zorder=6)
    ax.set_title(f"{name}\nn={len(strokes)}", fontsize=9, fontweight='bold')
    ax.set_xlabel("Cycle normalisé (%)", fontsize=8)
    ax.set_ylabel("acc_x (m/s²)", fontsize=8)
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, alpha=0.3)

savefig(fig, 'fig09_auc_patterns_individuels.png')

# ── Fig 10 : Superposition + barplots AUC ────────────────────────────────────
print("Fig 10 – Comparaison AUC superposée + barplots...")
fig = plt.figure(figsize=(20, 12))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
fig.suptitle("Comparaison inter-athlètes – AUC du coup de pagaie (250m)",
             fontsize=14, fontweight='bold')

ax1 = fig.add_subplot(gs[0, :])
for name, strokes in data_motif.items():
    mu, sd, _ = mean_profile(strokes)
    c = COLORS[name]
    ax1.fill_between(x_norm, mu-sd, mu+sd, alpha=0.07, color=c)
    ax1.plot(x_norm, mu, color=c, lw=2.5, label=name.split()[-1])
ax1.axhline(0, color='k', lw=1, ls='--', alpha=0.5)
ax1.set_xlabel("Cycle normalisé (%)", fontsize=11)
ax1.set_ylabel("acc_x (m/s²)", fontsize=11)
ax1.set_title("Superposition des profils moyens", fontsize=12)
ax1.legend(fontsize=9, ncol=4)
ax1.grid(True, alpha=0.3)

for ax, metric, label in [
    (fig.add_subplot(gs[1, 0]), 'auc_pos', 'AUC+ (m/s) – Phase propulsion'),
    (fig.add_subplot(gs[1, 1]), 'auc_neg', '|AUC-| (m/s) – Phase freinage'),
]:
    names_s = [n.split()[-1] for n in data_motif]
    vals    = [to_df(s)[metric].abs().mean() for s in data_motif.values()]
    errs    = [to_df(s)[metric].abs().std()  for s in data_motif.values()]
    ax.bar(names_s, vals, yerr=errs, color=[COLORS[n] for n in data_motif],
           edgecolor='white', capsize=4)
    ax.axhline(np.mean(vals), color='k', ls='--', lw=1.5, alpha=0.7,
               label=f'Moy. groupe: {np.mean(vals):.4f}')
    ax.set_title(label, fontsize=10, fontweight='bold')
    ax.set_xticks(range(len(names_s)))
    ax.set_xticklabels(names_s, rotation=40, ha='right', fontsize=8)
    ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)

savefig(fig, 'fig10_auc_comparaison.png')

# ── Fig 11 : Scatter AUC+ vs |AUC-| ─────────────────────────────────────────
print("Fig 11 – Scatter AUC+ vs |AUC-|...")
fig, ax = plt.subplots(figsize=(11, 8))
fig.suptitle("Relation AUC+ (propulsion) vs |AUC-| (freinage) par coup\n"
             "Chaque point = un coup de pagaie  |  ♦ = centroïde athlète",
             fontsize=13, fontweight='bold')

for name, strokes in data_motif.items():
    df_s = to_df(strokes)
    c    = COLORS[name]
    ax.scatter(df_s['auc_pos'], df_s['auc_neg'].abs(),
               alpha=0.3, s=20, color=c,
               label=f"{name.split()[-1]} (+{df_s['auc_pos'].mean():.3f} / -{df_s['auc_neg'].abs().mean():.3f})")
    ax.scatter(df_s['auc_pos'].mean(), df_s['auc_neg'].abs().mean(),
               color=c, s=180, zorder=5, edgecolors='black', lw=1.5, marker='D')

lim = max(to_df(s)['auc_pos'].max() for s in data_motif.values())
ax.plot([0, lim], [0, lim], 'k--', lw=1.2, alpha=0.4, label='AUC+ = |AUC-|')
ax.set_xlabel("AUC+ (m/s) – Propulsion", fontsize=11)
ax.set_ylabel("|AUC-| (m/s) – Freinage", fontsize=11)
ax.legend(fontsize=8, loc='upper left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
savefig(fig, 'fig11_scatter_auc.png')

# ── Fig 12 : Tableau AUC récapitulatif ───────────────────────────────────────
print("Fig 12 – Tableau AUC récapitulatif...")
auc_rows = []
for name, strokes in data_motif.items():
    df_s = to_df(strokes)
    auc_rows.append([
        name, len(df_s),
        f"{df_s['auc_pos'].mean():.4f}", f"{df_s['auc_pos'].std():.4f}",
        f"{df_s['auc_neg'].abs().mean():.4f}", f"{df_s['auc_neg'].abs().std():.4f}",
        f"{df_s['auc_net'].mean():.4f}",
        f"{df_s['auc_abs'].mean():.4f}",
        f"{df_s['auc_pos'].mean() / df_s['auc_abs'].mean() * 100:.1f}%",
    ])

cols_auc = ['Athlète', 'N coups', 'AUC+ moy', 'AUC+ std',
            '|AUC-| moy', '|AUC-| std', 'AUC_net moy', 'AUC_abs moy', 'Ratio prop.']
fig, ax = plt.subplots(figsize=(16, 4))
ax.axis('off')
tbl = ax.table(cellText=auc_rows, colLabels=cols_auc, cellLoc='center', loc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.9)
for j in range(len(cols_auc)):
    tbl[(0,j)].set_facecolor('#1a4a6b')
    tbl[(0,j)].set_text_props(color='white', fontweight='bold')
for i in range(1, len(auc_rows)+1):
    bg = '#f0f4f8' if i % 2 == 0 else 'white'
    for j in range(len(cols_auc)): tbl[(i,j)].set_facecolor(bg)
ax.set_title("Tableau récapitulatif AUC  |  toutes valeurs en m/s  |  250m (motif_id)",
             fontsize=12, fontweight='bold', pad=20)
plt.tight_layout()
savefig(fig, 'fig12_tableau_auc.png')


# =============================================================================
# SECTION 4 – ANALYSE LONGITUDINALE 2000m (ZWILLER TAO)
# =============================================================================
print()
print("Chargement 2000m – Zwiller Tao...")
name_2k, fname_2k = ATHLETE_2000
df2k    = load(fname_2k)
s2k     = extract_strokes_motif(df2k)
df_s2k  = to_df(s2k)
df_s2k['D_rel'] = df_s2k['D_start'] - df_s2k['D_start'].min()
print(f"  {name_2k} (2000m) : {len(s2k)} coups extraits")

# ── Fig 13 : Heatmap des profils normalisés ───────────────────────────────────
print("Fig 13 – Heatmap 2000m...")
fig, ax = plt.subplots(figsize=(18, 8))
mat = np.vstack([s['acc_norm'] for s in s2k])
im  = ax.imshow(mat, aspect='auto', origin='upper', cmap='RdBu_r', vmin=-6, vmax=6)
plt.colorbar(im, ax=ax, label='acc_x (m/s²)', shrink=0.8)
ax.set_xlabel("Cycle normalisé (%)", fontsize=11)
ax.set_ylabel("Coup # (ordre chronologique)", fontsize=11)
ax.set_xticks(np.linspace(0, N_NORM-1, 6))
ax.set_xticklabels([f'{int(v)}%' for v in np.linspace(0, 100, 6)])
# Ajouter des lignes pour les quarts
d_max = df_s2k['D_rel'].max()
for q_pct in [0.25, 0.5, 0.75]:
    idx_q = int(q_pct * len(s2k))
    ax.axhline(idx_q, color='white', lw=1.5, ls='--', alpha=0.7,
               label=f'{int(q_pct*d_max)}m')
ax.legend(fontsize=9)
ax.set_title(f"{name_2k} – Heatmap des {len(s2k)} profils de coups normalisés (2000m)",
             fontsize=13, fontweight='bold')
plt.tight_layout()
savefig(fig, 'fig13_heatmap_2000m.png')

# ── Fig 14 : Évolution métriques sur 2000m ───────────────────────────────────
print("Fig 14 – Évolution métriques 2000m...")
fig = plt.figure(figsize=(18, 14))
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.35)
fig.suptitle(f"{name_2k} – Évolution des métriques au fil des 2000m (motif_id)",
             fontsize=14, fontweight='bold')

plots_2k = [
    ('pic_acc',    'Pic accélération (m/s²)',    '#2196F3', False, gs[0,0]),
    ('pic_down',   '|Pic décélération| (m/s²)',  '#f44336', True,  gs[0,1]),
    ('duration',   'Durée du coup (s)',           '#4CAF50', False, gs[1,0]),
    ('d_stroke',   'Distance/coup (m)',           '#FF9800', False, gs[1,1]),
    ('auc_pos',    'AUC+ (m/s) – Propulsion',    '#7B1FA2', False, gs[2,0]),
    ('auc_abs',    'AUC_abs (m/s) – Effort total','#00796B', False, gs[2,1]),
]
for metric, label, color, absval, gspec in plots_2k:
    ax = fig.add_subplot(gspec)
    x  = df_s2k['D_rel'].values
    y  = df_s2k[metric].abs().values if absval else df_s2k[metric].values
    ax.scatter(x, y, alpha=0.15, s=6, color=color)
    xs, ys = rma(x, y, w=30)
    ax.plot(xs, ys, color=color, lw=2.8, label='Moy. glissante (30 coups)')
    z = np.polyfit(x, y, 1)
    ax.plot(np.sort(x), np.poly1d(z)(np.sort(x)), 'k--', lw=1.3, alpha=0.6,
            label=f'Tendance : {z[0]:+.5f}/m')
    for dm in [500, 1000, 1500]:
        ax.axvline(dm, color='gray', ls=':', lw=1, alpha=0.5)
    ax.set_xlabel("Distance (m)", fontsize=9)
    ax.set_ylabel(label, fontsize=9)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

savefig(fig, 'fig14_evolution_metriques_2000m.png')

# ── Fig 15 : Profils moyens par quart de course ───────────────────────────────
print("Fig 15 – Profils par quart de course 2000m...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(f"{name_2k} – Profil moyen du coup par quart de course (2000m)",
             fontsize=14, fontweight='bold')

quarters = [(0,500),(500,1000),(1000,1500),(1500,d_max)]
q_colors = ['#2c7bb6','#74add1','#fdae61','#d7191c']
q_labels = ['0–500m','500–1000m','1000–1500m','1500–2000m']

for ax, strokes_list, method in [(axes[0], s2k, 'motif_id'),
                                  (axes[1], s2k, 'motif_id (zoom AUC)')]:
    d_starts = df_s2k['D_rel'].values
    for (d_lo, d_hi), qc, ql in zip(quarters, q_colors, q_labels):
        q_s = [s for s, ds in zip(strokes_list, d_starts) if d_lo <= ds < d_hi]
        if q_s:
            mu, sd, _ = mean_profile(q_s)
            df_q      = to_df(q_s)
            ax.fill_between(x_norm, mu-sd, mu+sd, alpha=0.15, color=qc)
            ax.plot(x_norm, mu, color=qc, lw=2.5,
                    label=f"{ql} (n={len(q_s)}, AUC+={df_q['auc_pos'].mean():.4f})")
    ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.5)
    ax.set_xlabel("Cycle normalisé (%)", fontsize=10)
    ax.set_ylabel("acc_x (m/s²)", fontsize=10)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

axes[0].set_title("Profils moyens superposés", fontsize=11)
# Deuxième graphe : différence par rapport au 1er quart
d_starts = df_s2k['D_rel'].values
profiles  = []
for (d_lo, d_hi) in quarters:
    q_s = [s for s, ds in zip(s2k, d_starts) if d_lo <= ds < d_hi]
    if q_s:
        mu, _, _ = mean_profile(q_s)
        profiles.append(mu)

ref = profiles[0]
axes[1].set_title("Différence par rapport au 1er quart (0–500m)", fontsize=11)
for mu, qc, ql in zip(profiles[1:], q_colors[1:], q_labels[1:]):
    diff = mu - ref
    axes[1].fill_between(x_norm, diff, 0, alpha=0.25, color=qc)
    axes[1].plot(x_norm, diff, color=qc, lw=2.5, label=ql)
axes[1].axhline(0, color='k', lw=1, ls='-', alpha=0.5)
axes[1].set_xlabel("Cycle normalisé (%)", fontsize=10)
axes[1].set_ylabel("Δ acc_x vs début de course (m/s²)", fontsize=10)
axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
savefig(fig, 'fig15_profils_quarts_2000m.png')


# =============================================================================
# RÉSUMÉ FINAL
# =============================================================================
print()
print("=" * 60)
print("✅ ANALYSE TERMINÉE – 15 figures produites")
print(f"   Sauvegardées dans : {OUT_DIR}")
print("=" * 60)
print()
print("Résumé AUC+ (propulsion) par athlète :")
for name, strokes in data_motif.items():
    df_s = to_df(strokes)
    ratio = df_s['auc_pos'].mean() / df_s['auc_abs'].mean() * 100
    print(f"  {name:25s} | AUC+={df_s['auc_pos'].mean():.4f} m/s"
          f" | AUC_abs={df_s['auc_abs'].mean():.4f} m/s"
          f" | Ratio={ratio:.1f}%")

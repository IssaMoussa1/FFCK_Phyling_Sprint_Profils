"""
=============================================================================
COMPARAISON DES COUPS INDIVIDUELS ENTRE ATHLÈTES
=============================================================================
Pour chaque athlète :
  - Extraction de TOUS les coups (signal acc_x normalisé sur 0→100%)
  - Affichage de chaque coup en semi-transparent + profil moyen en gras
  - Comparaison directe des profils moyens entre tous les athlètes

FIGURES PRODUITES :
  Fig A – Grille individuelle : tous les coups superposés + moyenne
           (une case par athlète, 2 méthodes = 2 figures)
  Fig B – Comparaison des moyennes : tous athlètes sur un même graphe
           avec bandes de confiance
  Fig C – Heatmap individuelle : variabilité coup par coup (ordre distance)
  Fig D – Dendrogramme de similarité des profils moyens
=============================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
DATA_DIR = r'C:\Users\imoussa\OneDrive - FFCK\Documents\Application\FFCK_Sprint_Profil'
OUT_DIR  = r'C:\Users\imoussa\OneDrive - FFCK\Documents\Application\FFCK_Sprint_Profil\outputs'

ATHLETES = {
    'Bavenkoff Viktor':   'bavenkoff_viktor-20260218_024153-sel_250.csv',
    'Gilhard Tom':        'gilhard_tom-20260215_101526-sel_250.csv',
    'Martin Noe':         'martin_noe-20260215_103255-sel_250.csv',
    'Polet Theophile':    'polet_theophile-20260215_103845-sel_250.csv',
    'Siabas Simon':       'siabas_simon_anatole-20260215_101514-sel_250.csv',
    'Zappaterra Clement': 'zappaterra_clement-20260215_111129-sel_250.csv',
    'Zoualegh Nathan':    'zoualegh_nathan-20260215_084430-sel_250.csv',
    'Zwiller Tao':        'zwiller_tao-20260215_102746-sel_250.csv',
}

N_NORM   = 200   # points de normalisation
FS       = 100   # fréquence Hz
ALPHA_IND = 0.08 # transparence des coups individuels

AFFICHER = True
SAUVER   = True
os.makedirs(OUT_DIR, exist_ok=True)

COLORS = {n: plt.cm.tab10(i / len(ATHLETES)) for i, n in enumerate(ATHLETES)}
x_norm = np.linspace(0, 100, N_NORM)


# =============================================================================
# FONCTIONS
# =============================================================================

def load(fname):
    return pd.read_csv(os.path.join(DATA_DIR, fname))

def extract_motif(df, n_norm=N_NORM):
    """Extraction via motif_id — retourne liste de tableaux normalisés + métadonnées."""
    result = []
    for mid, grp in df.groupby('motif_id'):
        grp = grp.sort_values('T')
        acc = grp['acc_x'].values
        if len(acc) < 5:
            continue
        acc_norm = interp1d(np.linspace(0,1,len(acc)), acc)(np.linspace(0,1,n_norm))
        result.append({
            'acc_norm': acc_norm,
            'D_start':  float(grp['D'].iloc[0]),
            'D_end':    float(grp['D'].iloc[-1]),
            'duration': float(grp['T'].iloc[-1] - grp['T'].iloc[0]),
        })
    return result

def extract_accx(df, n_norm=N_NORM, min_period_s=0.3):
    """Extraction via zero-crossing de acc_x filtré."""
    b, a  = butter(2, 2.0 / (FS / 2), btype='low')
    df    = df.sort_values('T').copy()
    acc_f = filtfilt(b, a, df['acc_x'].values)
    t     = df['T'].values
    D     = df['D'].values
    sign  = np.sign(acc_f)
    cross = np.where((sign[:-1] < 0) & (sign[1:] >= 0))[0]
    mg    = int(min_period_s * FS)
    filt  = [cross[0]] if len(cross) > 0 else []
    for c in cross[1:]:
        if c - filt[-1] >= mg:
            filt.append(c)
    result = []
    for i in range(len(filt) - 1):
        i0, i1 = filt[i], filt[i+1]
        seg = acc_f[i0:i1]
        if len(seg) < 5:
            continue
        acc_norm = interp1d(np.linspace(0,1,len(seg)), seg)(np.linspace(0,1,n_norm))
        result.append({
            'acc_norm': acc_norm,
            'D_start':  D[i0],
            'D_end':    D[i1-1],
            'duration': t[i1-1] - t[i0],
        })
    return result

def get_matrix(strokes):
    """Retourne la matrice (n_coups × n_norm) et les D_start triés."""
    mat      = np.vstack([s['acc_norm'] for s in strokes])
    d_starts = np.array([s['D_start']  for s in strokes])
    order    = np.argsort(d_starts)
    return mat[order], d_starts[order]

def savefig(fig, fname):
    if SAUVER:
        fig.savefig(os.path.join(OUT_DIR, fname), dpi=150, bbox_inches='tight')
    if AFFICHER:
        plt.show()
    plt.close(fig)


# =============================================================================
# CHARGEMENT
# =============================================================================
print("=" * 65)
print("Extraction des coups pour tous les athlètes...")
print("=" * 65)

data = {}   # data[name] = {'motif': [...], 'accx': [...]}
for name, fname in ATHLETES.items():
    df           = load(fname)
    s_mot        = extract_motif(df)
    s_ax         = extract_accx(df)
    data[name]   = {'motif': s_mot, 'accx': s_ax}
    print(f"  {name:25s} | motif: {len(s_mot):3d} coups | accx: {len(s_ax):3d} coups")
print()


# =============================================================================
# FIG A — Grille individuelle : tous les coups + moyenne
#          (version motif_id ET version zero-crossing)
# =============================================================================
for method_key, method_label, fig_name in [
    ('motif', 'motif_id Phyling',   'figA1_coups_individuels_motif.png'),
    ('accx',  'zero-crossing acc_x','figA2_coups_individuels_accx.png'),
]:
    print(f"Fig A – Coups individuels ({method_label})...")

    fig, axes = plt.subplots(2, 4, figsize=(22, 10), sharey=False)
    fig.suptitle(
        f"Tous les coups de pagaie superposés par athlète – {method_label} (250m)\n"
        "Trait fin = coup individuel  |  Trait épais = profil moyen  |  Ombrage = ±1σ",
        fontsize=13, fontweight='bold'
    )

    for idx, (name, fname) in enumerate(ATHLETES.items()):
        ax      = axes.flatten()[idx]
        strokes = data[name][method_key]
        mat, d_starts = get_matrix(strokes)
        mu  = mat.mean(axis=0)
        sd  = mat.std(axis=0)
        c   = COLORS[name]
        n   = len(strokes)

        # Colorier chaque coup selon sa position dans la course (début→fin)
        cmap     = plt.cm.RdYlGn
        d_rel    = d_starts - d_starts.min()
        d_rel_n  = d_rel / (d_rel.max() + 1e-9)   # 0→1

        for i, (acc_i, dr) in enumerate(zip(mat, d_rel_n)):
            color_i = cmap(dr)
            ax.plot(x_norm, acc_i, color=color_i, lw=0.6, alpha=ALPHA_IND, zorder=1)

        # Bande ±1σ
        ax.fill_between(x_norm, mu - sd, mu + sd, alpha=0.20, color=c, zorder=2)
        # Profil moyen
        ax.plot(x_norm, mu, color=c, lw=3, zorder=3, label=f'Moyenne (n={n})')
        ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.5, zorder=2)

        # Marqueurs pics
        ax.scatter(x_norm[np.argmax(mu)], mu.max(),
                   color='#1B5E20', s=80, zorder=5, marker='^')
        ax.scatter(x_norm[np.argmin(mu)], mu.min(),
                   color='#B71C1C', s=80, zorder=5, marker='v')

        ax.set_title(name, fontsize=10, fontweight='bold')
        ax.set_xlabel("Cycle normalisé (%)", fontsize=8)
        ax.set_ylabel("acc_x (m/s²)", fontsize=8)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=7)

    # Barre de couleur distance
    sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=Normalize(0, 250))
    sm.set_array([])
    fig.colorbar(sm, ax=axes, label='Distance relative (m)', shrink=0.5, pad=0.02)
    savefig(fig, fig_name)

print()


# =============================================================================
# FIG B — Comparaison directe des profils moyens entre athlètes
#          (panneau gauche = motif_id, panneau droit = zero-crossing)
# =============================================================================
print("Fig B – Comparaison des profils moyens entre athlètes...")

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle(
    "Comparaison des profils moyens de coup entre athlètes (250m)\n"
    "Trait plein = moyenne  |  Ombrage = ±1σ",
    fontsize=14, fontweight='bold'
)

for ax, method_key, method_label in [
    (axes[0], 'motif', 'Méthode motif_id Phyling'),
    (axes[1], 'accx',  'Méthode zero-crossing acc_x'),
]:
    for name in ATHLETES:
        strokes   = data[name][method_key]
        mat, _    = get_matrix(strokes)
        mu        = mat.mean(axis=0)
        sd        = mat.std(axis=0)
        c         = COLORS[name]
        short     = name.split()[-1]

        ax.fill_between(x_norm, mu - sd, mu + sd, alpha=0.10, color=c)
        ax.plot(x_norm, mu, color=c, lw=2.5, label=short)

    ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.5)
    ax.set_title(method_label, fontsize=12)
    ax.set_xlabel("Cycle normalisé (%)", fontsize=11)
    ax.set_ylabel("acc_x (m/s²)", fontsize=11)
    ax.legend(fontsize=9, ncol=2, loc='upper right')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
savefig(fig, 'figB_comparaison_moyennes.png')


# =============================================================================
# FIG C — Heatmap par athlète : variabilité coup par coup (triés par distance)
#          (2 lignes = motif / accx, 8 colonnes = athlètes)
# =============================================================================
print("Fig C – Heatmaps individuelles (variabilité coup par coup)...")

for method_key, method_label, fig_name in [
    ('motif', 'motif_id',   'figC1_heatmap_coups_motif.png'),
    ('accx',  'zero-crossing', 'figC2_heatmap_coups_accx.png'),
]:
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    fig.suptitle(
        f"Heatmap des coups normalisés par athlète – {method_label} (250m)\n"
        "Chaque ligne = un coup  |  Trié par distance (haut=début, bas=fin)\n"
        "Bleu = décélération  |  Rouge = accélération",
        fontsize=12, fontweight='bold'
    )

    vmax = max(
        np.abs(get_matrix(data[n][method_key])[0]).max()
        for n in ATHLETES
    )

    for idx, name in enumerate(ATHLETES):
        ax      = axes.flatten()[idx]
        mat, _  = get_matrix(data[name][method_key])
        n_str   = len(mat)

        im = ax.imshow(mat, aspect='auto', origin='upper',
                       cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_title(f"{name}\n(n={n_str})", fontsize=9, fontweight='bold')
        ax.set_xlabel("Cycle normalisé (%)", fontsize=8)
        ax.set_ylabel("Coup # (ordre distance)", fontsize=8)
        ax.set_xticks(np.linspace(0, N_NORM-1, 6))
        ax.set_xticklabels([f'{int(v)}%' for v in np.linspace(0, 100, 6)], fontsize=7)
        ax.tick_params(axis='y', labelsize=7)
        plt.colorbar(im, ax=ax, label='acc_x (m/s²)', shrink=0.8)

    plt.tight_layout()
    savefig(fig, fig_name)


# =============================================================================
# FIG D — Dendrogramme : similarité des profils moyens entre athlètes
# =============================================================================
print("Fig D – Dendrogramme de similarité des profils moyens...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(
    "Similarité des profils moyens de coup entre athlètes (distance euclidienne)\n"
    "Plus les athlètes sont proches dans l'arbre, plus leurs patterns sont similaires",
    fontsize=13, fontweight='bold'
)

for ax, method_key, method_label in [
    (axes[0], 'motif', 'motif_id Phyling'),
    (axes[1], 'accx',  'zero-crossing acc_x'),
]:
    names   = list(ATHLETES.keys())
    means   = np.vstack([
        get_matrix(data[n][method_key])[0].mean(axis=0)
        for n in names
    ])
    # Distance euclidienne entre profils moyens
    dist_mat = pdist(means, metric='euclidean')
    Z        = linkage(dist_mat, method='ward')

    labels_short = [n.split()[-1] for n in names]
    dend = dendrogram(
        Z,
        labels=labels_short,
        ax=ax,
        leaf_rotation=40,
        leaf_font_size=10,
        color_threshold=0.6 * max(Z[:, 2]),
    )
    # Colorier les labels selon les couleurs athlètes
    for lbl, tick in zip(ax.get_xticklabels(), ax.get_xticklabels()):
        lbl_text = tick.get_text()
        for name in names:
            if name.split()[-1] == lbl_text:
                tick.set_color(COLORS[name])

    ax.set_title(method_label, fontsize=11)
    ax.set_ylabel("Distance (dissimilarité)", fontsize=10)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
savefig(fig, 'figD_dendrogramme_similarite.png')


# =============================================================================
# FIG E — Vue synthétique : moyenne ± enveloppe min/max de TOUS les athlètes
# =============================================================================
print("Fig E – Enveloppe globale et profils individuels...")

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle(
    "Enveloppe complète des coups : min/max/moyenne par athlète (250m)",
    fontsize=14, fontweight='bold'
)

for ax, method_key, method_label in [
    (axes[0], 'motif', 'motif_id Phyling'),
    (axes[1], 'accx',  'zero-crossing acc_x'),
]:
    for name in ATHLETES:
        strokes   = data[name][method_key]
        mat, _    = get_matrix(strokes)
        mu        = mat.mean(axis=0)
        env_min   = mat.min(axis=0)
        env_max   = mat.max(axis=0)
        c         = COLORS[name]
        short     = name.split()[-1]

        # Enveloppe min/max très transparente
        ax.fill_between(x_norm, env_min, env_max, alpha=0.04, color=c)
        # Profil moyen
        ax.plot(x_norm, mu, color=c, lw=2.5, label=short)

    ax.axhline(0, color='k', lw=1, ls='--', alpha=0.5)
    ax.set_title(method_label, fontsize=12)
    ax.set_xlabel("Cycle normalisé (%)", fontsize=11)
    ax.set_ylabel("acc_x (m/s²)", fontsize=11)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
savefig(fig, 'figE_enveloppe_globale.png')


print()
print("=" * 65)
print("✅ TERMINÉ – 7 figures produites dans :", OUT_DIR)
print("=" * 65)
print()
print("  figA1_coups_individuels_motif.png  – tous les coups + moyenne (motif_id)")
print("  figA2_coups_individuels_accx.png   – tous les coups + moyenne (zero-crossing)")
print("  figB_comparaison_moyennes.png      – comparaison inter-athlètes")
print("  figC1_heatmap_coups_motif.png      – heatmap variabilité (motif_id)")
print("  figC2_heatmap_coups_accx.png       – heatmap variabilité (zero-crossing)")
print("  figD_dendrogramme_similarite.png   – arbre de similarité des profils")
print("  figE_enveloppe_globale.png         – enveloppe min/max de tous les coups")

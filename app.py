"""
=============================================================================
APPLICATION STREAMLIT — Analyse des coups de pagaie
Maxi-Phyling · FFCK Sprint · Méthode creux locaux
=============================================================================
Lancement : python -m streamlit run app.py
=============================================================================
Onglets :
  ① Signal            – signal brut + KPI
  ② Analyse indiv.    – profils, quarts, heatmap
  ③ Comparaison       – enveloppe, scatter, dendro, matrice
  ④ Métriques         – tableaux + distributions + export
=============================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Ellipse, Patch
from scipy.signal import butter, filtfilt, find_peaks
from scipy.interpolate import interp1d
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr
import streamlit as st
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

ATHLETES_FILES = {
    'Bavenkoff Viktor':   {'250m': 'bavenkoff_viktor-20260218_024153-sel_250.csv',
                           '2000m': 'bavenkoff_viktor-20260218_024153-sel_2000.csv'},
    'Gilhard Tom':        {'250m': 'gilhard_tom-20260215_101526-sel_250.csv',
                           '2000m': 'gilhard_tom-20260215_101526-sel_2000.csv'},
    'Polet Theophile':    {'250m': 'polet_theophile-20260215_103845-sel_250.csv',
                           '2000m': 'polet_theophile-20260215_103845-sel_2000.csv'},
    'Siabas Simon':       {'250m': 'siabas_simon_anatole-20260215_101514-sel_250.csv',
                           '2000m': 'siabas_simon_anatole-20260215_101514-sel_2000.csv'},
    'Zappaterra Clement': {'250m': 'zappaterra_clement-20260215_111129-sel_250.csv',
                           '2000m': 'zappaterra_clement-20260215_111129-sel_2000.csv'},
    'Zoualegh Nathan':    {'250m': 'zoualegh_nathan-20260215_084430-sel_250.csv',
                           '2000m': 'zoualegh_nathan-20260215_084430-sel_2000.csv'},
    'Zwiller Tao':        {'250m': 'zwiller_tao-20260215_102746-sel_250.csv',
                           '2000m': 'zwiller_tao-20260215_102746-sel_2000.csv'},
}

FC_SMOOTH  = 3.0
MIN_DIST_S = 0.25
MIN_PEAK_H = 0.2
N_NORM     = 200
FS         = 100

C_POS  = '#2E7D32'
C_NEG  = '#C62828'
C_MEAN = '#1565C0'
C_BG   = '#F8F9FA'
C_GRID = '#E0E0E0'

QUART_COLORS = ['#1E88E5', '#43A047', '#FB8C00', '#E53935']
QUART_LABELS = ['1er quart (0–25%)', '2ème quart (25–50%)',
                '3ème quart (50–75%)', '4ème quart (75–100%)']

xn = np.linspace(0, 100, N_NORM)

METRIC_META = {
    'auc_pos':    ('AUC+ (m/s)',       'Impulsion de propulsion par coup'),
    'auc_neg':    ('|AUC-| (m/s)',      'Impulsion de freinage par coup'),
    'sym_ratio':  ('Sym ratio',         'Équilibre propulsion/freinage  (idéal < 0.7)'),
    'rfd':        ('RFD (m/s³)',        'Explosivité du catch'),
    'jerk_rms':   ('Jerk RMS (m/s³)',   'Fluidité du coup  (bas = plus fluide)'),
    'pos_pic_pct':('Position pic (%)',  'Timing de la propulsion dans le cycle'),
    'fwhm_s':     ('FWHM (s)',          'Durée de la propulsion intense'),
    'duration':   ('Durée coup (s)',    'Durée totale du cycle de pagaie'),
    'd_stroke':   ('Distance/coup (m)', 'Distance parcourue par coup'),
}

# ─────────────────────────────────────────────────────────────────────────────
# TEXTES EXPLICATIFS (tooltips / points d'interrogation)
# ─────────────────────────────────────────────────────────────────────────────
HELP_SIGNAL = """
**Signal acc_x — accélération longitudinale du bateau**

Le capteur IMU du Maxi-Phyling mesure l'accélération du bateau selon son axe de déplacement (avant-arrière), à 200 Hz.

- **acc_x > 0** (vert) : le bateau accélère → la pagaie pousse sur l'eau de manière efficace (phase de propulsion)
- **acc_x < 0** (rouge) : le bateau ralentit → résistance hydrodynamique, phase passive entre deux coups

**Pourquoi c'est utile ?**  
La forme de ce signal reflète directement la technique de pagaie. Un pic élevé et étroit = catch explosif. Une décélération faible entre deux coups = bonne glisse ou cadence soutenue.

**Frontière entre coups :**  
Chaque coup est délimité par les creux locaux (minimum d'accélération), qui correspondent au moment précis où la pagaie entre dans l'eau — point de transition entre la phase passive et la phase active.
"""

HELP_DETECTION = """
**Paramètres de détection des coups**

La détection utilise un algorithme en 3 étapes :

1. **Lissage passe-bas (fc, Hz)** : filtre les vibrations parasites du signal brut avant de chercher des pics. Une valeur plus basse = plus de lissage. Valeur conseillée : 3 Hz. Si des faux coups sont détectés (bruit), augmentez légèrement.

2. **Distance minimale entre pics (s)** : durée minimale entre deux pics successifs — évite de détecter deux pics dans le même coup. Correspond à peu près à 60/cadence_max (ex. : cadence 120 cpm → 0.50 s). Valeur conseillée : 0.25 s.

3. **Hauteur minimale du pic (m/s²)** : un pic doit dépasser ce seuil pour être compté comme un coup valide. Permet d'ignorer les micro-oscillations en dehors des phases de pagayage. Valeur conseillée : 0.2 m/s².

**Attention :** ces paramètres ne doivent être modifiés que si le nombre de coups détectés semble aberrant (trop élevé = bruit, trop faible = coups manqués).
"""

HELP_OMBRAGE = """
**Que signifient les ombrages ±1σ et ±2σ ?**

σ (sigma) est l'écart-type : il mesure la dispersion des coups autour du profil moyen.

- **±1σ (ombrage sombre)** : 68 % des coups se trouvent dans cette zone. C'est la plage de variabilité normale.  
  *Exemple : si à 30 % du cycle le profil moyen est 3.5 m/s² avec σ = 0.4, alors 68 % des coups produisent entre 3.1 et 3.9 m/s² à cet instant.*

- **±2σ (ombrage clair)** : 95 % des coups se trouvent dans cette zone. Les coups hors de cet ombrage sont des coups atypiques (départ, coups sous fatigue extrême, erreur technique).

**Interprétation pour l'entraîneur :**  
Un ombrage ±1σ étroit = athlète très régulier (bonne reproductibilité technique).  
Un ombrage ±1σ large = variabilité importante coup-à-coup (fatigue, instabilité, apprentissage).
"""

HELP_QUARTS = """
**Découpage en quarts de course**

La course est divisée en 4 segments égaux selon la distance parcourue (pas le temps).

- **Panneau gauche** : profil moyen du coup pour chaque quart. Permet de voir si la forme du coup change au fil de la course.
- **Panneau droit** : différence entre le profil de chaque quart et le 1er quart (référence = début de course).
  - Valeur positive = plus d'accélération qu'au départ (athlète en montée en puissance)
  - Valeur négative = moins d'accélération (fatigue, adaptation rythmique)

*Exemple typique en 250 m : l'accélération au catch (0–30 % du cycle) diminue dans les deux derniers quarts — signe de fatigue du catch.*
"""

HELP_HEATMAP = """
**Heatmap des coups normalisés**

Chaque ligne horizontale représente un coup de pagaie.  
L'axe horizontal = le cycle normalisé (0 % = entrée pagaie, 100 % = entrée suivante).  
La couleur = l'accélération : rouge foncé = forte propulsion, bleu foncé = fort freinage.

Les coups sont triés du premier (haut) au dernier (bas).

**Ce qu'on cherche :**  
- Des lignes horizontales régulières = technique stable  
- Un changement de couleur progressif du haut vers le bas = évolution de la technique au fil de la course (ex. : pic de propulsion qui se déplace ou s'affaiblit)
- Des lignes très différentes des voisines = coups atypiques (virage, relance, coup manqué)
"""


# ─────────────────────────────────────────────────────────────────────────────
# FONCTIONS CORE
# ─────────────────────────────────────────────────────────────────────────────

def ath_color(name, name_list):
    idx = list(name_list).index(name) if name in name_list else 0
    return plt.cm.tab10(idx / max(len(name_list), 1))


@st.cache_data(show_spinner=False)
def load_and_detect(fname, fc=FC_SMOOTH, min_d=MIN_DIST_S, min_h=MIN_PEAK_H):
    df  = pd.read_csv(os.path.join(DATA_DIR, fname)).sort_values('T').copy()
    acc = df['acc_x'].values; t = df['T'].values; D = df['D'].values
    spd = df['speed'].values if 'speed' in df.columns else np.full(len(t), np.nan)
    b, a = butter(2, fc / (FS / 2), btype='low')
    sm   = filtfilt(b, a, acc)
    peaks, _ = find_peaks(sm, height=min_h, distance=int(min_d * FS))
    troughs = [peaks[i] + np.argmin(sm[peaks[i]:peaks[i+1]])
               for i in range(len(peaks) - 1)]
    strokes = []
    for i in range(len(troughs) - 1):
        i0, i1 = troughs[i], troughs[i+1]
        sa, st_, sD = acc[i0:i1], t[i0:i1], D[i0:i1]
        if len(sa) < 8: continue
        an      = interp1d(np.linspace(0,1,len(sa)), sa)(np.linspace(0,1,N_NORM))
        auc_pos = float(np.trapezoid(np.clip(sa,0,None), st_))
        auc_neg = float(np.trapezoid(np.clip(sa,None,0), st_))
        auc_abs = float(np.trapezoid(np.abs(sa), st_))
        idx_pk  = int(np.argmax(sa))
        rfd     = float(sa[idx_pk]/max(st_[idx_pk]-st_[0],1e-6)) if idx_pk>0 else np.nan
        jerk    = float(np.sqrt(np.mean((np.diff(sa)*FS)**2)))
        pos_pic = float(idx_pk/len(sa)*100)
        above   = np.where(sa >= sa[idx_pk]/2)[0]
        fwhm    = float(st_[above[-1]]-st_[above[0]]) if len(above)>1 else np.nan
        sym     = float(abs(auc_neg)/auc_pos) if auc_pos>0 else np.nan
        strokes.append({'D_start':float(sD[0]),'D_end':float(sD[-1]),
                        'duration':float(st_[-1]-st_[0]),
                        'pic_acc':float(np.max(sa)),'pic_down':float(np.min(sa)),
                        't_acc_frac':float(np.sum(sa>0))/len(sa),
                        'd_stroke':float(sD[-1]-sD[0]),
                        'speed_moy':float(np.nanmean(spd[i0:i1])),
                        'auc_pos':auc_pos,'auc_neg':auc_neg,'auc_abs':auc_abs,
                        'rfd':rfd,'jerk_rms':jerk,'pos_pic_pct':pos_pic,
                        'fwhm_s':fwhm,'sym_ratio':sym,'acc_norm':an.tolist()})
    raw = {'T':t.tolist(),'acc_x':acc.tolist(),'D':D.tolist()}
    return strokes, raw


def to_df(strokes):
    return pd.DataFrame([{k:v for k,v in s.items() if k!='acc_norm'} for s in strokes])


def apply_filters(strokes, d_range, s_lo, s_hi):
    if not strokes: return []
    d0  = min(s['D_start'] for s in strokes)
    out = [s for s in strokes if (d0+d_range[0]) <= s['D_start'] <= (d0+d_range[1])]
    lo, hi = max(0, s_lo-1), min(len(out), s_hi)
    return out[lo:hi]


def get_mat(strokes):
    mat = np.vstack([s['acc_norm'] for s in strokes])
    return mat[np.argsort([s['D_start'] for s in strokes])]


def mean_sd(strokes):
    mat = get_mat(strokes); return mat.mean(0), mat.std(0)


def get_quarters(strokes):
    d0,d1 = min(s['D_start'] for s in strokes), max(s['D_end'] for s in strokes)
    step  = (d1-d0)/4
    return [[s for s in strokes if d0+q*step <= s['D_start'] < d0+(q+1)*step]
            for q in range(4)]


def rolling_mean(x, y, w=15):
    idx = np.argsort(x)
    return x[idx], pd.Series(y[idx]).rolling(w,center=True,min_periods=1).mean().values


def confidence_ellipse(x, y, ax, n_std=1.5, **kw):
    if len(x) < 4: return
    cov = np.cov(x,y); p = cov[0,1]/np.sqrt(cov[0,0]*cov[1,1])
    ell = Ellipse((0,0), width=np.sqrt(1+p)*2, height=np.sqrt(1-p)*2, **kw)
    T   = (transforms.Affine2D().rotate_deg(45)
           .scale(np.sqrt(cov[0,0])*n_std, np.sqrt(cov[1,1])*n_std)
           .translate(x.mean(), y.mean()))
    ell.set_transform(T+ax.transData); ax.add_patch(ell)


def style_ax(ax):
    ax.set_facecolor(C_BG)
    ax.grid(True, color=C_GRID, linewidth=0.7, alpha=0.8)
    ax.spines[['top','right']].set_visible(False)

def style_fig(fig):
    fig.patch.set_facecolor(C_BG)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────────────────────────────────────

def fig_signal_zoom(raw, strokes, t_start, t_dur, c):
    t_all = np.array(raw['T']); acc_all = np.array(raw['acc_x']); D_all = np.array(raw['D'])
    t0, t1 = t_all.min()+t_start, t_all.min()+t_start+t_dur
    mask   = (t_all>=t0)&(t_all<=t1)
    t_z, acc_z, D_z = t_all[mask]-t0, acc_all[mask], D_all[mask]
    fig, ax = plt.subplots(figsize=(14,4)); style_fig(fig); style_ax(ax)
    ax.plot(t_z, acc_z, color='#BDBDBD', lw=0.9, zorder=1)
    ax.axhline(0, color='#757575', lw=0.8, ls='--', alpha=0.6, zorder=2)
    if D_z.size > 0:
        for k, s in enumerate([s for s in strokes
                                if s['D_start']>=D_z.min() and s['D_end']<=D_z.max()+1]):
            seg = (D_all[mask]>=s['D_start'])&(D_all[mask]<=s['D_end'])
            if seg.sum()<3: continue
            t_s, acc_s = t_z[seg], acc_z[seg]; ck = plt.cm.tab10(k%10)
            ax.axvspan(t_s[0],t_s[-1],alpha=0.06,color=ck,zorder=1)
            ax.plot(t_s,acc_s,color=ck,lw=2.2,zorder=4)
            ax.fill_between(t_s,acc_s,0,where=acc_s>0,alpha=0.28,color=C_POS,zorder=3)
            ax.fill_between(t_s,acc_s,0,where=acc_s<0,alpha=0.28,color=C_NEG,zorder=3)
            ax.axvline(t_s[0],color=ck,lw=1.0,ls=':',alpha=0.7,zorder=5)
            pk = np.argmax(acc_s)
            ax.scatter(t_s[pk],acc_s[pk],color=ck,s=55,zorder=6,marker='^',
                       edgecolors='white',linewidths=0.7)
    ax.set_xlabel('Temps relatif (s)',fontsize=10)
    ax.set_ylabel('acc_x (m/s²)',fontsize=10)
    ax.set_title('Accélération longitudinale (acc_x) avec détection des coups par creux locaux\n'
                 '▲ = pic de propulsion  ·  Vert = accélération (propulsion)  ·  Rouge = décélération (freinage)',
                 fontsize=11, fontweight='bold')
    ax.legend(handles=[Patch(color=C_POS,alpha=0.55,label='Propulsion (acc_x > 0)'),
                       Patch(color=C_NEG,alpha=0.55,label='Freinage / glisse (acc_x < 0)')],
              fontsize=9,loc='upper right',framealpha=0.85)
    plt.tight_layout(); return fig


def fig_profil_auc(strokes, name, c):
    """Profil moyen avec AUC et ombrages ±1σ / ±2σ."""
    mu, sd = mean_sd(strokes); df_s = to_df(strokes)
    fig, ax = plt.subplots(figsize=(9,5)); style_fig(fig); style_ax(ax)
    ax.fill_between(xn,mu-2*sd,mu+2*sd,alpha=0.07,color=c,label='±2σ (95% des coups)')
    ax.fill_between(xn,mu-sd,mu+sd,alpha=0.20,color=c,label='±1σ (68% des coups)')
    ax.fill_between(xn,mu,0,where=mu>0,alpha=0.38,color=C_POS,
                    label=f'Propulsion · AUC+ = {df_s["auc_pos"].mean():.4f} m/s')
    ax.fill_between(xn,mu,0,where=mu<0,alpha=0.38,color=C_NEG,
                    label=f'Freinage  · |AUC-| = {df_s["auc_neg"].abs().mean():.4f} m/s')
    ax.plot(xn,mu,color=c,lw=3,zorder=5,label='Profil moyen')
    ax.axhline(0,color='#616161',lw=0.8,ls='--',alpha=0.6)
    ax.scatter(xn[np.argmax(mu)],mu.max(),color=C_POS,s=110,zorder=7,
               marker='^',edgecolors='white',linewidths=1)
    ax.scatter(xn[np.argmin(mu)],mu.min(),color=C_NEG,s=110,zorder=7,
               marker='v',edgecolors='white',linewidths=1)
    ax.text(0.015,0.975,
            f'n = {len(strokes)} coups\nSym ratio = {df_s["sym_ratio"].mean():.3f}\n'
            f'% cycle prop. = {df_s["t_acc_frac"].mean()*100:.1f}%\nPic acc = {df_s["pic_acc"].mean():.2f} m/s²',
            transform=ax.transAxes,fontsize=8.5,va='top',family='monospace',
            bbox=dict(boxstyle='round,pad=0.4',facecolor='white',alpha=0.82,edgecolor=C_GRID))
    ax.set_xlabel('Cycle normalisé (%)  ·  0% = entrée pagaie',fontsize=10)
    ax.set_ylabel('acc_x (m/s²)',fontsize=10)
    ax.set_title(f'Profil moyen du coup — {name}',fontsize=11,fontweight='bold')
    ax.legend(fontsize=8.5,loc='lower right',framealpha=0.85)
    plt.tight_layout(); return fig


def fig_tous_coups_no_cbar(strokes, name, c):
    """Coups individuels + profil moyen, SANS colorbar ni légende distance."""
    mat = get_mat(strokes); mu, sd = mat.mean(0), mat.std(0)
    d_s  = np.sort([s['D_start'] for s in strokes])
    d_n  = (d_s-d_s.min())/(d_s.max()-d_s.min()+1e-9)
    fig, ax = plt.subplots(figsize=(8,5)); style_fig(fig); style_ax(ax)
    for ai, dr in zip(mat, d_n):
        ax.plot(xn,ai,color=plt.cm.RdYlGn(dr),lw=0.55,alpha=0.15,zorder=1)
    ax.fill_between(xn,mu-sd,mu+sd,alpha=0.22,color=C_MEAN,zorder=2)
    ax.plot(xn,mu,color=C_MEAN,lw=3.5,zorder=3,label=f'Profil moyen  (n={len(strokes)})')
    ax.axhline(0,color='#616161',lw=0.8,ls='--',alpha=0.6)
    ax.scatter(xn[np.argmax(mu)],mu.max(),color=C_POS,s=90,zorder=5,marker='^')
    ax.scatter(xn[np.argmin(mu)],mu.min(),color=C_NEG,s=90,zorder=5,marker='v')
    ax.set_xlabel('Cycle normalisé (%)',fontsize=10)
    ax.set_ylabel('acc_x (m/s²)',fontsize=10)
    ax.set_title(f'Coups individuels\n(rouge=début → vert=fin)',fontsize=11,fontweight='bold')
    ax.legend(fontsize=9)
    plt.tight_layout(); return fig


def fig_heatmap(strokes, name, c):
    mat = get_mat(strokes); vmax = np.abs(mat).max()
    fig, ax = plt.subplots(figsize=(9,5)); style_fig(fig); style_ax(ax)
    im = ax.imshow(mat,aspect='auto',origin='upper',cmap='RdBu_r',vmin=-vmax,vmax=vmax)
    plt.colorbar(im,ax=ax,label='acc_x (m/s²)',shrink=0.85)
    ax.set_title(f'Heatmap des coups  (n={len(mat)}, triés par distance)',
                 fontsize=11,fontweight='bold',color=c)
    ax.set_xlabel('Cycle normalisé (%)',fontsize=10)
    ax.set_ylabel('Numéro de coup',fontsize=10)
    ax.set_xticks(np.linspace(0,N_NORM-1,6))
    ax.set_xticklabels([f'{int(v)}%' for v in np.linspace(0,100,6)])
    plt.tight_layout(); return fig


def fig_quarts_individuel(strokes, name):
    quarters = get_quarters(strokes)
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(14,5)); style_fig(fig)
    style_ax(ax1); style_ax(ax2)
    profiles = []
    for q_str,qc,ql in zip(quarters,QUART_COLORS,QUART_LABELS):
        if not q_str: profiles.append(None); continue
        mat_q = np.vstack([s['acc_norm'] for s in q_str])
        mu_q  = mat_q.mean(0); sd_q = mat_q.std(0)
        auc_q = np.mean([s['auc_pos'] for s in q_str])
        profiles.append(mu_q)
        ax1.fill_between(xn,mu_q-sd_q,mu_q+sd_q,alpha=0.10,color=qc)
        ax1.plot(xn,mu_q,color=qc,lw=2.5,
                 label=f'{ql}  (n={len(q_str)}, AUC+={auc_q:.4f})')
    ax1.axhline(0,color='#616161',lw=0.8,ls='--',alpha=0.6)
    ax1.set_xlabel('Cycle normalisé (%)',fontsize=10)
    ax1.set_ylabel('acc_x (m/s²)',fontsize=10)
    ax1.set_title('Profils moyens par quart',fontsize=11,fontweight='bold')
    ax1.legend(fontsize=8,loc='lower right')

    ref = profiles[0]
    if ref is not None:
        for mu_q,qc,ql in zip(profiles[1:],QUART_COLORS[1:],QUART_LABELS[1:]):
            if mu_q is None: continue
            diff = mu_q-ref
            ax2.fill_between(xn,diff,0,alpha=0.18,color=qc)
            ax2.plot(xn,diff,color=qc,lw=2.5,label=f'{ql} − 1er')
        ax2.axhline(0,color='k',lw=1,alpha=0.5)
    ax2.set_xlabel('Cycle normalisé (%)',fontsize=10)
    ax2.set_ylabel('Δ acc_x vs 1er quart (m/s²)',fontsize=10)
    ax2.set_title('Différence vs début de course\n(+ = plus puissant · − = fatigue)',
                  fontsize=11,fontweight='bold')
    ax2.legend(fontsize=8)
    fig.suptitle(f'{name}  ·  Évolution du coup par quart de course',
                 fontsize=12,fontweight='bold')
    plt.tight_layout(); return fig


def fig_evolution(strokes_dict, metric, name_list, w):
    ylabel,title = METRIC_META[metric]
    fig, ax = plt.subplots(figsize=(13,5)); style_fig(fig); style_ax(ax)
    for name in name_list:
        strokes = strokes_dict.get(name,[])
        if not strokes: continue
        df_s = to_df(strokes); df_s['D_rel'] = df_s['D_start']-df_s['D_start'].min()
        c = ath_color(name,name_list)
        x,y = df_s['D_rel'].values, df_s[metric].values
        mask = np.isfinite(y)
        if mask.sum()<3: continue
        ax.scatter(x[mask],y[mask],alpha=0.10,s=12,color=c,zorder=1)
        xs,ys = rolling_mean(x[mask],y[mask],w)
        ax.plot(xs,ys,color=c,lw=2.5,label=name.split()[-1],zorder=3)
    ax.set_xlabel('Distance relative (m)',fontsize=11)
    ax.set_ylabel(ylabel,fontsize=11)
    ax.set_title(f'{title}\nMoyenne glissante sur {w} coups · points individuels en transparence',
                 fontsize=11,fontweight='bold')
    ax.legend(fontsize=9,ncol=2,framealpha=0.85)
    plt.tight_layout(); return fig


def fig_quarts_multi(strokes_dict, name_list):
    n = len(name_list)
    if n==0: return plt.figure()
    fig,axes = plt.subplots(1,n,figsize=(5.5*n,5),sharey=True); style_fig(fig)
    axes_list = [axes] if n==1 else axes.tolist()
    for ax,name in zip(axes_list,name_list):
        style_ax(ax); strokes = strokes_dict.get(name,[])
        if not strokes:
            ax.text(0.5,0.5,'—',ha='center',va='center',transform=ax.transAxes); continue
        for q_str,qc,ql in zip(get_quarters(strokes),QUART_COLORS,QUART_LABELS):
            if not q_str: continue
            mu_q = np.vstack([s['acc_norm'] for s in q_str]).mean(0)
            ax.plot(xn,mu_q,color=qc,lw=2.5,label=ql[:9])
        ax.axhline(0,color='#616161',lw=0.8,ls='--',alpha=0.6)
        ax.set_title(name.split()[-1],fontsize=11,fontweight='bold',
                     color=ath_color(name,name_list))
        ax.set_xlabel('Cycle normalisé (%)',fontsize=9)
        if ax is axes_list[0]: ax.set_ylabel('acc_x (m/s²)',fontsize=9)
        ax.legend(fontsize=7,loc='lower right')
    fig.suptitle('Profil par quart de course — comparaison inter-athlètes',
                 fontsize=13,fontweight='bold')
    plt.tight_layout(); return fig


def fig_enveloppe_superposee(strokes_dict, name_list):
    fig, ax = plt.subplots(figsize=(13,6)); style_fig(fig); style_ax(ax)
    for name in name_list:
        strokes = strokes_dict.get(name,[])
        if not strokes: continue
        mu,sd = mean_sd(strokes); c = ath_color(name,name_list)
        df_s = to_df(strokes)
        ax.fill_between(xn,mu-sd,mu+sd,alpha=0.10,color=c)
        ax.plot(xn,mu,color=c,lw=2.8,
                label=f'{name.split()[-1]}  (AUC+={df_s["auc_pos"].mean():.4f})')
    ax.axhline(0,color='#616161',lw=1,ls='--',alpha=0.5)
    ax.set_xlabel('Cycle normalisé (%)  ·  0% = entrée pagaie',fontsize=11)
    ax.set_ylabel('acc_x (m/s²)',fontsize=11)
    ax.set_title('Comparaison des profils moyens de coup  ·  Ombrage ±1σ',
                 fontsize=12,fontweight='bold')
    ax.legend(fontsize=9,ncol=2,framealpha=0.85)
    plt.tight_layout(); return fig


def fig_scatter(strokes_dict, name_list, metric_y):
    ylabel = METRIC_META.get(metric_y,(metric_y,''))[0]
    fig, ax = plt.subplots(figsize=(11,7)); style_fig(fig); style_ax(ax)
    for name in name_list:
        strokes = strokes_dict.get(name,[])
        if not strokes: continue
        df_s = to_df(strokes); c = ath_color(name,name_list)
        x = df_s['auc_pos'].values; y = df_s[metric_y].abs().values
        mask = np.isfinite(x)&np.isfinite(y); x,y = x[mask],y[mask]
        if len(x)<3: continue
        ax.scatter(x,y,alpha=0.15,s=18,color=c,zorder=2)
        confidence_ellipse(x,y,ax,n_std=1.5,facecolor=c,alpha=0.07,
                           edgecolor=c,linewidth=1.5,linestyle='--',zorder=3)
        cx,cy = x.mean(),y.mean()
        ax.scatter(cx,cy,color=c,s=230,zorder=5,edgecolors='white',linewidths=1.5,
                   marker='D',label=f'{name.split()[-1]}')
        ax.annotate(name.split()[-1],(cx,cy),textcoords='offset points',xytext=(8,5),
                    fontsize=9.5,color=c,fontweight='bold')
    if metric_y=='auc_neg':
        vals = [to_df(strokes_dict[n])['auc_pos'].max()
                for n in name_list if strokes_dict.get(n)]
        if vals:
            lim = max(vals)*1.05
            ax.plot([0,lim],[0,lim],'k--',lw=1,alpha=0.25,label='AUC+ = |AUC-|')
    ax.set_xlabel('AUC+ (m/s)  ·  Impulsion de propulsion',fontsize=11)
    ax.set_ylabel(ylabel,fontsize=11)
    ax.set_title(f'AUC+ vs {ylabel}\nPoint = 1 coup · ♦ = centroïde · Ellipse ±1.5σ',
                 fontsize=11,fontweight='bold')
    ax.legend(fontsize=8.5,ncol=2,framealpha=0.85)
    plt.tight_layout(); return fig


def fig_dendrogrammes(strokes_dict, name_list):
    valid = [n for n in name_list if strokes_dict.get(n)]
    if len(valid)<2:
        fig,ax=plt.subplots()
        ax.text(0.5,0.5,'Sélectionnez ≥ 2 athlètes',ha='center',va='center',
                transform=ax.transAxes,fontsize=13)
        return fig
    means_d = {n:np.vstack([s['acc_norm'] for s in strokes_dict[n]]).mean(0) for n in valid}
    def feat(n):
        df_s=to_df(strokes_dict[n])
        return [df_s['auc_pos'].mean(),df_s['auc_neg'].abs().mean(),
                df_s['rfd'].mean()/100,df_s['jerk_rms'].mean()/1000,
                df_s['pos_pic_pct'].mean()/100,df_s['sym_ratio'].mean(),
                df_s['duration'].mean()]
    fig,axes=plt.subplots(1,2,figsize=(16,6)); style_fig(fig)
    for ax in axes: style_ax(ax)
    fig.suptitle('Similarité entre athlètes  ·  Clustering hiérarchique (Ward)',
                 fontsize=13,fontweight='bold')
    for ax,(title,dist) in zip(axes,[
        ('Forme du coup\n(corrélation profils normalisés)',
         pdist([means_d[n] for n in valid],metric='correlation')),
        ('Métriques scalaires\n(AUC+, AUC-, RFD, Jerk, pos_pic, sym, durée)',
         pdist([feat(n) for n in valid],metric='euclidean'))]):
        Z = linkage(dist,method='ward')
        dendrogram(Z,labels=[n.split()[-1] for n in valid],ax=ax,
                   leaf_rotation=30,leaf_font_size=11,
                   color_threshold=0.6*max(Z[:,2]))
        for lbl in ax.get_xticklabels():
            for n in valid:
                if n.split()[-1]==lbl.get_text():
                    lbl.set_color(ath_color(n,name_list))
        ax.set_title(title,fontsize=10,fontweight='bold')
        ax.set_ylabel('Distance (Ward)',fontsize=9)
    plt.tight_layout(); return fig


def fig_matrice(strokes_dict, name_list):
    """Matrice de corrélation avec échelle adaptée aux corrélations positives."""
    valid = [n for n in name_list if strokes_dict.get(n)]
    if len(valid)<2:
        fig,ax=plt.subplots()
        ax.text(0.5,0.5,'Sélectionnez ≥ 2 athlètes',ha='center',va='center',
                transform=ax.transAxes,fontsize=13)
        return fig
    means_d = {n:np.vstack([s['acc_norm'] for s in strokes_dict[n]]).mean(0) for n in valid}
    n_v = len(valid)
    corr = np.array([[pearsonr(means_d[a],means_d[b])[0] for b in valid] for a in valid])
    short = [n.split()[-1] for n in valid]

    # Échelle adaptative : vmin = plancher légèrement sous la valeur min hors diagonale
    off_diag = corr[~np.eye(n_v,dtype=bool)]
    vmin_auto = max(0.0, off_diag.min() - 0.05)
    vmax_auto = 1.0

    fig,ax = plt.subplots(figsize=(max(6,n_v*1.3),max(5,n_v*1.1)))
    style_fig(fig); style_ax(ax)
    im = ax.imshow(corr,cmap='YlOrRd',vmin=vmin_auto,vmax=vmax_auto)
    ax.set_xticks(range(n_v)); ax.set_xticklabels(short,rotation=40,ha='right',fontsize=10)
    ax.set_yticks(range(n_v)); ax.set_yticklabels(short,fontsize=10)
    for i in range(n_v):
        for j in range(n_v):
            ax.text(j,i,f'{corr[i,j]:.3f}',ha='center',va='center',fontsize=10,fontweight='bold',
                    color='white' if corr[i,j] > (vmin_auto+vmax_auto)*0.7 else '#212121')
    plt.colorbar(im,ax=ax,
                 label=f'Corrélation de Pearson  (échelle : {vmin_auto:.2f} → 1.00)',
                 shrink=0.85)
    ax.set_title('Matrice de corrélation des profils moyens\n'
                 'Échelle adaptée aux valeurs positives observées',
                 fontsize=11,fontweight='bold')
    plt.tight_layout(); return fig


def fig_distribution(strokes_dict, metric, name_list):
    ylabel = METRIC_META.get(metric,(metric,metric))[0]
    fig,ax = plt.subplots(figsize=(12,4)); style_fig(fig); style_ax(ax)
    for name in name_list:
        strokes=strokes_dict.get(name,[])
        if not strokes: continue
        c=ath_color(name,name_list); vals=to_df(strokes)[metric].dropna()
        ax.hist(vals,bins=28,alpha=0.42,color=c,label=name.split()[-1],density=True)
        ax.axvline(vals.mean(),color=c,lw=2.2,ls='--')
    ax.set_xlabel(ylabel,fontsize=11); ax.set_ylabel('Densité',fontsize=11)
    ax.set_title(f'Distribution de {ylabel}  ·  Trait pointillé = moyenne',
                 fontsize=11,fontweight='bold')
    ax.legend(fontsize=9,framealpha=0.85)
    plt.tight_layout(); return fig


# ─────────────────────────────────────────────────────────────────────────────
# APP STREAMLIT
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title='Pagaie Sprint — FFCK', page_icon='🛶',
                   layout='wide', initial_sidebar_state='expanded')

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=DM+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
.app-title{font-size:1.9rem;font-weight:700;color:#0d2137;letter-spacing:-0.5px;}
.app-sub{font-size:0.88rem;color:#546e7a;margin-bottom:1.2rem;}
.sh{font-size:1.05rem;font-weight:700;color:#0d2137;margin:1.2rem 0 0.4rem;
    border-left:3px solid #1976D2;padding-left:10px;}
.note{background:#e8f4fd;border-left:3px solid #1976D2;border-radius:0 8px 8px 0;
      padding:9px 13px;font-size:0.84rem;color:#0d2137;margin:0.5rem 0;}
.kpi-card{background:#f0f4f8;border-radius:10px;padding:12px 16px;text-align:center;
          border:1px solid #dde3ea;}
.kpi-lbl{font-size:0.70rem;color:#78909c;text-transform:uppercase;letter-spacing:0.05em;}
.kpi-val{font-size:1.4rem;font-weight:700;color:#0d2137;font-family:'DM Mono',monospace;}
</style>""", unsafe_allow_html=True)


# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('## 🛶 Sprint Canoë-Kayak')
    st.caption('Analyse des coups de pagaie · Maxi-Phyling')
    st.divider()

    st.markdown('**Athlètes**')
    all_names = list(ATHLETES_FILES.keys())
    selected  = st.multiselect('Sélectionner', all_names, default=all_names[:3])
    distance  = st.selectbox('Épreuve', ['250m', '2000m'])

    st.divider()
    st.markdown('**Filtres**')
    d_max   = 2000 if distance=='2000m' else 250
    d_range = st.slider('Fenêtre de distance (m)', 0, d_max, (0, d_max), 10)

    # Compteur numérique pour les coups (remplace le slider)
    st.markdown('**Sélection des coups**')
    col_lo, col_hi = st.columns(2)
    with col_lo:
        s_lo = st.number_input('Coup début', min_value=1, max_value=400,
                               value=1, step=1, help='Numéro du premier coup à inclure')
    with col_hi:
        s_hi = st.number_input('Coup fin', min_value=1, max_value=400,
                               value=400, step=1, help='Numéro du dernier coup à inclure')
    if s_lo > s_hi:
        st.warning('Début > Fin — vérifiez les numéros de coups.')

    st.divider()
    st.markdown('**Fenêtre signal**',
                help='Ajuste la portion du signal affichée dans l\'onglet Signal')
    t_start = st.slider('Début zoom (s)', 0.0, 120.0, 3.0, 0.5)
    t_dur   = st.slider('Durée fenêtre (s)', 2.0, 15.0, 5.0, 0.5)

    st.markdown('**Évolution temporelle**')
    roll_w = st.slider('Moy. glissante (coups)', 3, 40, 15)

    # Paramètres détection avec aide détaillée
    with st.expander('⚙️ Détection avancée  ℹ️'):
        st.markdown(HELP_DETECTION)
        st.divider()
        fc = st.slider('Lissage fc (Hz)', 1.0, 8.0, FC_SMOOTH, 0.5,
                       help='Fréquence de coupure du filtre passe-bas. Réduire si faux pics détectés.')
        md = st.slider('Distance min entre pics (s)', 0.15, 0.5, MIN_DIST_S, 0.05,
                       help='Durée minimale entre deux pics = 60/cadence_max')
        mh = st.slider('Hauteur min du pic (m/s²)', 0.05, 1.0, MIN_PEAK_H, 0.05,
                       help='Seuil pour ignorer les oscillations parasites')

    st.divider()
    st.markdown("""**Métriques — guide rapide**
| Métrique | Signification |
|---|---|
| AUC+ | Impulsion propulsion |
| \|AUC-\| | Impulsion freinage |
| Sym ratio | \|AUC-\|/AUC+ (idéal < 0.7) |
| RFD | Explosivité catch |
| Jerk | Fluidité coup |
| Pos. pic | Timing propulsion |
| CV AUC+ | Consistance coup/coup |""")


# ── CHARGEMENT ───────────────────────────────────────────────────────────────
if not selected:
    st.warning('Sélectionnez au moins un athlète.')
    st.stop()

raw_strokes, raw_signals = {}, {}
with st.spinner('Chargement et détection des coups…'):
    for name in selected:
        fname = ATHLETES_FILES[name].get(distance, '')
        fpath = os.path.join(DATA_DIR, fname) if fname else ''
        if fname and os.path.exists(fpath):
            s, sig = load_and_detect(fname, fc, md, mh)
            raw_strokes[name]=s; raw_signals[name]=sig
        else:
            raw_strokes[name]=[]; raw_signals[name]={}
            st.warning(f'Fichier {distance} introuvable : {name}')

filt_strokes = {n: apply_filters(raw_strokes[n], d_range, s_lo, s_hi) for n in selected}
valid_names  = [n for n in selected if filt_strokes.get(n)]


# ── HEADER ───────────────────────────────────────────────────────────────────
st.markdown('<div class="app-title">🛶 Analyse des coups de pagaie</div>',
            unsafe_allow_html=True)
st.markdown(f'<div class="app-sub">Méthode : creux locaux · {distance} · '
            f'{len(valid_names)} athlète(s) chargé(s)</div>', unsafe_allow_html=True)


# ── ONGLETS ───────────────────────────────────────────────────────────────────
t1, t2, t3, t4 = st.tabs([
    '① Signal',
    '② Analyse individuelle',
    '③ Comparaison',
    '④ Métriques',
])


# ══ ① SIGNAL ════════════════════════════════════════════════════════════════
with t1:
    # Point d'interrogation informatif sur le signal
    with st.expander('ℹ️ Qu\'est-ce que ce signal ? Comment lire ce graphique ?'):
        st.markdown(HELP_SIGNAL)

    st.markdown('<div class="sh">Signal brut acc_x avec coups détectés</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="note">Chaque couleur = un coup · '
                'Frontière = creux local (minimum d\'accélération entre deux pics) · '
                '▲ = pic de propulsion · Vert = acc > 0 · Rouge = acc < 0</div>',
                unsafe_allow_html=True)

    ath_s = (st.selectbox('Athlète', valid_names, key='s_ath')
             if len(valid_names) > 1 else (valid_names[0] if valid_names else None))
    if ath_s and raw_signals.get(ath_s):
        st.pyplot(fig_signal_zoom(raw_signals[ath_s], filt_strokes[ath_s],
                                  t_start, t_dur, ath_color(ath_s, selected)))
    else:
        st.info('Aucun signal disponible.')

    # KPI
    if valid_names:
        st.markdown('<div class="sh">Indicateurs de synthèse</div>',
                    unsafe_allow_html=True)
        KPI = [('auc_pos','AUC+ (m/s)','{:.4f}'),('auc_neg','|AUC-| (m/s)','{:.4f}'),
               ('sym_ratio','Sym ratio','{:.3f}'),('rfd','RFD (m/s³)','{:.2f}'),
               ('jerk_rms','Jerk RMS','{:.1f}'),('duration','Durée (s)','{:.3f}')]
        for name in valid_names:
            df_s = to_df(filt_strokes[name])
            st.markdown(f'**{name}** &nbsp; `{len(df_s)} coups`')
            cols = st.columns(len(KPI))
            for col,(m,lbl,fmt) in zip(cols,KPI):
                v = df_s[m].mean()
                if np.isfinite(v):
                    col.markdown(
                        f'<div class="kpi-card"><div class="kpi-lbl">{lbl}</div>'
                        f'<div class="kpi-val">{fmt.format(v)}</div></div>',
                        unsafe_allow_html=True)
            st.write('')


# ══ ② ANALYSE INDIVIDUELLE ══════════════════════════════════════════════════
with t2:
    if not valid_names:
        st.info('Aucun coup disponible.')
    else:
        ath = (st.selectbox('Athlète', valid_names, key='ind_ath')
               if len(valid_names) > 1 else valid_names[0])
        strokes = filt_strokes.get(ath, [])
        if not strokes:
            st.warning('Aucun coup dans la sélection.')
        else:
            c = ath_color(ath, selected)

            # ── Ligne 1 : coups individuels (gauche) + profil moyen (droite) ──
            st.markdown('<div class="sh">Coups individuels et profil moyen</div>',
                        unsafe_allow_html=True)
            with st.expander('ℹ️ Que signifient les ombrages ±1σ et ±2σ ?'):
                st.markdown(HELP_OMBRAGE)

            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(fig_tous_coups_no_cbar(strokes, ath, c))
            with col2:
                st.pyplot(fig_profil_auc(strokes, ath, c))

            # ── Ligne 2 : quarts de course ────────────────────────────────────
            st.markdown('<div class="sh">Évolution par quart de course</div>',
                        unsafe_allow_html=True)
            with st.expander('ℹ️ Comment lire ce graphique ?'):
                st.markdown(HELP_QUARTS)
            st.pyplot(fig_quarts_individuel(strokes, ath))

            # ── Ligne 3 : heatmap (gauche) + espace vide (droite) ────────────
            st.markdown('<div class="sh">Heatmap des coups</div>',
                        unsafe_allow_html=True)
            with st.expander('ℹ️ Comment lire la heatmap ?'):
                st.markdown(HELP_HEATMAP)

            col_hm, col_empty = st.columns([1, 1])
            with col_hm:
                st.pyplot(fig_heatmap(strokes, ath, c))
            with col_empty:
                # Encart récapitulatif des métriques avancées
                st.markdown('**Récapitulatif métriques pour cet athlète**')
                df_s = to_df(strokes)
                def fmt_m(col, d=3):
                    return f'{df_s[col].mean():.{d}f} ± {df_s[col].std():.{d}f}'
                cv_auc = df_s['auc_pos'].std()/df_s['auc_pos'].mean()*100
                recap = {
                    'AUC+ (m/s)':          fmt_m('auc_pos', 4),
                    '|AUC-| (m/s)':        fmt_m('auc_neg', 4),
                    'Sym ratio':           fmt_m('sym_ratio', 3),
                    'RFD (m/s³)':          fmt_m('rfd', 2),
                    'Jerk RMS (m/s³)':     f'{df_s["jerk_rms"].mean():.1f} ± {df_s["jerk_rms"].std():.1f}',
                    'Position pic (%)':    fmt_m('pos_pic_pct', 1),
                    'FWHM (ms)':           f'{df_s["fwhm_s"].mean()*1000:.0f} ± {df_s["fwhm_s"].std()*1000:.0f}',
                    'CV AUC+ (%)':         f'{cv_auc:.1f}%',
                    'Durée coup (s)':      fmt_m('duration', 3),
                    'Distance/coup (m)':   fmt_m('d_stroke', 2),
                }
                df_recap = pd.DataFrame.from_dict(recap, orient='index',
                                                   columns=['Moyenne ± écart-type'])
                st.dataframe(df_recap, use_container_width=True)


# ══ ③ COMPARAISON ════════════════════════════════════════════════════════════
with t3:
    if len(valid_names) < 2:
        st.info('Sélectionnez ≥ 2 athlètes pour la comparaison.')
    else:
        # 1. Profils moyens superposés
        st.markdown('<div class="sh">1 · Profils moyens superposés</div>',
                    unsafe_allow_html=True)
        st.pyplot(fig_enveloppe_superposee(filt_strokes, valid_names))

        # 2. Évolution par quart (comparaison)
        st.markdown('<div class="sh">2 · Profils par quart de course — comparaison</div>',
                    unsafe_allow_html=True)
        st.pyplot(fig_quarts_multi(filt_strokes, valid_names))

        # 3. Évolution des métriques
        st.markdown('<div class="sh">3 · Évolution des métriques au fil de la distance</div>',
                    unsafe_allow_html=True)
        metrics_sel = st.multiselect(
            'Métriques',
            options=list(METRIC_META.keys()),
            default=['auc_pos','sym_ratio','rfd'],
            format_func=lambda x: METRIC_META[x][1],
            key='cmp_metrics'
        )
        for m in metrics_sel:
            st.pyplot(fig_evolution(filt_strokes, m, valid_names, roll_w))
            st.write('')

        # 4. Scatter
        st.markdown('<div class="sh">4 · Relation propulsion / autre métrique</div>',
                    unsafe_allow_html=True)
        scatter_y = st.selectbox(
            'Axe Y',
            ['auc_neg','sym_ratio','rfd','jerk_rms','pos_pic_pct'],
            format_func=lambda x: METRIC_META[x][1], key='sc_y')
        st.pyplot(fig_scatter(filt_strokes, valid_names, scatter_y))

        # 5. Similarité
        st.markdown('<div class="sh">5 · Similarité entre athlètes</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="note">Branches proches = athlètes similaires · '
                    'Gauche = forme du coup · Droite = métriques de puissance et timing</div>',
                    unsafe_allow_html=True)
        st.pyplot(fig_dendrogrammes(filt_strokes, valid_names))

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown('**Corrélation des profils moyens**')
            st.markdown('<div class="note">Échelle adaptée aux valeurs observées — '
                        'les différences subtiles entre athlètes sont ainsi visibles.</div>',
                        unsafe_allow_html=True)
            st.pyplot(fig_matrice(filt_strokes, valid_names))
        with col_b:
            st.markdown('**Distributions comparées**')
            dm = st.selectbox('Métrique', list(METRIC_META.keys()),
                              format_func=lambda x: METRIC_META[x][0], key='dm4')
            st.pyplot(fig_distribution(filt_strokes, dm, valid_names))


# ══ ④ MÉTRIQUES ══════════════════════════════════════════════════════════════
with t4:
    if not valid_names:
        st.info('Aucun coup disponible.')
    else:
        st.markdown('<div class="sh">Métriques principales</div>', unsafe_allow_html=True)
        rows1 = []
        for name in valid_names:
            df_s = to_df(filt_strokes[name])
            def f(col, d=4): return f'{df_s[col].mean():.{d}f} ± {df_s[col].std():.{d}f}'
            rows1.append({'Athlète':name,'N coups':len(df_s),
                          'AUC+ (m/s)':f('auc_pos'),'|AUC-| (m/s)':f('auc_neg'),
                          'Sym ratio':f('sym_ratio',3),'Durée (s)':f('duration',3),
                          'Dist/coup (m)':f('d_stroke',2),
                          '% prop.':f"{df_s['t_acc_frac'].mean()*100:.1f}%"})
        st.dataframe(pd.DataFrame(rows1).set_index('Athlète'), use_container_width=True)

        st.markdown('<div class="sh">Métriques avancées</div>', unsafe_allow_html=True)
        st.caption('RFD = explosivité catch · Jerk = fluidité · FWHM = durée propulsion intense · CV = consistance')
        rows2 = []
        for name in valid_names:
            df_s = to_df(filt_strokes[name])
            def f(col, d=2): return f'{df_s[col].mean():.{d}f} ± {df_s[col].std():.{d}f}'
            cv_a = df_s['auc_pos'].std()/df_s['auc_pos'].mean()*100
            cv_j = df_s['jerk_rms'].std()/df_s['jerk_rms'].mean()*100
            rows2.append({'Athlète':name,'RFD (m/s³)':f('rfd'),
                          'Jerk RMS':f('jerk_rms',1),
                          'Position pic (%)':f('pos_pic_pct',1),
                          'FWHM (ms)':f'{df_s["fwhm_s"].mean()*1000:.0f} ± {df_s["fwhm_s"].std()*1000:.0f}',
                          'Sym ratio':f('sym_ratio',3),
                          'CV AUC+':f'{cv_a:.1f}%','CV Jerk':f'{cv_j:.1f}%'})
        st.dataframe(pd.DataFrame(rows2).set_index('Athlète'), use_container_width=True)

        st.markdown('<div class="sh">Distributions</div>', unsafe_allow_html=True)
        dm5 = st.selectbox('Métrique', list(METRIC_META.keys()),
                           format_func=lambda x: METRIC_META[x][0], key='dm5')
        st.pyplot(fig_distribution(filt_strokes, dm5, valid_names))

        st.markdown('<div class="sh">Export CSV</div>', unsafe_allow_html=True)
        all_rows = []
        for name in valid_names:
            df_s = to_df(filt_strokes[name]); df_s.insert(0,'athlete',name)
            all_rows.append(df_s)
        if all_rows:
            df_exp = pd.concat(all_rows, ignore_index=True)
            st.download_button(
                '⬇️ Télécharger les données (CSV)',
                data=df_exp.to_csv(index=False).encode('utf-8'),
                file_name=f'pagaie_creux_{distance}.csv',
                mime='text/csv')
            st.caption(f'{len(df_exp)} coups · {len(df_exp.columns)-1} métriques par coup')

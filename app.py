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
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Ellipse, Patch
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr, spearmanr
import streamlit as st
import streamlit.components.v1 as components
import warnings
warnings.filterwarnings('ignore')

from phyling_client import clear_phyling_disk_cache
from registry_service import load_registre as _registry_load_registre
from stroke_analysis import (
    apply_filters,
    cache_path as _analysis_cache_path,
    get_mat,
    get_quarters,
    load_and_detect as _analysis_load_and_detect,
    load_with_cache as _analysis_load_with_cache,
    mean_sd,
    rolling_mean,
    to_df,
)


def render_calendar(available_dates, year, month, date_data=None, selected_dates=None):
    """
    Calendrier HTML interactif. Cliquer sur un jour le sélectionne/désélectionne.
    Les jours sélectionnés sont mis en évidence.
    Communique avec Streamlit via postMessage → window.parent.
    """
    import calendar as _cal
    MONTHS = ['','Janvier','Février','Mars','Avril','Mai','Juin',
              'Juillet','Août','Septembre','Octobre','Novembre','Décembre']

    if selected_dates is None:
        selected_dates = set(available_dates)

    days_with_data = {d for d in available_dates if d.year == year and d.month == month}
    sel_strs = {d.strftime('%Y-%m-%d') for d in selected_dates}
    cal = _cal.monthcalendar(year, month)

    rows = ""
    for week in cal:
        cells = ""
        for d in week:
            if d == 0:
                cells += '<td></td>'
                continue
            from datetime import date as _date
            day_date  = _date(year, month, d)
            day_str   = day_date.strftime('%Y-%m-%d')
            has_data  = day_date in days_with_data
            is_sel    = day_str in sel_strs

            if has_data:
                entries = (date_data or {}).get(day_date, [])
                tooltip_lines = list(entries[:4])
                if len(entries) > 4:
                    tooltip_lines.append('...')
                tooltip = '&#10;'.join(tooltip_lines)
                dot_color = '#1E88E5' if is_sel else '#546E7A'
                bg        = '#1E3A5F' if is_sel else '#1A2332'
                border    = '2px solid #1E88E5' if is_sel else '1px solid #2d3f55'
                cells += (
                    '<td class="cal-has-data" data-date="{}" '
                    'onclick="toggleDate(this)" title="{}" '
                    'style="background:{};border:{};border-radius:6px;cursor:pointer">'
                    '{}<div class="cal-dot" style="background:{}"></div></td>'
                ).format(day_str, tooltip, bg, border, d, dot_color)
            else:
                cells += '<td class="cal-day">{}</td>'.format(d)
        rows += '<tr>{}</tr>'.format(cells)

    # Passer les dates sélectionnées au JS
    sel_json = '[' + ','.join('"' + s + '"' for s in sorted(sel_strs)) + ']'

    html = """
<style>
.cal-wrap{{font-family:'DM Sans',sans-serif;width:100%;position:relative;}}
.cal-title{{text-align:center;font-size:0.76rem;font-weight:700;
            color:#90CAF9;margin-bottom:5px;text-transform:uppercase;letter-spacing:.05em}}
.cal-table{{width:100%;border-collapse:separate;border-spacing:2px;font-size:0.72rem;}}
.cal-table th{{color:#546E7A;text-align:center;padding:2px;font-weight:600;}}
.cal-table td{{text-align:center;padding:3px 2px;color:#CFD8DC;}}
.cal-has-data{{color:#FFFFFF;font-weight:700;transition:all .15s;}}
.cal-has-data:hover{{opacity:0.85;}}
.cal-dot{{width:5px;height:5px;border-radius:50%;margin:1px auto 0;transition:background .15s;}}
.cal-day{{}}
.cal-tooltip{{
  display:none;position:absolute;z-index:9999;
  background:#1A2332;border:1px solid #2d3f55;border-radius:8px;
  padding:8px 12px;font-size:0.72rem;color:#E3F2FD;
  box-shadow:0 4px 16px rgba(0,0,0,.5);
  pointer-events:none;white-space:pre-line;min-width:180px;max-width:260px;
  left:110%;top:0;
}}
.cal-hint{{font-size:0.65rem;color:#546E7A;text-align:center;margin-top:4px;}}
</style>
<div class="cal-wrap">
  <div class="cal-title">{month_name} {year}</div>
  <table class="cal-table">
    <tr><th>L</th><th>M</th><th>M</th><th>J</th><th>V</th><th>S</th><th>D</th></tr>
    {rows}
  </table>
  <div class="cal-hint">Cliquez sur un jour pour filtrer</div>
</div>
<script>
var selectedDates = {sel_json};

function toggleDate(el) {{
  var d = el.getAttribute('data-date');
  var idx = selectedDates.indexOf(d);
  if (idx === -1) {{
    selectedDates.push(d);
    el.style.background = '#1E3A5F';
    el.style.border = '2px solid #1E88E5';
    el.querySelector('.cal-dot').style.background = '#1E88E5';
  }} else {{
    selectedDates.splice(idx, 1);
    el.style.background = '#1A2332';
    el.style.border = '1px solid #2d3f55';
    el.querySelector('.cal-dot').style.background = '#546E7A';
  }}
  // Envoyer la sélection à Streamlit via query params
  var joined = selectedDates.sort().join(',');
  window.parent.postMessage({{
    type: 'streamlit:setComponentValue',
    value: joined
  }}, '*');
  // Fallback : modifier l'URL parent
  try {{
    var url = new URL(window.parent.location.href);
    url.searchParams.set('cal_dates', joined);
    window.parent.history.replaceState({{}}, '', url.toString());
    // Déclencher un event pour que Streamlit recharge
    window.parent.dispatchEvent(new Event('popstate'));
  }} catch(e) {{}}
}}

// Tooltip
(function(){{
  document.querySelectorAll('.cal-has-data[title]').forEach(function(el){{
    var tt = document.createElement('div');
    tt.className = 'cal-tooltip';
    tt.textContent = el.getAttribute('title').replace(/&#10;/g,'\n');
    el.style.position = 'relative';
    el.appendChild(tt);
    el.addEventListener('mouseenter', function(){{ tt.style.display = 'block'; }});
    el.addEventListener('mouseleave', function(){{ tt.style.display = 'none'; }});
  }});
}})();
</script>
""".format(month_name=MONTHS[month], year=year, rows=rows, sel_json=sel_json)
    return html


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
    """Coups individuels + profil moyen, sans colorbar."""
    mat = get_mat(strokes); mu, sd = mat.mean(0), mat.std(0)
    d_s = np.sort([s['D_start'] for s in strokes])
    d_n = (d_s - d_s.min()) / (d_s.max() - d_s.min() + 1e-9)
    fig, ax = plt.subplots(figsize=(8, 5)); style_fig(fig); style_ax(ax)
    for ai, dr in zip(mat, d_n):
        ax.plot(xn, ai, color=plt.cm.RdYlGn(dr), lw=0.6, alpha=0.18, zorder=1)
    ax.fill_between(xn, mu-sd, mu+sd, alpha=0.22, color=C_MEAN, zorder=2)
    ax.plot(xn, mu, color=C_MEAN, lw=3.5, zorder=3, label='Profil moyen  (n={})'.format(len(strokes)))
    ax.axhline(0, color='#616161', lw=0.8, ls='--', alpha=0.6)
    ax.scatter(xn[np.argmax(mu)], mu.max(), color=C_POS, s=90, zorder=5, marker='^')
    ax.scatter(xn[np.argmin(mu)], mu.min(), color=C_NEG, s=90, zorder=5, marker='v')
    ax.set_xlabel('Cycle normalise (%)', fontsize=10)
    ax.set_ylabel('acc_x (m/s²)', fontsize=10)
    ax.set_title(
        'Coups individuels  (n={})\nrouge = debut  ->  vert = fin'.format(len(strokes)),
        fontsize=11, fontweight='bold'
    )
    ax.legend(fontsize=9)
    plt.tight_layout(); return fig


def fig_heatmap(strokes, name, c):
    mat = get_mat(strokes); vmax = np.abs(mat).max()
    fig, ax = plt.subplots(figsize=(7,4)); style_fig(fig); style_ax(ax)
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
    fig, ax1 = plt.subplots(figsize=(7,4)); style_fig(fig); style_ax(ax1)
    for q_str,qc,ql in zip(quarters,QUART_COLORS,QUART_LABELS):
        if not q_str: continue
        mat_q = np.vstack([s['acc_norm'] for s in q_str])
        mu_q  = mat_q.mean(0); sd_q = mat_q.std(0)
        auc_q = np.mean([s['auc_pos'] for s in q_str])
        ax1.fill_between(xn,mu_q-sd_q,mu_q+sd_q,alpha=0.10,color=qc)
        ax1.plot(xn,mu_q,color=qc,lw=2.5,
                 label=f'{ql}  (n={len(q_str)}, AUC+={auc_q:.4f})')
    ax1.axhline(0,color='#616161',lw=0.8,ls='--',alpha=0.6)
    ax1.set_xlabel('Cycle normalisé (%)',fontsize=10)
    ax1.set_ylabel('acc_x (m/s²)',fontsize=10)
    ax1.set_title(f'{name}  ·  Profils moyens par quart de course',
                  fontsize=11,fontweight='bold')
    ax1.legend(fontsize=8,loc='lower right')
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
    dist = pdist([means_d[n] for n in valid], metric='correlation')
    Z    = linkage(dist, method='ward')
    fig, ax = plt.subplots(figsize=(max(8, len(valid)*1.2), 5))
    style_fig(fig); style_ax(ax)
    dendrogram(Z, labels=[n.split()[-1] for n in valid], ax=ax,
               leaf_rotation=30, leaf_font_size=11,
               color_threshold=0.6*max(Z[:,2]))
    for lbl in ax.get_xticklabels():
        for n in valid:
            if n.split()[-1]==lbl.get_text():
                lbl.set_color(ath_color(n,name_list))
    ax.set_title('Similarité entre athlètes — forme du coup (acc_x)',
                 fontsize=11, fontweight='bold')
    ax.set_ylabel('Distance (Ward)', fontsize=9)
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


def fig_vitesse(raw_signals_dict, strokes_dict, name_list, w):
    """Profil de vitesse GPS filtrée au fil de la distance."""
    fig, ax = plt.subplots(figsize=(13, 5)); style_fig(fig); style_ax(ax)
    for name in name_list:
        raw = raw_signals_dict.get(name, {})
        if not raw or 'speed' not in raw: continue
        strokes = strokes_dict.get(name, [])
        if not strokes: continue
        D_all   = np.array(raw['D'])
        spd_all = np.array(raw['speed'])
        d0 = min(s['D_start'] for s in strokes)
        d1 = max(s['D_end']   for s in strokes)
        mask = (D_all >= d0) & (D_all <= d1) & np.isfinite(spd_all) & (spd_all > 0)
        if mask.sum() < 10: continue
        D_rel = D_all[mask] - d0
        spd   = spd_all[mask]
        c     = ath_color(name, name_list)
        ax.plot(D_rel, spd, color=c, lw=0.6, alpha=0.18, zorder=1)
        w_gps = max(10, w * 8)
        spd_smooth = pd.Series(spd).rolling(w_gps, center=True, min_periods=1).mean().values
        v_moy = spd_smooth.mean()
        ax.plot(D_rel, spd_smooth, color=c, lw=2.8, zorder=3,
                label=f'{name.split()[-1]}  (moy. {v_moy:.1f} km/h)')
    ax.set_xlabel('Distance relative (m)', fontsize=11)
    ax.set_ylabel('Vitesse (km/h)', fontsize=11)
    ax.set_title('Profil de vitesse au fil de la course\n'
                 'Trait lissé + signal brut en transparence  ·  speed = GPS filtré passe-bas',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, ncol=2, framealpha=0.85)
    plt.tight_layout(); return fig


def fig_vitesse_vs_metrique(strokes_dict, name_list, metric, w):
    """Vitesse par coup vs métrique technique, sur le même axe de distance."""
    ylabel, title = METRIC_META[metric]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    style_fig(fig); style_ax(ax1); style_ax(ax2)
    for name in name_list:
        strokes = strokes_dict.get(name, [])
        if not strokes: continue
        df_s = to_df(strokes)
        df_s['D_rel'] = df_s['D_start'] - df_s['D_start'].min()
        c = ath_color(name, name_list)
        x  = df_s['D_rel'].values
        y1 = df_s['speed_moy'].values
        y2 = df_s[metric].values
        m1 = np.isfinite(y1) & (y1 > 0); m2 = np.isfinite(y2)
        if m1.sum() > 3:
            ax1.scatter(x[m1], y1[m1], alpha=0.10, s=12, color=c, zorder=1)
            xs, ys = rolling_mean(x[m1], y1[m1], w)
            ax1.plot(xs, ys, color=c, lw=2.5, label=name.split()[-1], zorder=3)
        if m2.sum() > 3:
            ax2.scatter(x[m2], y2[m2], alpha=0.10, s=12, color=c, zorder=1)
            xs, ys = rolling_mean(x[m2], y2[m2], w)
            ax2.plot(xs, ys, color=c, lw=2.5, zorder=3)
    ax1.set_ylabel('Vitesse moy./coup (km/h)', fontsize=10)
    ax1.set_title('Vitesse par coup', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=9, ncol=2, framealpha=0.85)
    ax2.set_ylabel(ylabel, fontsize=10)
    ax2.set_xlabel('Distance relative (m)', fontsize=10)
    ax2.set_title(title, fontsize=11, fontweight='bold')
    fig.suptitle('Vitesse vs technique — un coup "pas joli" peut-il aller vite ?',
                 fontsize=12, fontweight='bold')
    plt.tight_layout(); return fig


def fig_scatter_vitesse(strokes_dict, name_list, metric_x):
    """Scatter : métrique technique vs vitesse par coup."""
    xlabel = METRIC_META.get(metric_x, (metric_x, ''))[0]
    fig, ax = plt.subplots(figsize=(11, 7)); style_fig(fig); style_ax(ax)
    for name in name_list:
        strokes = strokes_dict.get(name, [])
        if not strokes: continue
        df_s = to_df(strokes); c = ath_color(name, name_list)
        x = df_s[metric_x].values
        y = df_s['speed_moy'].values
        mask = np.isfinite(x) & np.isfinite(y) & (y > 0)
        x, y = x[mask], y[mask]
        if len(x) < 3: continue
        ax.scatter(x, y, alpha=0.15, s=18, color=c, zorder=2)
        confidence_ellipse(x, y, ax, n_std=1.5, facecolor=c, alpha=0.07,
                           edgecolor=c, linewidth=1.5, linestyle='--', zorder=3)
        cx, cy = x.mean(), y.mean()
        ax.scatter(cx, cy, color=c, s=230, zorder=5, edgecolors='white',
                   linewidths=1.5, marker='D',
                   label=f'{name.split()[-1]}  ({cy:.1f} km/h)')
        ax.annotate(name.split()[-1], (cx, cy), textcoords='offset points',
                    xytext=(8, 5), fontsize=9.5, color=c, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel('Vitesse moy. par coup (km/h)', fontsize=11)
    ax.set_title(f'{xlabel} vs Vitesse par coup\n♦ = centroïde · Ellipse ±1.5σ',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8.5, ncol=2, framealpha=0.85)
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



def build_tab_temporel(valid_names, filt_strokes, raw_signals, selected, roll_w):
    st.markdown(
        '<div class="note">'
        '<b>Onglet temporaire — usage pedagogique</b><br>'
        "L'axe X est le <b>temps reel en secondes</b> depuis le debut du coup (non normalise 0-100 %).<br>"
        '<b>Avantage :</b> intuitif, on voit la duree reelle du coup.<br>'
        '<b>Limite :</b> la moyenne est tronquee a la duree du coup le plus court — '
        "c'est exactement ce probleme que resout le cycle normalise dans les autres onglets."
        '</div>',
        unsafe_allow_html=True
    )

    col_a, col_b = st.columns(2)
    with col_a:
        ath_t5 = (
            st.selectbox('Athlete', valid_names, key='t5_ath')
            if valid_names else None
        )
    strokes_t5 = filt_strokes.get(ath_t5, []) if ath_t5 else []
    with col_b:
        if strokes_t5:
            coup_num = st.number_input(
                'Numero du coup', min_value=1, max_value=len(strokes_t5),
                value=1, step=1,
                help='Utilisez les fleches ou tapez le numero'
            )
        else:
            coup_num = 1

    if not strokes_t5:
        st.info('Aucun coup disponible.')
        return

    c_t5       = ath_color(ath_t5, selected)
    stroke_sel = strokes_t5[int(coup_num) - 1]
    raw_t5     = raw_signals.get(ath_t5, {})
    if not raw_t5:
        return

    T_all   = np.array(raw_t5['T'])
    acc_all = np.array(raw_t5['acc_x'])
    D_all   = np.array(raw_t5['D'])

    # Donnees du coup selectionne
    mask_coup = (D_all >= stroke_sel['D_start']) & (D_all <= stroke_sel['D_end'])
    t_coup    = T_all[mask_coup]
    acc_coup  = acc_all[mask_coup]
    if len(t_coup) < 4:
        st.warning('Trop peu de points.')
        return

    t_rel = t_coup - t_coup[0]
    dur   = float(t_rel[-1])
    pk_i  = int(np.argmax(acc_coup))

    # Profil moyen en temps absolu (tronque a dur_min)
    all_durations = [s['duration'] for s in strokes_t5]
    dur_min = float(min(all_durations))
    dur_max = float(max(all_durations))
    n_pts   = max(4, int(dur_min * 100))
    t_mean_axis = np.linspace(0, dur_min, n_pts)

    mats = []
    d_starts = []
    for s in strokes_t5:
        mask_s = (D_all >= s['D_start']) & (D_all <= s['D_end'])
        acc_s  = acc_all[mask_s]
        if len(acc_s) >= n_pts:
            mats.append(acc_s[:n_pts])
            d_starts.append(s['D_start'])

    show_mean = len(mats) >= 3

    # ── Disposition cote a cote comme l'onglet Analyse individuelle ──────
    st.markdown('<div class="sh">Coups individuels et profil moyen — axe en secondes reelles</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # ── Gauche : tous les coups individuels + coup selectionne mis en avant
    with col1:
        d_s_all = np.array(d_starts)
        d_n_all = (d_s_all - d_s_all.min()) / (d_s_all.max() - d_s_all.min() + 1e-9) if len(d_s_all) > 1 else np.zeros(len(d_s_all))

        fig_ind, ax_ind = plt.subplots(figsize=(8, 5))
        style_fig(fig_ind); style_ax(ax_ind)

        # Tous les coups a leur duree COMPLETE (pas tronques)
        for s_i, dn_i in zip(strokes_t5, d_n_all):
            mask_si = (D_all >= s_i['D_start']) & (D_all <= s_i['D_end'])
            t_si    = T_all[mask_si]
            acc_si  = acc_all[mask_si]
            if len(t_si) > 1:
                ax_ind.plot(t_si - t_si[0], acc_si,
                            color=plt.cm.RdYlGn(dn_i), lw=0.6, alpha=0.18, zorder=1)

        # Coup selectionne mis en avant
        ax_ind.plot(t_rel, acc_coup, color=c_t5, lw=2.8, zorder=4,
                    label='Coup n{} ({:.3f} s)'.format(int(coup_num), dur))
        ax_ind.scatter(t_rel[pk_i], acc_coup[pk_i], color=C_POS,
                       s=100, zorder=6, marker='^', edgecolors='white', linewidths=1)

        # Profil moyen si dispo
        if show_mean:
            mat_m  = np.vstack(mats)
            mu_abs = mat_m.mean(0)
            ax_ind.plot(t_mean_axis, mu_abs, color=C_MEAN, lw=2.5, ls='--',
                        zorder=3, label='Profil moyen ({} coups)'.format(len(mats)))

        ax_ind.axhline(0, color='#616161', lw=0.8, ls='--', alpha=0.6)
        ax_ind.set_xlabel('Temps depuis le debut du coup (s)', fontsize=10)
        ax_ind.set_ylabel('acc_x (m/s2)', fontsize=10)
        ax_ind.set_title(
            'Coups individuels  (n={})\nrouge = debut  ->  vert = fin'.format(len(mats)),
            fontsize=11, fontweight='bold'
        )
        ax_ind.legend(fontsize=9)
        plt.tight_layout()
        st.pyplot(fig_ind)

    # ── Droite : profil moyen avec AUC + ombrages
    with col2:
        if show_mean:
            mat_m  = np.vstack(mats)
            mu_abs = mat_m.mean(0)
            sd_abs = mat_m.std(0)

            fig_moy, ax_moy = plt.subplots(figsize=(8, 5))
            style_fig(fig_moy); style_ax(ax_moy)

            # Ombrages
            ax_moy.fill_between(t_mean_axis, mu_abs - 2*sd_abs, mu_abs + 2*sd_abs,
                                 alpha=0.07, color=c_t5, label='+-2sigma (95% des coups)')
            ax_moy.fill_between(t_mean_axis, mu_abs - sd_abs, mu_abs + sd_abs,
                                 alpha=0.20, color=c_t5, label='+-1sigma (68% des coups)')

            # AUC zones (sur le profil moyen tronque)
            auc_pos_mean = np.trapezoid(np.clip(mu_abs, 0, None), t_mean_axis)
            auc_neg_mean = abs(np.trapezoid(np.clip(mu_abs, None, 0), t_mean_axis))
            ax_moy.fill_between(t_mean_axis, mu_abs, 0, where=(mu_abs > 0),
                                 alpha=0.38, color=C_POS,
                                 label='Propulsion  AUC+={:.4f} m/s'.format(auc_pos_mean))
            ax_moy.fill_between(t_mean_axis, mu_abs, 0, where=(mu_abs < 0),
                                 alpha=0.38, color=C_NEG,
                                 label='Freinage  |AUC-|={:.4f} m/s'.format(auc_neg_mean))

            ax_moy.plot(t_mean_axis, mu_abs, color=c_t5, lw=3, zorder=5, label='Profil moyen')
            ax_moy.scatter(t_mean_axis[np.argmax(mu_abs)], mu_abs.max(),
                           color=C_POS, s=110, zorder=7, marker='^', edgecolors='white', linewidths=1)
            ax_moy.scatter(t_mean_axis[np.argmin(mu_abs)], mu_abs.min(),
                           color=C_NEG, s=110, zorder=7, marker='v', edgecolors='white', linewidths=1)
            ax_moy.axhline(0, color='#616161', lw=0.8, ls='--', alpha=0.6)

            # Encart metriques
            df_s = to_df(strokes_t5)
            ax_moy.text(0.015, 0.975,
                'n = {} coups\nSym ratio = {:.3f}\n% cycle prop. = {:.1f}%\nPic acc = {:.2f} m/s2'.format(
                    len(mats),
                    df_s['sym_ratio'].mean(),
                    df_s['t_acc_frac'].mean()*100,
                    df_s['pic_acc'].mean()
                ),
                transform=ax_moy.transAxes, fontsize=8.5, va='top', family='monospace',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.82, edgecolor=C_GRID)
            )

            ax_moy.set_xlabel('Temps depuis le debut du coup (s)', fontsize=10)
            ax_moy.set_ylabel('acc_x (m/s2)', fontsize=10)
            ax_moy.set_title(
                'Profil moyen — {}\ntronque a {:.3f} s (coup le plus court)'.format(ath_t5, dur_min),
                fontsize=11, fontweight='bold'
            )
            ax_moy.legend(fontsize=8.5, loc='lower right', framealpha=0.85)
            plt.tight_layout()
            st.pyplot(fig_moy)
        else:
            st.info('Pas assez de coups pour calculer un profil moyen.')

    # Note pedagogique
    st.markdown(
        '<div class="note">'
        '<b>Pourquoi le profil moyen est-il tronque a {:.3f} s ?</b> '
        'Les coups durent entre {:.3f} s et {:.3f} s. '
        'Pour moyenner point a point, tous les tableaux doivent avoir la meme longueur. '
        'On tronque donc a la duree minimale — on perd la fin des coups plus longs.<br>'
        '<b>Le cycle normalise (0-100 %) dans les autres onglets evite ce probleme '
        'en etirant ou comprimant chaque coup pour lui donner la meme longueur.</b>'
        '</div>'.format(dur_min, min(all_durations), max(all_durations)),
        unsafe_allow_html=True
    )


# ══ ONGLET PERFORMANCE ════════════════════════════════════════════════════════

ACC_METRICS = [
    ('pic_acc',    'Pic acc (m/s²)',       'Force brute — valeur max de l\'acc propulsive'),
    ('pic_down',   'Pic freinage (m/s²)',   'Pic de décélération (négatif = fort freinage)'),
    ('auc_pos',    'AUC+ (m/s)',            'Impulsion propulsive totale par coup'),
    ('auc_neg',    'AUC- (m/s)',            'Impulsion de freinage totale par coup'),
    ('sym_ratio',  'Sym ratio',             'Équilibre freinage/propulsion |AUC-|/AUC+'),
    ('rfd',        'RFD (m/s³)',            'Rate of Force Development — montée au pic'),
    ('t_acc_frac', '% cycle prop.',         'Fraction du cycle en acc positive'),
    ('pos_pic_pct','Position pic (%)',      'À quel % du cycle le pic survient'),
    ('jerk_rms',   'Jerk RMS (m/s³)',       'Irrégularité / fluidité du signal'),
    ('fwhm_s',     'FWHM (s)',              'Durée du pic à mi-hauteur (largeur propulsion)'),
    ('cadence',    'Cadence (cpm)',          'Fréquence de pagaie'),
]

PERF_COL = '#FFD600'   # jaune doré pour la colonne vitesse


def style_ax_light(ax):
    """Style pour fonds clairs — tout le texte en sombre."""
    ax.set_facecolor('#FFFFFF')
    ax.grid(True, color='#E0E0E0', linewidth=0.7, alpha=0.9)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#BDBDBD')
    ax.tick_params(colors='#1A237E', labelsize=9, which='both')
    ax.xaxis.label.set_color('#1A237E')
    ax.yaxis.label.set_color('#1A237E')
    ax.title.set_color('#0D2137')

def style_fig_light(fig):
    """Fond blanc pour les figures de l'onglet Performance."""
    fig.patch.set_facecolor('#FFFFFF')


def build_tab_performance(valid_names, filt_strokes, selected):
    if len(valid_names) < 3:
        st.warning('Il faut au moins 3 athlètes pour cette analyse.')
        return

    # ── Calcul du tableau athlètes × métriques ──────────────────────────────
    rows = []
    for name in valid_names:
        strokes = filt_strokes.get(name, [])
        if not strokes:
            continue
        df_s = to_df(strokes)
        row  = {'Athlète': name}
        row['v_moy (km/h)'] = round(df_s['speed_moy'].mean(), 3)
        row['cadence']      = round(df_s['d_stroke'].apply(
            lambda x: df_s['speed_moy'].mean() / x * (1000/3600) if x > 0 else np.nan
        ).pipe(lambda _: df_s['speed_moy'].mean() / df_s['d_stroke'].mean() * (1000/3600)), 2)
        # remplacer cadence calculée par la vraie cadence depuis duration
        row['cadence'] = round(60 / df_s['duration'].mean(), 1)
        for key, label, _ in ACC_METRICS:
            if key == 'cadence':
                row[key] = row['cadence']
            elif key in df_s.columns:
                row[key] = round(df_s[key].mean(), 4)
            else:
                row[key] = np.nan
        # d_stroke ajouté explicitement (pas dans ACC_METRICS car tautologie avec v_moy)
        row['d_stroke'] = round(df_s['d_stroke'].mean(), 4) if 'd_stroke' in df_s.columns else np.nan
        rows.append(row)

    if len(rows) < 3:
        st.warning('Pas assez d\'athlètes avec des données.')
        return

    df_ath = pd.DataFrame(rows).sort_values('v_moy (km/h)', ascending=False).reset_index(drop=True)
    df_ath.index = df_ath.index + 1  # rang 1, 2, 3...

    v_arr   = df_ath['v_moy (km/h)'].values.astype(float)
    names   = df_ath['Athlète'].values
    n_ath   = len(names)
    metric_keys = [k for k, _, _ in ACC_METRICS]

    # ── SECTION 1 : Classement ───────────────────────────────────────────────
    st.markdown('<div class="sh">① Classement général</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="note">Chaque ligne = moyenne sur l\'ensemble des coups de la sélection. '
        'Trié par vitesse décroissante. La colonne vitesse est la variable à expliquer.</div>',
        unsafe_allow_html=True
    )

    # Affichage stylisé
    display_cols = ['Athlète', 'v_moy (km/h)'] + metric_keys
    df_display   = df_ath[display_cols].copy()

    # Colorier la colonne vitesse via gradient
    def highlight_speed(s):
        vmin, vmax = s.min(), s.max()
        normed = (s - vmin) / (vmax - vmin + 1e-9)
        styles = []
        for v in normed:
            g = int(50 + v * 150)
            styles.append(f'background-color: rgba(21,101,192,{v*0.6:.2f}); color: white; font-weight: bold')
        return styles

    st.dataframe(
        df_display.style.apply(highlight_speed, subset=['v_moy (km/h)']),
        use_container_width=True, height=320
    )

    # ── SECTION 2 : Corrélations ─────────────────────────────────────────────
    st.markdown('<div class="sh">② Corrélations avec la vitesse moyenne</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="note">'
        '<b>Pearson</b> = corrélation linéaire. '
        '<b>Spearman</b> = corrélation de rang (plus robuste aux outliers, ne suppose pas la linéarité). '
        'Avec n=' + str(n_ath) + ' athlètes, un r > 0.75 correspond à p < 0.05 — '
        'les barres d\'erreur sont larges, les tendances sont indicatives.'
        '</div>',
        unsafe_allow_html=True
    )

    corr_rows = []
    for key, label, desc in ACC_METRICS:
        if key not in df_ath.columns:
            continue
        x = df_ath[key].values.astype(float)
        mask = ~np.isnan(x) & ~np.isnan(v_arr)
        if mask.sum() < 3:
            continue
        r_p, p_p = pearsonr(x[mask], v_arr[mask])
        r_s, p_s = spearmanr(x[mask], v_arr[mask])
        corr_rows.append({
            'key': key, 'label': label, 'desc': desc,
            'pearson': r_p, 'p_pearson': p_p,
            'spearman': r_s, 'p_spearman': p_s,
        })
    if not corr_rows:
        st.warning(
            'Pas assez de données pour les corrélations. '
            'Colonnes disponibles : ' + str(list(df_ath.columns))
        )
        return
    df_corr = pd.DataFrame(corr_rows).sort_values('pearson', key=np.abs, ascending=False)

    # Figure corrélations
    fig_corr, axes = plt.subplots(1, 2, figsize=(14, 5))
    style_fig_light(fig_corr)

    for ax_i, (col, title) in enumerate(zip(['pearson', 'spearman'], ['Pearson r', 'Spearman ρ'])):
        ax = axes[ax_i]; style_ax_light(ax)
        vals  = df_corr[col].values
        labs  = df_corr['label'].values
        pvals = df_corr[f'p_{col}'].values
        colors = [C_POS if v > 0 else C_NEG for v in vals]
        bars  = ax.barh(range(len(vals)), vals, color=colors, alpha=0.82, edgecolor='#E0E0E0', lw=0.5)
        # Significance stars
        for j, (v, p) in enumerate(zip(vals, pvals)):
            star = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            x_pos = v + 0.03 * np.sign(v) if v != 0 else 0.03
            ax.text(x_pos, j, star, va='center', ha='left' if v >= 0 else 'right',
                    fontsize=9, color='#263238', fontweight='bold')
            # Valeur
            ax.text(-np.sign(v)*0.02, j, f'{v:.2f}', va='center',
                    ha='right' if v >= 0 else 'left', fontsize=8, color='#37474F')
        ax.set_yticks(range(len(labs)))
        ax.set_yticklabels(labs, fontsize=9)
        ax.axvline(0, color='#616161', lw=0.8)
        ax.axvline(0.75, color='#546E7A', lw=0.8, ls=':', alpha=0.6)
        ax.axvline(-0.75, color='#546E7A', lw=0.8, ls=':', alpha=0.6)
        ax.set_xlim(-1.1, 1.1)
        ax.set_xlabel('Corrélation avec v_moy', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig_corr)

    # Heatmap inter-métriques
    with st.expander('🔍 Heatmap des corrélations inter-métriques (redondances)'):
        metric_cols = [k for k in metric_keys if k in df_ath.columns]
        mat_corr = df_ath[metric_cols].astype(float).T.values
        # Pearson entre métriques
        n_m = len(metric_cols)
        corr_mat = np.eye(n_m)
        for i in range(n_m):
            for j in range(i+1, n_m):
                xi = mat_corr[i]; xj = mat_corr[j]
                mask = ~np.isnan(xi) & ~np.isnan(xj)
                if mask.sum() >= 3:
                    r, _ = pearsonr(xi[mask], xj[mask])
                    corr_mat[i, j] = corr_mat[j, i] = r

        fig_hm, ax_hm = plt.subplots(figsize=(11, 9))
        style_fig_light(fig_hm); style_ax_light(ax_hm)
        im = ax_hm.imshow(corr_mat, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        plt.colorbar(im, ax=ax_hm, shrink=0.8, label='Pearson r')
        labels_short = [l for _, l, _ in ACC_METRICS if _ or True]
        labels_short = [next(l for k2, l, _ in ACC_METRICS if k2 == k) for k in metric_cols]
        ax_hm.set_xticks(range(n_m)); ax_hm.set_xticklabels(labels_short, rotation=45, ha='right', fontsize=9)
        ax_hm.set_yticks(range(n_m)); ax_hm.set_yticklabels(labels_short, fontsize=9)
        for i in range(n_m):
            for j in range(n_m):
                ax_hm.text(j, i, f'{corr_mat[i,j]:.2f}', ha='center', va='center',
                           fontsize=7.5, color='white' if abs(corr_mat[i,j]) > 0.55 else '#0D2137')
        ax_hm.set_title('Corrélations entre métriques (Pearson)', fontsize=11, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig_hm)
        st.markdown(
            '<div class="note">Les paires très corrélées entre elles (|r| > 0.8) apportent '
            'une information redondante. Ex : si AUC+ et pic_acc corrèlent à 0.95, '
            'ils mesurent essentiellement la même chose.</div>',
            unsafe_allow_html=True
        )

    # ── SECTION 3 : Scatter grid ─────────────────────────────────────────────
    st.markdown('<div class="sh">③ Scatter plots — v_moy vs chaque métrique</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="note">Chaque point = un athlète. '
        'Droite de régression OLS. R² et p-value indiqués. '
        'Les athlètes sont colorés selon leur rang de vitesse (bleu foncé = plus rapide).</div>',
        unsafe_allow_html=True
    )

    valid_metrics = [(k, l, d) for k, l, d in ACC_METRICS if k in df_ath.columns
                     and not df_ath[k].isna().all()]
    n_cols = 3
    n_rows = int(np.ceil(len(valid_metrics) / n_cols))
    fig_sc, axes_sc = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 4))
    style_fig_light(fig_sc)
    axes_flat = axes_sc.flatten() if n_rows > 1 else axes_sc

    speed_norm = Normalize(vmin=v_arr.min(), vmax=v_arr.max())
    cmap_speed = plt.cm.plasma

    for idx_m, (key, label, desc) in enumerate(valid_metrics):
        ax = axes_flat[idx_m]; style_ax_light(ax)
        x  = df_ath[key].values.astype(float)
        mask = ~np.isnan(x)
        xm, ym, nm = x[mask], v_arr[mask], names[mask]

        # Scatter
        for xi, yi, ni in zip(xm, ym, nm):
            c = cmap_speed(speed_norm(yi))
            ax.scatter(xi, yi, color=c, s=110, zorder=4, edgecolors='white', linewidths=0.8)
            ax.annotate(ni.split('_')[1] if '_' in ni else ni.split()[0],
                        (xi, yi), textcoords='offset points', xytext=(5, 4),
                        fontsize=7.5, color='#0D2137', fontweight='bold')

        # Régression
        if len(xm) >= 3:
            coeffs = np.polyfit(xm, ym, 1)
            x_line = np.linspace(xm.min(), xm.max(), 100)
            ax.plot(x_line, np.polyval(coeffs, x_line),
                    color=C_MEAN, lw=1.8, ls='--', alpha=0.8, zorder=3)
            r, p = pearsonr(xm, ym)
            r2 = r**2
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
            ax.text(0.97, 0.97, f'R²={r2:.2f}  {sig}',
                    transform=ax.transAxes, fontsize=8, va='top', ha='right',
                    color='white',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=C_MEAN, alpha=0.85))

        ax.set_xlabel(label, fontsize=9)
        ax.set_ylabel('v_moy (km/h)', fontsize=9)
        ax.set_title(desc, fontsize=8.5, fontstyle='italic', pad=4)

    # Masquer les axes vides
    for idx_m in range(len(valid_metrics), len(axes_flat)):
        axes_flat[idx_m].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig_sc)

    # ── SECTION 4 : Cadence vs d_stroke ──────────────────────────────────────
    st.markdown('<div class="sh">④ Stratégie de vitesse — Cadence vs Distance par coup</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="note">'
        '<b>v_moy = cadence × d_stroke / 60 × (1000/3600)</b> — c\'est une identité mathématique. '
        'Ce graphique pose la question : <b>qui va vite parce qu\'il pagaie vite, '
        'et qui va vite parce qu\'il avance loin à chaque coup ?</b> '
        'Les deux axes sont indépendants de v_moy et apportent une vraie information stratégique.'
        '</div>',
        unsafe_allow_html=True
    )

    fig_strat, ax_st = plt.subplots(figsize=(9, 6))
    style_fig_light(fig_strat); style_ax_light(ax_st)

    cad_arr   = df_ath['cadence'].values.astype(float)
    dst_arr   = df_ath['d_stroke'].values.astype(float)

    for i, (xi, yi, ni, vi) in enumerate(zip(cad_arr, dst_arr, names, v_arr)):
        c = cmap_speed(speed_norm(vi))
        ax_st.scatter(xi, yi, color=c, s=200, zorder=4, edgecolors='white', linewidths=1.2)
        ax_st.annotate(ni.split('_')[1] if '_' in ni else ni.split()[0],
                       (xi, yi), textcoords='offset points', xytext=(8, 5),
                       fontsize=9.5, fontweight='bold', color='#0D2137')

    # Lignes iso-vitesse (hyperboles v = cad * d / 60 * 1000/3600)
    cad_range = np.linspace(cad_arr.min() * 0.9, cad_arr.max() * 1.1, 300)
    for v_iso in np.arange(np.floor(v_arr.min() - 0.5), np.ceil(v_arr.max() + 1), 0.5):
        d_iso = v_iso * 3600 / 1000 * 60 / cad_range
        mask_iso = (d_iso > 0) & (d_iso < dst_arr.max() * 1.5) & (d_iso > dst_arr.min() * 0.5)
        if mask_iso.sum() > 10:
            ax_st.plot(cad_range[mask_iso], d_iso[mask_iso],
                       color='#616161', lw=0.7, ls=':', alpha=0.5)
            mid = mask_iso.sum() // 2
            ax_st.text(cad_range[mask_iso][mid], d_iso[mask_iso][mid],
                       f'{v_iso:.1f}', fontsize=7, color='#888888', alpha=0.8,
                       ha='center', va='bottom')

    ax_st.set_xlabel('Cadence (cpm)', fontsize=11)
    ax_st.set_ylabel('Distance par coup (m)', fontsize=11)
    ax_st.set_title('Stratégie de vitesse : Cadence vs Amplitude\n(lignes pointillées = iso-vitesse en km/h)',
                    fontsize=12, fontweight='bold')

    sm = ScalarMappable(cmap=cmap_speed, norm=speed_norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax_st, label='v_moy (km/h)', shrink=0.8)
    plt.tight_layout()
    st.pyplot(fig_strat)

    # d_stroke vs AUC+ : est-ce que la force prédit l'amplitude ?
    with st.expander('🔍 La force (AUC+) prédit-elle l\'amplitude par coup ?'):
        fig_fa, ax_fa = plt.subplots(figsize=(8, 5))
        style_fig_light(fig_fa); style_ax_light(ax_fa)
        auc_pos_arr = df_ath['auc_pos'].values.astype(float)
        mask_fa = ~np.isnan(auc_pos_arr) & ~np.isnan(dst_arr)
        for xi, yi, ni, vi in zip(auc_pos_arr[mask_fa], dst_arr[mask_fa],
                                   names[mask_fa], v_arr[mask_fa]):
            c = cmap_speed(speed_norm(vi))
            ax_fa.scatter(xi, yi, color=c, s=150, zorder=4, edgecolors='white', lw=1)
            ax_fa.annotate(ni.split('_')[1] if '_' in ni else ni.split()[0],
                           (xi, yi), xytext=(5, 4), textcoords='offset points',
                           fontsize=9, color='#0D2137', fontweight='bold')
        if mask_fa.sum() >= 3:
            coeffs = np.polyfit(auc_pos_arr[mask_fa], dst_arr[mask_fa], 1)
            xr = np.linspace(auc_pos_arr[mask_fa].min(), auc_pos_arr[mask_fa].max(), 100)
            ax_fa.plot(xr, np.polyval(coeffs, xr), color=C_MEAN, lw=2, ls='--')
            r, p = pearsonr(auc_pos_arr[mask_fa], dst_arr[mask_fa])
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
            ax_fa.text(0.97, 0.97, f'R²={r**2:.2f}  {sig}',
                       transform=ax_fa.transAxes, fontsize=10, va='top', ha='right',
                       color='white',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=C_MEAN, alpha=0.7))
        ax_fa.set_xlabel('AUC+ — impulsion propulsive (m/s)', fontsize=11)
        ax_fa.set_ylabel('Distance par coup (m)', fontsize=11)
        ax_fa.set_title('AUC+ vs d_stroke : la force prédit-elle l\'amplitude ?',
                        fontsize=11, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig_fa)
        st.markdown(
            '<div class="note">Si R² est faible : l\'amplitude dépend d\'autre chose que de la force '
            '(hydrodynamique, technique de sortie, trajectoire). '
            'Si R² est fort : plus on pousse fort, plus on avance loin.</div>',
            unsafe_allow_html=True
        )

    # ── SECTION 5 : Analyse avancée ──────────────────────────────────────────
    st.markdown('<div class="sh">⑤ Analyse avancée — Leave-one-out & Profil radar</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="note">'
        'Avec n=' + str(n_ath) + ' athlètes, toute régression multiple overfite immédiatement. '
        'On utilise une approche <b>Leave-One-Out</b> : pour chaque métrique, on prédit '
        'le rang de vitesse de chaque athlète en s\'entraînant sur les n-1 autres. '
        'Le score = corrélation moyenne entre rang prédit et rang réel.'
        '</div>',
        unsafe_allow_html=True
    )

    # LOO ranking
    loo_rows = []
    ranks_real = np.argsort(np.argsort(-v_arr)).astype(float)  # 0 = plus rapide
    for key, label, desc in ACC_METRICS:
        if key not in df_ath.columns:
            continue
        x = df_ath[key].values.astype(float)
        mask = ~np.isnan(x)
        if mask.sum() < 3:
            continue
        preds = np.full(n_ath, np.nan)
        for i in range(n_ath):
            if not mask[i]:
                continue
            idx_train = [j for j in range(n_ath) if j != i and mask[j]]
            if len(idx_train) < 3:
                continue
            x_tr = x[idx_train]; v_tr = v_arr[idx_train]
            c = np.polyfit(x_tr, v_tr, 1)
            preds[i] = np.polyval(c, x[i])
        valid_pred = ~np.isnan(preds)
        if valid_pred.sum() >= 3:
            r_loo, p_loo = pearsonr(preds[valid_pred], v_arr[valid_pred])
            loo_rows.append({'key': key, 'label': label, 'r_loo': r_loo, 'p_loo': p_loo})

    if loo_rows:
        df_loo = pd.DataFrame(loo_rows).sort_values('r_loo', key=np.abs, ascending=False)
        fig_loo, ax_loo = plt.subplots(figsize=(9, 5))
        style_fig_light(fig_loo); style_ax_light(ax_loo)
        vals_loo = df_loo['r_loo'].values
        labs_loo = df_loo['label'].values
        pvals_loo = df_loo['p_loo'].values
        colors_loo = [C_POS if v > 0 else C_NEG for v in vals_loo]
        ax_loo.barh(range(len(vals_loo)), vals_loo, color=colors_loo, alpha=0.82,
                    edgecolor='#E0E0E0', lw=0.5)
        for j, (v, p) in enumerate(zip(vals_loo, pvals_loo)):
            star = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
            ax_loo.text(v + 0.02*np.sign(v), j, f'{v:.2f} {star}',
                        va='center', ha='left' if v >= 0 else 'right',
                        fontsize=8.5, color='#263238')
        ax_loo.set_yticks(range(len(labs_loo)))
        ax_loo.set_yticklabels(labs_loo, fontsize=9)
        ax_loo.axvline(0, color='#616161', lw=0.8)
        ax_loo.set_xlim(-1.1, 1.3)
        ax_loo.set_xlabel('r LOO (prédiction croisée du rang)', fontsize=10)
        ax_loo.set_title('Leave-One-Out — quelle métrique prédit le mieux la vitesse ?',
                         fontsize=11, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig_loo)

    # Profil radar : plus rapide vs plus lent vs Clement
    st.markdown('<div class="sh">Profil radar — comparaison des extrêmes</div>',
                unsafe_allow_html=True)

    radar_keys = [k for k, _, _ in ACC_METRICS
                  if k in df_ath.columns and not df_ath[k].isna().all()
                  and k != 'cadence']
    radar_labels = [next(l for k2, l, _ in ACC_METRICS if k2 == k) for k in radar_keys]

    # Normaliser 0-1 pour le radar
    radar_df = df_ath.set_index('Athlète')[radar_keys].astype(float)
    radar_norm = (radar_df - radar_df.min()) / (radar_df.max() - radar_df.min() + 1e-9)

    # Athlètes à afficher : 1er, dernier + Clement si présent
    fastest = df_ath.iloc[0]['Athlète']
    slowest = df_ath.iloc[-1]['Athlète']
    clement_name = next((n for n in names if 'zappaterra' in n.lower() or 'clement' in n.lower()), None)
    radar_athletes = list(dict.fromkeys([fastest, slowest] + ([clement_name] if clement_name else [])))

    N = len(radar_keys)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig_rad, ax_rad = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    style_fig_light(fig_rad)
    ax_rad.set_facecolor('#FAFAFA')
    ax_rad.spines['polar'].set_color('#BDBDBD')
    ax_rad.tick_params(colors='#263238', labelsize=9)
    ax_rad.set_thetagrids(np.degrees(angles[:-1]), radar_labels, fontsize=9, color='#263238')
    ax_rad.yaxis.set_tick_params(colors='#263238')
    ax_rad.set_ylim(0, 1)
    ax_rad.grid(color='#BDBDBD', alpha=0.5)

    radar_colors = [C_POS, C_NEG, PERF_COL]
    for i, ath in enumerate(radar_athletes):
        if ath not in radar_norm.index:
            continue
        vals_r = radar_norm.loc[ath, radar_keys].values.tolist()
        vals_r += vals_r[:1]
        rank_v = df_ath[df_ath['Athlète'] == ath].index[0]
        label_r = '{} (rang {}, {:.2f} km/h)'.format(
            ath.split('_')[1] if '_' in ath else ath.split()[0],
            rank_v,
            df_ath[df_ath['Athlète'] == ath]['v_moy (km/h)'].values[0]
        )
        color_r = radar_colors[i % len(radar_colors)]
        ax_rad.plot(angles, vals_r, color=color_r, lw=2.5, label=label_r)
        ax_rad.fill(angles, vals_r, color=color_r, alpha=0.10)

    ax_rad.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15),
                  fontsize=9, framealpha=0.7)
    ax_rad.set_title('Profil radar normalisé\n(0 = min groupe, 1 = max groupe)',
                     fontsize=11, fontweight='bold', color='#263238', pad=20)
    st.pyplot(fig_rad)

    st.markdown(
        '<div class="note">'
        '<b>Comment lire ce radar :</b> les valeurs sont normalisées au sein du groupe — '
        '1 = meilleur du groupe sur cette métrique, 0 = moins bon. '
        'Ce n\'est pas une valeur absolue. L\'objectif est de voir '
        'si le profil du plus rapide est systématiquement "plus grand" que celui du plus lent, '
        'ou si certaines métriques vont dans le sens contraire.'
        '</div>',
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────────────────────
# APP STREAMLIT
# ─────────────────────────────────────────────────────────────────────────────

def _get_phyling_api_key():
    return st.secrets.get("PHYLING_API_KEY", "") or os.environ.get("PHYLING_API_KEY", "")


def _has_phyling_api_key():
    return bool(_get_phyling_api_key())


def _cache_path(fname, fc, min_d, min_h):
    return _analysis_cache_path(fname, fc, min_d, min_h)


def load_with_cache(fname, fc, min_d, min_h, flip=False):
    return _analysis_load_with_cache(
        fname,
        fc,
        min_d,
        min_h,
        flip=flip,
        api_key=_get_phyling_api_key(),
    )


@st.cache_data(show_spinner=False)
def load_and_detect(fname, fc=FC_SMOOTH, min_d=MIN_DIST_S, min_h=MIN_PEAK_H,
                    flip=False):
    return _analysis_load_and_detect(
        fname,
        fc=fc,
        min_d=min_d,
        min_h=min_h,
        flip=flip,
        api_key=_get_phyling_api_key(),
    )


@st.cache_data(ttl=300, show_spinner=False)
def load_registre(use_api=False, return_status=False):
    return _registry_load_registre(
        api_key=_get_phyling_api_key(),
        use_api=use_api,
        return_status=return_status,
    )

st.set_page_config(page_title='Pagaie Sprint — FFCK', page_icon='🛶',
                   layout='wide', initial_sidebar_state='expanded')


# ─────────────────────────────────────────────────────────────────────────────
# AUTHENTIFICATION
# ─────────────────────────────────────────────────────────────────────────────

# Credentials : définis dans .streamlit/secrets.toml ou variables d'environnement
# secrets.toml :  [auth]
#                 users = {coach1 = "mdp1", coach2 = "mdp2"}
def _get_users():
    try:
        return dict(st.secrets["auth"]["users"])
    except Exception:
        pass
    # Fallback : variable d'environnement AUTH_USERS="user1:pwd1,user2:pwd2"
    raw = os.environ.get("AUTH_USERS", "")
    if raw:
        users = {}
        for pair in raw.split(","):
            if ":" in pair:
                u, p = pair.split(":", 1)
                users[u.strip()] = p.strip()
        return users
    # Valeur par défaut (à changer avant déploiement)
    return {"coach": "ffck2026", "admin": "phyling2026"}

def check_login():
    """Affiche la page de login si non authentifié. Retourne True si connecté."""
    if st.session_state.get("authenticated"):
        return True

    st.markdown("""
<style>
.login-box{max-width:380px;margin:80px auto;background:#1a2332;border-radius:16px;
           padding:40px 36px;border:1px solid #2d3f55;box-shadow:0 8px 32px rgba(0,0,0,0.4)}
.login-title{font-size:1.5rem;font-weight:700;color:#FFFFFF;text-align:center;margin-bottom:4px}
.login-sub{font-size:0.85rem;color:#90A4AE;text-align:center;margin-bottom:28px}
</style>""", unsafe_allow_html=True)

    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.markdown('<div class="login-box">', unsafe_allow_html=True)
        st.markdown('<div class="login-title">🛶 Sprint Kayak</div>', unsafe_allow_html=True)
        st.markdown('<div class="login-sub">Analyse Maxi-Phyling · FFCK</div>', unsafe_allow_html=True)

        username = st.text_input("Identifiant", key="login_user", placeholder="Votre identifiant")
        password = st.text_input("Mot de passe", type="password", key="login_pwd", placeholder="••••••••")

        if st.button("Connexion", use_container_width=True, type="primary"):
            users = _get_users()
            if username in users and users[username] == password:
                st.session_state["authenticated"] = True
                st.session_state["username"] = username
                st.rerun()
            else:
                st.error("Identifiant ou mot de passe incorrect.")

        st.markdown('</div>', unsafe_allow_html=True)
    return False

if not check_login():
    st.stop()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=DM+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
.app-title{font-size:1.9rem;font-weight:700;color:#FFFFFF;letter-spacing:-0.5px;}
.app-sub{font-size:0.88rem;color:#AAAAAA;margin-bottom:1.2rem;}
.sh{font-size:1.05rem;font-weight:700;color:#FFFFFF;margin:1.2rem 0 0.4rem;
    border-left:3px solid #1976D2;padding-left:10px;}
.note{background:#e8f4fd;border-left:3px solid #1976D2;border-radius:0 8px 8px 0;
      padding:9px 13px;font-size:0.84rem;color:#0d2137;margin:0.5rem 0;}
.kpi-card{background:#f0f4f8;border-radius:10px;padding:12px 16px;text-align:center;
          border:1px solid #dde3ea;}
.kpi-lbl{font-size:0.70rem;color:#78909c;text-transform:uppercase;letter-spacing:0.05em;}
.kpi-val{font-size:1.4rem;font-weight:700;color:#0d2137;font-family:'DM Mono',monospace;}
.portal-title{font-size:2rem;font-weight:800;color:#FFFFFF;margin-bottom:0.2rem;}
.portal-sub{font-size:0.95rem;color:#AAAAAA;margin-bottom:1.4rem;}
.portal-card{background:#f0f4f8;border:1px solid #dde3ea;border-radius:12px;padding:18px 18px;
             min-height:170px;display:flex;flex-direction:column;justify-content:space-between;}
.portal-card h3{color:#0d2137;font-size:1.05rem;margin:0 0 0.4rem;font-weight:800;}
.portal-card p{color:#536270;font-size:0.86rem;margin:0 0 1rem;line-height:1.55;}
</style>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# APPLICATION PORTAL
# ─────────────────────────────────────────────────────────────────────────────

APP_OPTIONS = {
    "portal": "Portail",
    "sprint": "Analyse Maxi-Phyling",
    "la28": "Préparation LA28",
}


def _switch_app(app_key):
    st.session_state["active_app"] = app_key
    st.rerun()


def _portal_sidebar():
    with st.sidebar:
        st.markdown("## 🛶 FFCK Sprint")
        uname = st.session_state.get("username", "")
        if uname:
            st.caption(f"Connecté : {uname}")

        current = st.session_state.get("active_app", "portal")
        labels = list(APP_OPTIONS.values())
        keys = list(APP_OPTIONS.keys())
        selected_label = st.selectbox(
            "Application",
            labels,
            index=keys.index(current) if current in keys else 0,
        )
        selected_key = keys[labels.index(selected_label)]
        if selected_key != current:
            _switch_app(selected_key)

        if st.button("Déconnexion", use_container_width=True):
            st.session_state["authenticated"] = False
            st.session_state.pop("active_app", None)
            st.rerun()


def _render_portal():
    _portal_sidebar()
    st.markdown('<div class="portal-title">Portail FFCK Sprint</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="portal-sub">Choisir un outil après connexion.</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
<div class="portal-card">
  <div>
    <h3>Analyse Maxi-Phyling</h3>
    <p>Analyse des coups de pagaie, sessions, profils individuels et comparaisons.</p>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )
        if st.button("Ouvrir l'analyse sprint", use_container_width=True, type="primary"):
            _switch_app("sprint")

    with col2:
        st.markdown(
            """
<div class="portal-card">
  <div>
    <h3>Préparation LA28</h3>
    <p>Comparatif interactif des sites de préparation finale face à Long Beach.</p>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )
        if st.button("Ouvrir l'outil LA28", use_container_width=True, type="primary"):
            _switch_app("la28")


def _render_la28_tool():
    _portal_sidebar()
    st.markdown('<div class="app-title">Préparation finale Sprint — LA28</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-sub">Outil comparatif des sites de préparation face au Marine Stadium de Long Beach.</div>',
        unsafe_allow_html=True,
    )
    html_path = Path(__file__).parent / "assets" / "comparatif-sites-la28-v2.html"
    if not html_path.exists():
        st.error("Fichier HTML LA28 introuvable dans assets/.")
        return
    components.html(html_path.read_text(encoding="utf-8"), height=1800, scrolling=True)


active_app = st.session_state.get("active_app", "portal")
if active_app == "portal":
    _render_portal()
    st.stop()
if active_app == "la28":
    _render_la28_tool()
    st.stop()


# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('## 🛶 Sprint Canoë-Kayak')
    uname = st.session_state.get('username', '')
    if uname:
        st.caption(f'Connecté : {uname}')
    sprint_app_label = st.selectbox(
        "Application",
        list(APP_OPTIONS.values()),
        index=list(APP_OPTIONS.keys()).index("sprint"),
        key="active_app_selector_sprint",
    )
    sprint_app_key = list(APP_OPTIONS.keys())[list(APP_OPTIONS.values()).index(sprint_app_label)]
    if sprint_app_key != "sprint":
        _switch_app(sprint_app_key)
    if st.button('Déconnexion', use_container_width=True):
        st.session_state['authenticated'] = False
        st.session_state.pop("active_app", None)
        st.rerun()
    st.caption('Analyse des coups de pagaie · Maxi-Phyling')
    st.divider()

    has_phyling_api = _has_phyling_api_key()

    # Bouton de rafraîchissement
    refresh_label = '🔄 Actualiser depuis Phyling' if has_phyling_api else '🔄 Recharger les données locales'
    refresh_help = (
        "Recharge la liste des sessions depuis l'API Phyling"
        if has_phyling_api else
        "Recharge le registre et les CSV présents dans data/"
    )
    if st.button(refresh_label, use_container_width=True, help=refresh_help):
        st.cache_data.clear()
        if has_phyling_api:
            clear_phyling_disk_cache()
        st.rerun()

    use_api = has_phyling_api
    data_source = 'API Phyling' if use_api else 'dossier data/'
    with st.spinner(f'Chargement des sessions ({data_source})...'):
        df_registre, api_status = load_registre(use_api=use_api, return_status=True)
    n_sessions  = len(df_registre)
    n_athletes  = df_registre['athlete'].nunique() if not df_registre.empty else 0
    st.caption(f'{n_sessions} session(s) · {n_athletes} athlète(s) · {data_source}')
    if use_api:
        if api_status.get("ok"):
            st.caption(
                f"Phyling OK · {api_status.get('kayak_selections', 0)} sélection(s) · "
                f"{api_status.get('pages', 0)} page(s)"
            )
            if api_status.get("from_disk_cache"):
                st.caption("Liste Phyling chargée depuis le cache rapide")
            if api_status.get("full_record_fallbacks", 0):
                st.caption(f"{api_status.get('full_record_fallbacks', 0)} record(s) chargés sans sélection détaillée")
        else:
            st.warning(
                f"Phyling non chargé : {api_status.get('message', 'erreur inconnue')} · "
                f"fallback local {api_status.get('local_rows', 0)} session(s)"
            )
    else:
        st.info("Mode local : ajoutez PHYLING_API_KEY dans les secrets Streamlit pour charger tout l'historique Phyling.")

    df_filt = df_registre.copy()

    from datetime import datetime as _dt
    import streamlit.components.v1 as _components

    # Helper : valeurs propres d'une colonne (sans vide/nan)
    def _vals(col):
        if col not in df_filt.columns:
            return []
        return sorted(df_filt[col].replace('nan', '').replace('', float('nan'))
                      .dropna().unique().tolist())

    # ── Calendrier + filtre date ────────────────────────────────────────────────
    st.markdown('**Date**')
    from datetime import datetime as _dt
    import streamlit.components.v1 as _components
    import calendar as _cal

    all_dates_str = df_filt['date'].replace('', float('nan')).dropna().unique().tolist()
    parsed_dates  = []
    for d in all_dates_str:
        try:
            parsed_dates.append(_dt.strptime(str(d), '%Y-%m-%d').date())
        except Exception:
            pass

    if parsed_dates:
        unique_dates = sorted(set(parsed_dates))

        # ── Période : Du / Au ────────────────────────────────────────────────
        st.markdown('**Période**')
        show_full_history = st.checkbox('Tout l\'historique', value=False, key='show_full_history')
        from datetime import timedelta as _td
        default_from = unique_dates[0] if show_full_history else max(
            unique_dates[0], unique_dates[-1] - _td(days=30)
        )
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            date_from = st.date_input('Du', value=default_from,
                                      min_value=unique_dates[0],
                                      max_value=unique_dates[-1],
                                      key='date_from')
        with col_d2:
            date_to = st.date_input('Au', value=unique_dates[-1],
                                    min_value=unique_dates[0],
                                    max_value=unique_dates[-1],
                                    key='date_to')
        df_filt = df_filt[
            (df_filt['date'] >= date_from.strftime('%Y-%m-%d')) &
            (df_filt['date'] <= date_to.strftime('%Y-%m-%d'))
        ]

    # ── Athlètes → Session → Sélections ──────────────────────────────────────
    st.markdown('**Athlètes**')
    all_athletes = sorted(df_filt['athlete'].dropna().unique().tolist())
    sel_athletes = st.multiselect('Athlètes', all_athletes, default=[],
                                  key='sel_athletes', label_visibility='collapsed')
    distance = ''

    # Pour chaque athlète sélectionné : choisir une session puis les sélections
    # ATHLETES_FILES : clé = label unique, valeur = fichier_key (rec_id:sel_id)
    ATHLETES_FILES = {}   # label_unique → fichier_key
    session_labels = {}   # label_unique → label court pour affichage

    MAX_SELECTIONS = 8    # limite pour garder les graphiques lisibles

    for ath in sel_athletes:
        # Grouper les lignes par session (date + heure)
        df_ath = df_filt[df_filt['athlete'] == ath].sort_values(['date','heure','sel'],
                                                                  ascending=[False,False,True])
        if df_ath.empty:
            continue

        # Construire la liste des sessions uniques (date + heure)
        sessions_uniq = []
        seen = set()
        for _, row in df_ath.iterrows():
            sess_key = (row['date'], row['heure'])
            if sess_key not in seen:
                seen.add(sess_key)
                n_sel = len(df_ath[(df_ath['date']==row['date']) &
                                    (df_ath['heure']==row['heure'])])
                try:
                    from datetime import datetime as _ddt
                    date_fr = _ddt.strptime(row['date'], '%Y-%m-%d').strftime('%d/%m')
                except Exception:
                    date_fr = row['date']
                heure_short = row['heure'][:5] if row.get('heure') else ''
                label_sess = f"{date_fr} · {heure_short}  ({n_sel} sél.)"
                sessions_uniq.append({
                    'key':   sess_key,
                    'label': label_sess,
                    'date':  row['date'],
                    'heure': row['heure'],
                })

        if not sessions_uniq:
            continue

        # Sélecteur de session
        st.markdown(f'<div style="font-size:0.78rem;font-weight:500;'
                    f'color:#90CAF9;margin:6px 0 2px">{ath.split()[-1]}</div>',
                    unsafe_allow_html=True)
        sess_labels = [s['label'] for s in sessions_uniq]
        sess_key_sel = 'sess_' + ath.replace(' ','_')
        chosen_sess_label = st.selectbox('Session', sess_labels, index=0,
                                          key=sess_key_sel,
                                          label_visibility='collapsed')
        chosen_sess = sessions_uniq[sess_labels.index(chosen_sess_label)]

        # Sélections disponibles pour cette session
        df_sess = df_ath[
            (df_ath['date']  == chosen_sess['date']) &
            (df_ath['heure'] == chosen_sess['heure'])
        ].sort_values('sel')

        # Construire les labels de sélection avec commentaire si dispo
        sel_options = []
        for _, row in df_sess.iterrows():
            num  = row.get('sel', '1')
            note = str(row.get('notes', '') or '').strip()
            lbl  = f"Sél. {num}" + (f" — {note[:30]}" if note else "")
            sel_options.append({'label': lbl, 'fichier': row['fichier']})

        # Checkboxes pour choisir les sélections
        for opt in sel_options:
            cb_key = 'cb_' + opt['fichier'].replace(':','_')
            col_cb, col_flip = st.columns([5, 1])
            with col_cb:
                checked = st.checkbox(opt['label'], value=False, key=cb_key)
            with col_flip:
                flip_key = 'flip_' + opt['fichier'].replace(':','_')
                is_flip = st.checkbox('↩', value=False, key=flip_key,
                                      help='Capteur place a l envers - corrige acc_x')
            if checked and len(ATHLETES_FILES) < MAX_SELECTIONS:
                try:
                    date_fr2 = _ddt.strptime(chosen_sess['date'],'%Y-%m-%d').strftime('%d/%m')
                except Exception:
                    date_fr2 = chosen_sess['date']
                num = opt['label'].split('—')[0].strip().replace('Sél. ','S')
                unique_label = f"{ath.split()[-1]} {date_fr2} {num}"
                ATHLETES_FILES[unique_label] = (opt['fichier'], is_flip)
                session_labels[unique_label] = opt['label']

    # Liste finale des labels actifs
    selected = list(ATHLETES_FILES.keys())
    st.divider()
    _is_admin = st.session_state.get('username', '') == 'admin'
    if _is_admin:
        st.markdown('**Fenêtre signal**',
                    help='Ajuste la portion du signal affichée dans l\'onglet Signal')
        t_start = st.slider('Début zoom (s)', 0.0, 120.0, 3.0, 0.5)
        t_dur   = st.slider('Durée fenêtre (s)', 2.0, 15.0, 5.0, 0.5)
    else:
        t_start = 3.0
        t_dur   = 5.0

    roll_w = 15  # valeur fixe (paramètre interne)

    # Paramètres détection avec aide détaillée
    fc, md, mh = FC_SMOOTH, MIN_DIST_S, MIN_PEAK_H  # paramètres fixes

    st.divider()
    st.markdown("""**Métriques — guide rapide**
| Métrique | Ce que ça mesure |
|---|---|
| AUC+ | Force de propulsion par coup |
| AUC- | Freinage involontaire de la pagaie |
| Sym ratio | Équilibre entre propulsion et freinage (idéal < 0.7) |
| RFD | Rapidité de mise en puissance au départ du coup |
| Jerk | Fluidité et régularité du mouvement |
| Pos. pic | Moment où le coup est le plus fort (tôt ou tard) |
| CV AUC+ | Régularité des coups sur toute la course |""")


# ── CHARGEMENT ───────────────────────────────────────────────────────────────
# Filtres de coups : valeurs par défaut (sliders supprimés pour simplifier l'UI)
d_max   = 2000 if distance == '2000m' else (1000 if distance == '1000m' else
          500  if distance == '500m'  else 250)
d_range = (0, d_max)
s_lo    = 1
s_hi    = 400

if not selected:
    st.warning('Sélectionnez au moins une sélection.')
    st.stop()

# Chargement parallèle des sélections
def _load_one(label):
    entry = ATHLETES_FILES.get(label, '')
    if not entry:
        return label, [], {}, False
    # entry peut être un tuple (fname, flip) ou juste fname
    if isinstance(entry, tuple):
        fname, flip = entry
    else:
        fname, flip = entry, False
    flip_suffix = '_flip' if flip else ''
    cp = _cache_path(fname + flip_suffix, fc, md, mh)
    from_cache = os.path.exists(cp)
    s, sig = load_with_cache(fname, fc, md, mh, flip=flip)
    return label, s, sig, from_cache

raw_strokes, raw_signals = {}, {}
_cache_hits = 0
def _get_fname_flip(lbl):
    entry = ATHLETES_FILES.get(lbl, '')
    if isinstance(entry, tuple): return entry
    return entry, False
_n_cached = sum(1 for lbl in selected
                if (lambda f,fl: f and os.path.exists(
                    _cache_path(f + ('_flip' if fl else ''), fc, md, mh)))
                (*_get_fname_flip(lbl)))
_n_to_dl  = len(selected) - _n_cached
spinner_msg = (f'Téléchargement de {_n_to_dl} sélection(s) depuis Phyling…'
               if _n_to_dl > 0 else 'Chargement depuis le cache…')

with st.spinner(spinner_msg):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_load_one, lbl): lbl for lbl in selected}
        for future in as_completed(futures):
            try:
                lbl, s, sig, from_cache = future.result()
                raw_strokes[lbl] = s
                raw_signals[lbl] = sig
                if from_cache:
                    _cache_hits += 1
                if not s:
                    st.warning(f'Données introuvables pour {lbl}')
            except Exception:
                lbl = futures[future]
                raw_strokes[lbl] = []
                raw_signals[lbl] = {}

if _cache_hits == len(selected) and len(selected) > 0:
    st.sidebar.caption('⚡ Cache — chargement instantané')
elif _cache_hits > 0:
    st.sidebar.caption(f'⚡ {_cache_hits}/{len(selected)} depuis le cache')

filt_strokes = {n: apply_filters(raw_strokes[n], d_range, s_lo, s_hi) for n in selected}
valid_names  = [n for n in selected if filt_strokes.get(n)]


# ── HEADER ───────────────────────────────────────────────────────────────────
st.markdown('<div class="app-title">🛶 Analyse des coups de pagaie</div>',
            unsafe_allow_html=True)
st.markdown(f'<div class="app-sub">Méthode : creux locaux · {len(valid_names)} sélection(s) chargée(s)</div>', unsafe_allow_html=True)


# ── ONGLETS ───────────────────────────────────────────────────────────────────
t1, t2, t3 = st.tabs([
    '① Signal',
    '② Analyse individuelle',
    '③ Comparaison',
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
        col_kpi_title, col_kpi_toggle = st.columns([5, 1])
        with col_kpi_title:
            st.markdown('<div class="sh">Indicateurs de synthèse</div>',
                        unsafe_allow_html=True)
        with col_kpi_toggle:
            highlight_on = st.toggle('Highlight', value=True, key='kpi_highlight',
                                     help='Mettre en avant le meilleur résultat par métrique')

        # Métriques affichées : (colonne_df, label, format, higher_is_better)
        # Pour auc_neg : valeur toujours <= 0, meilleur = le plus proche de 0 = False
        KPI = [
            ('auc_pos',   'AUC+',        '{:.4f}', True),
            ('auc_neg',   'AUC-',        '{:.4f}', False),
            ('sym_ratio', 'Sym ratio',   '{:.3f}', False),
            ('rfd',       'RFD',         '{:.2f}', True),
            ('duration',  'Durée coup',  '{:.3f}', False),
            ('jerk_rms',  'Jerk RMS',    '{:.1f}', False),
        ]

        # Calculer les moyennes par athlète pour chaque métrique + vitesse
        kpi_data = {}
        for name in valid_names:
            df_s = to_df(filt_strokes[name])
            row = {m: df_s[m].mean() for m, *_ in KPI if m in df_s.columns}
            # Vitesse moyenne depuis raw_signals
            raw = raw_signals.get(name, {})
            if raw and 'speed' in raw:
                spd = np.array(raw['speed'])
                spd = spd[np.isfinite(spd) & (spd > 0)]
                row['v_moy'] = float(spd.mean()) if len(spd) > 0 else np.nan
            else:
                row['v_moy'] = np.nan
            kpi_data[name] = row

        # Identifier le meilleur par colonne
        def best_athlete(metric, higher_is_better):
            vals = {n: kpi_data[n].get(metric, np.nan) for n in valid_names}
            vals = {n: v for n, v in vals.items() if np.isfinite(v)}
            if not vals:
                return None
            if metric == 'auc_neg':
                # Plus proche de 0 = moins de freinage = meilleur
                return min(vals, key=lambda n: abs(vals[n]))
            return max(vals, key=vals.get) if higher_is_better else min(vals, key=vals.get)

        best_speed = best_athlete('v_moy', True)
        best_per_kpi = {m: best_athlete(m, hib) for m, _, _, hib in KPI}

        # CSS cartes
        CARD_BEST = ('background:#1B5E20;border:2px solid #43A047;'
                     'border-radius:10px;padding:12px 16px;text-align:center;')
        CARD_NORM = ('background:#f0f4f8;border:1px solid #dde3ea;'
                     'border-radius:10px;padding:12px 16px;text-align:center;')
        LBL_BEST  = 'font-size:0.70rem;color:#A5D6A7;text-transform:uppercase;letter-spacing:0.05em;'
        LBL_NORM  = 'font-size:0.70rem;color:#78909c;text-transform:uppercase;letter-spacing:0.05em;'
        VAL_BEST  = 'font-size:1.4rem;font-weight:700;color:#FFFFFF;font-family:"DM Mono",monospace;'
        VAL_NORM  = 'font-size:1.4rem;font-weight:700;color:#0d2137;font-family:"DM Mono",monospace;'

        for name in valid_names:
            df_s      = to_df(filt_strokes[name])
            v_moy_val = kpi_data[name].get('v_moy', np.nan)
            v_str     = '{:.1f} km/h'.format(v_moy_val) if np.isfinite(v_moy_val) else '—'
            is_fastest = highlight_on and (name == best_speed)
            speed_badge = (' ★' if is_fastest else '')

            st.markdown(
                f'**{name}**&nbsp; `{len(df_s)} coups` &nbsp;'
                f'<span style="background:#E3F2FD;border-radius:6px;padding:2px 8px;'
                f'font-size:0.85rem;color:#0D47A1;font-weight:600">'
                f'{v_str}{speed_badge}</span>',
                unsafe_allow_html=True)

            cols = st.columns(len(KPI))
            for col, (m, lbl, fmt, hib) in zip(cols, KPI):
                v = kpi_data[name].get(m, np.nan)
                if np.isfinite(v):
                    is_best = highlight_on and (name == best_per_kpi.get(m))
                    card  = CARD_BEST if is_best else CARD_NORM
                    lbl_s = LBL_BEST  if is_best else LBL_NORM
                    val_s = VAL_BEST  if is_best else VAL_NORM
                    col.markdown(
                        f'<div style="{card}">'
                        f'<div style="{lbl_s}">{lbl}</div>'
                        f'<div style="{val_s}">{fmt.format(v)}</div>'
                        f'</div>',
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

            # ── Ligne 2 : quarts (gauche) + heatmap (droite) ────────────────
            st.markdown('<div class="sh">Évolution par quart de course · Heatmap</div>',
                        unsafe_allow_html=True)
            col_q, col_hm = st.columns(2)
            with col_q:
                with st.expander('ℹ️ Comment lire ce graphique ?'):
                    st.markdown(HELP_QUARTS)
                st.pyplot(fig_quarts_individuel(strokes, ath))
            with col_hm:
                with st.expander('ℹ️ Comment lire la heatmap ?'):
                    st.markdown(HELP_HEATMAP)
                st.pyplot(fig_heatmap(strokes, ath, c))


# ══ ③ COMPARAISON ════════════════════════════════════════════════════════════
with t3:
    if len(valid_names) < 2:
        st.info('Sélectionnez ≥ 2 sélections pour la comparaison.')
    else:
        # Toggle vue détaillée / synthétique
        col_tog1, col_tog2 = st.columns([3,1])
        with col_tog2:
            vue_synth = st.toggle('Vue synthétique', value=False, key='vue_synth',
                                   help='Activé : une courbe moyenne par athlète · '
                                        'Désactivé : une courbe par sélection')

        # En vue synthétique : regrouper les sélections par préfixe athlète
        if vue_synth:
            # Extraire le nom de base (avant la date) de chaque label
            from collections import defaultdict as _dd
            groups = _dd(list)
            for lbl in valid_names:
                base = lbl.split(' ')[0]  # ex "MARTIN" depuis "MARTIN 13/04 S1"
                groups[base].append(lbl)
            # Créer des strokes agrégés par athlète
            synth_strokes = {}
            for base, lbls in groups.items():
                merged = []
                for lbl in lbls:
                    merged.extend(filt_strokes.get(lbl, []))
                if merged:
                    synth_strokes[base] = merged
            display_strokes = synth_strokes
            display_names   = list(synth_strokes.keys())
        else:
            display_strokes = filt_strokes
            display_names   = valid_names

        # 1. Profils moyens superposés
        st.markdown('<div class="sh">1 · Profils moyens superposés</div>',
                    unsafe_allow_html=True)
        st.pyplot(fig_enveloppe_superposee(display_strokes, display_names))

        # 2. Évolution par quart (comparaison)
        st.markdown('<div class="sh">2 · Profils par quart de course</div>',
                    unsafe_allow_html=True)
        st.pyplot(fig_quarts_multi(display_strokes, display_names))

        # 3. Dendrogramme
        st.markdown('<div class="sh">3 · Similarité — clustering hiérarchique</div>',
                    unsafe_allow_html=True)
        if len(display_names) >= 2:
            st.pyplot(fig_dendrogrammes(display_strokes, display_names))
        else:
            st.info('Sélectionnez au moins 2 entrées pour afficher le dendrogramme.')

# Onglets ④ ⑤ ⑥ masqués — décommenter pour activer

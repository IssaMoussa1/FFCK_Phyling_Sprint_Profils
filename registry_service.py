"""Session registry and local metadata handling."""

import os
import re

import numpy as np
import pandas as pd

from config import DATA_DIR, REGISTRE
from phyling_client import fetch_phyling_records_with_status, has_api_key


COLS_BASE = ['fichier', 'athlete', 'distance', 'date', 'heure', 'sel', 'notes']
COLS_META = ['discipline', 'sexe', 'categorie', 'bateau', 'type_course', 'lieu']
COLS_ALL = COLS_BASE + COLS_META


def default_api_status(api_key):
    return {
        "enabled": has_api_key(api_key),
        "ok": False,
        "pages": 0,
        "total_api": None,
        "raw_records": 0,
        "kayak_selections": 0,
        "records_with_selections": 0,
        "full_record_fallbacks": 0,
        "message": "API non interrogée",
    }


def _parse_filename(fname):
    """Extrait athlete, distance, date depuis le nom de fichier. Robuste."""
    base = os.path.splitext(os.path.basename(fname))[0]
    dist_vals = {'250', '500', '750', '1000', '2000'}

    m_date = re.search(r'([0-9]{8})_([0-9]{6})', base)
    if m_date:
        date_raw, heure_raw = m_date.group(1), m_date.group(2)
        date_str = '{}-{}-{}'.format(date_raw[:4], date_raw[4:6], date_raw[6:])
        heure_str = '{}:{}'.format(heure_raw[:2], heure_raw[2:4])
        name_part = base[:m_date.start()].strip('-_')
        suffix = base[m_date.end():].strip('-_')

        m_dist = re.search(r'([0-9]+)m', suffix, re.IGNORECASE)
        dist = m_dist.group(1) + 'm' if m_dist else ''

        if not dist:
            m_sd = re.search(r'sel[_-]([0-9]+)', suffix, re.IGNORECASE)
            if m_sd and m_sd.group(1) in dist_vals:
                dist = m_sd.group(1) + 'm'

        if not dist:
            for tok in re.split(r'[-_]', suffix):
                if tok in dist_vals:
                    dist = tok + 'm'
                    break

        m_sel = re.search(r'sel[_-]([0-9]+)', suffix, re.IGNORECASE)
        sel = m_sel.group(1) if m_sel else '1'

        segs = [s.strip('_') for s in name_part.split('-') if s.strip('_')]
        aths = [' '.join(w.capitalize() for w in s.split('_')) for s in segs]
        athlete = ' / '.join(aths) if len(aths) > 1 else (aths[0] if aths else '')
        if not athlete:
            return None
        return {'athlete': athlete, 'distance': dist, 'date': date_str,
                'heure': heure_str, 'sel': sel, 'format': 'nouveau',
                'n_athletes': len(aths)}

    m_old = re.match(r'^([a-z][a-z0-9_]+?)([0-9]{8})_?([0-9]{6})sel[_-]([0-9]+)$',
                     base, re.IGNORECASE)
    if m_old:
        name_raw, date_raw, heure_raw, dist_raw = m_old.groups()
        athlete = ' '.join(w.capitalize() for w in name_raw.strip('_').split('_'))
        date_str = '{}-{}-{}'.format(date_raw[:4], date_raw[4:6], date_raw[6:])
        heure_str = '{}:{}'.format(heure_raw[:2], heure_raw[2:4])
        dist = dist_raw + 'm' if dist_raw in dist_vals else ''
        return {'athlete': athlete, 'distance': dist, 'date': date_str,
                'heure': heure_str, 'sel': '1' if dist else dist_raw,
                'format': 'ancien', 'n_athletes': 1}
    return None


def scan_data_dir(data_dir=DATA_DIR):
    """Parcourt data_dir et retourne les CSV Maxi-Phyling reconnus."""
    rows = []
    if not os.path.isdir(data_dir):
        return rows
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith('.csv') or fname == 'registre.csv':
            continue
        info = _parse_filename(fname)
        if info is None:
            continue
        rows.append({
            'fichier': fname,
            'athlete': info['athlete'],
            'distance': info['distance'],
            'date': info['date'],
            'heure': info.get('heure', ''),
            'sel': info.get('sel', '1'),
            'notes': '',
        })
    return rows


COMMENT_DICT = {
    'K': ('discipline', 'Kayak'),
    'C': ('discipline', 'Canoë'),
    'H': ('sexe', 'H'),
    'D': ('sexe', 'F'),
    'FA': ('type_course', 'Finale A'),
    'FB': ('type_course', 'Finale B'),
    'SF': ('type_course', 'Demi-finale'),
    'BSM': ('lieu', 'Boulogne-sur-Mer'),
}
CATEGORIE_PATTERN = re.compile(r'\b(U\d{2})\b', re.IGNORECASE)
BATEAU_PATTERN = re.compile(r'\b([KC][124])\b', re.IGNORECASE)


def parse_comment(comment):
    meta = {
        'discipline': '',
        'sexe': '',
        'categorie': 'Senior',
        'bateau': '',
        'type_course': '',
        'lieu': '',
    }
    if not comment or not isinstance(comment, str):
        return meta

    for tok in comment.upper().split():
        if tok in COMMENT_DICT:
            field, val = COMMENT_DICT[tok]
            meta[field] = val

    m_cat = CATEGORIE_PATTERN.search(comment)
    if m_cat:
        meta['categorie'] = m_cat.group(1).upper()

    m_bat = BATEAU_PATTERN.search(comment)
    if m_bat:
        meta['bateau'] = m_bat.group(1).upper()
        if not meta['discipline']:
            meta['discipline'] = 'Kayak' if meta['bateau'].startswith('K') else 'Canoë'

    return meta


def parse_zip_metadata(zip_path):
    """Lit le maxi_database.xlsx dans un zip et retourne les métadonnées."""
    import io
    import zipfile

    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            names = zf.namelist()
            xlsx_name = next((n for n in names if n.endswith('maxi_database.xlsx')), None)
            if not xlsx_name:
                return None
            with zf.open(xlsx_name) as f:
                xl = pd.ExcelFile(io.BytesIO(f.read()))

            meta = {}
            if 'Record' in xl.sheet_names:
                df_rec = xl.parse('Record').fillna('')
                if not df_rec.empty:
                    row = df_rec.iloc[0]
                    comment = str(row.get('comment', ''))
                    meta.update(parse_comment(comment))
                    meta['comment_raw'] = comment
                    if not meta['discipline'] and str(row.get('sport', '')).lower() == 'kayak':
                        meta['discipline'] = 'Kayak'
                    if not meta['bateau']:
                        try:
                            import json as _json
                            od = _json.loads(str(row.get('other_data', '{}')))
                            boat = od.get('boat', '')
                            if boat:
                                meta['bateau'] = boat.upper()
                        except Exception:
                            pass

            if 'User' in xl.sheet_names:
                df_usr = xl.parse('User').fillna('')
                athletes = []
                for _, u in df_usr.iterrows():
                    fn = str(u.get('firstname', '')).strip().capitalize()
                    ln = str(u.get('lastname', '')).strip().capitalize()
                    if fn or ln:
                        athletes.append('{} {}'.format(fn, ln).strip())
                meta['athletes_zip'] = athletes

            return meta
    except Exception:
        return None


def enrich_registre_from_zips(df_reg, data_dir=DATA_DIR):
    """Enrichit le registre local avec les métadonnées des ZIP."""
    zips_dir = os.path.join(data_dir, 'zips')
    if not os.path.isdir(zips_dir):
        return df_reg

    for c in COLS_META:
        if c not in df_reg.columns:
            df_reg[c] = ''

    for zip_fname in os.listdir(zips_dir):
        if not zip_fname.endswith('.zip'):
            continue
        zip_path = os.path.join(zips_dir, zip_fname)
        meta = parse_zip_metadata(zip_path)
        if not meta:
            continue

        m_zip = re.search(r'([0-9]{8})_([0-9]{6})', zip_fname)
        if not m_zip:
            continue
        date_raw, heure_raw = m_zip.groups()
        date_str = '{}-{}-{}'.format(date_raw[:4], date_raw[4:6], date_raw[6:])
        heure_str = '{}:{}'.format(heure_raw[:2], heure_raw[2:4])
        mask = (df_reg['date'] == date_str) & (df_reg['heure'].str.startswith(heure_str[:5]))
        if mask.sum() == 0:
            continue

        for col in COLS_META:
            val = meta.get(col, '')
            if val:
                df_reg.loc[mask & (df_reg[col] == ''), col] = val

    return df_reg


def _load_local_registre(data_dir=DATA_DIR, registre_path=REGISTRE):
    df_local = pd.DataFrame(columns=COLS_ALL)
    if os.path.exists(registre_path):
        try:
            for enc in ('utf-8-sig', 'utf-8', 'latin-1'):
                try:
                    df_local = pd.read_csv(registre_path, dtype=str, encoding=enc,
                                           sep=None, engine='python').fillna('')
                    df_local.columns = [c.lstrip('\ufeff').strip() for c in df_local.columns]
                    break
                except UnicodeDecodeError:
                    continue
        except Exception:
            df_local = pd.DataFrame(columns=COLS_ALL)

    for c in COLS_ALL:
        if c not in df_local.columns:
            df_local[c] = ''
    if not df_local.empty and 'fichier' in df_local.columns:
        df_local = df_local[df_local['fichier'].apply(
            lambda f: bool(f) and (':' in str(f) or os.path.exists(os.path.join(data_dir, str(f))))
        )].copy()

    scanned = pd.DataFrame(scan_data_dir(data_dir))
    if not scanned.empty:
        for c in COLS_ALL:
            if c not in scanned.columns:
                scanned[c] = ''
        existing = set(df_local['fichier'].values) if not df_local.empty else set()
        new_files = scanned[~scanned['fichier'].isin(existing)]
        if not new_files.empty:
            df_local = pd.concat([df_local, new_files[COLS_ALL]], ignore_index=True)

    needs_zip_enrichment = (
        df_local.empty
        or any(c not in df_local.columns for c in COLS_META)
        or df_local[COLS_META].replace('', np.nan).isna().all(axis=None)
    )
    if not df_local.empty and needs_zip_enrichment:
        df_local = enrich_registre_from_zips(df_local, data_dir)

    return df_local


def load_registre(api_key="", use_api=False, return_status=False,
                  data_dir=DATA_DIR, registre_path=REGISTRE):
    """
    Charge le registre API en priorité, puis local en fallback.
    Retourne éventuellement un statut API pour diagnostic UI.
    """
    api_status = default_api_status(api_key)

    if use_api and has_api_key(api_key):
        api_records, api_status = fetch_phyling_records_with_status(
            api_key=api_key,
            page_size=500,
            days_back=None,
        )
        if api_records:
            df_reg = pd.DataFrame(api_records)
            for c in COLS_ALL:
                if c not in df_reg.columns:
                    df_reg[c] = ''
            df_reg = df_reg.drop_duplicates(subset=['fichier'], keep='first').reset_index(drop=True)
            if return_status:
                api_status["local_rows"] = 0
                api_status["final_rows"] = len(df_reg)
                return df_reg, api_status
            return df_reg

    df_local = _load_local_registre(data_dir, registre_path)
    if use_api and not has_api_key(api_key):
        api_status["message"] = "PHYLING_API_KEY absente"

    df_reg = df_local.copy()
    for c in COLS_ALL:
        if c not in df_reg.columns:
            df_reg[c] = ''
    if not df_reg.empty:
        df_reg = df_reg.drop_duplicates(subset=['fichier'], keep='first').reset_index(drop=True)

    try:
        if not df_reg.empty and os.path.isdir(data_dir):
            local_rows = df_reg[~df_reg['fichier'].astype(str).str.contains(':', regex=False)]
            if not local_rows.empty:
                local_rows[COLS_ALL].to_csv(registre_path, index=False, encoding='utf-8')
    except Exception:
        pass

    if return_status:
        api_status["local_rows"] = len(df_local)
        api_status["final_rows"] = len(df_reg)
        return df_reg, api_status
    return df_reg


def get_athletes_list(df_reg):
    if df_reg.empty:
        return []
    return sorted(df_reg['athlete'].dropna().unique().tolist())


def get_sessions_for_athlete(df_reg, athlete, distance):
    mask = (df_reg['athlete'] == athlete) & (df_reg['distance'] == distance)
    sub = df_reg[mask].sort_values(['date', 'heure', 'sel'])
    sessions = []
    for _, row in sub.iterrows():
        parts = []
        if row.get('date'):
            parts.append(row['date'])
        if row.get('heure'):
            parts.append(row['heure'])
        if row.get('sel') and row['sel'] != '1':
            parts.append('sel ' + row['sel'])
        label = ' — '.join(parts) if parts else row['fichier']
        sessions.append({
            'label': label,
            'fichier': row['fichier'],
            'date': row.get('date', ''),
            'heure': row.get('heure', ''),
            'sel': row.get('sel', '1'),
        })
    return sessions


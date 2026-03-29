"""
update_registre.py
==================
Script de mise a jour automatique du registre CSV.

Usage :
    python update_registre.py

Lancez ce script depuis le dossier FFCK_Sprint_Profil apres avoir copie
de nouveaux CSV dans data/ et les zips correspondants dans data/zips/.

Le script :
1. Detecte les nouveaux CSV dans data/
2. Lit les zips dans data/zips/ pour extraire les metadonnees automatiquement
   (discipline, sexe, categorie, bateau, lieu, type_course, noms athletes)
3. Complete le dictionnaire META avec les nouveaux athletes trouves dans les zips
4. Ajoute les nouvelles lignes au registre.csv

Les champs non trouves automatiquement sont laisses vides pour completion
manuelle dans VS Code.
"""

import os, re, io, json, zipfile, pandas as pd
from datetime import datetime

# Chemins
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, 'data')
ZIPS_DIR   = os.path.join(DATA_DIR, 'zips')
REGISTRE   = os.path.join(DATA_DIR, 'registre.csv')

DIST_VALS  = {'250', '500', '750', '1000', '2000'}

# Dictionnaire des commentaires Phyling
COMMENT_DICT = {
    'K':    ('discipline', 'Kayak'),
    'C':    ('discipline', 'Canoë'),
    'H':    ('sexe', 'H'),
    'D':    ('sexe', 'F'),
    'FA':   ('type_course', 'Finale A'),
    'FB':   ('type_course', 'Finale B'),
    'SF':   ('type_course', 'Demi-finale'),
    'BSM':  ('lieu', 'Boulogne-sur-Mer'),
    'VSM':  ('lieu', 'Vaires-sur-Marne'),
    'CAEN': ('lieu', 'Caen'),
}

# Noms avec tiret dans le prenom
NOMS_TIRET = {'nowakowski_ana-lucia': 'Nowakowski Ana-Lucia'}

# Metadonnees connues — VERSION DE L'UTILISATEUR avec ses modifications
META = {
    'Bavenkoff Viktor':         {'sexe':'H', 'categorie':'Relève',  'discipline':'Kayak'},
    'Bonnavaud Marine':         {'sexe':'F', 'categorie':'Senior',  'discipline':'Kayak'},
    'Cuenot Jules':             {'sexe':'H', 'categorie':'Relève',  'discipline':'Canoe'},
    'Dagouneau Lisa':           {'sexe':'F', 'categorie':'',        'discipline':'Canoe'},
    'Dubut Capucine':           {'sexe':'F', 'categorie':'',        'discipline':'Kayak'},
    'Freslon Pauline':          {'sexe':'F', 'categorie':'',        'discipline':'Kayak'},
    'Gaudin Pierre':            {'sexe':'H', 'categorie':'Relève',  'discipline':'Canoe'},
    'Gilhard Tom':              {'sexe':'H', 'categorie':'Relève',  'discipline':'Kayak'},
    'Guedes Tanguy':            {'sexe':'H', 'categorie':'U18',     'discipline':'Kayak'},
    'Henry Steven':             {'sexe':'H', 'categorie':'',        'discipline':'Kayak'},
    'Hostens Manon':            {'sexe':'F', 'categorie':'',        'discipline':'Kayak'},
    'Jeannest Nathan':          {'sexe':'H', 'categorie':'',        'discipline':'Canoe'},
    'Keller Guillaume':         {'sexe':'H', 'categorie':'',        'discipline':'Kayak'},
    'Koch Quillian':            {'sexe':'H', 'categorie':'Senior',  'discipline':'Kayak'},
    'Lanee Marin':              {'sexe':'H', 'categorie':'',        'discipline':'Kayak'},
    'Laroche Lucas':            {'sexe':'H', 'categorie':'',        'discipline':'Canoe'},
    'Lefoulon Loulia':          {'sexe':'F', 'categorie':'',        'discipline':'Kayak'},
    'Lefoulon Salya':           {'sexe':'F', 'categorie':'Senior',  'discipline':'Kayak'},
    'Leonard Loic':             {'sexe':'H', 'categorie':'',        'discipline':'Canoe'},
    'Le Petit Telio':           {'sexe':'H', 'categorie':'U18',     'discipline':'Kayak'},
    'Le Souef Arthur':          {'sexe':'H', 'categorie':'',        'discipline':'Kayak'},
    'Lomberget Adam':           {'sexe':'H', 'categorie':'Relève',  'discipline':'Kayak'},
    'Mangot Lilou':             {'sexe':'F', 'categorie':'',        'discipline':'Kayak'},
    'Margely Maxime':           {'sexe':'H', 'categorie':'Senior',  'discipline':'Kayak'},
    'Martin Noe':               {'sexe':'H', 'categorie':'Senior',  'discipline':'Kayak'},
    'Maumy Nais':               {'sexe':'F', 'categorie':'Relève',  'discipline':'Kayak'},
    'Mouget Francis':           {'sexe':'H', 'categorie':'',        'discipline':'Kayak'},
    'Muzeau Mahot':             {'sexe':'H', 'categorie':'',        'discipline':'Canoe'},
    'Nicot Chloe':              {'sexe':'F', 'categorie':'',        'discipline':'Canoe'},
    'Nogues Elouan':            {'sexe':'H', 'categorie':'',        'discipline':'Kayak'},
    'Nowakowski Ana-Lucia':     {'sexe':'F', 'categorie':'',        'discipline':'Kayak'},
    'Paoletti Vanina':          {'sexe':'F', 'categorie':'',        'discipline':'Kayak'},
    'Pflieger Amelie':          {'sexe':'F', 'categorie':'Relève',  'discipline':'Kayak'},
    'Poitoux Cathy':            {'sexe':'F', 'categorie':'',        'discipline':'Kayak'},
    'Polet Theophile':          {'sexe':'H', 'categorie':'Relève',  'discipline':'Kayak'},
    'Romney Ruben':             {'sexe':'H', 'categorie':'Relève',  'discipline':'Kayak'},
    'Siabas Simon Anatole':     {'sexe':'H', 'categorie':'Relève',  'discipline':'Kayak'},
    'Wagner Yann':              {'sexe':'H', 'categorie':'',        'discipline':'Kayak'},
    'Zappaterra Clement':       {'sexe':'H', 'categorie':'Senior',  'discipline':'Kayak'},
    'Zoualegh Nathan':          {'sexe':'H', 'categorie':'Relève',  'discipline':'Kayak'},
    'Zwiller Tao':              {'sexe':'H', 'categorie':'Senior',  'discipline':'Kayak'},
}


# ══════════════════════════════════════════════════════════════════════════════
# LECTURE DES ZIPS
# ══════════════════════════════════════════════════════════════════════════════

def _parse_comment(comment):
    """Extrait les metadonnees depuis le champ comment du maxi_database."""
    meta = {'discipline':'', 'sexe':'', 'categorie':'Senior',
            'bateau':'', 'type_course':'', 'lieu':''}
    if not comment or not isinstance(comment, str):
        return meta
    for tok in comment.upper().split():
        if tok in COMMENT_DICT:
            field, val = COMMENT_DICT[tok]
            meta[field] = val
    m_cat = re.search(r'\b(U\d{2})\b', comment, re.IGNORECASE)
    if m_cat:
        meta['categorie'] = m_cat.group(1).upper()
    return meta


def _parse_zip(zip_path):
    """Lit le maxi_database.xlsx dans un zip et retourne les metadonnees."""
    result = {'athletes':[], 'discipline':'', 'sexe':'', 'categorie':'Senior',
              'bateau':'', 'type_course':'', 'lieu':'', 'comment_raw':''}
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            xlsx_name = next((n for n in zf.namelist()
                              if n.endswith('maxi_database.xlsx')), None)
            if not xlsx_name:
                return result
            with zf.open(xlsx_name) as f:
                xl = pd.ExcelFile(io.BytesIO(f.read()))

        if 'User' in xl.sheet_names:
            df_u = xl.parse('User').fillna('')
            for _, row in df_u.iterrows():
                fn = str(row.get('firstname', '')).strip().capitalize()
                ln = str(row.get('lastname',  '')).strip().capitalize()
                if fn or ln:
                    result['athletes'].append(f"{fn} {ln}".strip())

        if 'Record' in xl.sheet_names:
            df_r = xl.parse('Record').fillna('')
            if not df_r.empty:
                row     = df_r.iloc[0]
                comment = str(row.get('comment', ''))
                result['comment_raw'] = comment
                result.update(_parse_comment(comment))
                try:
                    od   = json.loads(str(row.get('other_data', '{}')))
                    boat = od.get('boat', '')
                    if boat:
                        result['bateau'] = boat.upper()
                        if not result['discipline']:
                            result['discipline'] = ('Kayak' if boat.startswith('K')
                                                    else 'Canoë')
                except Exception:
                    pass
                sport = str(row.get('sport', '')).lower()
                if not result['discipline']:
                    if sport == 'kayak':
                        result['discipline'] = 'Kayak'
                    elif 'canoe' in sport:
                        result['discipline'] = 'Canoë'

    except Exception as e:
        print(f"  ⚠️  Erreur zip {os.path.basename(zip_path)} : {e}")
    return result


def build_zip_index():
    """
    Parcourt data/zips/ et construit un index :
    { (date_str, heure_HH:MM) : metadata_dict }
    """
    index = {}
    if not os.path.isdir(ZIPS_DIR):
        return index
    for zname in sorted(os.listdir(ZIPS_DIR)):
        if not zname.endswith('.zip'):
            continue
        m = re.search(r'([0-9]{8})_([0-9]{6})', zname)
        if not m:
            continue
        date_raw, heure_raw = m.group(1), m.group(2)
        key = ('{}-{}-{}'.format(date_raw[:4], date_raw[4:6], date_raw[6:]),
               '{}:{}'.format(heure_raw[:2], heure_raw[2:4]))
        meta = _parse_zip(os.path.join(ZIPS_DIR, zname))
        meta['zip_name'] = zname
        index[key] = meta
    return index


def enrich_meta_from_zips(zip_index):
    """Enrichit META avec les nouveaux athletes trouves dans les zips."""
    global META
    added = []
    for key, meta in zip_index.items():
        for nom in meta.get('athletes', []):
            if nom and nom not in META:
                sexe = meta.get('sexe', '')
                if not sexe:
                    raw = meta.get('comment_raw', '').upper()
                    if re.search(r'\bH\b', raw):  sexe = 'H'
                    elif re.search(r'\bD\b', raw): sexe = 'F'
                META[nom] = {
                    'sexe':       sexe,
                    'categorie':  meta.get('categorie', ''),
                    'discipline': meta.get('discipline', ''),
                }
                added.append(nom)
    if added:
        print(f"  Nouveaux athletes ajoutes a META ({len(added)}) :")
        for a in added:
            print(f"    + {a} -> {META[a]}")


# ══════════════════════════════════════════════════════════════════════════════
# PARSING DES NOMS DE FICHIERS CSV
# ══════════════════════════════════════════════════════════════════════════════

def _parse_filename(fname):
    """Extrait les infos depuis le nom de fichier CSV."""
    base = os.path.splitext(os.path.basename(fname))[0]

    m = re.search(r'([0-9]{8})_([0-9]{6})', base)
    if not m:
        return None

    date_raw, heure_raw = m.group(1), m.group(2)
    date_str  = '{}-{}-{}'.format(date_raw[:4], date_raw[4:6], date_raw[6:])
    heure_str = '{}:{}'.format(heure_raw[:2], heure_raw[2:4])
    name_part = base[:m.start()].strip('-_')
    suffix    = base[m.end():].strip('-_')

    # Distance
    m_dist = re.search(r'([0-9]+)m', suffix, re.IGNORECASE)
    dist   = m_dist.group(1) + 'm' if m_dist else ''
    if not dist:
        m_sd = re.search(r'sel[_-]([0-9]+)', suffix, re.IGNORECASE)
        if m_sd and m_sd.group(1) in DIST_VALS:
            dist = m_sd.group(1) + 'm'

    m_sel = re.search(r'sel[_-]([0-9]+)', suffix, re.IGNORECASE)
    sel   = m_sel.group(1) if m_sel else '1'

    # Lieu depuis le suffixe du nom de fichier
    lieu = ''
    sl = suffix.lower()
    if 'vsm' in sl:   lieu = 'Vaires-sur-Marne'
    elif 'bsm' in sl: lieu = 'Boulogne-sur-Mer'

    # Noms des athletes — gerer les tirets dans les prenoms
    name_clean   = name_part
    replacements = {}
    for key, val in NOMS_TIRET.items():
        if key in name_clean:
            placeholder = key.replace('-', '_TIRET_')
            name_clean  = name_clean.replace(key, placeholder)
            replacements[placeholder] = val

    segs = [s.strip('_') for s in name_clean.split('-') if s.strip('_')]
    athletes = []
    for seg in segs:
        if seg in replacements:
            athletes.append(replacements[seg])
        else:
            athletes.append(' '.join(w.capitalize() for w in seg.split('_')))

    athlete = ' / '.join(athletes) if len(athletes) > 1 else (athletes[0] if athletes else '')
    if not athlete:
        return None

    return {
        'athlete':       athlete,
        'distance':      dist,
        'date':          date_str,
        'heure':         heure_str,
        'heure_prefix':  heure_str[:5],
        'sel':           sel,
        'lieu':          lieu,
        'n_athletes':    len(athletes),
    }


# ══════════════════════════════════════════════════════════════════════════════
# REGISTRE
# ══════════════════════════════════════════════════════════════════════════════

def load_registre():
    """Charge le registre existant ou cree un DataFrame vide."""
    cols = ['fichier', 'athlete', 'distance', 'date', 'heure', 'sel',
            'notes', 'discipline', 'sexe', 'categorie', 'bateau', 'type_course', 'lieu']
    if os.path.exists(REGISTRE):
        for enc in ('utf-8-sig', 'utf-8', 'latin-1'):
            try:
                df = pd.read_csv(REGISTRE, sep=None, engine='python',
                                 encoding=enc, dtype=str).fillna('')
                df.columns = [c.lstrip('\ufeff').strip() for c in df.columns]
                def fix(x):
                    if not isinstance(x, str): return x
                    try: return x.encode('latin-1').decode('utf-8')
                    except: return x
                for col in df.columns:
                    df[col] = df[col].apply(fix)
                def fix_date(d):
                    for fmt in ('%d/%m/%Y', '%Y-%m-%d', '%d-%m-%Y'):
                        try: return datetime.strptime(str(d).strip(), fmt).strftime('%Y-%m-%d')
                        except: pass
                    return d
                df['date'] = df['date'].apply(fix_date)
                for c in cols:
                    if c not in df.columns:
                        df[c] = ''
                return df
            except UnicodeDecodeError:
                continue
    return pd.DataFrame(columns=cols)


def scan_new_files(df_existing):
    """Detecte les CSV dans data/ absents du registre."""
    connus   = set(df_existing['fichier'].tolist())
    nouveaux = []
    for fname in sorted(os.listdir(DATA_DIR)):
        if not fname.endswith('.csv') or fname == 'registre.csv':
            continue
        if fname in connus:
            continue
        info = _parse_filename(fname)
        if info is None:
            print(f"  ⚠️  Non reconnu (ignore) : {fname}")
            continue
        nouveaux.append((fname, info))
    return nouveaux


def build_row(fname, info, zip_index):
    """
    Construit une ligne de registre.
    Priorite : zip > META > valeur par defaut.
    """
    ath      = info['athlete']
    segs     = ath.split(' / ')
    n        = len(segs)
    zip_key  = (info['date'], info['heure_prefix'])
    zip_meta = zip_index.get(zip_key, {})

    # Bateau : zip > calcul depuis le nombre d'athletes
    bateau = zip_meta.get('bateau', '')
    if not bateau:
        bateau = 'K4' if n >= 4 else ('K2' if n == 2 else 'K1')

    # Discipline, sexe, categorie : zip > META > vide
    meta_ath   = META.get(segs[0], {})
    discipline = zip_meta.get('discipline') or meta_ath.get('discipline', 'Kayak')
    sexe       = zip_meta.get('sexe')        or meta_ath.get('sexe', '')
    categorie  = zip_meta.get('categorie')   or meta_ath.get('categorie', '')

    # Lieu : nom de fichier > zip > vide
    lieu = info.get('lieu') or zip_meta.get('lieu', '')

    # Type de course : zip > nom de fichier
    type_course = zip_meta.get('type_course', '')
    if not type_course and 'course_2' in fname:
        type_course = 'Course 2'

    return {
        'fichier':     fname,
        'athlete':     ath,
        'distance':    info['distance'],
        'date':        info['date'],
        'heure':       info['heure'],
        'sel':         info['sel'],
        'notes':       'capteur inversé' if 'Martin Noe' in ath else '',
        'discipline':  discipline,
        'sexe':        sexe,
        'categorie':   categorie,
        'bateau':      bateau,
        'type_course': type_course,
        'lieu':        lieu,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=== Mise a jour du registre CSV ===\n")

    df = load_registre()
    print(f"Registre actuel : {len(df)} lignes")

    # Lire les zips et enrichir META automatiquement
    print("\nLecture des zips dans data/zips/ ...")
    zip_index = build_zip_index()
    if zip_index:
        print(f"  {len(zip_index)} zip(s) trouves")
        enrich_meta_from_zips(zip_index)
    else:
        print("  Aucun zip trouve (metadonnees depuis META uniquement)")

    # Detecter les nouveaux CSV
    nouveaux = scan_new_files(df)
    if not nouveaux:
        print("\nAucun nouveau fichier detecte. Le registre est a jour.")
        return

    print(f"\n{len(nouveaux)} nouveau(x) fichier(s) detecte(s) :")
    rows     = []
    manquants = []
    for fname, info in nouveaux:
        row     = build_row(fname, info, zip_index)
        rows.append(row)
        zip_ok  = (info['date'], info['heure_prefix']) in zip_index
        src     = '[zip]' if zip_ok else '[META]'
        dist_str = row['distance'] or '(distance manquante)'
        print(f"  + {row['athlete']}")
        print(f"    {dist_str} | {row['date']} | {row['lieu'] or '?'} | "
              f"bateau:{row['bateau']} sexe:{row['sexe'] or '?'} {src}")
        if not row['distance'] or not row['sexe']:
            manquants.append(fname)

    df_final = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    df_final.to_csv(REGISTRE, index=False, encoding='utf-8')

    print(f"\nRegistre sauvegarde : {len(df_final)} lignes total ✓")
    if manquants:
        print(f"\n>>> {len(manquants)} ligne(s) a completer dans VS Code :")
        print("    - distance (si vide)")
        print("    - sexe / categorie (si nouvel athlete sans zip)")
        print("    - lieu (si non detecte automatiquement)")
    print()


if __name__ == '__main__':
    main()

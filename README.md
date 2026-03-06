# 🛶 Analyse des coups de pagaie — FFCK Sprint

Application d'analyse biomécanique des coups de pagaie en canoë-kayak sprint, développée pour la Fédération Française de Canoë-Kayak.

Basée sur les données du capteur **Maxi-Phyling** (Phyling), en collaboration avec la FFCK et l'équipe de France Olympique.

---

## Aperçu

L'application permet d'analyser, comparer et suivre la technique de pagaie de plusieurs athlètes à partir des données brutes du capteur embarqué.

**4 onglets :**
- **① Signal** — visualisation du signal d'accélération brut avec détection des coups
- **② Analyse individuelle** — profil du coup, évolution par quart de course, heatmap
- **③ Comparaison** — profils superposés, scatter, dendrogramme, matrice de corrélation
- **④ Métriques** — tableaux complets, distributions, export CSV

---

## Méthode de détection

Les coups de pagaie sont délimités par les **creux locaux** du signal `acc_x` — les minimums d'accélération situés entre deux pics positifs consécutifs. Ce point correspond physiquement au moment d'entrée de la pagaie dans l'eau.

---

## Métriques calculées

| Métrique | Description | Unité |
|---|---|---|
| **AUC+** | Impulsion de propulsion par coup | m/s |
| **\|AUC-\|** | Impulsion de freinage par coup | m/s |
| **Sym ratio** | \|AUC-\| / AUC+ — équilibre propulsion/freinage | — |
| **RFD** | Rate of Force Development — explosivité du catch | m/s³ |
| **Jerk RMS** | Régularité et fluidité du coup | m/s³ |
| **Position pic** | Timing de la propulsion dans le cycle | % |
| **FWHM** | Durée de la phase de propulsion intense | s |
| **CV AUC+** | Consistance coup-à-coup | % |
| **Distance/coup** | Distance parcourue par coup | m |

---

## Installation locale

### Prérequis

- Python 3.9 ou supérieur
- pip

### Installation

```bash
git clone https://github.com/VOTRE_USERNAME/NOM_DU_REPO.git
cd NOM_DU_REPO
pip install -r requirements.txt
```

### Structure des données

Placez vos fichiers CSV dans un dossier `data/` à la racine du projet :

```
ffck-pagaie/
├── app.py
├── requirements.txt
├── data/
│   ├── bavenkoff_viktor-20260218_024153-sel_250.csv
│   ├── bavenkoff_viktor-20260218_024153-sel_2000.csv
│   ├── gilhard_tom-20260215_101526-sel_250.csv
│   └── ...
```

Les fichiers CSV doivent respecter la convention de nommage :
```
{nom_prenom}-{date}-sel_{distance}.csv
```

### Lancement

```bash
python -m streamlit run app.py
```

L'application s'ouvre automatiquement sur [http://localhost:8501](http://localhost:8501).

---

## Données

Les fichiers CSV proviennent du capteur **Maxi-Phyling** (Phyling, Palaiseau). Chaque fichier contient un enregistrement par athlète, échantillonné à 100 Hz, avec les variables suivantes :

- `acc_x`, `acc_y`, `acc_z` — accélération sur les 3 axes (m/s²)
- `gyro_x`, `gyro_y`, `gyro_z` — vitesse angulaire (deg/s)
- `speed`, `speed_gps`, `speed_i` — vitesse (km/h)
- `D` — distance parcourue (m)
- `T` — temps depuis le début de l'enregistrement (s)
- `latitude`, `longitude`, `altitude` — position GPS
- `cadence`, `d_stroke`, `motif_id` — indicateurs de pagaie calculés par le capteur

> ⚠️ **Note :** Martin Noé est exclu des analyses — son capteur a été installé à l'envers, le signal `acc_x` est inversé.

---

## Paramètres de détection

Modifiables depuis la barre latérale de l'application :

| Paramètre | Valeur par défaut | Rôle |
|---|---|---|
| Lissage fc | 3.0 Hz | Fréquence de coupure du filtre passe-bas |
| Distance min entre pics | 0.25 s | Équivaut à une cadence max de 240 cpm |
| Hauteur min du pic | 0.2 m/s² | Seuil pour ignorer le bruit |

---

## Déploiement (Streamlit Cloud)

L'application est déployée sur Streamlit Community Cloud.  
Accès réservé aux membres de l'équipe FFCK Sprint — contactez l'administrateur pour obtenir un accès.

---

## Stack technique

- **Python** — traitement du signal et calcul des métriques
- **Streamlit** — interface web
- **SciPy** — filtrage, détection de pics, clustering
- **Matplotlib** — visualisations
- **Pandas / NumPy** — manipulation des données

---

## Capteur

**Maxi-Phyling** — Phyling, Palaiseau (France)  
GPS multi-bande 10 Hz · IMU 200 Hz · Poids 120 g  
Validé par la Fédération Internationale de Canoë-Kayak (ICF)  
[phyling.fr](https://www.phyling.fr)

---

## Auteur

Développé pour la **Fédération Française de Canoë-Kayak**  
Usage interne — données confidentielles

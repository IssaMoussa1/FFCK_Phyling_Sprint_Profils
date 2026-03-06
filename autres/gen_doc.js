const fs = require('fs');
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  HeadingLevel, AlignmentType, BorderStyle, WidthType, ShadingType,
  VerticalAlign, PageNumber, Header, Footer
} = require('docx');

// ─── Helpers ───────────────────────────────────────────────────────────────

const WHITE     = 'FFFFFF'; 
const BLUE      = '305CDE';
const BLUE_DARK = 'FFFFFF';
const BLUE_LIGHT= 'E3F0FD';
const GREEN     = '2E7D32';
const RED       = 'C62828';
const ORANGE    = 'E65100';
const GREY_LIGHT= 'F5F7FA';
const GREY_BG   = 'EEF2F7';

const h1 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_1,
  children: [new TextRun({ text, bold: true, size: 32, color: WHITE, font: 'Arial' })],
  spacing: { before: 360, after: 160 },
  border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: WHITE } }
});

const h2 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_2,
  children: [new TextRun({ text, bold: true, size: 26, color: WHITE, font: 'Arial' })],
  spacing: { before: 280, after: 100 },
});

const h3 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_3,
  children: [new TextRun({ text, bold: true, size: 22, color: WHITE, font: 'Arial' })],
  spacing: { before: 200, after: 80 },
});

const body = (text, opts = {}) => new Paragraph({
  children: [new TextRun({ text, size: 22, font: 'Arial', ...opts })],
  spacing: { before: 60, after: 60 },
});

const bold_body = (label, val) => new Paragraph({
  children: [
    new TextRun({ text: label + ' : ', bold: true, size: 22, font: 'Arial' }),
    new TextRun({ text: val, size: 22, font: 'Arial' }),
  ],
  spacing: { before: 60, after: 60 },
});

const bullet = (text, indent = 0) => new Paragraph({
  children: [new TextRun({ text, size: 22, font: 'Arial' })],
  bullet: { level: indent },
  spacing: { before: 40, after: 40 },
});

const note = (text) => new Paragraph({
  children: [new TextRun({ text: '💡 ' + text, size: 20, font: 'Arial', italics: true, color: '444444' })],
  shading: { type: ShadingType.CLEAR, color: 'auto', fill: BLUE_LIGHT },
  spacing: { before: 100, after: 100 },
  indent: { left: 200 },
  border: { left: { style: BorderStyle.SINGLE, size: 16, color: BLUE } }
});

const example = (text) => new Paragraph({
  children: [new TextRun({ text: '📊 Exemple : ' + text, size: 20, font: 'Arial', italics: true, color: '2E5300' })],
  shading: { type: ShadingType.CLEAR, color: 'auto', fill: 'EDF7ED' },
  spacing: { before: 100, after: 100 },
  indent: { left: 200 },
  border: { left: { style: BorderStyle.SINGLE, size: 16, color: '43A047' } }
});

const sep = () => new Paragraph({
  children: [new TextRun({ text: '' })],
  spacing: { before: 120, after: 120 }
});

// Table helper
function makeTable(headers, rows) {
  const headerCells = headers.map(h => new TableCell({
    children: [new Paragraph({
      children: [new TextRun({ text: h, bold: true, size: 20, font: 'Arial', color: 'FFFFFF' })],
      alignment: AlignmentType.CENTER,
    })],
    shading: { type: ShadingType.CLEAR, color: 'auto', fill: BLUE },
    verticalAlign: VerticalAlign.CENTER,
    margins: { top: 100, bottom: 100, left: 120, right: 120 },
  }));

  const tableRows = [new TableRow({ children: headerCells })];

  rows.forEach((row, ri) => {
    const cells = row.map(cell => new TableCell({
      children: [new Paragraph({
        children: [new TextRun({ text: cell, size: 20, font: 'Arial' })],
        alignment: AlignmentType.LEFT,
      })],
      shading: { type: ShadingType.CLEAR, color: 'auto', fill: ri % 2 === 0 ? 'FFFFFF' : GREY_LIGHT },
      margins: { top: 80, bottom: 80, left: 120, right: 120 },
    }));
    tableRows.push(new TableRow({ children: cells }));
  });

  return new Table({
    rows: tableRows,
    width: { size: 100, type: WidthType.PERCENTAGE },
  });
}

// ─── DOCUMENT ──────────────────────────────────────────────────────────────

const doc = new Document({
  styles: {
    default: { document: { run: { font: 'Arial', size: 22 } } },
    paragraphStyles: [
      { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 32, bold: true, font: 'Arial', color: WHITE },
        paragraph: { spacing: { before: 360, after: 160 }, outlineLevel: 0 } },
      { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 26, bold: true, font: 'Arial', color: WHITE },
        paragraph: { spacing: { before: 280, after: 100 }, outlineLevel: 1 } },
      { id: 'Heading3', name: 'Heading 3', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 22, bold: true, font: 'Arial', color: WHITE },
        paragraph: { spacing: { before: 200, after: 80 }, outlineLevel: 2 } },
    ]
  },
  sections: [{
    properties: {
      page: {
        size: { width: 11906, height: 16838 },
        margin: { top: 1134, right: 1134, bottom: 1134, left: 1134 }
      }
    },
    headers: {
      default: new Header({
        children: [new Paragraph({
          children: [
            new TextRun({ text: 'Documentation Analyse Coups de Pagaie — Maxi-Phyling FFCK Sprint', size: 16, font: 'Arial', color: '888888' })
          ],
          border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: 'CCCCCC' } }
        })]
      })
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          children: [
            new TextRun({ text: 'Page ', size: 16, font: 'Arial', color: '888888' }),
            new TextRun({ children: [PageNumber.CURRENT], size: 16, font: 'Arial', color: '888888' }),
            new TextRun({ text: ' / ', size: 16, font: 'Arial', color: '888888' }),
            new TextRun({ children: [PageNumber.TOTAL_PAGES], size: 16, font: 'Arial', color: '888888' }),
          ],
          alignment: AlignmentType.RIGHT,
          border: { top: { style: BorderStyle.SINGLE, size: 4, color: 'CCCCCC' } }
        })]
      })
    },
    children: [

      // ── PAGE DE TITRE ────────────────────────────────────────────────────
      new Paragraph({
        children: [new TextRun({ text: 'Documentation', size: 28, font: 'Arial', color: '888888', bold: false })],
        alignment: AlignmentType.CENTER, spacing: { before: 800, after: 80 }
      }),
      new Paragraph({
        children: [new TextRun({ text: 'Analyse des coups de pagaie', size: 52, bold: true, font: 'Arial', color: BLUE_DARK })],
        alignment: AlignmentType.CENTER, spacing: { before: 0, after: 80 }
      }),
      new Paragraph({
        children: [new TextRun({ text: 'Métriques, graphiques et interprétation', size: 28, font: 'Arial', color: BLUE, italics: true })],
        alignment: AlignmentType.CENTER, spacing: { before: 0, after: 200 }
      }),
      new Paragraph({
        children: [new TextRun({ text: '─────────────────────────────', size: 22, color: BLUE })],
        alignment: AlignmentType.CENTER, spacing: { before: 80, after: 80 }
      }),
      new Paragraph({
        children: [new TextRun({ text: 'Maxi-Phyling · FFCK Sprint · Application Streamlit', size: 22, font: 'Arial', color: '555555' })],
        alignment: AlignmentType.CENTER, spacing: { before: 80, after: 80 }
      }),
      new Paragraph({
        children: [new TextRun({ text: 'Méthode de détection : creux locaux', size: 22, font: 'Arial', color: '555555', italics: true })],
        alignment: AlignmentType.CENTER, spacing: { before: 40, after: 600 }
      }),

      // ── 1. CONTEXTE ───────────────────────────────────────────────────────
      h1('1. Contexte et données sources'),

      h2('1.1 Le capteur Maxi-Phyling'),
      body('Le Maxi-Phyling est un boîtier GPS+IMU développé par Phyling en collaboration avec la FFCK. Il est fixé sur la coque du bateau dans le sens de déplacement.'),
      body('Il mesure à 200 Hz l\'accélération sur les 3 axes (acc_x, acc_y, acc_z) et la vitesse angulaire (gyro_x, gyro_y, gyro_z). Le GPS enregistre position, vitesse et altitude à 10 Hz.'),
      sep(),

      h2('1.2 Signal principal : acc_x'),
      body('L\'axe X est l\'axe de déplacement du bateau (avant ↔ arrière).'),
      bullet('acc_x > 0 : le bateau accélère — la pagaie est en phase de propulsion active'),
      bullet('acc_x < 0 : le bateau ralentit — phase passive entre deux coups (glisse, résistance hydrodynamique)'),
      note('Le signal acc_x reflète directement la qualité technique du coup : intensité du catch, durée de propulsion, freinage passif entre deux coups.'),
      example('Un athlète avec un bon catch produit un pic acc_x bref et intense (~6–8 m/s²) suivi d\'une décélération modérée (~-2 à -3 m/s²). Un pagayeur moins technique présente un pic plus étalé et plus bas.'),
      sep(),

      h2('1.3 Méthode de détection des coups : creux locaux'),
      body('La frontière entre deux coups est définie par le creux local (minimum d\'accélération) situé entre deux pics positifs consécutifs.'),
      body('Algorithme en 3 étapes :'),
      bullet('Lissage passe-bas du signal brut (filtre Butterworth, fc = 3 Hz) pour supprimer le bruit haute fréquence'),
      bullet('Détection des pics positifs via scipy.signal.find_peaks avec seuils de hauteur et de distance minimale'),
      bullet('Creux = argmin de acc_x entre chaque paire de pics consécutifs'),
      body('Chaque coup = intervalle [creux_i → creux_{i+1}]. Le creux correspond physiquement au moment où la pagaie entre dans l\'eau — point de transition biomécanique naturel.'),
      note('Ce choix est préférable à motif_id (frontière variable) et au passage par zéro (frontière arbitraire) car le creux a une signification physique reproductible.'),
      sep(),

      // ── 2. MÉTRIQUES DE BASE ───────────────────────────────────────────────
      h1('2. Métriques de base'),

      h2('2.1 AUC+ — Impulsion de propulsion'),
      bold_body('Définition', 'Intégrale de acc_x sur la partie positive du coup (acc_x > 0). Unité : m/s.'),
      bold_body('Formule', 'AUC+ = ∫ max(acc_x, 0) · dt  sur [creux_i → creux_{i+1}]'),
      bold_body('Signification physique', 'Correspond à l\'impulsion de force vers l\'avant par coup. Plus AUC+ est élevé, plus le coup est puissant.'),
      bold_body('Valeurs typiques', '0.35 – 0.55 m/s en 250 m pour un kayakiste de haut niveau'),
      example('Zwiller obtient AUC+ = 0.47 m/s vs Gilhard = 0.38 m/s → Zwiller produit 24 % plus d\'impulsion propulsive par coup.'),
      note('AUC+ dépend à la fois de l\'amplitude du pic (technique) et de la durée de la phase positive (cadence, style).'),
      sep(),

      h2('2.2 |AUC-| — Impulsion de freinage'),
      bold_body('Définition', 'Valeur absolue de l\'intégrale de acc_x sur la partie négative du coup (acc_x < 0). Unité : m/s.'),
      bold_body('Formule', '|AUC-| = |∫ min(acc_x, 0) · dt|'),
      bold_body('Signification physique', 'Quantifie la décélération passive du bateau entre deux coups. Reflète la résistance hydrodynamique et la prise de vitesse inter-coups.'),
      body('Un |AUC-| élevé n\'est pas forcément mauvais : il peut signifier que le bateau était très rapide et que la résistance est forte. C\'est la relation avec AUC+ qui importe.'),
      example('Si AUC+ = 0.45 m/s et |AUC-| = 0.28 m/s, le bateau récupère 62 % de son accélération à chaque coup.'),
      sep(),

      h2('2.3 Sym ratio — Ratio de symétrie'),
      bold_body('Définition', 'Rapport moyen entre le freinage et la propulsion par coup. Sans unité.'),
      bold_body('Formule', 'sym_ratio = mean( |AUC-ᵢ| / AUC+ᵢ )  pour chaque coup i'),
      bold_body('Valeurs de référence', '0.45 – 0.65 pour un haut niveau · < 0.45 peu probable physiquement · > 0.75 = beaucoup de freinage relatif'),
      bold_body('Important', 'sym_ratio ≠ |mean(AUC-)| / mean(AUC+). Ces deux formules donnent des résultats proches mais différents (inégalité de Jensen : la moyenne d\'un ratio ≠ le ratio des moyennes). L\'écart entre les deux est un indicateur indirect de la variabilité inter-coups.'),
      example('Un athlète avec sym_ratio = 0.52 : pour chaque 1 m/s de propulsion, il y a 0.52 m/s de freinage. En comparant mean(|AUC-|)/mean(AUC+) = 0.54, l\'écart de 0.02 indique une légère variabilité coup-à-coup.'),
      sep(),

      h2('2.4 AUC_abs — Effort mécanique total'),
      bold_body('Définition', 'Intégrale de |acc_x| sur tout le coup. Unité : m/s.'),
      bold_body('Formule', 'AUC_abs = ∫ |acc_x| · dt = AUC+ + |AUC-|'),
      bold_body('Signification', 'Représente l\'effort mécanique total subi par le bateau, propulsion + freinage. Utilisé pour calculer le % de propulsion.'),
      bold_body('% cycle propulsion', 'AUC+ / AUC_abs × 100 → proportion du cycle où acc_x > 0 en énergie'),
      sep(),

      h2('2.5 Durée du coup'),
      bold_body('Définition', 'Temps entre deux creux consécutifs. Unité : secondes.'),
      bold_body('Lien avec la cadence', 'Cadence (cpm) ≈ 60 / durée'),
      example('Durée = 0.48 s → cadence ≈ 125 cpm. Durée = 0.54 s → cadence ≈ 111 cpm.'),
      sep(),

      h2('2.6 Distance par coup'),
      bold_body('Définition', 'Distance GPS parcourue entre deux creux consécutifs. Unité : mètres.'),
      bold_body('Calcul', 'sD[-1] - sD[0] pour chaque coup (méthode de Vincenty)'),
      bold_body('Interprétation', 'Reflète l\'efficacité propulsive : un coup loin = propulsion efficace ET/OU vitesse élevée. Combiné à la cadence, définit la vitesse : v = distance/coup × cadence.'),
      example('Distance/coup = 2.55 m, cadence = 118 cpm → vitesse ≈ 18.1 km/h.'),
      sep(),

      // ── 3. MÉTRIQUES AVANCÉES ─────────────────────────────────────────────
      h1('3. Métriques avancées'),

      h2('3.1 RFD — Rate of Force Development (Explosivité)'),
      bold_body('Définition', 'Pente de montée en accélération jusqu\'au pic positif. Unité : m/s³.'),
      bold_body('Formule', 'RFD = acc_x[pic] / (t_pic - t_creux_début)'),
      bold_body('Signification', 'Mesure la vitesse à laquelle le pagayeur engage la propulsion au catch. Un RFD élevé = catch agressif et explosif. Discriminant de la technique au niveau élite.'),
      bold_body('Valeurs typiques', '15 – 40 m/s³ selon le niveau et la distance'),
      example('Bavenkoff RFD = 32.5 m/s³ vs Polet RFD = 21.0 m/s³ → Bavenkoff engage la propulsion 55 % plus vite. Après 200 m, si le RFD de Bavenkoff chute à 26 m/s³, c\'est un indicateur de fatigue du catch.'),
      note('Le RFD peut être amélioré par des exercices de puissance explosive (catch statique, résistances élastiques).'),
      sep(),

      h2('3.2 Jerk RMS — Régularité et fluidité'),
      bold_body('Définition', 'Racine de la moyenne du carré de la dérivée de acc_x, calculée à FS=100 Hz. Unité : m/s³.'),
      bold_body('Formule', 'Jerk RMS = √( mean( (Δacc_x × FS)² ) )'),
      bold_body('Signification', 'Mesure la régularité et la fluidité du coup. Un jerk faible = coup lisse, contrôlé, fluide. Un jerk élevé = coup haché, avec discontinuités (pagaie qui glisse, mauvais appui).'),
      bold_body('Valeurs typiques', '80 – 200 m/s³ · différences entre athlètes souvent > 20 %'),
      example('Si Jerk = 95 m/s³ au premier quart et 130 m/s³ au dernier quart → la fatigue dégrade la fluidité du coup de 37 %. Cible : maintenir le jerk stable sur toute la course.'),
      note('Jerk et RFD sont complémentaires : un athlète peut avoir un RFD élevé (explosif) ET un jerk élevé (peu fluide). L\'objectif est d\'avoir RFD élevé ET jerk faible.'),
      sep(),

      h2('3.3 Position du pic (%)'),
      bold_body('Définition', 'Position relative du pic d\'accélération dans le cycle normalisé. Unité : % (0–100).'),
      bold_body('Formule', 'pos_pic = argmax(acc_x) / len(acc_x) × 100'),
      bold_body('Signification', 'Indique à quel moment du cycle la propulsion maximale est atteinte.'),
      makeTable(
        ['Valeur', 'Interprétation', 'Profil technique'],
        [
          ['< 25 %', 'Pic très tôt dans le cycle', 'Catch explosif, pagaie perpendiculaire, propulsion en début de coup'],
          ['25–35 %', 'Pic précoce (zone haute performance)', 'Technique élite typique'],
          ['35–45 %', 'Pic médian', 'Catch moins marqué, ou pagaie inclinée'],
          ['> 45 %', 'Pic tardif', 'Pagaie qui traîne, engagement tardif — souvent sous fatigue'],
        ]
      ),
      sep(),
      example('Siabas pos_pic = 28 % → catch très marqué en début de cycle. Polet pos_pic = 42 % → propulsion plus tardive, signature technique différente.'),
      sep(),

      h2('3.4 FWHM — Largeur du pic à mi-hauteur'),
      bold_body('Définition', 'Durée pendant laquelle acc_x dépasse la moitié de sa valeur maximale. Unité : secondes (affiché en ms).'),
      bold_body('Formule', 'FWHM = t[acc_x ≥ pic/2][-1] - t[acc_x ≥ pic/2][0]'),
      bold_body('Signification', 'Mesure la durée de la phase de propulsion intense. Un FWHM court = coup explosif et bref (effort concentré). Un FWHM long = effort soutenu sur une plus grande plage.'),
      example('FWHM = 80 ms : la propulsion intense dure 80 ms sur un coup de ~450 ms total (18 % du cycle). FWHM = 120 ms pour le même athlète en fin de course → la phase propulsive s\'étale (signe de perte d\'explosivité ou de compensation).'),
      sep(),

      h2('3.5 CV AUC+ — Coefficient de variation de la propulsion'),
      bold_body('Définition', 'Écart-type de AUC+ divisé par la moyenne, exprimé en %. Mesure la consistance coup-à-coup.'),
      bold_body('Formule', 'CV AUC+ = std(AUC+) / mean(AUC+) × 100'),
      makeTable(
        ['Valeur', 'Interprétation'],
        [
          ['< 10 %', 'Très régulier — technique reproductible'],
          ['10–20 %', 'Variabilité modérée — normal en compétition'],
          ['20–30 %', 'Variabilité élevée — fatigue ou instabilité technique'],
          ['> 30 %', 'Forte variabilité — problème technique ou conditions externes'],
        ]
      ),
      sep(),
      example('En 250 m : Zwiller CV = 12 %, Zoualegh CV = 22 %. Zoualegh est moins régulier, probablement dû à une fatigue accélérée ou un rythme de course non optimal.'),
      sep(),

      // ── 4. GRAPHIQUES ─────────────────────────────────────────────────────
      h1('4. Graphiques — description et interprétation'),

      h2('4.1 Signal acc_x avec coups détectés'),
      bold_body('Description', 'Représentation temporelle du signal d\'accélération brut (non normalisé) sur une fenêtre de quelques secondes.'),
      body('Chaque coup est coloré différemment. Les zones vertes = acc_x > 0 (propulsion). Les zones rouges = acc_x < 0 (freinage/glisse). Le triangle ▲ marque le pic de chaque coup.'),
      bold_body('Ce qu\'on cherche',
        'Des coups réguliers et bien délimités. Des pics bien définis et répétables. Une décélération inter-coups modérée.'),
      bold_body('Signaux d\'alerte',
        'Coups mal délimités (creux peu marqués) → signal bruité ou cadence très élevée. Pics de hauteurs très variables → instabilité technique.'),
      example('Sur 5 secondes à 120 cpm, on voit ~10 coups. Si les 3 derniers coups ont des pics 30 % plus bas que les premiers, c\'est de la fatigue.'),
      sep(),

      h2('4.2 Profil moyen du coup avec AUC'),
      bold_body('Description', 'Le signal de chaque coup est normalisé sur 0–100 % (interpolation linéaire à 200 points). On trace la moyenne et les ombrages.'),
      body('Zone verte = AUC+ (propulsion). Zone rouge = |AUC-| (freinage). Trait central = profil moyen.'),
      bold_body('Ombrages',
        '±1σ (sombre) : 68 % des coups sont dans cette zone. ±2σ (clair) : 95 % des coups. σ = écart-type coup-à-coup à chaque point du cycle.'),
      example('Si à 30 % du cycle le profil moyen est 4.2 m/s² avec σ = 0.3 m/s², cela signifie que 68 % des coups produisent entre 3.9 et 4.5 m/s² à ce moment précis. Un σ = 0.8 m/s² signale une forte variabilité à cet instant.'),
      bold_body('Indicateurs clés', 'Encart : n (nombre de coups), sym_ratio, % cycle propulsion, pic d\'accélération moyen.'),
      sep(),

      h2('4.3 Coups individuels superposés'),
      bold_body('Description', 'Tous les coups normalisés tracés simultanément, colorés selon leur position dans la course (rouge = début, vert = fin).'),
      bold_body('Ce qu\'on cherche', 'Des coups qui se superposent bien = technique reproductible. Un changement progressif de couleur (rouge → vert) sans dégradation = bonne gestion de l\'effort.'),
      example('Si les coups rouges (début) sont au-dessus des verts (fin) dans la zone 20–40 % → l\'athlète "lâche" progressivement sa propulsion en fin de course.'),
      sep(),

      h2('4.4 Heatmap des coups normalisés'),
      bold_body('Description', 'Chaque ligne = un coup. L\'axe horizontal = cycle normalisé. La couleur = acc_x (rouge foncé = forte propulsion, bleu foncé = fort freinage). Les coups sont triés du premier au dernier.'),
      bold_body('Lecture', 'Une heatmap "propre" a des lignes rouges régulières dans la zone 20–50 % du cycle (pic de propulsion). Des lignes irrégulières = variabilité inter-coups.'),
      bold_body('Ce qu\'on cherche',
        'Un bloc rouge compact et stable → très bonne reproductibilité. Un bloc rouge qui se déplace ou s\'affaiblit vers le bas → évolution de la technique au fil de la course.'),
      example('Si les 20 premières lignes ont leur rouge le plus intense à ~28 % et les 20 dernières à ~35 %, le pic se décale en fin de course : signe que l\'athlète engage sa propulsion plus tard sous fatigue.'),
      sep(),

      h2('4.5 Évolution par quart de course'),
      bold_body('Description', 'La course est divisée en 4 segments égaux par distance. Pour chaque segment, on trace le profil moyen du coup.'),
      body('Panneau gauche : profils absolus des 4 quarts. Panneau droit : différence entre chaque quart et le 1er quart (référence = début de course).'),
      bold_body('Interprétation différence vs 1er quart',
        'Valeur positive à un % du cycle → le coup est plus puissant qu\'au début à cet instant. Valeur négative → dégradation par rapport au début.'),
      example('En 250 m, si le 4ème quart montre -0.8 m/s² à 25 % du cycle → le catch a perdu 0.8 m/s² d\'accélération en fin de course. Si +0.3 m/s² à 60 % → la phase finale du coup reste forte, le pagayeur "tire" jusqu\'au bout.'),
      sep(),

      h2('4.6 Enveloppe comparée inter-athlètes'),
      bold_body('Description', 'Profils moyens de plusieurs athlètes tracés sur le même graphique, avec ombrage ±1σ pour chacun.'),
      bold_body('Ce qu\'on cherche', 'Des formes de coup structurellement différentes (pic plus tôt/tard, amplitude différente). Des ±1σ étroits = reproducibilité. Des chevauchements entre athlètes → techniques similaires.'),
      example('Si l\'ombrage de Bavenkoff ne chevauche pas celui de Gilhard, leurs techniques sont statistiquement différentes. S\'ils se chevauchent fortement, ils paddlent de manière similaire.'),
      sep(),

      h2('4.7 Graphique de dispersion (Scatter)'),
      bold_body('Description', 'Nuage de points où chaque point = un coup. Axe X = AUC+ (propulsion). Axe Y = métrique choisie. Ellipse de confiance ±1.5σ. ♦ = centroïde de l\'athlète.'),
      bold_body('Ellipse de confiance', 'Représente la zone où ~87 % des coups d\'un athlète se situent. Sa forme indique la corrélation entre les deux métriques. Ellipse allongée diagonalement → les deux métriques évoluent ensemble.'),
      bold_body('Centroïde (♦)', 'Valeur moyenne de l\'athlète. C\'est la "signature" de cet athlète pour ces deux métriques.'),
      example('Scatter AUC+ vs |AUC-| : si le centroïde d\'un athlète est en haut à droite (forte propulsion, fort freinage), il est puissant mais dans un bateau rapide. Un centroïde en bas à droite (forte propulsion, faible freinage) est idéal : puissant et glissant.'),
      sep(),

      h2('4.8 Dendrogramme de similarité'),
      bold_body('Description', 'Arbre hiérarchique (clustering Ward) regroupant les athlètes selon leur similarité. Deux versions : (1) forme du profil normalisé, (2) métriques scalaires.'),
      bold_body('Lecture', 'Plus les branches se rejoignent bas (distance faible), plus les athlètes sont similaires. Les groupes naturels apparaissent comme des sous-arbres.'),
      bold_body('Métriques scalaires utilisées', 'AUC+, |AUC-|, RFD, Jerk RMS, Position pic, Sym ratio, Durée (normalisés pour mise à l\'échelle)'),
      example('Si Bavenkoff et Zwiller se rejoignent très bas dans le dendrogramme de forme, leurs profils de coup sont quasi-identiques en forme (mais pas nécessairement en amplitude).'),
      note('Un athlète seul (branche longue) a un profil unique — peut être intéressant à analyser individuellement pour identifier des caractéristiques techniques distinctives.'),
      sep(),

      h2('4.9 Matrice de corrélation'),
      bold_body('Description', 'Matrice n×n montrant la corrélation de Pearson entre les profils moyens normalisés de chaque paire d\'athlètes.'),
      bold_body('Échelle', 'Adaptée dynamiquement aux valeurs observées (généralement > 0.8 pour des kayakistes de même niveau). L\'échelle part du minimum observé hors-diagonale pour maximiser la différenciation visuelle.'),
      bold_body('Interprétation', 'r = 1.00 → profils identiques. r = 0.95 → très similaires. r = 0.80 → formes différentes. r < 0.70 → techniques structurellement différentes.'),
      example('Si tous les athlètes ont r > 0.92, la coloration sur une échelle 0–1 ne montrerait que du vert. Avec une échelle adaptée (0.90–1.00), les différences subtiles (0.91 vs 0.97) deviennent visibles.'),
      sep(),

      h2('4.10 Évolution des métriques au fil de la distance'),
      bold_body('Description', 'Pour chaque coup, la valeur d\'une métrique est tracée en fonction de la distance relative. Points individuels (transparents) + moyenne glissante (trait épais).'),
      bold_body('Moyenne glissante', 'Calculée sur N coups consécutifs (défaut : 15). Lisse les variations coup-à-coup pour révéler la tendance de fond.'),
      bold_body('Utilisation principale', 'Détecter la fatigue (déclin progressif), les seuils (chute brutale à un endroit), les patterns de course (pic de propulsion en milieu de course).'),
      example('AUC+ sur 2000 m : si la courbe descend de 0.45 à 0.38 m/s entre 500 m et 1500 m puis remonte à 0.43 m/s → creux en milieu de course, typique d\'une prise de risque au départ trop importante.'),
      sep(),

      // ── 5. SYMÉTRIE ET RELATION AUC ───────────────────────────────────────
      h1('5. Relations entre métriques — points clés'),

      h2('5.1 Pourquoi sym_ratio ≠ |mean(AUC-)| / mean(AUC+)'),
      body('Ces deux formules semblent équivalentes mais donnent des résultats légèrement différents. C\'est un exemple de l\'inégalité de Jensen.'),
      body('Formule A (sym_ratio dans le code) : mean( |AUC-ᵢ| / AUC+ᵢ )  — moyenne des ratios individuels.'),
      body('Formule B (calcul manuel souvent fait) : |mean(AUC-)| / mean(AUC+) — ratio des moyennes.'),
      body('La différence est mesurable lorsqu\'un athlète est irrégulier (grands écarts entre coups). Un athlète très régulier aura A ≈ B. L\'écart |A - B| est donc lui-même un indicateur de variabilité.'),
      example('Athlète A : AUC+ = [0.40, 0.50] m/s, |AUC-| = [0.24, 0.28] m/s. Formule A = mean(0.24/0.40, 0.28/0.50) = mean(0.60, 0.56) = 0.58. Formule B = 0.26/0.45 = 0.578. Écart = 0.002 (négligeable ici). Avec plus de variabilité, l\'écart augmente.'),
      sep(),

      h2('5.2 Vitesse, cadence et distance par coup'),
      body('Ces trois variables sont liées par la relation fondamentale :'),
      new Paragraph({
        children: [new TextRun({ text: '  Vitesse (m/s) = Distance/coup (m) × Cadence (coups/s)', size: 22, font: 'Courier New', bold: true })],
        spacing: { before: 100, after: 100 },
        indent: { left: 400 }
      }),
      body('En kayak de vitesse, deux stratégies coexistent :'),
      bullet('Haute cadence + distance/coup moyenne → athlètes "tourneurs"'),
      bullet('Cadence modérée + grande distance/coup → athlètes "pousseurs"'),
      example('120 cpm × 2.50 m/coup = 5.00 m/s = 18.0 km/h. 110 cpm × 2.73 m/coup = 5.00 m/s = 18.0 km/h. Même vitesse, stratégies différentes.'),
      sep(),

      h2('5.3 RFD vs Pic d\'accélération'),
      body('Le RFD et le pic d\'accélération mesurent des choses différentes :'),
      bullet('Pic = valeur maximale atteinte → intensité de la propulsion'),
      bullet('RFD = pente = vitesse d\'atteinte du pic → explosivité du catch'),
      body('Un athlète peut avoir un pic élevé avec un RFD faible (propulsion lente à monter) ou un RFD élevé avec un pic modéré (catch explosif mais court).'),
      sep(),

      // ── 6. GUIDE D'INTERPRÉTATION ─────────────────────────────────────────
      h1('6. Guide d\'interprétation pour l\'entraîneur'),

      h2('6.1 Analyse individuelle — ordre de lecture recommandé'),
      body('Protocole recommandé pour analyser un athlète :'),
      bullet('1. Signal acc_x → vérifier que la détection est correcte (bons creux, nombre de coups cohérent)'),
      bullet('2. Profil moyen + AUC → identifier la forme du coup, l\'AUC+ moyen, le sym_ratio'),
      bullet('3. Heatmap → évaluer la reproductibilité coup-à-coup (ombrage ±1σ étroit ?)'),
      bullet('4. Quarts de course → détecter fatigue, montée en régime, variations tactiques'),
      bullet('5. Évolution des métriques → suivre RFD, jerk et AUC+ sur la distance'),
      sep(),

      h2('6.2 Signaux de fatigue'),
      makeTable(
        ['Signal observé', 'Métrique concernée', 'Interprétation probable'],
        [
          ['AUC+ diminue progressivement', 'AUC+', 'Fatigue musculaire — perte de puissance propulsive'],
          ['RFD chute en 2ème moitié', 'RFD', 'Perte d\'explosivité du catch — typique de la fatigue neuromusculaire'],
          ['Position pic se décale vers droite', 'pos_pic_pct', 'Engagement tardif — pagaie qui traîne sous fatigue'],
          ['Jerk RMS augmente', 'Jerk RMS', 'Perte de fluidité — coups moins contrôlés'],
          ['CV AUC+ augmente', 'CV AUC+', 'Irrégularité croissante — stratégie ou fatigue'],
          ['Profil quart 4 < quart 1', 'Profil par quart', 'Affaissement global de la propulsion en fin de course'],
        ]
      ),
      sep(),

      h2('6.3 Comparaison inter-athlètes — que chercher ?'),
      bullet('Dendrogramme : identifier les "groupes" techniques naturels au sein de l\'équipe'),
      bullet('Scatter AUC+ vs |AUC-| : qui produit le plus de propulsion sans freinage ?'),
      bullet('Profils superposés : à quelle phase du cycle chaque athlète est-il le plus fort ?'),
      bullet('Matrice de corrélation : r > 0.95 → techniques très proches (adapter ensemble). r < 0.85 → techniques différentes (adapter individuellement)'),
      sep(),

      h2('6.4 Martin Noé — note d\'exclusion'),
      new Paragraph({
        children: [new TextRun({
          text: '⚠️ Martin Noé est exclu de toutes les analyses. Son capteur a été installé à l\'envers (sens inversé par rapport à l\'axe de déplacement). Son signal acc_x est le miroir du signal réel — les pics positifs correspondent en réalité au freinage et inversement. Ses données ne peuvent pas être comparées aux autres athlètes sans correction préalable.',
          size: 22, font: 'Arial', color: '8B0000'
        })],
        shading: { type: ShadingType.CLEAR, color: 'auto', fill: 'FFF3E0' },
        spacing: { before: 100, after: 100 },
        indent: { left: 200 },
        border: { left: { style: BorderStyle.SINGLE, size: 16, color: RED } }
      }),
      sep(),

      // ── GLOSSAIRE ─────────────────────────────────────────────────────────
      h1('Glossaire'),
      makeTable(
        ['Terme', 'Définition'],
        [
          ['acc_x', 'Accélération longitudinale du bateau (axe de déplacement), en m/s²'],
          ['AUC', 'Area Under the Curve — intégrale d\'un signal. AUC+ = zone positive, AUC- = zone négative'],
          ['Catch', 'Moment d\'entrée de la pagaie dans l\'eau — début de la phase de propulsion'],
          ['Creux local', 'Minimum d\'accélération entre deux pics positifs consécutifs — frontière entre coups'],
          ['Cycle normalisé', 'Représentation du coup sur 0–100 % par interpolation, indépendante de la durée'],
          ['CV', 'Coefficient de Variation : std/mean × 100. Mesure la dispersion relative'],
          ['FWHM', 'Full Width at Half Maximum — largeur d\'un pic mesurée à mi-hauteur'],
          ['IMU', 'Inertial Measurement Unit — centrale inertielle (accéléromètre + gyroscope)'],
          ['Jerk', 'Dérivée de l\'accélération par rapport au temps (variation d\'accélération)'],
          ['Pic d\'accélération', 'Valeur maximale de acc_x sur un coup — intensité maximale de la propulsion'],
          ['RFD', 'Rate of Force Development — vitesse de montée en force/accélération'],
          ['Sym ratio', 'Ratio |AUC-|/AUC+ — mesure le freinage relatif à la propulsion'],
          ['σ (sigma)', 'Écart-type — mesure de dispersion autour de la moyenne'],
          ['Ward', 'Méthode de clustering hiérarchique minimisant la variance intra-groupe'],
        ]
      ),
      sep(),

      new Paragraph({
        children: [new TextRun({ text: 'Document généré pour usage interne FFCK Sprint — Confidentiel', size: 18, font: 'Arial', color: 'AAAAAA', italics: true })],
        alignment: AlignmentType.CENTER,
        spacing: { before: 600 }
      }),
    ]
  }]
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync('documentation_pagaie.docx', buf);
  console.log('OK');
});

# Préparation LA28 - Cloudflare Pages

Ce dossier contient la version statique plein écran de l'outil comparatif LA28.

## Déploiement Cloudflare Pages

1. Aller dans Cloudflare Dashboard > Workers & Pages > Pages.
2. Créer un projet avec connexion GitHub.
3. Sélectionner le repo `IssaMoussa1/FFCK_Sprint_Profil`.
4. Paramètres de build :
   - Framework preset : `None`
   - Build command : laisser vide
   - Build output directory : `cloudflare-pages-la28`
5. Lancer le déploiement.

Cloudflare Pages publiera `index.html` comme page principale. Les prochains `git push` sur la branche connectée redéploieront automatiquement l'outil.

## Protection Cloudflare Access

Option la plus simple :

1. Ouvrir le projet Pages dans Cloudflare.
2. Aller dans Settings > Access policy.
3. Activer la protection Access.
4. Ajouter une politique `Allow` limitée aux emails ou domaines autorisés.
5. Tester l'accès avant de partager l'URL.

Option domaine personnalisé :

1. Ajouter un domaine ou sous-domaine personnalisé au projet Pages.
2. Aller dans Cloudflare Zero Trust > Access > Applications.
3. Créer une application `Self-hosted` sur ce domaine.
4. Ajouter une politique `Allow` limitée aux emails ou domaines autorisés.

## Lien depuis le portail Streamlit

Quand l'URL Cloudflare est créée, ajouter ce secret dans Streamlit Cloud :

```toml
LA28_TOOL_URL = "https://ffck-sprint-la28.pages.dev"
```

Le bouton `Préparation LA28` du portail Streamlit ouvrira alors directement la page Cloudflare plein écran.

## Notes

- Le fichier `_headers` ajoute des en-têtes de sécurité et bloque l'intégration de la page dans une iframe.
- L'outil est statique : les données sont dans `index.html`, pas dans une base distante.
- Pour mettre à jour l'outil, modifier `index.html`, puis pousser le changement sur GitHub.

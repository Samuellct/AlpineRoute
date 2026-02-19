# AlpineRoute

Calcul d'itinéraires optimaux pour l'alpinisme hors-piste dans les Alpes (France et étranger). Genère des traces GPS en tenant compte de la topographie réelle du terrain à partir de modèles numeriques de terrain (Lidar MNT).

## Fonctionnalités

- Téléchargement automatique des données MNT IGN Lidar HD (50 cm natif, échantillonnage 0.5 - 10 m pour le projet)
- Fallback MapTiler pour les zones non couvertes par le lidar
- Calcul de pente, orientation et rugosité via Horn's method (scipy)
- Fonction de cout multi-critères : pente, altitude, aspect/saison, glacier, rugosité, couverture du sol
- Intégration des contours glaciaires RGI 7.0 et du landcover ESA WorldCover 10 m
- Pathfinding Dijkstra sur grille (skimage)
- Export GPX et GeoJSON
- Lien backend avec API REST, calcul asynchrone et suivi de progression (SSE)

## Stack technique

**Backend** : Python 3.11 (Miniforge), FastAPI, NumPy, SciPy, Rasterio, GDAL, scikit-image, GeoPandas, Shapely

**Frontend** : Vite, React, MapLibreGL, TailwindCSS

**Donnees** : IGN Lidar HD MNT, RGI 7.0, ESA WorldCover 10 m, MapTiler

## Structure du projet

```
AlpineRoute/
├── backend/
│   └── alpineroute/
│       ├── config.py          # constantes et paramètres
│       ├── utils.py           # fonctions partagées
│       ├── dem/               # téléchargement DEM, analyse terrain
│       ├── cost/              # surface de cout, landcover
│       ├── routing/           # pathfinding, export GPX/GeoJSON
│       ├── api/               # FastAPI endpoints
│       └── db/                # SQLite schema + CRUD
├── frontend/                  # app React
├── docs/                      # documentation du projet
├── docker/                    # Dockerfile backend
└── docker-compose.yml         # lancement local (backend + frontend)
```

## Installation

Le plus simple avec Docker :

```bash
git clone https://github.com/Samuellct/AlpineRoute.git
cd AlpineRoute
docker compose up --build
```

Frontend sur http://localhost:3000, API sur http://localhost:8000/docs.

Pour l'installation locale (conda + npm), voir le [guide d'installation complet](docs/installation.md).

## Utilisation

```bash
# avec Docker
docker compose up

# ou en local (deux terminaux)
# Terminal 1 : backend
conda activate alpineroute
uvicorn backend.alpineroute.api.main:app --reload --port 8000

# Terminal 2 : frontend
cd frontend
npm run dev
```

## Données et licences

| Source | Resolution | Couverture | Licence |
|--------|-----------|------------|---------|
| IGN Lidar HD MNT | 50 cm | France métropolitaine | Etalab 2.0 |
| MapTiler | ~10 m | Mondial | Api free tier|
| RGI 7.0 | Contours vectoriels | Mondial | CC-BY 4.0 |
| ESA WorldCover v200 | 10 m | Mondial | CC-BY 4.0 |

## Statut du projet

V1 du projet entièrement fonctionnel en local. Plsieurs correctifs déjà prévus pour la V1.1 (ajouts de calques, bugs interface web).

Voir les details dans la [documentation technique](docs/README.md).

## Licence

[MIT](LICENSE)
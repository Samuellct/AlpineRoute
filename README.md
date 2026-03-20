# AlpineRoute

Calcul d'itinéraires optimaux pour l'alpinisme dans les Alpes. Générer des traces GPS en tenant compte de la topographie réelle du terrain à partir de données lidar MNT haute résolution.

## Fonctionnalités

- Routage adaptatif : réseau OSM (Valhalla), hybride (réseau + hors-piste), ou raster pur sur grille lidar
- Graphe de traces GPX alpinisme avec connexion au réseau OSM
- Téléchargement automatique des données lidar MNT IGN (50 cm natif, échantillonnage 0.5 - 10 m)
- Fallback Copernicus GLO-30 pour les zones hors France
- Fonction de coût multi-critères : pente, altitude, radiation solaire, glacier, rugosité, couverture du sol, sentiers OSM, barrières
- Intégration contours glaciaires RGI 7.0, landcover ESA WorldCover 10 m, sentiers/barrières OSM
- Recherche de lieux par nom (Nominatim)
- Pathfinding Dijkstra sur grille (skimage)
- Export GPX
- API REST, calcul asynchrone, suivi de progression SSE

## Stack technique

**Backend** : Python 3.11 (Miniforge), FastAPI, NumPy, SciPy, Rasterio, GDAL, scikit-image, GeoPandas, Shapely

**Frontend** : Vite, React, MapLibreGL, TailwindCSS

**Routage réseau** : Valhalla (Docker), PBF Alps via Geofabrik

**Données** : IGN Lidar HD MNT, RGI 7.0, ESA WorldCover 10 m, OSM PBF Alps, Copernicus GLO-30

## Structure du projet

```
AlpineRoute/
├── backend/
│   └── alpineroute/
│       ├── config.py          # constantes et paramètres
│       ├── pipeline.py        # orchestration du calcul
│       ├── utils.py           # fonctions partagées
│       ├── dem/               # téléchargement lidar MNT, analyse terrain
│       ├── cost/              # surface de cout, radiation, sentiers, barrières
│       ├── routing/           # pathfinding, Valhalla, hybride, export
│       ├── alpine/            # indexation traces GPX, graphe overlay
│       ├── api/               # FastAPI endpoints
│       └── db/                # SQLite schema + CRUD
├── frontend/                  # app React
├── data/gpx/                  # traces alpinisme (index.json)
├── docs/                      # documentation
├── docker/                    # Dockerfile backend
└── docker-compose.yml         # backend + frontend + Valhalla
```

## Installation

Le plus simple avec Docker :

```bash
git clone https://github.com/Samuellct/AlpineRoute.git
cd AlpineRoute
docker compose up --build
```

Frontend sur http://localhost:3000, API sur http://localhost:8000/docs.

Pour l'installation locale (conda + npm), voir le [guide d'installation](docs/installation.md).

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

## Donnees et licences

| Source | Resolution | Couverture | Licence |
|--------|-----------|------------|---------|
| IGN Lidar HD MNT | 50 cm | France metropolitaine | Etalab 2.0 |
| Copernicus GLO-30 | ~30 m | Mondial | Libre |
| RGI 7.0 | Contours vectoriels | Mondial | CC-BY 4.0 |
| ESA WorldCover v200 | 10 m | Mondial | CC-BY 4.0 |
| OSM PBF Alps | Reseau routier | Arc alpin | ODbL |

## Statut du projet

V2, en cours de validation.

Voir les details dans la [documentation technique](docs/README.md).

## Licence

[MIT](LICENSE)

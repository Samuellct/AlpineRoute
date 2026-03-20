# Installation

Deux Méthodes : Docker ou installation locale.

---

## Méthode 1 : Docker (recommandée)

Pré-requis : [Docker](https://docs.docker.com/get-docker/) avec Docker Compose.

```bash
git clone https://github.com/Samuellct/AlpineRoute.git
cd AlpineRoute
docker compose up --build
```

Ca lance le backend (port 8000), le frontend (port 3000) et Valhalla (port 8002).

- Frontend : http://localhost:3000
- API docs : http://localhost:8000/docs
- Health check : http://localhost:8000/health

**Valhalla** : le conteneur Valhalla télécharge le PBF Alps et build les tiles au premier lancement (env 5-10 min). Les lignes sont persistantes dans un volume Docker. Sans Valhalla, le pipeline fonctionne en mode raster pur.

Les donnees lidar et la base SQLite sont stockées dans un volume Docker `alpineroute-data`. Elles persistent entre les arrêts/relances des containers.

Pour tout arrêter :

```bash
docker compose down          # garde les data
docker compose down -v       # supprime aussi le volume (reset)
```

---

## Méthode 2 : Installation locale

### Pré-requis

- Python 3.11+ (via [Miniforge](https://github.com/conda-forge/miniforge))
- Node.js 22+ et npm
- Git
- ~2 Go d'espace disque pour les tuiles Lidar (varie selon la zone)

### Backend

Les libs géospatiales (GDAL, raster, fiona) doivent être installées via conda-forge, surtout sur Windows.

```bash
# créer l'env conda
conda create -n alpineroute python=3.11
conda activate alpineroute
conda install -c conda-forge gdal rasterio numpy scipy scikit-image geopandas shapely pyproj fiona owslib

# deps API
pip install fastapi "uvicorn[standard]" httpx gpxpy pydantic
```

Lancer le backend :

```bash
cd AlpineRoute/backend
conda activate alpineroute
uvicorn alpineroute.api.main:app --reload --port 8000
```

Vérifier : `curl http://localhost:8000/health` doit retourner `{"status":"ok"}`.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Le frontend tourne sur http://localhost:5173 en dev (avec proxy automatique vers le backend sur le port 8000 via vite.config.ts).

### Notes par OS

**Windows** : conda-forge obligatoire pour GDAL/rasterio. Ne pas mixer pip et conda pour les deps geospatiales, ca casse les DLL. Si `import rasterio` échoue avec une erreur DLL, réinstaller proprement via conda.

**Linux** : les wheels pip pour rasterio fonctionnent en général. Alternative : `apt install gdal-bin libgdal-dev` puis `pip install rasterio`.

---

## Premier lancement

Au premier calcul de route, le backend télécharge automatiquement les tuiles MNT depuis l'api IGN (Lidar HD). Ca prend quelques minutes selon la zone et la connexion. Les tuiles sont cachées dans `data/cache/dem/` et réutilisées ensuite.

Structure du dossier data apres un premier calcul :

```
data/
├── cache/
│   ├── dem/          # tuiles lidar MNT IGN (.tif)
│   ├── cost/         # surfaces de coût pré-calculées (.npz)
│   ├── radiation/    # angles d'horizon + radiation mensuelle
│   └── osm/          # sentiers/barrières OSM (.gpkg)
├── derived/          # grilles calculées (pente, rugosite...)
├── gpx/              # traces alpinisme (index.json + fichiers GPX)
├── rgi/              # contours glaciaires
└── alpineroute.db    # base SQLite
```

---

## Variables d'environnement

Optionnelles, valeurs par défaut adaptées au dev local.

| Variable | Defaut | Description |
|----------|--------|-------------|
| `ALPINEROUTE_DATA_DIR` | `<repo>/data` | Dossier data (cache lidar, SQLite) |
| `ALPINEROUTE_DB_PATH` | `<data>/alpineroute.db` | Chemin base SQLite |
| `ALPINEROUTE_HOST` | `127.0.0.1` | Adresse d'écoute du backend |
| `ALPINEROUTE_PORT` | `8000` | Port du backend |
| `ALPINEROUTE_CORS_ORIGINS` | `*` | Origins CORS (séparés par des virgules) |
| `ALPINEROUTE_VALHALLA_URL` | `http://localhost:8002` | URL du service Valhalla |
| `ALPINEROUTE_LOG_LEVEL` | `INFO` | Niveau de log backend (DEBUG, INFO, WARNING) |

---

## Troubleshooting

**`ImportError: cannot import name '_gdal_array'`** ou erreur DLL rasterio sur Windows : l'install GDAL est cassée. Supprimer l'env conda et réinstaller proprement :
```bash
conda deactivate
conda env remove -n alpineroute
# puis reprendre la section Backend ci-dessus
```

**Port 8000 deja utilise** : soit un autre process tourne dessus, soit le backend a crash sans se fermer proprement. `lsof -i :8000` (Linux) ou `netstat -ano | findstr 8000` (Windows) pour trouver le PID.

**Timeout au téléchargement lidar** : l'API IGN peut être lente ou down temporairement. Relancer le calcul, le cache reprendra là où il en était. Si ça persiste, vérifier la connexion ou tester avec une bbox plus petite.

**Docker build lent** : le premier build telecharge les images Docker + pip install toutes les deps geospatiales (~700 MB). Les builds suivants utilisent le cache Docker et sont quasi-instantanes tant que requirements.txt ne change pas.

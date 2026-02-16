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

Ca lance le backend (port 8000) et le frontend (port 3000).

- Frontend : http://localhost:3000
- API docs : http://localhost:8000/docs
- Health check : http://localhost:8000/health

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
│   └── dem/          # tuiles DEM IGN (.tif)
├── derived/          # grilles calculees (pente, rugosite...)
├── rgi/              # contours glaciaires (telecharges auto)
└── alpineroute.db    # base SQLite (routes sauvegardees, zones)
```

---

## Variables d'environnement

Optionnelles, valeurs par défaut adaptées au dev local.

| Variable | Defaut | Description |
|----------|--------|-------------|
| `ALPINEROUTE_DATA_DIR` | `<repo>/data` | Dossier data (DEM cache, SQLite) |
| `ALPINEROUTE_DB_PATH` | `<data>/alpineroute.db` | Chemin base SQLite |
| `ALPINEROUTE_HOST` | `127.0.0.1` | Adresse d'écoute du backend |
| `ALPINEROUTE_PORT` | `8000` | Port du backend |
| `ALPINEROUTE_CORS_ORIGINS` | `*` | Origins CORS (séparés par des virgules) |

---

## Troubleshooting

**`ImportError: cannot import name '_gdal_array'`** ou erreur DLL rasterio sur Windows : l'install GDAL est cassée. Supprimer l'env conda et réinstaller proprement :
```bash
conda deactivate
conda env remove -n alpineroute
# puis reprendre la section Backend ci-dessus
```

**Port 8000 deja utilise** : soit un autre process tourne dessus, soit le backend a crash sans se fermer proprement. `lsof -i :8000` (Linux) ou `netstat -ano | findstr 8000` (Windows) pour trouver le PID.

**Timeout au telechargement DEM** : l'API IGN peut etre lente ou down temporairement. Relancer le calcul, le cache reprendra la ou il en etait. Si ca persiste, verifier la connexion ou tester avec une bbox plus petite.

**Docker build lent** : le premier build telecharge les images Docker + pip install toutes les deps geospatiales (~700 MB). Les builds suivants utilisent le cache Docker et sont quasi-instantanes tant que requirements.txt ne change pas.

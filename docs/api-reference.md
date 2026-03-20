# Reference API

## Démarrage

```bash
conda activate alpineroute
cd backend
uvicorn alpineroute.api.main:app --reload --port 8000
```

La doc OpenAPI interactive est disponible sur `http://localhost:8000/docs`.

## Base URL

```
http://localhost:8000
```

## Endpoints

### GET /health

Verifie que le serveur est up.

**Réponse** :
```json
{
  "status": "ok"
}
```

**Exemple** :
```bash
curl http://localhost:8000/health
```

---

### POST /calculate

Calcul de route synchrone. Bloquant, renvoie le resultat directement.

**Body** :
```json
{
  "start_lat": 45.8786,
  "start_lon": 6.8875,
  "end_lat": 45.8527,
  "end_lon": 6.8943,
  "resolution": 1.0,
  "month": 7,
  "acclimatized": true,
  "n_alternatives": 0,
  "save": true,
  "name": "Requin - Midi"
}
```

**Réponse** :
```json
{
  "status": "ok",
  "route": {
    "type": "Feature",
    "geometry": {
      "type": "LineString",
      "coordinates": [[6.8875, 45.8786, 3532.1], ...]
    },
    "properties": {
      "route_index": 0,
      "is_optimal": true,
      "distance_km": 4.52,
      "dplus_m": 1203,
      "dminus_m": 487,
      "time_tobler_h": 3.2,
      "glacier_pct": 34.5,
      "cost_total": 9823.5,
      "n_points": 4520,
      "resolution_m": 1.0
    }
  },
  "computation_time_s": 120.3,
  "saved_route_id": 42,
  "routes": [...],
  "n_routes": 3
}
```

Les champs `routes`, `n_routes` ne sont présents que si `n_alternatives > 0`.

**Erreurs** :
- 400 : coordonnées hors zone valide, résolution invalide, paramètres incohérents
- 502 : échec de téléchargement Lidar (timeout réseau, dalle IGN manquante)
- 500 : erreur de calcul (grille trop grande)

**Exemple** :
```bash
curl -X POST http://localhost:8000/calculate \
  -H "Content-Type: application/json" \
  -d '{"start_lat": 45.8786, "start_lon": 6.8875, "end_lat": 45.8527, "end_lon": 6.8943}'
```

---

### POST /calculate-async

Lance un calcul asynchrone. Retourne immediatement un `job_id` pour suivre la progression via SSE.

**Body** : identique a `/calculate`.

**Réponse** :
```json
{
  "job_id": "a1b2c3d4"
}
```

**Exemple** :
```bash
curl -X POST http://localhost:8000/calculate-async \
  -H "Content-Type: application/json" \
  -d '{"start_lat": 45.8786, "start_lon": 6.8875, "end_lat": 45.8527, "end_lon": 6.8943}'
```

---

### GET /progress/{job_id}

Stream SSE (Server-Sent Events) de la progression d'un calcul async.

**Etapes** : `network` -> `gpx_graph` -> `bbox` -> `cache` -> `dem` -> `terrain` -> `worldcover` -> `glacier` -> `radiation` -> `osm` -> `cost` -> `zones` -> `pathfinding` -> `result`

En cas de cache hit, les etapes `dem` a `glacier` sont sautees (progression rapide).

**Format des evenements** :

Progression en cours :
```
data: {"progress": 45, "step": "pathfinding", "message": "pathfinding (45%)", "status": "running"}
```
\
Terminé (inclut le résultat complet) :
```
data: {"progress": 100, "step": "done", "message": "Termine", "status": "completed", "result": {...}}
```
\
Erreur :
```
data: {"progress": 0, "step": "init", "message": "Point hors grille", "status": "error"}
```

---

### GET /routes

Liste les routes sauvegardes, avec filtres optionnels.

**Query params** :

| Param | Type | Description |
|-------|------|-------------|
| `lon_min`, `lat_min`, `lon_max`, `lat_max` | float | Bbox WGS84 |
| `date_from` | string | Date ISO min (ex: `2025-01-01`) |
| `date_to` | string | Date ISO max |
| `min_distance_m` | float | Distance minimale en m |
| `max_distance_m` | float | Distance maximale en m |
| `limit` | int | Nombre max de résultats (defaut 50, max 500) |
| `offset` | int | Offset pour pagination (defaut 0) |

**Réponse** :
```json
{
  "routes": [
    {
      "id": 1,
      "created_at": "2025-07-15T10:30:00Z",
      "name": "Requin - Midi",
      "start_lat": 45.8786,
      "start_lon": 6.8875,
      "end_lat": 45.8527,
      "end_lon": 6.8943,
      "distance_m": 4520.0,
      "dplus_m": 1203.0,
      "dminus_m": 487.0,
      "time_tobler_h": 3.2,
      "glacier_pct": 34.5,
      "cost_total": 9823.5,
      "computation_time_s": 120.3
    }
  ],
  "count": 1
}
```

**Exemple** :
```bash
curl "http://localhost:8000/routes?limit=10&offset=0"
```

---

### GET /routes/{route_id}

Récupère une route par ID, avec le GeoJSON complet.

**Réponse** : objet route complet (mêmes champs que la liste).

**Erreurs** : 404 si la route n'existe pas

**Exemple** :
```bash
curl http://localhost:8000/routes/42
```

---

### DELETE /routes/{route_id}

Supprime une route.

**Réponse** :
```json
{"status": "deleted", "id": 42}
```

**Erreurs** : 404 si la route n'existe pas.

---

### GET /routes/{route_id}/gpx

Exporte une route au format GPX.

**Content-Type** : `application/gpx+xml`

Le fichier GPX contient un track avec les coordonnées 3D (lon, lat, élévation) extraites du GeoJSON. Le header `Content-Disposition` propose un nom de fichier `route_{id}.gpx`.

**Erreurs** :
- 404 : route inexistante ou sans GeoJSON

**Exemple** :
```bash
curl -o route_42.gpx http://localhost:8000/routes/42/gpx
```

---

### GET /zones

Liste les zones utilisateur.

**Query params** :

| Param | Type | Description |
|-------|------|-------------|
| `zone_type` | string | Filtre par type (`crevasse`, `serac`, `cornice`, `rockfall`, `forbidden`, `custom`) |
| `active_only` | bool | Si `true`, ne retourne que les zones actives (defaut `false`) |

**Réponse** :
```json
{
  "zones": [
    {
      "id": 1,
      "created_at": "2025-07-15T10:00:00Z",
      "name": "Crevasses Bossons",
      "zone_type": "crevasse",
      "cost_multiplier": 100.0,
      "geojson": {"type": "Polygon", "coordinates": [...]},
      "active": 1
    }
  ],
  "count": 1
}
```

---

### POST /zones

Crée une nouvelle zone.

**Body** (JSON) :
```json
{
  "name": "Crevasses Bossons",
  "zone_type": "crevasse",
  "cost_multiplier": 100.0,
  "geojson": {
    "type": "Polygon",
    "coordinates": [[[6.85, 45.88], [6.86, 45.88], [6.86, 45.87], [6.85, 45.87], [6.85, 45.88]]]
  },
  "active": true
}
```

**Réponse** (201) :
```json
{"status": "created", "id": 5}
```

**Erreurs** : 422 si le body est invalide (zone_type inconnu, champs manquants).

---

### GET /zones/{zone_id}

Récupère une zone par ID. Le champ `geojson` est parsé en objet (pas en string).

**Erreurs** : 404 si la zone n'existe pas.

---

### PUT /zones/{zone_id}

Mise a jour partielle d'une zone.

**Body** (JSON) :
```json
{
  "name": "Crevasses Bossons (maj)",
  "cost_multiplier": 200.0
}
```

**Réponse** :
```json
{"status": "updated", "id": 5}
```

**Erreurs** :
- 400 : body vide
- 404 : zone inexistante

---

### DELETE /zones/{zone_id}

Supprime une zone.

**Réponse** :
```json
{"status": "deleted", "id": 5}
```

**Erreurs** : 404 si la zone n'existe pas.

---

### GET /glaciers

Retourne les contours glaciaires RGI dans une bounding box.

**Query params** :

| Param | Type | Description |
|-------|------|-------------|
| `bbox` | string (requis) | `xmin,ymin,xmax,ymax` en WGS84 |

**Réponse** :
```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {"type": "Polygon", "coordinates": [...]},
      "properties": {"glac_name": "Mer de Glace", "area_km2": 30.4}
    }
  ]
}
```

**Erreurs** :
- 400 : bbox invalide (format incorrect, valeurs hors limites)
- 500 : erreur interne

**Exemple** :
```bash
curl "http://localhost:8000/glaciers?bbox=6.85,45.85,6.95,45.92"
```

---

### GET /cost-surface

Retourne la derniere surface de cout calculee sous forme d'image PNG.

**Query params** : aucun

**Réponse** : image PNG (RGBA, echelle log vert->rouge). Headers custom :
- `X-Bounds-South` / `X-Bounds-North` / `X-Bounds-West` / `X-Bounds-East` : emprise WGS84

**Erreurs** :
- 404 : aucun calcul precedent (pas de surface en cache)

**Exemple** :
```bash
curl -o cost.png http://localhost:8000/cost-surface
```

---

### POST /admin/invalidate-cache

Invalide le cache des surfaces de cout pre-calculees.

**Body** (optionnel) :
```json
{
  "xmin": 950000, "ymin": 6490000,
  "xmax": 960000, "ymax": 6500000
}
```

Si un body est fourni (bbox L93), seules les entrees dont la bbox intersecte sont supprimees. Sans body, tout le cache est vide.

**Reponse** :
```json
{"status": "ok", "invalidated": 3}
```

**Exemple** :
```bash
curl -X POST http://localhost:8000/admin/invalidate-cache
```

---

### GET /admin/cache-stats

Statistiques du cache des surfaces de cout.

**Reponse** :
```json
{
  "entries": 5,
  "total_size_mb": 234.7,
  "age_min_days": 0.1,
  "age_max_days": 12.3
}
```

**Exemple** :
```bash
curl http://localhost:8000/admin/cache-stats
```

---

## Modèles

### RouteRequest

| Champ | Type | Defaut | Description |
|-------|------|--------|-------------|
| `start_lat` | float | requis | Latitude du depart (WGS84, range 43-48) |
| `start_lon` | float | requis | Longitude du depart (WGS84, range 4-9) |
| `end_lat` | float | requis | Latitude de l'arrivee |
| `end_lon` | float | requis | Longitude de l'arrivee |
| `resolution` | float | 1.0 | Résolution Lidar en m (0.5, 1.0, 2.0, 5.0, 10.0) |
| `month` | int | 7 | Mois pour le facteur aspect/saison (1-12) |
| `acclimatized` | bool | true | Si l'utilisateur est acclimaté |
| `n_alternatives` | int | 0 | Nombre de routes alternatives (0-5) |
| `save` | bool | true | Sauvegarder automatiquement la route en DB |
| `name` | string | null | Nom optionnel pour la route |

### ZoneCreate

| Champ | Type | Defaut | Description |
|-------|------|--------|-------------|
| `name` | string | requis | Nom de la zone |
| `zone_type` | string | requis | Type parmi : `crevasse`, `serac`, `cornice`, `rockfall`, `forbidden`, `custom` |
| `cost_multiplier` | float | 100.0 | Multiplicateur de cout (1.0 = neutre, 1000 = quasi-interdit) |
| `geojson` | object | requis | GeoJSON Polygon ou MultiPolygon |
| `active` | bool | true | Zone active ou non |

### ZoneUpdate

| Champ | Type | Description |
|-------|------|-------------|
| `name` | string | Nouveau nom |
| `zone_type` | string | Nouveau type |
| `cost_multiplier` | float | Nouveau multiplicateur |
| `geojson` | object | Nouvelle geometrie |
| `active` | bool | Activer/desactiver |

## Notes

- Les coordonnees sont en WGS84 (EPSG:4326) en entrée/sortie. Le calcul interne se fait en Lambert-93.
- Le GeoJSON retourné contient des coordonnees 3D (lon, lat, élévation).
- Si la route dépasse 5000 points, elle est sous-echantillonnée coté serveur.
- La validation des inputs (lat/lon, résolutions autorisées, types de zone) est faite par Pydantic. Les erreurs de validation retournent un 422.

# Changelog

## [2.0.0-alpha.2]

Rework complet du moteur de calcul. Passage de 100% raster à graphes (Valhalla) + raster.

### Added
- Docker Valhalla pour fraph reseau OSM
- Client Python Valhalla : route, locate, status (routing/network.py)
- Config VALHALLA_BASE_URL configurable par env var
- Tests unitaires + integration network

### Changed
- docker-compose : ajout service valhalla + depends_on backend

## [2.0.0-alpha]

### Added
- Parametre `routing_mode` dans l'API (1=hors-piste pur, 2=OSM+hors-piste)
- ajout sentiers OSM dans le pipeline : trail_cost multiplié dans la surface de cout
- ajout barrieres OSM : rivières/autoroutes bloquées
- Etape SSE "osm" dans la progression du calcul

### Changed
- routing_mode=2 par defaut (OSM active)
- Redistribution des poids SSE pour inclure l'etape OSM
- Multiplicateurs trail fortement reduits (road 0.55->0.18, paved 0.50->0.15, etc.) pour un ratio sentier/hors-piste de 5-7x
- Penalite de proximite (x2.5) autour des sentiers
- Buffers trail elargis (road 2->3m, trail 1->1.5m)

## [1.3.0]

### Added
- Module OSM trails : dl Overpass, classification 12 niveaux (sac_scale, tracktype, surface), rasterisation sur grille lidar
- Module OSM barriers : rivières, autoroutes infranchissables, ruisseaux pénalisés, détection ponts/gué
- Cache local .gpkg pour les donnees OSM avec TTL 30j
- Tests unitaires trails et barriers

## [1.2.0]

### Changed
- Suppression OFF_TRAIL_FACTOR
- modifs multiplicateurs WorldCover : valeurs ajustées pour forêt, buissons, moraine, etc, eau et batiments infranchissables (1e6)

### Added
- Tests temporaires WorldCover : verif eau/bati infranchissables

## [1.1.1]

### Added
- environment.yml pour setup micromamba

### Fixed
- Typage ElevationProfile : formatter/labelFormatter Recharts + onMouseMove
- Cast geojson zones dans RouteMap (Record<string, unknown> au lieu de any)
- loadZones déclaré avant useEffect dans ZonePanel
- Cast inutile retiré dans useOverlays
- eslint-disable ciblé pour useApp dans context.tsx (react-refresh/only-export-components)

## [1.1.0]

### Added
- Warning API si mode isotrope avec D+/D- > 500m
- script benchmark + save JSON

### Changed
- Pénalité pente progressive 35-55 deg (remplace seuils durs 45/60 deg)
- Hypoxie à deux niveaux : 0.01/1000m (1500-2500m) et 0.03/1000m (>2500m)
- Gradient clippe a [-10, 10] avant calcul Tobler

## [1.0.3]

### Fixed
- Fuite file handles dans build_mosaic (ExitStack)
- useEffect deps manquantes dans HistoryPanel

### Changed
- Extraction getSelectedRoute dans types.ts
- Extraction hooks calques depuis RouteMap

## [1.0.2]

### Changed
- Nettoyage des commentaires python
- Ajouts de quelques infos dans la doc

## [1.0.1]

### Fixed
- Buffer routes alternatives en mètres (50m minimum qq soit la resolution)
- Alignement raster WorldCover via rasterio.warp au lieu de scipy.ndimage.zoom
- Doc : endpoint /glaciers et /cost-surface documentés

## [1.0.0]

Première version fonctionnelle en local.

### Added
- Pipeline de calcul complet : requête utilisateur -> GeoJSON
- Téléchargement automatique des dalles MNT IGN Lidar HD (0.5 - 10 m)
- Fallback MNT via MapTiler pour les zones hors couverture Lidar
- Analyse terrain : pente (Horn), orientation, rugosité (scipy)
- Surface de cout multi-critères : pente, altitude, aspect/saison, glacier, rugosité, couverture du sol
- Intégration contours glaciaires RGI 7.0
- Integration landcover ESA WorldCover 10 m
- Pathfinding Dijkstra isotrope sur grille implicite (skimage)
- Pathfinding Dijkstra anisotrope (scipy.sparse)
- Routes alternatives via méthode de pénalité
- API REST fastAPI avec calcul asynchrone et progression SSE
- Sauvegarde automatique des routes en SQLite
- Historique des routes avec rechargement
- Zones utilisateur (polygones d'exclusion/pénalité)
- Frontend React + MapLibreGL avec relief 3D
- Profil altimétrique avec synchro carte
- Sélection de fonds de carte IGN (topo, satellite)
- Calques superposables (pentes, glaciers, surface de cout)
- Sidebar avec onglets calcul / historique
- Docker Compose (backend + frontend nginx)

### Fixed
- Boucle SSE : les erreurs pipeline sont maintenant transmises au front (tracking status en plus du progress)
- Bug export GeoJSON sur routes longues
- Buffer API passé de 10 min a 24 h

### Removed
- Fond de carte "Pentes" (doublon avec le calque pentes)
- Calque "Hillshade" (doublon avec relief 3D)
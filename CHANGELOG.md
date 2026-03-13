# Changelog

## [2.0.0-beta]

### Added
- Recherche textuelle Nominatim pour départ/arrivée
- Affichage stratégie de routage (Reseau/Hybride/Raster) avec badge coloré
- Affichage des couches de donnée utilisées
- warning si Valhalla indisponible
- Calque traces alpinisme (couleur par cotation, tooltip hover)
- Calque segments terrain (pointille jaune, tooltip hover)
- Hook useNominatim avec debounce 500ms
- Appels API fetchAlpineRoutesGeoJSON, fetchSegmentsGeoJSON

### Changed
- Champs départ/arrivée : champ texte recherche Nominatim + coordonnées au clic carte
- Types TS : strategy, coverage, snap, layers_used, valhalla_available dans RouteResult
- OverlayId élargi : alpine-routes, segments

## [2.0.0-alpha.7]

### Changed
- PBF Valhalla : rhone-alpes -> alps-latest (arc alpin complet)
- VALID_LAT/LON_RANGE élargi (41-50 / -2-18) pour accepter les autres massifs français
- Healthcheck Docker Valhalla : start_period 300s -> 800s

### Added
- Garde couverture PBF : skip Valhalla si points hors bbox Alps
- Ghost route : rejet boucles (first==last <10m) et routes <3 points
- Champs réponse API : coverage, snap_start_m, snap_end_m, warnings
- Documentation procédure maj PBF (docs/data-sources.md)

### Fixed
- Warnings pas toujours inclus dans la réponse API (network path)

## [2.0.0-alpha.6]

### Fixed
- Vérification distance snap Valhalla (rejet routes fantome + arrêts prématurés)
- mode hybride : approche Valhalla + raster terminal (sensé suivre les sentiers OSM avec trail cost)
- assemble_route() intègre les stats Valhalla (distance, temps)
- exit_point utilise coords[-1] de la route Valhalla
- KeyError 'approach' quand find_network_entry entrait dans le bloc continuation
- Lignes droites à travers les montagnes (assemble_bridged_route supprimé)
- Overpass timeout 504 sur grandes bboxes (timeout 60 > 180s)
- Buffer trails alpins invisibles a 1m de résolution (0.5->3.0m)
- Multiplicateurs T4/T5/T6 trop faibles (T4:0.55->0.30, T5:0.70->0.35, T6:0.85->0.45)

### Added
- find_network_exit/entry pour le routage hybride
- parse_locate_snap pour exploiter /locate Valhalla
- Config: SNAP_MAX_DISTANCE_M, GHOST_ROUTE_MIN_DISTANCE_KM, HYBRID_BBOX_MARGIN_M
- Retry Overpass avec delai progressif (3 tentatives)
- Warning explicite quand trail_cost=None (aucun sentier OSM charge)

## [2.0.0-alpha.5]

### Added
- Segments terrain : rasterisation GPX custom + merge avec sentiers OSM
- Ponts raster : detection detours Valhalla + pathfinding local

### Changed
- Schema terrain_segments : ajout colonnes start/end lat/lon

## [2.0.0-alpha.4]

### Added
- Indexation traces GPX (index.json + sync SQLite)
- Module alpine/routes.py : parsing GPX, conversion GeoJSON, cotations
- Tables alpine_routes + terrain_segments
- Endpoints /alpine-routes, /alpine-routes/summits, /alpine-routes/{id}, /alpine-routes/{id}/gpx, /admin/reload-index
- POST /admin/reload-index

## [2.0.0-alpha.3]

### Added
- modif complète du pipeline, passe à un mode unique adaptatif : tentative Valhalla automatique avant raster
- Détection de détours excessif avec ratio vol d'oiseau
- Champs `strategy`, `valhalla_available`, `layers_used` dans la reponse API

### Changed
- Redistribution poids SSE (étape "network" ajoutée)

### Removed
- routing_mode dans RouteRequest

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
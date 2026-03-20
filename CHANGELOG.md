# Changelog

## [2.0.1]

Correctifs logique moteur de routage : re-entree Valhalla apres les gaps OSM.

### Fixed
- GPX partial : egress Valhalla depuis la sortie GPX quand le reseau OSM reprend de l'autre cote du gap (ex: traversee glaciaire Mer de Glace). Permet Valhalla + GPX + Valhalla sans passer par le raster.
- GPX partial bidirectionnel : detection des portails cote destination (reverse), pas seulement cote depart. Couvre les cas ou la trace GPX s'etend vers le depart depuis un portail proche de la destination.
- Detection portails GPX : snap augmente de 200m a 350m pour mieux couvrir les zones de montagne ou le reseau OSM est epars.

### Added
- Fonction `_try_valhalla_egress()` dans le pipeline : centralise la validation egress (detour, snap) pour eviter la duplication entre forward et reverse partial.
- Assemblage reverse partial dans le pipeline : raster approach + GPX + Valhalla egress quand l'approach Valhalla echoue en mode reverse.

## [2.0.0]

Rework complet : passage de 100% raster a un routage adaptatif Valhalla + graphe GPX + raster.

### Added
- Routage hybride : Valhalla (graphe OSM) + raster hors-piste, détection automatique de la stratégie
- Graphe GPX overlay : traces alpinisme indexées formant un réseau topologique connecté au réseau OSM
- Routage multi-graphe : Valhalla approche + GPX milieu + Valhalla sortie, dégradation partielle si un seul portail
- Sentiers OSM intégrés dans la surface de coût (trail_cost multiplicatif, ratio ~9x sentier/hors-piste)
- Barrières OSM : rivières/autoroutes infranchissables, ruisseaux pénalisés, détection ponts
- Segments terrain : rasterisation GPX custom + merge sentiers OSM
- Radiation solaire : modèle physique avec ombres portées (horizons + irradiance directe), remplace f_aspect
- Hill slope : pénalité progressive pour les pentes latérales (devers > 25 deg)
- Recherche textuelle Nominatim pour départ/arrivée
- Cache surface de coût : pré-calcul par zone/résolution/mois
- Cache radiation : angles d'horizon permanents, radiation mensuelle par zone
- Indexation traces GPX (index.json + sync SQLite)
- Endpoints : /alpine-routes, /alpine-routes/summits, /terrain-segments/geojson, /admin/reload-index
- Endpoints admin : /admin/invalidate-cache, /admin/cache-stats
- Calque traces alpinisme (couleur par cotation) et segments terrain
- Affichage stratégie de routage (Reseau/Hybride/Raster) avec badge
- Docker Valhalla (PBF Alps, arc alpin complet FR/IT/CH/AT/SI/DE)
- Champs réponse API : strategy, valhalla_available, layers_used, coverage, snap, warnings

### Changed
- Pipeline adaptatif : tentative Valhalla puis GPX graph puis fallback raster
- Surface de cout : trail_cost découplé des facteurs terrain (sentiers non pénalisés par glacier/rugosité)
- Couts glacier augmentés (flat 3.0, moderate 5.0, steep 10.0, very_steep 25.0)
- Multiplicateurs cotation alpinisme ajustés (T4:0.30, T5:0.35, T6:0.45)
- Champs départ/arrivée : recherche Nominatim + coords au clic carte
- SSE : étapes network, gpx_graph, cache, osm, radiation dans la progression
- PBF Valhalla : rhone-alpes -> alps-latest
- Buffers trail élargis (OSM 3.0m, alpine 5.0m, segments GPX 4.0m)

### Fixed
- Snap Valhalla : rejet routes fantome, boucles, arrêts prématurés
- Détection détours excessifs routage réseau
- Overpass API : 3 endpoints en rotation + timeout 300s + retry progressif
- Fallback aspect si radiation échoue (KeyError bbox corrigé)

### Removed
- routing_mode dans RouteRequest (remplacé par détection automatique)
- assemble_bridged_route (lignes droites à travers les montagnes)

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
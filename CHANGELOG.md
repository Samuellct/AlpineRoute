# Changelog

## [1.0.0] - 2026-02-19

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
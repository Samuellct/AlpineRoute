# AlpineRoute

Outil de calcul d'itineraires optimaux pour l'alpinisme hors-track. L'idée de base est de générer "automatiquement" des traces GPS en prenant en compte la topographie reelle du terrain (pentes, orientation, glaciers, zones de danger) a partir de modeles numeriques de terrain en hd.

## Contexte

La plupart des outils de planification d'itinéraires existants (CalTopo, Strava, Gaia GPS) sont conçus pour la rando sur sentier. Pour l'alpinisme hors-sentier, la planification reste largement manuelle : on étudie la carte, on trace un itinéraire au jugé, et on espère que ça passe. AlpineRoute vise à automatiser cette étape en s'appuyant sur des données topo précises et des modèles scientifiques de coût de déplacement en montagne.

## Etat du projet

Le projet est encore à l'étape de recherches d'infos et de tests. Je prévois de tester les différents modules prévus en utilisant des données d'une zone connue (surement autour de l'Aiguille du Midi).

Les points à valider :
- Dl et traitement des donnees Lidar HD MNT de l'IGN
- Calcul des attributs de terrain (pente, orientation, rugosité, etc)
- Intégration des contours glaciaires (Randolph Glacier Inventory (?))
- Test du calcul de cout multi-critères
- Pathfinding sur grille raster avec skimage
- Export des resultats en GPX et GeoJSON
- Affichage sur une carte web

## Sources de donnees prévues

- **MNT** : Lidar HD (only france) a 50 cm de resolution (maybe sous-echantillonne à 1 ou 2 m pour les tests). Fallback prevu sur Copernicus DEM GLO-30 si les data ne sont pas disponibles.
- **Glaciers** : RGI 7.0
- **Couverture du sol** : ESA WorldCover 10 m
- **Itineraires de reference** : API CampToCamp v6 (pour comparaison/validation)

## Stack technique

### Phase de test
- Python 3.11 (Miniforge)
- Rasterio, GDAL, NumPy, SciPy, scikit-image
- GeoPandas, Shapely, pyproj
- FastAPI (endpoint minimal)
- Frontend basique : Vite + React + TypeScript, Leaflet

### V1 (prévu)
- Backend avec FastAPI
- Frontend avec Vite + React + Leaflet + TailwindCSS + Recharts
- PostgreSQL + PostGIS, Redis, Docker
- Deploiement ghpages + backend API

## Structure du projet

```
AlpineRoute/
├── Phase_00_testsInitiaux/    # scripts de test independants
├── frontend/                  # app React/Vite
└── README.md
```

## Licence

Ce projet utilise une licence [MIT](LICENSE).

Les données externes utilisées sont soumises a leurs licences respectives :
- MNT IGN Lidar HD : Licence Ouverte Etalab 2.0
- RGI 7.0 : CC-BY 4.0
- ESA WorldCover : CC-BY 4.0
- CampToCamp : CC-BY-SA
- OpenStreetMap : ODbL

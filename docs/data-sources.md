# Sources de donnees

## IGN Lidar HD MNT

Modele numerique de terrain derive du programme national Lidar HD de l'IGN.

- **Produit** : MNT Lidar HD (Modele Numerique de Terrain)
- **Resolution native** : 50 cm
- **Resolution de travail** : 1 m (resample bilineaire, ajustable dans `config.py`)
- **Couverture** : France metropolitaine
- **CRS** : EPSG:2154 (Lambert-93)
- **Format** : GeoTIFF, dalles de 1 km x 1 km
- **Acces** : API WFS pour la decouverte des dalles, WMS-R pour le telechargement.
- **Portail** : [cartes.gouv.fr](https://cartes.gouv.fr) (ex-Geoplateforme)
- **Licence** : Licence Ouverte Etalab 2.0 (reutilisation libre, attribution)

Le pipeline telecharge automatiquement les dalles qui intersectent la bbox de la requete, les assemble en mosaique et resample a la resolution demandee.

## Copernicus GLO-30

MNT global utilisé comme fallback quand les data IGN ne sont pas disponibles (zones hors France).

- **Produit** : Copernicus DEM GLO-30
- **Resolution** : ~30 m (1 arc-sec)
- **Couverture** : Mondiale
- **CRS** : EPSG:4326 (WGS84 natif), reprojete en L93 par le pipeline
- **Format** : GeoTIFF COG, tuiles de 1 deg x 1 deg
- **Acces** : S3 public (`copernicus-dem-30m.s3.eu-central-1.amazonaws.com`)
- **Licence** : Licence libre (Copernicus data policy)

La resolution est 60x inferieure au Lidar HD.

## RGI 7.0

Contours des glaciers pour le masque glacier dans la surface de cout.

- **Produit** : Randolph Glacier Inventory version 7.0
- **Contenu** : Contours vectoriels (polygones) de tous les glaciers du monde
- **Format** : Shapefile
- **Region utilisee** : RGI region 11 (Europe centrale)
- **Acces** : Telechargement depuis [NSIDC](https://nsidc.org/data/nsidc-0770)
- **Licence** : CC-BY 4.0
- **Reference** : RGI 7.0 Consortium (2023). *Randolph Glacier Inventory - A Dataset of Global Glacier Outlines, Version 7.0.* NSIDC.

Les contours sont rasterisés sur la grille lidar MNT (même résolution, même emprise) pour produire un masque glacier/non-glacier.

### Telechargement manuel des shapefiles RGI

NSIDC requiert un compte (Earthdata Login) pour acceder aux fichiers.

1. Creer un compte sur [Earthdata](https://urs.earthdata.nasa.gov/users/new)
2. Aller sur la page du dataset : https://nsidc.org/data/nsidc-0770
3. Telecharger l'archive de la **region 11** (Central Europe) :
   `RGI2000-v7.0-G-11_central_europe.zip` (~20 Mo)
4. Extraire le zip. On obtient un dossier avec les fichiers shapefile
   (`.shp`, `.shx`, `.dbf`, `.prj`, `.cpg`)
5. Placer les fichiers dans `data/rgi/` a la racine du projet :
   ```
   data/rgi/
     RGI2000-v7.0-G-11_central_europe.shp
     RGI2000-v7.0-G-11_central_europe.shx
     RGI2000-v7.0-G-11_central_europe.dbf
     RGI2000-v7.0-G-11_central_europe.prj
     RGI2000-v7.0-G-11_central_europe.cpg
   ```

Le backend detecte automatiquement les shapefiles dans `$DATA_DIR/rgi/`.
Si le dossier est vide ou absent, le masque glacier est ignore (pas de crash).

### Docker : montage des donnees RGI

Pour rendre les donnees disponibles dans le conteneur, monter un volume
dans `docker-compose.yml` :

```yaml
backend:
  volumes:
    - ./data:/app/data
```

Les shapefiles dans `data/rgi/` seront accessibles au backend dans `/app/data/rgi/`.

## ESA WorldCover

Couverture du sol a 10 m pour les multiplicateurs de cout par type de terrain.

- **Produit** : ESA WorldCover v200 (2021)
- **Resolution** : 10 m
- **Couverture** : Mondiale
- **Classes** : 11 classes de couverture du sol
- **CRS** : EPSG:4326 natif, reprojete en L93 (nearest neighbor pour donnees categorielles)
- **Format** : GeoTIFF COG
- **Acces** : S3 public via `/vsicurl/` (lecture fenetree, pas besoin de telecharger la tuile entiere)
- **Licence** : CC-BY 4.0
- **Reference** : Zanaga D. et al. (2022). *ESA WorldCover 10 m 2021 v200.*

Les multiplicateurs par classe sont definis dans `config.py` (`WORLDCOVER_MULTIPLIERS`). Voir [cost-function.md](cost-function.md) pour le detail.

## OSM PBF (Valhalla)

Réseau routier et sentiers utilisé par Valhalla pour le routage sur le réseau OSM.

- **Produit** : Extrait PBF Geofabrik
- **Zone** : Alps (`alps-latest.osm.pbf`)
- **Couverture** : Arc alpin complet FR/IT/CH/AT/SI/DE (~4E-17.5E, ~43N-49N)
- **Acces** : http://download.geofabrik.de/europe/alps-latest.osm.pbf
- **Mise a jour** : Geofabrik quotidien
- **Licence** : ODbL (OpenStreetMap)

### Procédure de mise à jour PBF

1. Modifier `VALHALLA_PBF_URL` dans `config.py` avec la nouvelle URL
2. Modifier `tile_urls` dans `docker-compose.yml`
3. Mettre à jour `VALHALLA_COVERAGE_BBOX` dans `config.py` si la zone change
4. Passer `force_rebuild=True` dans `docker-compose.yml`
5. `docker compose down && docker compose up -d` (delete l'ancien volume avec l'ancien fichier pbf)
6. Attendre le rebuild des tiles (~10-20 min pour Alps, ~5 min pour Rhone-Alpes)
7. Remettre `force_rebuild=False` après le build

## Nominatim (geocoding)

Recherche de lieux par nom dans les champs départ/arrivée du frontend.

- **Service** : API Nominatim publique (OpenStreetMap)
- **Usage** : recherche textuelle avec debounce 500ms cote frontend
- **Licence** : ODbL (OpenStreetMap)

## Traces GPX alpinisme

Traces de courses d'alpinisme indexées localement pour le graph overlay et l'affichage sur la carte.

- **Format** : fichiers GPX dans `data/gpx/`, indexés par `index.json`
- **Contenu** : routes completès et segments terrain avec cotation
- **Source** : traces personnelles

Voir [gpx-traces.md](gpx-traces.md) pour le format et l'ajout de traces.

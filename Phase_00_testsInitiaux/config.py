# config phase 00

import os

# ---- Chemins ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DEM_DIR = os.path.join(DATA_DIR, "dem")
RGI_DIR = os.path.join(DATA_DIR, "rgi")
DERIVED_DIR = os.path.join(DATA_DIR, "derived")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MAPS_DIR = os.path.join(OUTPUT_DIR, "maps")
GPX_DIR = os.path.join(OUTPUT_DIR, "gpx")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")

# ---- Zone de test Aiguille du Midi ----
# bbox WGS84 (lon/lat) pour les APIs web
BBOX_WGS84 = {
    "lon_min": 6.84,
    "lon_max": 6.92,
    "lat_min": 45.85,
    "lat_max": 45.90,
}

# bbox Lambert-93 (EPSG:2154) pour le traitement raster
BBOX_L93 = {
    "xmin": 1002000,
    "ymin": 6536000,
    "xmax": 1008000,
    "ymax": 6542000,
}

# crs
CRS_L93 = "EPSG:2154"
CRS_WGS84 = "EPSG:4326"

# ---- Resolution DEM ----
# 0.5 = natif Lidar HD, 1.0 = phase test, 2.0+ = debug rapide
DEM_RESOLUTION = 1.0  # metres

# ---- pts de test pour le pathfinding ----
# Gare arrivee telepherique Aiguille du Midi
START_POINT_WGS84 = (45.8793, 6.8874)  # (lat, lon)
# Refuge du Requin
END_POINT_WGS84 = (45.8845, 6.9297)    # (lat, lon)  

# ---- Params par defaut pour la surface de cout ----
DEFAULT_SEASON_MONTH = 7       # juillet
DEFAULT_ACCLIMATIZED = True
DEFAULT_LOAD_KG = 10

# ---- Divers ----
NODATA_VALUE = -9999.0

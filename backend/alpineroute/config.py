# config centralisee -- tous les params du pipeline

import os

# ---- chemins de base ----
# en dev: relatif au repo. En prod/docker: surcharge via env vars
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.environ.get("ALPINEROUTE_DATA_DIR", os.path.join(BASE_DIR, "data"))
DEM_CACHE_DIR = os.path.join(DATA_DIR, "cache", "dem")
RGI_DIR = os.path.join(DATA_DIR, "rgi")
DERIVED_DIR = os.path.join(DATA_DIR, "derived")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
DB_PATH = os.environ.get("ALPINEROUTE_DB_PATH", os.path.join(DATA_DIR, "alpineroute.db"))
GPX_DIR = os.path.join(DATA_DIR, "gpx")
GPX_INDEX_PATH = os.path.join(GPX_DIR, "index.json")

# ---- CRS ----
CRS_L93 = "EPSG:2154"
CRS_WGS84 = "EPSG:4326"

# ---- resolution lidar par defaut----
DEM_RESOLUTION = 1.0

# ---- nodata ----
NODATA_VALUE = -9999.0

# ---- IGN WFS/WMS-R ----
WFS_URL = "https://data.geopf.fr/wfs/ows"
WFS_TYPENAME = "IGNF_MNT-LIDAR-HD:dalle"
HTTP_TIMEOUT = 60
MAX_RETRIES = 3
RETRY_DELAY = 2.0

# ---- Copernicus GLO-30 (fallback hors France) ----
COPERNICUS_S3_BASE = "https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com"
COPERNICUS_RESOLUTION = 30.0  # metres (natif ~30m, 1 arcsec)
COPERNICUS_DL_TIMEOUT = 120

# ---- bbox dynamique ----
BBOX_MARGIN_M = 4000       # marge autour des pts depart/arrivee
BBOX_MAX_SIZE_M = 30000    # max 30km x 30km
BBOX_ALIGN_M = 1000        # aligner sur les dalles IGN (1km)

# ---- cost function : Tobler ----
TOBLER_BASE_SPEED_KMH = 6.0
TOBLER_OPTIMAL_GRADIENT = 0.05
GRADIENT_CLIP = 10.0  # |gradient| max avant Tobler

# ---- cost function : penalite progressive pour les pentes ----
STEEP_ONSET_DEG = 35          # deg debut penalite
STEEP_FULL_DEG = 55           # deg de penalite maximale
STEEP_MAX_MULTIPLIER = 20.0
SERAC_SLOPE_DEG = 35          # seuil seracs a midifier

# ---- cost function : altitude ----
HYPOXIA_ALTITUDE_THRESHOLD = 1500
HYPOXIA_MODERATE_THRESHOLD = 2500   # palier ou le taux augmente
HYPOXIA_RATE_MODERATE = 0.01        # 1500-2500m
HYPOXIA_RATE_ACCLIMATIZED = 0.03    # >2500m
HYPOXIA_RATE_NOT_ACCLIMATIZED = 0.063
HYPOXIA_MIN_CAPACITY = 0.3

# ---- cost function : aspect / saison ----
ASPECT_SUMMER_MONTHS = [6, 7, 8, 9]
ASPECT_SOUTH_PENALTY_MAX = 0.5      # +50 % max sur face sud en ete
ASPECT_SOUTH_SLOPE_THRESHOLD = 30
ASPECT_SOUTH_ALTITUDE_THRESHOLD = 2500
ASPECT_NORTH_PENALTY_MAX = 0.3      # +30 % max sur face nord en hiver
ASPECT_NORTH_SLOPE_THRESHOLD = 25

# ---- cost function : glacier ----
GLACIER_COST_FLAT = 1.3             # pente < 10 deg
GLACIER_COST_MODERATE = 2.0         # 10-20 deg
GLACIER_COST_STEEP = 4.0            # 20-30 deg
GLACIER_COST_VERY_STEEP = 10.0      # > 30 deg

# ---- cost function : rugosite ----
ROUGHNESS_CLAMP = 5.0               # max TRI en metres
ROUGHNESS_SCALE = 0.8               # cost = 1 + scale * TRI

# ---- cost function : WorldCover (ESA 2021) ----
WORLDCOVER_MULTIPLIERS = {
    10: 3.5,    # foret
    20: 2.5,    # buissons
    30: 1.1,    # herbe/alpage
    40: 1.5,    # cultures
    50: 1e6,    # bati - infranchissable
    60: 3.0,    # moraine/eboulis
    70: 1.5,    # neige/glace
    80: 1e6,    # eau - infranchissable
    90: 5.0,    # zone humide
    95: 3.0,    # mangrove
    100: 1.2,   # mousse/lichen
    0: 1.0,     # nodata
}
WORLDCOVER_URL_PATTERN = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"

# ---- OSM / sentiers & barrieres ----
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OVERPASS_TIMEOUT = 60
OSM_CACHE_DIR = os.path.join(DATA_DIR, "cache", "osm")
OSM_CACHE_TTL_DAYS = 30

TRAIL_COST_MULTIPLIERS = {
    "paved": 0.15,
    "gravel": 0.20,
    "road": 0.18,
    "trail_t1t2": 0.25,
    "trail_default": 0.30,
    "trail_t3": 0.40,
    "track_soft": 0.45,
    "trail_t4": 0.55,
    "trail_t5": 0.70,
    "trail_t6": 0.85,
}

TRAIL_BUFFER_M = {
    "road": 3.0,
    "trail": 1.5,
    "alpine": 0.5,
}

# penalite proximite sentier: px proches d'un sentier mais hors sentier
TRAIL_PROXIMITY_BUFFER_M = 8.0
TRAIL_PROXIMITY_PENALTY = 2.5

RIVER_BUFFER_M = 5.0
CANAL_BUFFER_M = 3.0
STREAM_BUFFER_M = 2.0
BRIDGE_BUFFER_M = 3.0
MOTORWAY_BUFFER_M = 5.0
STREAM_CROSSING_PENALTY = 6.0

# ---- Valhalla (routage reseau) ----
VALHALLA_BASE_URL = os.environ.get("ALPINEROUTE_VALHALLA_URL", "http://localhost:8002")
VALHALLA_TIMEOUT_S = 30
VALHALLA_MAX_HIKING_DIFFICULTY = 6   # echelle 0-6 Valhalla pr les chemins T1-T6 osm
VALHALLA_PBF_URL = "http://download.geofabrik.de/europe/france/rhone-alpes-latest.osm.pbf"
VALHALLA_DETOUR_THRESHOLD = 3.0   # ratio dist_valhalla / vol_oiseau
VALHALLA_MIN_DIRECT_M = 500       # en dessous, pas de test detour

# ponts raster (Phase 7) -- comble les gaps OSM courts
BRIDGE_MAX_DISTANCE_M = 300       # dist max directe pour tenter un pont
BRIDGE_DETOUR_RATIO = 2.5         # ratio leg/direct pour detecter un detour
BRIDGE_BBOX_MARGIN_M = 200        # marge autour du pont pour le pathfinding local

# ---- hillshade ----
HILLSHADE_AZIMUTH = 315
HILLSHADE_ALTITUDE = 45
HILLSHADE_VERT_EXAG = 2

# ---- export ----
SIMPLIFY_TOLERANCE_M = 5.0
UHD_DPI = 600

# ---- API ----
API_HOST = os.environ.get("ALPINEROUTE_HOST", "127.0.0.1")
API_PORT = int(os.environ.get("ALPINEROUTE_PORT", "8000"))
CORS_ORIGINS = os.environ.get("ALPINEROUTE_CORS_ORIGINS", "*").split(",")

# ---- validation inputs ----
VALID_LAT_RANGE = (43.0, 48.0)   # Alpes grosso modo
VALID_LON_RANGE = (4.0, 16.0)
VALID_RESOLUTIONS = (0.5, 1.0, 2.0, 5.0, 10.0)
MAX_GRID_PIXELS = 200_000_000    # garde-fou memoire
MAX_GRID_PIXELS_ANISO = int(os.environ.get(
    "ALPINEROUTE_MAX_GRID_ANISO", 50_000_000
))  # aniso gourmand en RAM, surcharger si machine costaud

# ---- warnings ----
ISOTROPIC_WARNING_DPLUS_M = 500  # seuil D+ pour warning isotrope

# ---- pathfinding ----
COST_NODATA_VALUE = 1e6     # cap pour eviter overflow
MAX_ROUTE_POINTS_API = 5000  # sous-echantillonnage si plus

# ---- routes alternatives (penalty method) ----
N_ALTERNATIVE_ROUTES = 3
PENALTY_MULTIPLIER = 5.0
PENALTY_BUFFER_PX = 15          # minimum garanti en px
PENALTY_BUFFER_M = 50           # buffer minimum en m
MAX_ALTERNATIVE_ROUTES = 5

# ---- zones utilisateur ----
FORBIDDEN_ZONE_MULTIPLIER = 1000.0
ZONE_TYPES = ("crevasse", "serac", "cornice", "rockfall", "forbidden", "custom")

# config centralisee -- tous les params du pipeline

import os

# ---- chemins de base ----
# en dev: relatif au repo. En prod/docker: surcharge via env vars
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.environ.get("ALPINEROUTE_DATA_DIR", os.path.join(BASE_DIR, "data"))
DEM_CACHE_DIR = os.path.join(DATA_DIR, "cache", "dem")       # dalles MNT telecharges
COST_CACHE_DIR = os.path.join(DATA_DIR, "cache", "cost")     # surfaces de cout serialisees npz
COST_CACHE_MAX_AGE_DAYS = 90    # TTL cache en jours
COST_CACHE_VERSION = "2.0.3"    # bump pour invalider les anciens caches
RGI_DIR = os.path.join(DATA_DIR, "rgi")         # shapefiles glaciers RGI 7.0
DERIVED_DIR = os.path.join(DATA_DIR, "derived")  # rasters derives (slope etc)
OUTPUT_DIR = os.path.join(BASE_DIR, "output")    # exports GPX/GeoJSON
DB_PATH = os.environ.get("ALPINEROUTE_DB_PATH", os.path.join(DATA_DIR, "alpineroute.db"))  # sqlite
GPX_DIR = os.path.join(DATA_DIR, "gpx")          # traces GPX importees
GPX_INDEX_PATH = os.path.join(GPX_DIR, "index.json")  # index des traces

# graphe GPX overlay (Phase 11)
GPX_SUBSAMPLE_M = 10          # sous-echantillonnage (avant 50, coupait les virages)
GPX_MERGE_TOLERANCE_M = 12    # fusion noeuds proches inter-traces (avant 30, ecrasait les lacets)
GPX_PORTAL_SNAP_M = 350       # max dist snap pour portal Valhalla
GPX_CORRIDOR_RATIO = 0.50     # largeur corridor = 50% de la dist start-end
GPX_CORRIDOR_MIN_M = 4000     # corridor minimum 4km
GPX_ROUTE_TRAIL_COST = 0.30   # trail_cost fixe pour les traces "route"

# ---- CRS ----
CRS_L93 = "EPSG:2154"    # Lambert-93, projection officielle France
CRS_WGS84 = "EPSG:4326"  # GPS classique (lat/lon)

# ---- resolution lidar par defaut ----
DEM_RESOLUTION = 1.0      # metres/pixel, cf CLAUDE.md (test phase = 1m)

# ---- nodata ----
NODATA_VALUE = -9999.0    # sentinel pour les px sans donnees

# ---- IGN WFS/WMS-R ----
WFS_URL = "https://data.geopf.fr/wfs/ows"        # Geoplateforme, ex wxs.ign.fr
WFS_TYPENAME = "IGNF_MNT-LIDAR-HD:dalle"          # couche dalles MNT Lidar HD
HTTP_TIMEOUT = 60       # timeout requetes WFS/download (sec)
MAX_RETRIES = 3         # nb retries sur erreur reseau
RETRY_DELAY = 2.0       # delai entre retries (sec)

# ---- Copernicus GLO-30 (fallback hors France) ----
COPERNICUS_S3_BASE = "https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com"  # S3 public
COPERNICUS_RESOLUTION = 30.0  # metres (natif ~30m, 1 arcsec)
COPERNICUS_DL_TIMEOUT = 120   # timeout download (sec)

# ---- bbox dynamique ----
BBOX_MARGIN_M = 4000       # marge autour des pts depart/arrivee
BBOX_MAX_SIZE_M = 30000    # max 30km x 30km
BBOX_ALIGN_M = 1000        # aligner sur les dalles IGN (1km)

# ---- cost function : Tobler ----
TOBLER_BASE_SPEED_KMH = 6.0    # vitesse max de la fct Tobler (terrain plat)
TOBLER_OPTIMAL_GRADIENT = 0.05  # pente optimale legere descente (~3 deg)
GRADIENT_CLIP = 10.0  # |gradient| max avant Tobler

# ---- cost function : penalite progressive pour les pentes ----
STEEP_ONSET_DEG = 35          # deg debut penalite
STEEP_FULL_DEG = 55           # deg de penalite maximale
STEEP_MAX_MULTIPLIER = 20.0   # x20 au-dela de STEEP_FULL_DEG
SERAC_SLOPE_DEG = 35          # seuil seracs a modifier

# ---- cost function : altitude ----
HYPOXIA_ALTITUDE_THRESHOLD = 1500       # debut effet altitude (m)
HYPOXIA_MODERATE_THRESHOLD = 2500       # palier ou le taux augmente
HYPOXIA_RATE_MODERATE = 0.01            # 1500-2500m
HYPOXIA_RATE_ACCLIMATIZED = 0.03        # >2500m, personne acclimatee
HYPOXIA_RATE_NOT_ACCLIMATIZED = 0.063   # >2500m, non acclimatee
HYPOXIA_MIN_CAPACITY = 0.3             # plancher capacite (jamais sous 30%)

# ---- cost function : aspect / saison ----
ASPECT_SUMMER_MONTHS = [6, 7, 8, 9]   # juin-sept = ete pour l'aspect
ASPECT_SOUTH_PENALTY_MAX = 0.5        # +50 % max sur face sud en ete
ASPECT_SOUTH_SLOPE_THRESHOLD = 30     # deg min pour appliquer la penalite sud
ASPECT_SOUTH_ALTITUDE_THRESHOLD = 2500  # altitude min pour penalite sud (m)
ASPECT_NORTH_PENALTY_MAX = 0.3        # +30 % max sur face nord en hiver
ASPECT_NORTH_SLOPE_THRESHOLD = 25     # deg min pour penalite nord

# ---- cost function : glacier ----
# couts eleves: crevasses, encordement, crampons obligatoires
GLACIER_COST_FLAT = 3.0             # pente < 10 deg (plat mais crevasses)
GLACIER_COST_MODERATE = 5.0         # 10-20 deg
GLACIER_COST_STEEP = 10.0           # 20-30 deg (zone de seracs)
GLACIER_COST_VERY_STEEP = 25.0      # > 30 deg (chutes de seracs, infranchissable)

# ---- cost function : rugosite ----
ROUGHNESS_CUT = 5.0               # max TRI en metres
ROUGHNESS_SCALE = 0.8               # cost = 1 + scale * TRI

# ---- cost function : hill slope (devers / pente laterale) ----
HILLSLOPE_ONSET_DEG = 25            # debut penalite
HILLSLOPE_SCALE = 0.8               # raideur de la penalite

# ---- radiation solaire ----
RADIATION_CACHE_DIR = os.path.join(DATA_DIR, "cache", "radiation")
RADIATION_N_AZIMUTHS = 36           # 36 directions = 10 deg par pas
RADIATION_DEM_RESOLUTION = 5.0      # resolution reduite pour les horizons
RADIATION_TIME_STEP_H = 0.5         # pas de temps integration journaliere (30 min)
RADIATION_HORIZON_MAX_DIST_M = 5000  # portee max calcul ombres (m)
RADIATION_SUMMER_PENALTY = 0.5      # penalite max face exposee en ete
RADIATION_WINTER_PENALTY = 0.3      # penalite max face ombree en hiver
RADIATION_SUMMER_MONTHS = [6, 7, 8, 9]
RADIATION_SLOPE_THRESHOLD = 15      # pas de penalite radiation sous ce seuil (deg)
RADIATION_ALTITUDE_THRESHOLD = 2000 # pas de penalite radiation sous cette altitude (m)

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
WORLDCOVER_URL_PATTERN = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"  # S3 public ESA

# ---- OSM / sentiers & barrieres ----
OVERPASS_URLS = [                # serveurs Overpass avec fallback
    "https://overpass-api.de/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
    "https://z.overpass-api.de/api/interpreter",
]
OVERPASS_URL = OVERPASS_URLS[0]  # compat retro
OVERPASS_TIMEOUT = 300           # avant 180, augmente pour gros bbox
OSM_CACHE_DIR = os.path.join(DATA_DIR, "cache", "osm")  # cache requetes Overpass
OSM_CACHE_TTL_DAYS = 30         # duree cache OSM (jours)

TRAIL_COST_MULTIPLIERS = {       # multiplicateur cout par type de sentier (<1 = bonus)
    "paved": 0.15,
    "gravel": 0.20,
    "road": 0.18,
    "trail_t1t2": 0.25,
    "trail_default": 0.30,
    "trail_t3": 0.40,
    "track_soft": 0.45,
    "trail_t4": 0.30,      # avant 0.55 trop faible vs terrain alpin
    "trail_t5": 0.35,      # avant 0.70
    "trail_t6": 0.45,      # avant 0.85
}

TRAIL_BUFFER_M = {               # largeur rasterisation sentier (m)
    "road": 3.0,
    "trail": 3.0,            # avant 1.5, trop etroit a 1m (3px -> decrochages)
    "alpine": 5.0,           # avant 3.0, elargir pour garder le pathfinder colle
}

# penalite proximite sentier: px proches d'un sentier mais hors sentier
TRAIL_PROXIMITY_BUFFER_M = 8.0
TRAIL_PROXIMITY_PENALTY = 5.0       # avant 2.5, renforce pour empecher les decrochages

# graphe local OSM (fallback quand Valhalla ne couvre pas la zone)
TRAIL_GRAPH_SUBSAMPLE_M = 8         # sous-echantillonnage (dense pour garder les lacets)
TRAIL_GRAPH_MERGE_M = 5             # fusion noeuds (conservateur pour eviter les raccourcis)
TRAIL_GRAPH_MAX_SNAP_M = 300        # snap max start/end sur le graphe

RIVER_BUFFER_M = 5.0              # largeur rasterisation barrieres (m)
CANAL_BUFFER_M = 3.0
STREAM_BUFFER_M = 2.0
BRIDGE_BUFFER_M = 3.0             # ponts = ouvertures dans les barrieres
MOTORWAY_BUFFER_M = 5.0
STREAM_CROSSING_PENALTY = 6.0     # penalite traversee torrent hors pont

# ---- Valhalla (routage reseau) ----
VALHALLA_BASE_URL = os.environ.get("ALPINEROUTE_VALHALLA_URL", "http://localhost:8002")  # conteneur Docker
VALHALLA_TIMEOUT_S = 30           # timeout appel /route (sec)
VALHALLA_MAX_HIKING_DIFFICULTY = 6   # echelle 0-6 Valhalla pr les chemins T1-T6 osm
VALHALLA_PBF_URL = "http://download.geofabrik.de/europe/alps-latest.osm.pbf"  # PBF Geofabrik Alps
VALHALLA_DETOUR_THRESHOLD = 3.0   # ratio dist_valhalla / vol_oiseau
VALHALLA_MIN_DIRECT_M = 500       # en dessous, pas de test detour

# ponts raster (Phase 7) -- comble les gaps OSM courts
BRIDGE_MAX_DISTANCE_M = 300       # dist max directe pour tenter un pont
BRIDGE_DETOUR_RATIO = 2.5         # ratio leg/direct pour detecter un detour
BRIDGE_BBOX_MARGIN_M = 200        # marge autour du pont pour le pathfinding local

# verification snap Valhalla -- dist max entre point demande et snap reseau
SNAP_MAX_DISTANCE_M = 500
GHOST_ROUTE_MIN_DISTANCE_KM = 1.0   # en dessous, route fantome detectee
HYBRID_BBOX_MARGIN_M = 4000         # marge bbox pour CAS B (assez large pour alternatives)

# bbox couverture PBF Alps (WGS84) -- west, south, east, north
# couvre l'arc alpin FR/IT/CH/AT/SI/DE, a ajuster si changement de PBF
VALHALLA_COVERAGE_BBOX = (4.0, 43.0, 17.5, 49.0)

# ---- hillshade (code mort, cf CLN-02) ----
HILLSHADE_AZIMUTH = 315       # direction lumiere (NW)
HILLSHADE_ALTITUDE = 45       # angle soleil (deg)
HILLSHADE_VERT_EXAG = 2       # exageration verticale

# ---- export ----
SIMPLIFY_TOLERANCE_M = 5.0    # tolerance Douglas-Peucker pour simplif trace
UHD_DPI = 600                 # DPI figures haute-res (code mort, cf CLN-03)

# ---- API ----
API_HOST = os.environ.get("ALPINEROUTE_HOST", "127.0.0.1")  # bind address
API_PORT = int(os.environ.get("ALPINEROUTE_PORT", "8000"))   # port uvicorn
CORS_ORIGINS = os.environ.get("ALPINEROUTE_CORS_ORIGINS", "*").split(",")  # origins autorises

# ---- validation inputs ----
VALID_LAT_RANGE = (41.0, 50.0)    # France + Alpes + Pyrenees + Benelux
VALID_LON_RANGE = (-2.0, 18.0)    # Atlantique -> Alpes orientales
VALID_RESOLUTIONS = (0.5, 1.0, 2.0, 5.0, 10.0)
MAX_GRID_PIXELS = 200_000_000     # garde-fou memoire (isotrope)
MAX_GRID_PIXELS_ANISO = int(os.environ.get(
    "ALPINEROUTE_MAX_GRID_ANISO", 50_000_000
))  # aniso gourmand en RAM, surcharger si machine costaud

# ---- warnings ----
ISOTROPIC_WARNING_DPLUS_M = 500  # seuil D+ pour warning isotrope

# ---- pathfinding ----
COST_NODATA_VALUE = 1e6     # cap pour eviter overflow
MAX_ROUTE_POINTS_API = 5000  # sous-echantillonnage si plus

# ---- routes alternatives (penalty method) ----
N_ALTERNATIVE_ROUTES = 3        # nb alternatives par defaut
PENALTY_MULTIPLIER = 5.0        # facteur penalite sur le chemin optimal
PENALTY_BUFFER_PX = 15          # minimum garanti en px
PENALTY_BUFFER_M = 50           # buffer minimum en m
MAX_ALTERNATIVE_ROUTES = 5      # max pour l'API

# ---- zones utilisateur ----
FORBIDDEN_ZONE_MULTIPLIER = 1000.0  # quasi-infranchissable
ZONE_TYPES = ("crevasse", "serac", "cornice", "rockfall", "forbidden", "custom")  # types valides

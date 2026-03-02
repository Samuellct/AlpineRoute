# fixtures partagees pour les tests
import os
import tempfile
import numpy as np
import pytest
from rasterio.transform import from_origin

from alpineroute.db.schema import init_db


# -- DEM synthetiques --

@pytest.fixture
def tiny_dem():
    """Plan incline 3000-3500m avec un peu de bruit."""
    rng = np.random.default_rng(42)
    rows, cols = 20, 20
    gradient = np.linspace(3000, 3500, rows).reshape(-1, 1)
    dem = np.broadcast_to(gradient, (rows, cols)).copy().astype(np.float32)
    dem += rng.normal(0, 2, dem.shape).astype(np.float32)
    return dem


@pytest.fixture
def flat_dem():
    """Terrain plat a 3000m pile."""
    return np.full((20, 20), 3000.0, dtype=np.float32)


@pytest.fixture
def fake_transform():
    """Affine 1m resolution centree sur Chamonix en L93."""
    # coin NW quelque part pres de l'Aiguille du Midi
    return from_origin(1001000.0, 6542000.0, 1.0, 1.0)


@pytest.fixture
def glacier_mask():
    """Masque glacier: bande centrale True."""
    mask = np.zeros((20, 20), dtype=bool)
    mask[8:12, :] = True
    return mask


# -- base SQLite temporaire --

@pytest.fixture
def tmp_db(tmp_path):
    """DB SQLite dans un dossier temporaire, tables creees."""
    db_file = str(tmp_path / "test.db")
    init_db(db_file)
    return db_file

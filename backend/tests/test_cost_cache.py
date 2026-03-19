# tests cache surface de cout

import os
import time
import tempfile

import numpy as np
import pytest
from unittest.mock import patch
from rasterio.transform import Affine

from alpineroute.cost.cache import (
    cost_cache_key, get_cached_cost, save_cost_cache,
    invalidate_cache, cache_stats,
)


@pytest.fixture
def tmp_cache_dir(tmp_path):
    cache_dir = str(tmp_path / "cost_cache")
    os.makedirs(cache_dir)
    with patch("alpineroute.cost.cache.COST_CACHE_DIR", cache_dir):
        yield cache_dir


@pytest.fixture
def sample_data():
    """Donnees test pour le cache."""
    shape = (100, 120)
    rng = np.random.default_rng(42)
    return {
        "cached_base": rng.random(shape).astype(np.float32) + 1.0,
        "slope_deg": rng.random(shape).astype(np.float32) * 45,
        "dem": rng.random(shape).astype(np.float32) * 3000 + 1000,
        "glacier_mask": (rng.random(shape) > 0.9),
        "nodata_mask": (rng.random(shape) > 0.95),
        "transform": Affine(1.0, 0, 950000, 0, -1.0, 6500000),
        "bbox_l93": {"xmin": 950000, "ymin": 6499880, "xmax": 950120, "ymax": 6500000},
        "resolution": 1.0,
        "month": 7,
    }


def _save(cache_dir, data, key=None):
    """Helper: save et retourne la cle."""
    if key is None:
        key = cost_cache_key(data["bbox_l93"], data["resolution"], data["month"])
    save_cost_cache(
        key, data["cached_base"], data["slope_deg"], data["dem"],
        data["glacier_mask"], data["nodata_mask"], data["transform"],
        data["bbox_l93"], data["resolution"], data["month"],
    )
    return key


def test_cache_key_deterministic():
    bbox = {"xmin": 950000, "ymin": 6499000, "xmax": 951000, "ymax": 6500000}
    k1 = cost_cache_key(bbox, 1.0, 7)
    k2 = cost_cache_key(bbox, 1.0, 7)
    assert k1 == k2
    assert len(k1) == 16
    # mois different = cle differente
    k3 = cost_cache_key(bbox, 1.0, 1)
    assert k3 != k1


def test_cache_miss(tmp_cache_dir):
    result = get_cached_cost("nonexistent_key_12")
    assert result is None


def test_roundtrip(tmp_cache_dir, sample_data):
    key = _save(tmp_cache_dir, sample_data)
    loaded = get_cached_cost(key)
    assert loaded is not None
    assert np.allclose(loaded["cached_base"], sample_data["cached_base"], atol=1e-5)
    assert np.allclose(loaded["slope_deg"], sample_data["slope_deg"], atol=1e-5)
    assert np.allclose(loaded["dem"], sample_data["dem"], atol=1e-5)
    assert np.array_equal(loaded["glacier_mask"], sample_data["glacier_mask"])
    assert np.array_equal(loaded["nodata_mask"], sample_data["nodata_mask"])
    # verif transform
    assert loaded["transform"].a == sample_data["transform"].a
    assert loaded["transform"].f == sample_data["transform"].f


def test_invalidate_all(tmp_cache_dir, sample_data):
    key = _save(tmp_cache_dir, sample_data)
    assert get_cached_cost(key) is not None
    n = invalidate_cache(None)
    assert n >= 1
    assert get_cached_cost(key) is None


def test_invalidate_by_bbox(tmp_cache_dir, sample_data):
    # entry 1 - zone originale
    key1 = _save(tmp_cache_dir, sample_data)

    # entry 2 - zone differente (loin)
    data2 = dict(sample_data)
    data2["bbox_l93"] = {"xmin": 800000, "ymin": 6400000, "xmax": 800120, "ymax": 6400100}
    key2 = cost_cache_key(data2["bbox_l93"], data2["resolution"], data2["month"])
    _save(tmp_cache_dir, data2, key=key2)

    # invalider seulement la zone 1
    n = invalidate_cache(sample_data["bbox_l93"])
    assert n == 1
    assert get_cached_cost(key1) is None
    assert get_cached_cost(key2) is not None


def test_ttl_expiration(tmp_cache_dir, sample_data):
    key = _save(tmp_cache_dir, sample_data)
    # forcer le mtime a 91 jours dans le passe
    npz_path = os.path.join(tmp_cache_dir, f"{key}.npz")
    old_time = time.time() - 91 * 86400
    os.utime(npz_path, (old_time, old_time))
    assert get_cached_cost(key) is None


def test_cache_stats_empty(tmp_cache_dir):
    s = cache_stats()
    assert s["entries"] == 0
    assert s["total_size_mb"] == 0


def test_cache_stats_populated(tmp_cache_dir, sample_data):
    _save(tmp_cache_dir, sample_data)
    # deuxieme entry
    data2 = dict(sample_data)
    data2["month"] = 1
    key2 = cost_cache_key(data2["bbox_l93"], data2["resolution"], data2["month"])
    _save(tmp_cache_dir, data2, key=key2)

    s = cache_stats()
    assert s["entries"] == 2
    assert s["total_size_mb"] > 0

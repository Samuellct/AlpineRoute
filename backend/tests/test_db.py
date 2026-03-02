# tests CRUD SQLite (routes + zones)
import json
import pytest

from alpineroute.db.crud import (
    save_route, get_route, list_routes, delete_route,
    save_zone, get_zone, list_zones, update_zone, delete_zone,
)


# ---- helpers ----

def _sample_route(**overrides):
    data = {
        "name": "test route",
        "start_lat": 45.88, "start_lon": 6.89,
        "end_lat": 45.85, "end_lon": 6.90,
        "resolution": 1.0, "month": 7,
        "acclimatized": True,
        "distance_m": 4500.0, "dplus_m": 1200.0, "dminus_m": 500.0,
        "time_tobler_h": 3.2, "glacier_pct": 12.5,
        "cost_total": 9800.0, "computation_time_s": 8.5,
        "geojson": json.dumps({"type": "Feature", "geometry": {"type": "LineString", "coordinates": [[6.89, 45.88, 3200]]}}),
    }
    data.update(overrides)
    return data


def _sample_zone(**overrides):
    data = {
        "name": "crevasse test",
        "zone_type": "crevasse",
        "cost_multiplier": 100.0,
        "geojson": {"type": "Polygon", "coordinates": [[[6.8, 45.9], [6.9, 45.9], [6.9, 45.8], [6.8, 45.8], [6.8, 45.9]]]},
        "active": True,
    }
    data.update(overrides)
    return data


# ---- routes ----

class TestRoutesCRUD:
    def test_save_returns_id(self, tmp_db):
        rid = save_route(tmp_db, _sample_route())
        assert rid > 0

    def test_get_returns_data(self, tmp_db):
        rid = save_route(tmp_db, _sample_route(name="requin-midi"))
        row = get_route(tmp_db, rid)
        assert row is not None
        assert row["name"] == "requin-midi"
        assert abs(row["distance_m"] - 4500.0) < 1e-3

    def test_list_with_limit(self, tmp_db):
        for i in range(5):
            save_route(tmp_db, _sample_route(name=f"route_{i}"))
        rows = list_routes(tmp_db, limit=3)
        assert len(rows) == 3

    def test_list_with_offset(self, tmp_db):
        for i in range(5):
            save_route(tmp_db, _sample_route(name=f"route_{i}"))
        rows = list_routes(tmp_db, limit=10, offset=3)
        assert len(rows) == 2

    def test_delete(self, tmp_db):
        rid = save_route(tmp_db, _sample_route())
        ok = delete_route(tmp_db, rid)
        assert ok
        assert get_route(tmp_db, rid) is None

    def test_delete_nonexistent(self, tmp_db):
        ok = delete_route(tmp_db, 999)
        assert not ok


# ---- zones ----

class TestZonesCRUD:
    def test_save_returns_id(self, tmp_db):
        zid = save_zone(tmp_db, _sample_zone())
        assert zid > 0

    def test_get_parses_geojson(self, tmp_db):
        zid = save_zone(tmp_db, _sample_zone())
        row = get_zone(tmp_db, zid)
        assert row is not None
        assert isinstance(row["geojson"], dict)
        assert row["geojson"]["type"] == "Polygon"

    def test_list_filter_by_type(self, tmp_db):
        save_zone(tmp_db, _sample_zone(zone_type="crevasse"))
        save_zone(tmp_db, _sample_zone(zone_type="serac", name="serac test"))
        rows = list_zones(tmp_db, zone_type="crevasse")
        assert all(z["zone_type"] == "crevasse" for z in rows)

    def test_update_partial(self, tmp_db):
        zid = save_zone(tmp_db, _sample_zone())
        ok = update_zone(tmp_db, zid, {"name": "renamed", "cost_multiplier": 50.0})
        assert ok
        row = get_zone(tmp_db, zid)
        assert row["name"] == "renamed"
        assert abs(row["cost_multiplier"] - 50.0) < 1e-3

    def test_delete(self, tmp_db):
        zid = save_zone(tmp_db, _sample_zone())
        ok = delete_zone(tmp_db, zid)
        assert ok
        assert get_zone(tmp_db, zid) is None

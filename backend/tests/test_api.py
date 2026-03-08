# tests API FastAPI (TestClient)
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from alpineroute.api.main import app
from alpineroute.db.schema import init_db


@pytest.fixture
def client(tmp_db):
    """TestClient avec DB temporaire."""
    with patch("alpineroute.api.main.DB_PATH", tmp_db), \
         patch("alpineroute.api.main.init_db", lambda: init_db(tmp_db)):
        with TestClient(app) as c:
            yield c


class TestHealth:
    def test_health_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


class TestRoutes:
    def test_routes_empty(self, client):
        r = client.get("/routes")
        assert r.status_code == 200
        data = r.json()
        assert data["count"] == 0
        assert data["routes"] == []


class TestGlaciers:
    def test_glaciers_bad_bbox(self, client):
        r = client.get("/glaciers?bbox=invalid")
        assert r.status_code == 400

    def test_glaciers_incomplete_bbox(self, client):
        r = client.get("/glaciers?bbox=1,2,3")
        assert r.status_code == 400


class TestCostSurface:
    def test_cost_surface_no_calc(self, client):
        """Pas de calcul lance -> 404."""
        r = client.get("/cost-surface")
        assert r.status_code == 404


class TestZonesCRUD:
    def _create_zone(self, client, name="test_zone"):
        payload = {
            "name": name,
            "zone_type": "crevasse",
            "cost_multiplier": 100.0,
            "geojson": {
                "type": "Polygon",
                "coordinates": [[[6.8, 45.8], [6.9, 45.8],
                                  [6.9, 45.9], [6.8, 45.9], [6.8, 45.8]]]
            },
            "active": True,
        }
        return client.post("/zones", json=payload)

    def test_create_zone(self, client):
        r = self._create_zone(client)
        assert r.status_code == 201
        assert r.json()["status"] == "created"

    def test_list_zones(self, client):
        self._create_zone(client, "zone_a")
        self._create_zone(client, "zone_b")
        r = client.get("/zones")
        assert r.status_code == 200
        assert r.json()["count"] == 2

    def test_get_zone(self, client):
        cr = self._create_zone(client)
        zone_id = cr.json()["id"]
        r = client.get(f"/zones/{zone_id}")
        assert r.status_code == 200
        assert r.json()["name"] == "test_zone"

    def test_update_zone(self, client):
        cr = self._create_zone(client)
        zone_id = cr.json()["id"]
        r = client.put(f"/zones/{zone_id}", json={"name": "renamed"})
        assert r.status_code == 200
        # verif
        r2 = client.get(f"/zones/{zone_id}")
        assert r2.json()["name"] == "renamed"

    def test_delete_zone(self, client):
        cr = self._create_zone(client)
        zone_id = cr.json()["id"]
        r = client.delete(f"/zones/{zone_id}")
        assert r.status_code == 200
        # verif supprimee
        r2 = client.get(f"/zones/{zone_id}")
        assert r2.status_code == 404

    def test_delete_zone_not_found(self, client):
        r = client.delete("/zones/9999")
        assert r.status_code == 404

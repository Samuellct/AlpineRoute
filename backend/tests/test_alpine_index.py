# tests index.json loading, sync SQLite, reload
import os
import json
import pytest
from unittest.mock import patch

from alpineroute.alpine.index import load_index, sync_to_sqlite, reload_index
from alpineroute.db.schema import init_db, get_connection

MINI_GPX = """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="test">
  <trk><name>test</name><trkseg>
    <trkpt lat="45.8326" lon="6.8652"><ele>3200</ele></trkpt>
    <trkpt lat="45.8340" lon="6.8670"><ele>3250</ele></trkpt>
    <trkpt lat="45.8355" lon="6.8690"><ele>3300</ele></trkpt>
  </trkseg></trk>
</gpx>
"""


@pytest.fixture
def gpx_tree(tmp_path):
    """Arborescence GPX + index.json dans tmp_path."""
    gpx_dir = tmp_path / "gpx"
    gpx_dir.mkdir()
    mb_dir = gpx_dir / "mont-blanc"
    mb_dir.mkdir()
    gpx_file = mb_dir / "cosmiques.gpx"
    gpx_file.write_text(MINI_GPX, encoding="utf-8")

    index = [
        {
            "type": "route",
            "gpx_file": "mont-blanc/cosmiques.gpx",
            "massif": "Mont-Blanc",
            "summit": "Aiguille du Midi",
            "voie": "Cosmiques",
            "grade": "AD",
        },
    ]
    index_path = gpx_dir / "index.json"
    index_path.write_text(json.dumps(index), encoding="utf-8")

    return {
        "gpx_dir": str(gpx_dir),
        "index_path": str(index_path),
        "gpx_file": str(gpx_file),
    }


class TestLoadIndex:
    def test_valid(self, gpx_tree):
        entries = load_index(gpx_tree["index_path"])
        assert len(entries) == 1
        assert entries[0]["type"] == "route"
        assert entries[0]["gpx_file"] == "mont-blanc/cosmiques.gpx"

    def test_missing_file(self, tmp_path):
        result = load_index(str(tmp_path / "nope.json"))
        assert result == []

    def test_invalid_entries_skipped(self, tmp_path):
        idx = tmp_path / "index.json"
        data = [
            {"type": "route", "gpx_file": "ok.gpx"},
            {"nope": True},  # pas de type/gpx_file
            {"type": "badtype", "gpx_file": "x.gpx"},  # type invalide
        ]
        idx.write_text(json.dumps(data), encoding="utf-8")
        entries = load_index(str(idx))
        assert len(entries) == 1


class TestSyncToSqlite:
    def test_one_route(self, gpx_tree, tmp_db):
        entries = load_index(gpx_tree["index_path"])
        with patch("alpineroute.alpine.index.GPX_DIR", gpx_tree["gpx_dir"]):
            result = sync_to_sqlite(entries, db_path=tmp_db)

        assert result["routes"] == 1
        assert result["segments"] == 0
        assert result["skipped"] == 0

        # verif en base
        conn = get_connection(tmp_db)
        conn.row_factory = lambda c, r: {col[0]: r[i] for i, col in enumerate(c.description)}
        row = conn.execute("SELECT * FROM alpine_routes").fetchone()
        conn.close()
        assert row["summit"] == "Aiguille du Midi"
        assert row["grade"] == "AD"
        assert row["distance_m"] > 0
        assert row["dplus_m"] == 100.0

    def test_missing_gpx_skipped(self, tmp_path, tmp_db):
        entries = [{"type": "route", "gpx_file": "ghost.gpx", "massif": "X"}]
        with patch("alpineroute.alpine.index.GPX_DIR", str(tmp_path)):
            result = sync_to_sqlite(entries, db_path=tmp_db)
        assert result["skipped"] == 1
        assert result["routes"] == 0

    def test_resync_replaces(self, gpx_tree, tmp_db):
        entries = load_index(gpx_tree["index_path"])
        with patch("alpineroute.alpine.index.GPX_DIR", gpx_tree["gpx_dir"]):
            sync_to_sqlite(entries, db_path=tmp_db)
            # re-sync: doit remplacer, pas dupliquer
            sync_to_sqlite(entries, db_path=tmp_db)

        conn = get_connection(tmp_db)
        count = conn.execute("SELECT COUNT(*) FROM alpine_routes").fetchone()[0]
        conn.close()
        assert count == 1


class TestReloadIndex:
    def test_returns_summary(self, gpx_tree, tmp_db):
        with patch("alpineroute.alpine.index.GPX_DIR", gpx_tree["gpx_dir"]), \
             patch("alpineroute.alpine.index.GPX_INDEX_PATH", gpx_tree["index_path"]):
            result = reload_index(db_path=tmp_db)
        assert "routes" in result
        assert "segments" in result
        assert "skipped" in result

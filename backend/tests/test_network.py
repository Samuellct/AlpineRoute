# tests client Valhalla (network.py)
import pytest
from unittest.mock import patch, MagicMock

from alpineroute.routing.network import (
    decode_valhalla_shape,
    valhalla_available,
    valhalla_route,
    valhalla_locate,
)
from alpineroute.utils import ValhallaError


# ---- decodeur polyline ----

class TestDecodeValhallaShape:
    def test_known_point(self):
        # Chamonix (45.9237, 6.8694) encode precision 6
        encoded = "gv}qvAoxgbL"
        pts = decode_valhalla_shape(encoded)
        assert len(pts) == 1
        assert abs(pts[0][0] - 45.9237) < 0.0001
        assert abs(pts[0][1] - 6.8694) < 0.0001

    def test_two_points(self):
        # Chamonix -> Montenvers
        encoded = "gv}qvAoxgbL_zM_p}A"
        pts = decode_valhalla_shape(encoded)
        assert len(pts) == 2
        assert abs(pts[1][0] - 45.9313) < 0.0001

    def test_empty_string(self):
        assert decode_valhalla_shape("") == []


# ---- valhalla_available ----

class TestValhallaAvailable:
    @patch("alpineroute.routing.network.httpx.get")
    def test_returns_true_on_200(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200)
        assert valhalla_available() is True

    @patch("alpineroute.routing.network.httpx.get")
    def test_returns_false_on_500(self, mock_get):
        mock_get.return_value = MagicMock(status_code=500)
        assert valhalla_available() is False

    @patch("alpineroute.routing.network.httpx.get")
    def test_returns_false_on_exception(self, mock_get):
        mock_get.side_effect = ConnectionError("refused")
        assert valhalla_available() is False


# ---- valhalla_route ----

class TestValhallaRoute:
    def _mock_response(self, status_code=200, json_data=None, text=""):
        resp = MagicMock()
        resp.status_code = status_code
        resp.json.return_value = json_data
        resp.text = text
        return resp

    def _success_json(self):
        return {
            "trip": {
                "summary": {"length": 4.2, "time": 3600},
                "legs": [{
                    "shape": "gv}qvAoxgbL_zM_p}A",
                }],
            }
        }

    @patch("alpineroute.routing.network.httpx.post")
    def test_success(self, mock_post):
        mock_post.return_value = self._mock_response(200, self._success_json())
        result = valhalla_route((45.92, 6.87), (45.93, 6.88))
        assert result is not None
        assert result["distance_km"] == 4.2
        assert result["duration_s"] == 3600
        assert len(result["coords"]) == 2
        assert "shape_encoded" in result

    @patch("alpineroute.routing.network.httpx.post")
    def test_no_route_returns_none(self, mock_post):
        mock_post.return_value = self._mock_response(400, text="no route")
        result = valhalla_route((45.92, 6.87), (45.93, 6.88))
        assert result is None

    @patch("alpineroute.routing.network.httpx.post")
    def test_server_error_raises(self, mock_post):
        mock_post.return_value = self._mock_response(500, text="internal error")
        with pytest.raises(ValhallaError):
            valhalla_route((45.92, 6.87), (45.93, 6.88))

    @patch("alpineroute.routing.network.httpx.post")
    def test_timeout_raises(self, mock_post):
        import httpx
        mock_post.side_effect = httpx.TimeoutException("timeout")
        with pytest.raises(ValhallaError, match="timeout"):
            valhalla_route((45.92, 6.87), (45.93, 6.88))

    @patch("alpineroute.routing.network.httpx.post")
    def test_max_difficulty_forwarded(self, mock_post):
        mock_post.return_value = self._mock_response(200, self._success_json())
        valhalla_route((45.92, 6.87), (45.93, 6.88), max_difficulty=3)
        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["costing_options"]["pedestrian"]["max_hiking_difficulty"] == 3


# ---- valhalla_locate ----

class TestValhallaLocate:
    @patch("alpineroute.routing.network.httpx.post")
    def test_success(self, mock_post):
        fake_data = [{"edges": [{"id": 123}]}]
        resp = MagicMock(status_code=200)
        resp.json.return_value = fake_data
        mock_post.return_value = resp
        result = valhalla_locate((45.92, 6.87))
        assert result == fake_data

    @patch("alpineroute.routing.network.httpx.post")
    def test_failure_returns_none(self, mock_post):
        mock_post.return_value = MagicMock(status_code=404)
        result = valhalla_locate((45.92, 6.87))
        assert result is None

    @patch("alpineroute.routing.network.httpx.post")
    def test_exception_returns_none(self, mock_post):
        mock_post.side_effect = ConnectionError("refused")
        result = valhalla_locate((45.92, 6.87))
        assert result is None


# ---- integration (skip si Valhalla absent) ----

@pytest.mark.integration
class TestValhallaIntegration:
    @pytest.fixture(autouse=True)
    def _skip_if_no_valhalla(self):
        if not valhalla_available():
            pytest.skip("Valhalla not running")

    def test_chamonix_montenvers(self):
        # Chamonix centre -> Montenvers (train du Montenvers)
        start = (45.9237, 6.8694)
        end = (45.9313, 6.9178)
        result = valhalla_route(start, end)
        assert result is not None
        assert 3 <= result["distance_km"] <= 15
        assert len(result["coords"]) > 5

    def test_locate_chamonix(self):
        result = valhalla_locate((45.9237, 6.8694))
        assert result is not None

"""Tests for FastAPI REST endpoints."""

import pytest
from httpx import ASGITransport, AsyncClient

from cre.api.app import app


@pytest.fixture
def client():
    from starlette.testclient import TestClient
    return TestClient(app)


class TestRoot:
    def test_root(self, client):
        r = client.get("/")
        assert r.status_code == 200
        data = r.json()
        assert data["version"] == "1.0.0"
        assert "disclaimer" in data


class TestEngines:
    def test_list_engines(self, client):
        r = client.get("/engines/")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list)
        assert len(data) == 4
        assert "merlin_1d" in data

    def test_get_engine(self, client):
        r = client.get("/engines/merlin_1d")
        assert r.status_code == 200
        data = r.json()
        assert data["name"] == "Merlin 1D"
        assert data["thrust_sl"] == 845000.0
        assert "disclaimer" in data

    def test_unknown_engine_404(self, client):
        r = client.get("/engines/rs_25")
        assert r.status_code == 404


class TestClusters:
    def test_list_clusters(self, client):
        r = client.get("/clusters/")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 4

    def test_get_cluster(self, client):
        r = client.get("/clusters/super_heavy")
        assert r.status_code == 200
        data = r.json()
        assert data["total_engines"] == 33
        assert len(data["rings"]) == 3
        assert "disclaimer" in data

    def test_unknown_cluster_404(self, client):
        r = client.get("/clusters/new_glenn")
        assert r.status_code == 404


class TestStabilitySweep:
    def test_default_sweep(self, client):
        r = client.post("/stability/sweep", json={})
        assert r.status_code == 200
        data = r.json()
        assert "tau" in data
        assert "n_crit" in data
        assert len(data["environments"]) == 2
        assert "disclaimer" in data

    def test_custom_sweep(self, client):
        r = client.post("/stability/sweep", json={
            "tau_min": 0.5e-3,
            "tau_max": 3e-3,
            "frequencies": [50.0, 100.0],
            "n_tau": 100,
        })
        assert r.status_code == 200
        data = r.json()
        assert len(data["tau"]) == 100
        assert len(data["frequencies"]) == 2


class TestDampingSpectrum:
    def test_default(self, client):
        r = client.post("/damping/spectrum", json={})
        assert r.status_code == 200
        data = r.json()
        assert data["n_engines"] == 20  # SH outer ring default
        assert len(data["environments"]) == 2
        assert "disclaimer" in data


class TestAmplificationSweep:
    def test_default(self, client):
        r = client.post("/amplification/sweep", json={})
        assert r.status_code == 200
        data = r.json()
        assert len(data["n_engines"]) == 40
        assert "disclaimer" in data


class TestPlots:
    def test_stability_plot(self, client):
        r = client.post("/plots/stability", json={"n_tau": 50})
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/png"
        assert len(r.content) > 1000  # Non-trivial PNG

    def test_damping_plot(self, client):
        r = client.post("/plots/damping", json={})
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/png"

    def test_amplification_plot(self, client):
        r = client.post("/plots/amplification", json={})
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/png"

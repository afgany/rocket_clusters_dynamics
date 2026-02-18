"""Tests for pre-loaded engine, cluster, and environment configurations."""

import pytest

from cre import configs
from cre.configs import (
    get_cluster,
    get_engine,
    get_environment,
    list_clusters,
    list_engines,
    list_environments,
)


class TestEngineConfigs:
    def test_all_four_engines_load(self):
        for name in ["merlin_1d", "raptor_2", "raptor_3", "rvac_2"]:
            engine = get_engine(name)
            assert engine.name
            assert engine.chamber_pressure > 0

    def test_merlin_specs(self):
        m = get_engine("merlin_1d")
        assert m.thrust_sl == 845_000.0
        assert m.chamber_pressure == 97e5
        assert m.injector_type == "pintle"
        assert m.cycle == "gas_generator"

    def test_raptor_2_specs(self):
        r = get_engine("raptor_2")
        assert r.thrust_sl == 2_256_000.0
        assert r.chamber_pressure == 300e5
        assert r.cycle == "ffscc"

    def test_rvac_no_sl_thrust(self):
        rv = get_engine("rvac_2")
        assert rv.thrust_sl is None
        assert rv.thrust_vac == 2_530_000.0

    def test_list_engines(self):
        names = list_engines()
        assert len(names) == 4
        assert "merlin_1d" in names

    def test_unknown_engine_raises(self):
        with pytest.raises(KeyError, match="Unknown engine"):
            get_engine("rs_25")

    def test_case_insensitive_lookup(self):
        assert get_engine("Merlin_1D") == get_engine("merlin_1d")

    def test_space_to_underscore_lookup(self):
        assert get_engine("merlin 1d") == get_engine("merlin_1d")

    def test_direct_access(self):
        assert configs.MERLIN_1D.name == "Merlin 1D"
        assert configs.RAPTOR_3.mass == 1525.0


class TestClusterConfigs:
    def test_all_four_clusters_load(self):
        for name in ["falcon_9", "falcon_heavy", "super_heavy", "starship"]:
            cluster = get_cluster(name)
            assert cluster.name
            assert cluster.total_engines > 0

    def test_falcon_9_geometry(self):
        f9 = get_cluster("falcon_9")
        assert f9.total_engines == 9
        assert f9.base_diameter == 3.66
        assert len(f9.rings) == 2
        assert f9.rings[1].n_engines == 8
        assert f9.rings[1].symmetry_group == "D8"

    def test_super_heavy_three_rings(self):
        sh = get_cluster("super_heavy")
        assert sh.total_engines == 33
        assert len(sh.rings) == 3
        ring_engines = [r.n_engines for r in sh.rings]
        assert ring_engines == [3, 10, 20]
        assert sh.rings[2].symmetry_group == "C20"
        assert sh.rings[2].gimbaling is False

    def test_starship_geometry(self):
        ss = get_cluster("starship")
        assert ss.total_engines == 6
        assert len(ss.rings) == 2

    def test_falcon_heavy_total(self):
        fh = get_cluster("falcon_heavy")
        assert fh.total_engines == 27

    def test_list_clusters(self):
        names = list_clusters()
        assert len(names) == 4

    def test_unknown_cluster_raises(self):
        with pytest.raises(KeyError, match="Unknown cluster"):
            get_cluster("new_glenn")

    def test_direct_access(self):
        assert configs.SUPER_HEAVY.base_diameter == 9.0


class TestEnvironmentConfigs:
    def test_earth_and_vacuum_load(self):
        earth = get_environment("earth_sl")
        vacuum = get_environment("lunar_vacuum")
        assert earth.ambient_pressure == 101_325.0
        assert vacuum.ambient_pressure == 0.0

    def test_earth_has_atmospheric_damping(self):
        earth = get_environment("earth_sl")
        assert earth.zeta_atmospheric > 0

    def test_vacuum_no_atmospheric_damping(self):
        vacuum = get_environment("lunar_vacuum")
        assert vacuum.zeta_atmospheric == 0.0

    def test_list_environments(self):
        names = list_environments()
        assert len(names) == 2
        assert "earth_sl" in names
        assert "lunar_vacuum" in names

    def test_default_damping(self):
        dp = configs.DEFAULT_DAMPING
        assert dp.zeta_internal == 0.015
        assert dp.zeta_coupling_max == 0.022

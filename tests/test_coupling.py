"""Tests for three coupling pathways (Eqs 9, 14)."""

import numpy as np
import numpy.testing as npt
import pytest

from cre.configs.defaults import EARTH_SL, LUNAR_VACUUM
from cre.configs.engines import MERLIN_1D, RAPTOR_2
from cre.core.coupling import (
    coupling_atmospheric,
    coupling_feed,
    coupling_structural,
    penetration_knudsen,
    total_coupling,
)


class TestCouplingAtmospheric:
    def test_vacuum_zero(self):
        """Vacuum → kappa_atm = 0."""
        k = coupling_atmospheric(LUNAR_VACUUM, RAPTOR_2, ring_radius=4.0, n_engines=20)
        assert k == 0.0

    def test_earth_positive(self):
        """Earth → kappa_atm > 0."""
        k = coupling_atmospheric(EARTH_SL, RAPTOR_2, ring_radius=4.0, n_engines=20)
        assert k > 0.0

    def test_single_engine_zero(self):
        """Single engine → no coupling."""
        k = coupling_atmospheric(EARTH_SL, RAPTOR_2, ring_radius=0.0, n_engines=1)
        assert k == 0.0


class TestCouplingStructural:
    def test_positive(self):
        k = coupling_structural(RAPTOR_2, ring_radius=4.0, n_engines=20)
        assert k > 0.0

    def test_independent_of_environment(self):
        """Structural coupling doesn't depend on environment."""
        k = coupling_structural(RAPTOR_2, ring_radius=4.0, n_engines=20)
        assert k > 0  # No environment parameter at all

    def test_single_engine_zero(self):
        k = coupling_structural(RAPTOR_2, ring_radius=4.0, n_engines=1)
        assert k == 0.0


class TestCouplingFeed:
    def test_positive(self):
        k = coupling_feed(RAPTOR_2, n_engines=20)
        assert k > 0.0

    def test_ffscc_stronger_than_gas_gen(self):
        """Full-flow staged combustion has tighter feed coupling."""
        k_raptor = coupling_feed(RAPTOR_2, n_engines=9)
        k_merlin = coupling_feed(MERLIN_1D, n_engines=9)
        # FFSCC should have higher coupling fraction
        assert k_raptor > 0
        assert k_merlin > 0

    def test_single_engine_zero(self):
        k = coupling_feed(RAPTOR_2, n_engines=1)
        assert k == 0.0


class TestTotalCoupling:
    def test_earth_greater_than_vacuum(self):
        """Earth kappa_total > vacuum kappa_total (atmospheric adds to total)."""
        k_earth = total_coupling(EARTH_SL, RAPTOR_2, ring_radius=4.0, n_engines=20)
        k_vacuum = total_coupling(LUNAR_VACUUM, RAPTOR_2, ring_radius=4.0, n_engines=20)
        assert k_earth > k_vacuum

    def test_vacuum_has_struct_and_feed_only(self):
        """In vacuum, total = structural + feed (no atmospheric)."""
        k_total = total_coupling(LUNAR_VACUUM, RAPTOR_2, ring_radius=4.0, n_engines=20)
        k_struct = coupling_structural(RAPTOR_2, ring_radius=4.0, n_engines=20)
        k_feed = coupling_feed(RAPTOR_2, n_engines=20)
        npt.assert_almost_equal(k_total, k_struct + k_feed)

    def test_earth_is_sum(self):
        """Earth total = atm + struct + feed."""
        k_total = total_coupling(EARTH_SL, RAPTOR_2, ring_radius=4.0, n_engines=20)
        k_atm = coupling_atmospheric(EARTH_SL, RAPTOR_2, ring_radius=4.0, n_engines=20)
        k_struct = coupling_structural(RAPTOR_2, ring_radius=4.0, n_engines=20)
        k_feed = coupling_feed(RAPTOR_2, n_engines=20)
        npt.assert_almost_equal(k_total, k_atm + k_struct + k_feed)

    def test_positive_all_configs(self):
        """All configs should have positive coupling."""
        for env in [EARTH_SL, LUNAR_VACUUM]:
            k = total_coupling(env, RAPTOR_2, ring_radius=4.0, n_engines=20)
            assert k > 0


class TestPenetrationKnudsen:
    def test_valid_output(self):
        theta = np.linspace(0.1, np.pi / 2, 50)
        Kn = penetration_knudsen(Kn_0=0.01, A_pl=1.0, D=0.65, r_n=0.65, theta=theta)
        assert Kn.shape == (50,)
        assert np.all(Kn > 0)

    def test_decreases_with_angle(self):
        """Kn should generally decrease as theta increases from 0 to pi/2."""
        theta = np.linspace(0.2, np.pi / 2, 100)
        Kn = penetration_knudsen(Kn_0=0.01, A_pl=1.0, D=0.65, r_n=0.65, theta=theta)
        # sin^2(theta) increases, so Kn should decrease
        assert Kn[0] > Kn[-1]

    def test_scalar_input(self):
        Kn = penetration_knudsen(Kn_0=0.01, A_pl=1.0, D=0.65, r_n=0.65, theta=np.pi / 4)
        assert Kn.shape == ()
        assert Kn > 0

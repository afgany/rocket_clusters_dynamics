"""Tests for damping spectrum analysis (Eqs 12-13)."""

import numpy as np
import numpy.testing as npt

from cre.configs.defaults import DEFAULT_DAMPING, EARTH_SL, LUNAR_VACUUM
from cre.core.damping import (
    breathing_mode_damping,
    critical_damping_threshold,
    damping_spectrum,
    damping_spectrum_multi_env,
    is_mode_stable,
)


class TestDampingSpectrum:
    def test_shape(self):
        zeta = damping_spectrum(33, DEFAULT_DAMPING, EARTH_SL)
        assert zeta.shape == (33,)

    def test_breathing_mode_lowest_vacuum(self):
        """Breathing mode (n=0) has lowest damping in vacuum."""
        zeta = damping_spectrum(33, DEFAULT_DAMPING, LUNAR_VACUUM)
        assert zeta[0] == np.min(zeta)

    def test_breathing_mode_no_coupling(self):
        """Breathing mode gets no coupling damping."""
        zeta = damping_spectrum(33, DEFAULT_DAMPING, LUNAR_VACUUM)
        expected = DEFAULT_DAMPING.zeta_internal + DEFAULT_DAMPING.zeta_nozzle + DEFAULT_DAMPING.zeta_feed
        npt.assert_almost_equal(zeta[0], expected)

    def test_earth_higher_than_vacuum(self):
        """Earth damping > vacuum damping for all modes."""
        zeta_earth = damping_spectrum(33, DEFAULT_DAMPING, EARTH_SL)
        zeta_vac = damping_spectrum(33, DEFAULT_DAMPING, LUNAR_VACUUM)
        assert np.all(zeta_earth > zeta_vac)

    def test_alternating_mode_maximum_coupling(self):
        """Mode N/2 (for even N) gets maximum coupling damping."""
        N = 20
        zeta = damping_spectrum(N, DEFAULT_DAMPING, LUNAR_VACUUM)
        # 1 - cos(2*pi*N/2/N) = 1 - cos(pi) = 2 → max coupling
        assert zeta[N // 2] == np.max(zeta)

    def test_symmetric_modes(self):
        """Mode n and N-n have same damping (degeneracy)."""
        N = 20
        zeta = damping_spectrum(N, DEFAULT_DAMPING, EARTH_SL)
        for n in range(1, N // 2):
            npt.assert_almost_equal(zeta[n], zeta[N - n])

    def test_all_positive(self):
        for N in [3, 9, 10, 20, 33]:
            zeta = damping_spectrum(N, DEFAULT_DAMPING, LUNAR_VACUUM)
            assert np.all(zeta > 0)


class TestDampingSpectrumMultiEnv:
    def test_output_shape(self):
        result = damping_spectrum_multi_env(33, DEFAULT_DAMPING, [EARTH_SL, LUNAR_VACUUM])
        assert result.zeta_total.shape == (2, 33)
        assert result.n_engines == 33
        assert result.environments == ["earth_sl", "lunar_vacuum"]

    def test_validated_false(self):
        result = damping_spectrum_multi_env(33, DEFAULT_DAMPING, [EARTH_SL])
        assert result.validated is False


class TestBreathingModeDamping:
    def test_vacuum(self):
        zeta = breathing_mode_damping(DEFAULT_DAMPING, LUNAR_VACUUM)
        expected = 0.015 + 0.020 + 0.005  # internal + nozzle + feed
        npt.assert_almost_equal(zeta, expected)

    def test_earth(self):
        zeta = breathing_mode_damping(DEFAULT_DAMPING, EARTH_SL)
        expected = 0.015 + 0.020 + 0.005 + 0.028  # + atmospheric
        npt.assert_almost_equal(zeta, expected)

    def test_matches_spectrum_n0(self):
        """Breathing mode function should match spectrum[0]."""
        zeta_direct = breathing_mode_damping(DEFAULT_DAMPING, EARTH_SL)
        zeta_spectrum = damping_spectrum(33, DEFAULT_DAMPING, EARTH_SL)
        npt.assert_almost_equal(zeta_direct, zeta_spectrum[0])


class TestCriticalDampingThreshold:
    def test_positive(self):
        zeta_c = critical_damping_threshold(
            n_crocco=1.0, tau=1e-3, omega_n=2 * np.pi * 2000
        )
        assert zeta_c > 0

    def test_higher_n_higher_threshold(self):
        z1 = critical_damping_threshold(n_crocco=0.5, tau=1e-3, omega_n=2 * np.pi * 2000)
        z2 = critical_damping_threshold(n_crocco=2.0, tau=1e-3, omega_n=2 * np.pi * 2000)
        assert z2 > z1


class TestIsModeStable:
    def test_breathing_mode_with_low_n(self):
        """Low Crocco n → breathing mode should be stable."""
        # Use very small n_crocco so threshold is tiny
        stable = is_mode_stable(
            mode_n=0, N=33, params=DEFAULT_DAMPING, environment=EARTH_SL,
            n_crocco=0.001, tau=1e-3, omega_n=2 * np.pi * 2000
        )
        assert stable is True

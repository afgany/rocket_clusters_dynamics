"""Tests for coupled oscillator normal mode eigenfrequencies (Eqs 5-6)."""

import numpy as np
import numpy.testing as npt

from cre.core.coupled_modes import (
    mode_frequency_ratios,
    normal_mode_frequencies,
    normal_mode_frequencies_squared,
)


class TestNormalModeFrequencies:
    def test_N1_single_frequency(self):
        """N=1 → single frequency = sqrt(k0/m)."""
        k0, m, kappa = 1e6, 100.0, 500.0
        omega = normal_mode_frequencies(k0, m, kappa, N=1)
        assert omega.shape == (1,)
        npt.assert_almost_equal(omega[0], np.sqrt(k0 / m))

    def test_kappa_zero_all_equal(self):
        """kappa=0 → no coupling → all frequencies equal."""
        k0, m = 1e6, 100.0
        omega = normal_mode_frequencies(k0, m, kappa=0.0, N=20)
        expected = np.sqrt(k0 / m)
        npt.assert_array_almost_equal(omega, expected)

    def test_breathing_mode_unshifted(self):
        """Mode n=0 (breathing) has omega_0 = sqrt(k0/m) regardless of kappa."""
        k0, m, kappa = 1e6, 100.0, 5000.0
        omega = normal_mode_frequencies(k0, m, kappa, N=33)
        npt.assert_almost_equal(omega[0], np.sqrt(k0 / m))

    def test_degeneracy_n_and_N_minus_n(self):
        """Modes n and N-n are degenerate (same frequency)."""
        k0, m, kappa, N = 1e6, 100.0, 5000.0, 20
        omega = normal_mode_frequencies(k0, m, kappa, N)
        for n in range(1, N // 2):
            npt.assert_almost_equal(omega[n], omega[N - n])

    def test_alternating_mode_maximum_shift(self):
        """Mode n=N/2 (for even N) has maximum frequency shift."""
        k0, m, kappa, N = 1e6, 100.0, 5000.0, 20
        omega = normal_mode_frequencies(k0, m, kappa, N)
        # n=N/2 gives cos(pi) = -1, so 1-cos = 2, maximum shift
        assert omega[N // 2] == np.max(omega)

    def test_frequencies_monotonic_first_half(self):
        """Frequencies increase from n=0 to n=N/2 for even N."""
        k0, m, kappa, N = 1e6, 100.0, 5000.0, 20
        omega = normal_mode_frequencies(k0, m, kappa, N)
        first_half = omega[:N // 2 + 1]
        assert np.all(np.diff(first_half) >= 0)

    def test_correct_shape(self):
        """Output shape is (N,)."""
        for N in [3, 9, 10, 20, 33]:
            omega = normal_mode_frequencies(1e6, 100.0, 1000.0, N)
            assert omega.shape == (N,)

    def test_super_heavy_per_ring(self):
        """Super Heavy: 3 separate analyses for 3, 10, 20 engines."""
        k0, m, kappa = 1e6, 100.0, 5000.0
        results = {}
        for N in [3, 10, 20]:
            omega = normal_mode_frequencies(k0, m, kappa, N)
            results[N] = omega
            assert omega.shape == (N,)
        # All breathing modes equal (same k0/m)
        npt.assert_almost_equal(results[3][0], results[10][0])
        npt.assert_almost_equal(results[10][0], results[20][0])


class TestNormalModeFrequenciesSquared:
    def test_omega_sq_breathing(self):
        """omega_0^2 = k0/m."""
        omega_sq = normal_mode_frequencies_squared(1e6, 100.0, 5000.0, N=10)
        npt.assert_almost_equal(omega_sq[0], 1e6 / 100.0)

    def test_omega_sq_alternating(self):
        """omega_{N/2}^2 = k0/m + 4*kappa/m."""
        k0, m, kappa, N = 1e6, 100.0, 5000.0, 10
        omega_sq = normal_mode_frequencies_squared(k0, m, kappa, N)
        expected = k0 / m + 4.0 * kappa / m
        npt.assert_almost_equal(omega_sq[N // 2], expected)


class TestModeFrequencyRatios:
    def test_breathing_ratio_is_one(self):
        """omega_0 / omega_0 = 1."""
        ratios = mode_frequency_ratios(kappa=5000.0, k0=1e6, N=20)
        npt.assert_almost_equal(ratios[0], 1.0)

    def test_kappa_zero_all_one(self):
        ratios = mode_frequency_ratios(kappa=0.0, k0=1e6, N=20)
        npt.assert_array_almost_equal(ratios, 1.0)

    def test_ratios_geq_one(self):
        ratios = mode_frequency_ratios(kappa=5000.0, k0=1e6, N=20)
        assert np.all(ratios >= 1.0 - 1e-10)

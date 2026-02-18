"""Tests for stability boundary computation (Eqs 10-11)."""

import numpy as np
import numpy.testing as npt

from cre.core.stability import (
    is_stable,
    n_critical,
    stability_boundary_sweep,
    stability_margin,
    zeta_minimum,
)


class TestNCritical:
    def test_vectorized(self):
        tau = np.linspace(0.1e-3, 5e-3, 100)
        n_c = n_critical(tau, alpha_total=0.1, omega=2 * np.pi * 50)
        assert n_c.shape == (100,)

    def test_positive(self):
        tau = np.linspace(0.5e-3, 4e-3, 100)
        n_c = n_critical(tau, alpha_total=0.1, omega=2 * np.pi * 135)
        assert np.all(n_c >= 0)

    def test_higher_alpha_higher_n_crit(self):
        """More damping → higher stability boundary."""
        tau = np.array([1e-3])
        n_c_low = n_critical(tau, alpha_total=0.05, omega=2 * np.pi * 135)
        n_c_high = n_critical(tau, alpha_total=0.15, omega=2 * np.pi * 135)
        assert n_c_high[0] > n_c_low[0]

    def test_clipped_at_singularities(self):
        """At sin(omega*tau)=0, n_crit is clipped, not infinite."""
        omega = 2 * np.pi * 50
        tau_singular = np.pi / omega  # sin(omega*tau) = sin(pi) = 0
        n_c = n_critical(np.array([tau_singular]), alpha_total=0.1, omega=omega)
        assert n_c[0] <= 20.0  # Clipped to max


class TestStabilityBoundarySweep:
    def test_output_shape(self):
        result = stability_boundary_sweep(
            tau_range=(0.1e-3, 5e-3),
            frequencies=[50.0, 135.0],
            alpha_earth=0.12,
            alpha_vacuum=0.06,
            n_tau=200,
        )
        assert result.tau.shape == (200,)
        assert result.n_crit.shape == (2, 2, 200)  # 2 envs, 2 freqs, 200 tau
        assert result.environments == ["earth_sl", "lunar_vacuum"]

    def test_vacuum_below_earth(self):
        """Vacuum n_crit < Earth n_crit (less damping → lower boundary)."""
        result = stability_boundary_sweep(
            tau_range=(0.5e-3, 4e-3),
            frequencies=[135.0],
            alpha_earth=0.12,
            alpha_vacuum=0.06,
        )
        # result.n_crit[0] = earth, [1] = vacuum
        # Vacuum should be strictly below earth at all tau
        earth = result.n_crit[0, 0, :]
        vacuum = result.n_crit[1, 0, :]
        # At non-singular points, vacuum < earth
        mask = (earth < 19.0) & (vacuum < 19.0)  # Exclude clipped points
        assert np.all(vacuum[mask] <= earth[mask])

    def test_validated_false(self):
        result = stability_boundary_sweep(
            tau_range=(0.1e-3, 5e-3),
            frequencies=[50.0],
            alpha_earth=0.1,
            alpha_vacuum=0.05,
        )
        assert result.validated is False

    def test_multiple_frequencies(self):
        result = stability_boundary_sweep(
            tau_range=(0.1e-3, 5e-3),
            frequencies=[50.0, 135.0, 56.0],
            alpha_earth=0.12,
            alpha_vacuum=0.06,
        )
        assert result.n_crit.shape[1] == 3


class TestIsStable:
    def test_stable_below_boundary(self):
        # Use high alpha and low omega to get n_crit >> 0.5
        assert is_stable(n=0.5, tau=1e-3, alpha_total=500.0, omega=2 * np.pi * 50)

    def test_unstable_above_boundary(self):
        assert not is_stable(n=100.0, tau=1e-3, alpha_total=0.001, omega=2 * np.pi * 135)


class TestStabilityMargin:
    def test_positive_when_stable(self):
        margin = stability_margin(n=0.5, tau=1e-3, alpha_total=500.0, omega=2 * np.pi * 50)
        assert margin > 0

    def test_negative_when_unstable(self):
        margin = stability_margin(n=100.0, tau=1e-3, alpha_total=0.001, omega=2 * np.pi * 135)
        assert margin < 0


class TestZetaMinimum:
    def test_positive(self):
        z = zeta_minimum(n=1.0, omega=2 * np.pi * 50, tau=1e-3,
                         omega_n=2 * np.pi * 2000)
        assert z > 0

    def test_zero_at_sin_zero(self):
        """At omega*tau = pi (or 0), sin=0 → zeta_min=0."""
        omega = 2 * np.pi * 50
        tau = np.pi / omega  # sin(omega*tau) = sin(pi) = 0
        z = zeta_minimum(n=1.0, omega=omega, tau=tau, omega_n=2 * np.pi * 2000)
        npt.assert_almost_equal(z, 0.0)

    def test_higher_n_higher_zeta(self):
        """Higher interaction index → more damping required."""
        z1 = zeta_minimum(n=0.5, omega=1000, tau=1e-3, omega_n=12000)
        z2 = zeta_minimum(n=2.0, omega=1000, tau=1e-3, omega_n=12000)
        assert z2 > z1

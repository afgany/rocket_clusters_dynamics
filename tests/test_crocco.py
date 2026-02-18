"""Tests for Crocco n-tau combustion response (Eq 2)."""

import numpy as np
import numpy.testing as npt

from cre.core.crocco import crocco_magnitude, crocco_phase, crocco_response


class TestCroccoResponse:
    def test_zero_at_tau_zero(self):
        """R(omega) = 0 when tau = 0 (no time lag)."""
        omega = np.linspace(100, 10_000, 50)
        R = crocco_response(omega, n=1.0, tau=0.0)
        npt.assert_array_almost_equal(R, 0.0)

    def test_magnitude_2n_at_omega_tau_pi(self):
        """|R| = 2n when omega*tau = pi (maximum response)."""
        n = 1.5
        tau = 1e-3  # 1 ms
        omega_max = np.pi / tau  # omega*tau = pi
        mag = crocco_magnitude(omega_max, n=n, tau=tau)
        npt.assert_almost_equal(mag, 2.0 * n, decimal=10)

    def test_magnitude_zero_at_omega_tau_2pi(self):
        """|R| = 0 when omega*tau = 2*pi (response crosses zero)."""
        tau = 1e-3
        omega_zero = 2.0 * np.pi / tau
        mag = crocco_magnitude(omega_zero, n=2.0, tau=tau)
        npt.assert_almost_equal(mag, 0.0, decimal=10)

    def test_vectorized_shape(self):
        """Output shape matches input omega array."""
        omega = np.linspace(100, 5000, 100)
        R = crocco_response(omega, n=1.0, tau=1e-3)
        assert R.shape == (100,)

    def test_scalar_input(self):
        """Scalar omega returns scalar-like array."""
        R = crocco_response(1000.0, n=1.0, tau=1e-3)
        assert R.shape == ()

    def test_response_is_complex(self):
        R = crocco_response(1000.0, n=1.0, tau=1e-3)
        assert np.issubdtype(R.dtype, np.complexfloating)

    def test_known_value(self):
        """Hand-calculated: R(1000, n=1, tau=1e-3) = 1*(1 - exp(-1j))."""
        R = crocco_response(1000.0, n=1.0, tau=1e-3)
        expected = 1.0 * (1.0 - np.exp(-1j * 1000.0 * 1e-3))
        npt.assert_almost_equal(R, expected)

    def test_n_scaling(self):
        """R scales linearly with n."""
        omega = 2000.0
        tau = 1e-3
        R1 = crocco_response(omega, n=1.0, tau=tau)
        R3 = crocco_response(omega, n=3.0, tau=tau)
        npt.assert_almost_equal(R3, 3.0 * R1)


class TestCroccoMagnitude:
    def test_analytical_formula(self):
        """|R| = 2n|sin(omega*tau/2)| — check against direct formula."""
        omega = np.linspace(500, 8000, 200)
        n, tau = 1.2, 0.8e-3
        mag = crocco_magnitude(omega, n, tau)
        expected = 2.0 * n * np.abs(np.sin(omega * tau / 2.0))
        npt.assert_array_almost_equal(mag, expected)

    def test_always_nonnegative(self):
        omega = np.linspace(0, 20_000, 500)
        mag = crocco_magnitude(omega, n=2.0, tau=1e-3)
        assert np.all(mag >= 0)


class TestCroccoPhase:
    def test_phase_at_omega_tau_pi(self):
        """At omega*tau = pi, R = n*(1 - exp(-i*pi)) = n*(1+1) = 2n (real positive).
        Phase should be 0."""
        tau = 1e-3
        omega = np.pi / tau
        phase = crocco_phase(omega, n=1.0, tau=tau)
        npt.assert_almost_equal(phase, 0.0, decimal=10)

    def test_phase_vectorized(self):
        omega = np.linspace(500, 5000, 50)
        phase = crocco_phase(omega, n=1.0, tau=1e-3)
        assert phase.shape == (50,)

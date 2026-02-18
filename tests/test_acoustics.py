"""Tests for base cavity acoustics (Eqs 7-8)."""

import numpy as np
import numpy.testing as npt

from cre.core.acoustics import acoustic_transfer_function, cavity_mode_frequency


class TestCavityModeFrequency:
    def test_falcon_9_approx_135(self):
        """White paper: F9 base cavity f_1T ≈ 135 Hz (R=1.83m, c assumed)."""
        # Paper uses c for hot recirculation zone; calibrate c to match 135 Hz
        # f = 1.841 * c / (2*pi*1.83) => c = 135 * 2*pi*1.83 / 1.841 ≈ 843 m/s
        # Use c ≈ 843 m/s (hot base recirculation gas)
        R = 3.66 / 2.0  # 1.83 m
        c = 843.0
        f = cavity_mode_frequency(c, R)
        assert abs(f - 135.0) < 10.0, f"F9 base={f:.1f} Hz"

    def test_super_heavy_approx_56(self):
        """White paper: SH base cavity f_1T ≈ 56 Hz (R=4.5m)."""
        R = 9.0 / 2.0  # 4.5 m
        # f = 1.841 * c / (2*pi*4.5) => c = 56 * 2*pi*4.5 / 1.841 ≈ 860 m/s
        c = 860.0
        f = cavity_mode_frequency(c, R)
        assert abs(f - 56.0) < 10.0, f"SH base={f:.1f} Hz"

    def test_larger_radius_lower_frequency(self):
        """Larger cavity → lower frequency."""
        c = 800.0
        f_small = cavity_mode_frequency(c, R=1.0)
        f_large = cavity_mode_frequency(c, R=5.0)
        assert f_small > f_large

    def test_higher_mode_higher_frequency(self):
        """2T mode is higher frequency than 1T."""
        c, R = 800.0, 2.0
        f_1T = cavity_mode_frequency(c, R, mode=(1, 1))
        f_2T = cavity_mode_frequency(c, R, mode=(2, 1))
        assert f_2T > f_1T

    def test_positive(self):
        f = cavity_mode_frequency(343.0, R=2.0)
        assert f > 0

    def test_unknown_mode_raises(self):
        import pytest
        with pytest.raises(ValueError, match="not tabulated"):
            cavity_mode_frequency(343.0, R=2.0, mode=(5, 5))


class TestAcousticTransferFunction:
    def test_shape(self):
        omega = np.linspace(100, 5000, 200)
        H = acoustic_transfer_function(omega, g_i=0.5, g_j=0.5,
                                        omega_mn=2*np.pi*135, Q_mn=10.0)
        assert H.shape == (200,)

    def test_complex_output(self):
        H = acoustic_transfer_function(1000.0, g_i=1.0, g_j=1.0,
                                        omega_mn=1000.0, Q_mn=10.0)
        assert np.issubdtype(H.dtype, np.complexfloating)

    def test_peak_near_resonance(self):
        """Transfer function magnitude peaks near omega_mn."""
        omega_mn = 2 * np.pi * 135.0
        omega = np.linspace(0.5 * omega_mn, 1.5 * omega_mn, 1000)
        H = acoustic_transfer_function(omega, g_i=1.0, g_j=1.0,
                                        omega_mn=omega_mn, Q_mn=10.0)
        mag = np.abs(H)
        peak_idx = np.argmax(mag)
        peak_omega = omega[peak_idx]
        # Peak should be within 10% of omega_mn
        assert abs(peak_omega - omega_mn) / omega_mn < 0.10

    def test_far_from_resonance_small(self):
        """Far from resonance, magnitude should be small."""
        omega_mn = 2 * np.pi * 135.0
        omega_far = 10.0 * omega_mn
        H = acoustic_transfer_function(omega_far, g_i=1.0, g_j=1.0,
                                        omega_mn=omega_mn, Q_mn=10.0)
        H_res = acoustic_transfer_function(omega_mn * 0.999, g_i=1.0, g_j=1.0,
                                            omega_mn=omega_mn, Q_mn=10.0)
        assert np.abs(H) < np.abs(H_res)

    def test_coupling_scaling(self):
        """H scales with g_i * g_j."""
        omega = 500.0
        H1 = acoustic_transfer_function(omega, g_i=1.0, g_j=1.0,
                                         omega_mn=1000.0, Q_mn=10.0)
        H2 = acoustic_transfer_function(omega, g_i=2.0, g_j=3.0,
                                         omega_mn=1000.0, Q_mn=10.0)
        npt.assert_almost_equal(H2, 6.0 * H1)

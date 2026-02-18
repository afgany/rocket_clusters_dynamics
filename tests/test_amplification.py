"""Tests for amplification factor computation."""

import numpy as np
import numpy.testing as npt

from cre.configs.defaults import DEFAULT_DAMPING
from cre.core.amplification import (
    amplification_ratio,
    amplification_sweep,
    coherent_amplification,
    damping_margin_ratio,
    incoherent_amplification,
)


class TestCoherentAmplification:
    def test_exact_values(self):
        """Table 3: coherent(N) = N."""
        npt.assert_almost_equal(coherent_amplification(6), 6.0)
        npt.assert_almost_equal(coherent_amplification(9), 9.0)
        npt.assert_almost_equal(coherent_amplification(27), 27.0)
        npt.assert_almost_equal(coherent_amplification(33), 33.0)

    def test_vectorized(self):
        N = np.array([6, 9, 27, 33])
        result = coherent_amplification(N)
        npt.assert_array_almost_equal(result, N.astype(float))


class TestIncoherentAmplification:
    def test_exact_values(self):
        """Table 3: incoherent(N) = sqrt(N)."""
        npt.assert_almost_equal(incoherent_amplification(6), np.sqrt(6))   # ~2.449
        npt.assert_almost_equal(incoherent_amplification(9), 3.0)          # sqrt(9)=3
        npt.assert_almost_equal(incoherent_amplification(33), np.sqrt(33)) # ~5.745

    def test_table_3_values(self):
        """Cross-check against Table 3 rounded values."""
        assert abs(incoherent_amplification(6) - 2.4) < 0.1
        assert abs(incoherent_amplification(9) - 3.0) < 0.1
        assert abs(incoherent_amplification(27) - 5.2) < 0.1
        assert abs(incoherent_amplification(33) - 5.7) < 0.1


class TestAmplificationRatio:
    def test_is_sqrt_N(self):
        N = np.array([6, 9, 27, 33])
        npt.assert_array_almost_equal(amplification_ratio(N), np.sqrt(N))

    def test_table_3_ratios(self):
        """Table 3: ratio = sqrt(N)."""
        assert abs(amplification_ratio(6) - 2.4) < 0.1
        assert abs(amplification_ratio(9) - 3.0) < 0.1
        assert abs(amplification_ratio(27) - 5.2) < 0.1
        assert abs(amplification_ratio(33) - 5.7) < 0.1


class TestDampingMarginRatio:
    def test_less_than_100(self):
        """Vacuum damping < Earth damping → ratio < 100%."""
        N = np.array([9, 27, 33])
        ratio = damping_margin_ratio(N, DEFAULT_DAMPING)
        assert np.all(ratio < 100.0)
        assert np.all(ratio > 0.0)

    def test_decreases_with_N(self):
        """Larger N → slightly lower margin (feed complexity)."""
        N = np.arange(1, 40)
        ratio = damping_margin_ratio(N, DEFAULT_DAMPING)
        # Should be generally decreasing (not necessarily strictly monotonic)
        assert ratio[0] > ratio[-1]


class TestAmplificationSweep:
    def test_output_shape(self):
        result = amplification_sweep(N_range=(1, 40), params=DEFAULT_DAMPING)
        assert result.n_engines.shape == (40,)
        assert result.coherent.shape == (40,)
        assert result.incoherent.shape == (40,)
        assert result.ratio.shape == (40,)
        assert result.damping_margin_ratio is not None
        assert result.damping_margin_ratio.shape == (40,)

    def test_validated_false(self):
        result = amplification_sweep(N_range=(1, 10), params=DEFAULT_DAMPING)
        assert result.validated is False

    def test_coherent_always_greater(self):
        """Coherent amplification > incoherent for N > 1."""
        result = amplification_sweep(N_range=(2, 40), params=DEFAULT_DAMPING)
        assert np.all(result.coherent > result.incoherent)

    def test_specific_vehicles(self):
        """Pre-computed for Falcon 9, FH, SH outer ring, Starship."""
        result = amplification_sweep(N_range=(1, 40), params=DEFAULT_DAMPING)
        # Find indices
        N = result.n_engines
        idx_6 = np.where(N == 6)[0][0]
        idx_9 = np.where(N == 9)[0][0]
        idx_27 = np.where(N == 27)[0][0]
        idx_33 = np.where(N == 33)[0][0]

        npt.assert_almost_equal(result.coherent[idx_6], 6.0)
        npt.assert_almost_equal(result.coherent[idx_9], 9.0)
        npt.assert_almost_equal(result.coherent[idx_33], 33.0)
        npt.assert_almost_equal(result.incoherent[idx_33], np.sqrt(33))

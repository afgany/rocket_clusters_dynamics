"""Tests for result container types."""

import numpy as np

from cre.models.results import (
    DISCLAIMER,
    AmplificationResult,
    DampingSpectrumResult,
    StabilitySweepResult,
)


class TestStabilitySweepResult:
    def test_instantiation(self):
        result = StabilitySweepResult(
            tau=np.linspace(0.1e-3, 5e-3, 50),
            n_crit=np.ones((2, 3, 50)),
            frequencies=np.array([50.0, 135.0, 56.0]),
            environments=["earth_sl", "lunar_vacuum"],
        )
        assert result.tau.shape == (50,)
        assert result.n_crit.shape == (2, 3, 50)
        assert len(result.environments) == 2

    def test_validated_false(self):
        result = StabilitySweepResult(
            tau=np.array([1.0]),
            n_crit=np.array([2.0]),
            frequencies=np.array([50.0]),
            environments=["earth_sl"],
        )
        assert result.validated is False
        assert result.disclaimer == DISCLAIMER

    def test_field_access(self):
        tau = np.array([0.001, 0.002])
        result = StabilitySweepResult(
            tau=tau,
            n_crit=np.array([1.5, 2.5]),
            frequencies=np.array([50.0]),
            environments=["earth_sl"],
        )
        assert np.array_equal(result.tau, tau)


class TestDampingSpectrumResult:
    def test_instantiation(self):
        N = 33
        result = DampingSpectrumResult(
            mode_indices=np.arange(N),
            zeta_total=np.random.rand(2, N),
            n_engines=N,
            environments=["earth_sl", "lunar_vacuum"],
        )
        assert result.mode_indices.shape == (33,)
        assert result.n_engines == 33
        assert result.validated is False

    def test_field_access(self):
        result = DampingSpectrumResult(
            mode_indices=np.array([0, 1, 2]),
            zeta_total=np.array([0.035, 0.057, 0.070]),
            n_engines=3,
            environments=["earth_sl"],
        )
        assert result.zeta_total[0] == 0.035


class TestAmplificationResult:
    def test_instantiation(self):
        N = np.arange(1, 41)
        result = AmplificationResult(
            n_engines=N,
            coherent=N.astype(float),
            incoherent=np.sqrt(N),
            ratio=np.sqrt(N),
            damping_margin_ratio=None,
        )
        assert result.n_engines.shape == (40,)
        assert result.validated is False
        assert result.disclaimer == DISCLAIMER

    def test_with_damping_margin(self):
        N = np.array([9, 27, 33])
        result = AmplificationResult(
            n_engines=N,
            coherent=N.astype(float),
            incoherent=np.sqrt(N),
            ratio=np.sqrt(N),
            damping_margin_ratio=np.array([0.55, 0.50, 0.48]),
        )
        assert result.damping_margin_ratio is not None
        assert result.damping_margin_ratio[0] == 0.55

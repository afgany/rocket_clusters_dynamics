"""Result containers for CRE computations.

All sweep outputs use NumPy ndarrays. Each result carries a `validated` flag
(always False — analytical model, not experimentally validated).
"""

from typing import NamedTuple

import numpy as np

DISCLAIMER = "Analytical model — not experimentally validated."


class StabilitySweepResult(NamedTuple):
    """Result of a stability boundary sweep in (n, tau) space."""

    tau: np.ndarray  # Sensitive time lag values [s]
    n_crit: np.ndarray  # Critical interaction index per (freq, env, tau)
    frequencies: np.ndarray  # Frequencies swept [Hz]
    environments: list[str]  # Environment names
    validated: bool = False
    disclaimer: str = DISCLAIMER


class DampingSpectrumResult(NamedTuple):
    """Per-mode damping ratios for an N-engine cluster."""

    mode_indices: np.ndarray  # n = 0, 1, ..., N-1
    zeta_total: np.ndarray  # Total damping per mode (may have env dimension)
    n_engines: int
    environments: list[str]
    validated: bool = False
    disclaimer: str = DISCLAIMER


class AmplificationResult(NamedTuple):
    """Coherent vs. incoherent amplification factors."""

    n_engines: np.ndarray  # Array of engine counts
    coherent: np.ndarray  # N × ΔF_single
    incoherent: np.ndarray  # √N × ΔF_single
    ratio: np.ndarray  # coherent / incoherent = √N
    damping_margin_ratio: np.ndarray | None  # vacuum/earth ζ ratio (optional)
    validated: bool = False
    disclaimer: str = DISCLAIMER

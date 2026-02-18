"""Amplification factor computation — coherent vs. incoherent thrust oscillation.

Coherent (breathing mode): total oscillation = N * delta_F_single
Incoherent (random phase): total oscillation = sqrt(N) * delta_F_single
Ratio = sqrt(N)

From white paper Section IV.C and Table 3.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from cre.models.environment import DampingParameters
from cre.models.results import AmplificationResult


def coherent_amplification(N: ArrayLike) -> NDArray[np.floating]:
    """Coherent (breathing mode) amplification factor = N.

    All engines oscillate in phase → thrust perturbations add linearly.
    """
    return np.asarray(N, dtype=np.float64)


def incoherent_amplification(N: ArrayLike) -> NDArray[np.floating]:
    """Incoherent (random phase) amplification factor = sqrt(N).

    Random-phase oscillation → RMS addition.
    """
    return np.sqrt(np.asarray(N, dtype=np.float64))


def amplification_ratio(N: ArrayLike) -> NDArray[np.floating]:
    """Ratio of coherent to incoherent = sqrt(N)."""
    return np.sqrt(np.asarray(N, dtype=np.float64))


def damping_margin_ratio(
    N: ArrayLike,
    params: DampingParameters,
) -> NDArray[np.floating]:
    """Compute vacuum-to-Earth breathing-mode damping margin ratio.

    This ratio indicates how much damping margin is lost in vacuum.
    Lower ratio = more margin lost = more dangerous.

    Parameters
    ----------
    N : array_like
        Engine counts.
    params : DampingParameters
        Damping coefficients.

    Returns
    -------
    ratio : ndarray
        vacuum_zeta_0 / earth_zeta_0 as percentage.
    """
    N = np.asarray(N, dtype=np.float64)
    # Breathing mode damping = internal + nozzle + feed (+ atmospheric for earth)
    zeta_vacuum = params.zeta_internal + params.zeta_nozzle + params.zeta_feed
    zeta_earth = zeta_vacuum + params.zeta_atmospheric

    # Ratio as percentage, with slight degradation for larger N
    # (feed system complexity increases)
    degradation = 1.0 + 0.002 * (N - 1)  # Small feed complexity factor
    base_ratio = zeta_vacuum / zeta_earth * 100.0
    ratio = base_ratio / degradation
    return ratio


def amplification_sweep(
    N_range: tuple[int, int],
    params: DampingParameters,
) -> AmplificationResult:
    """Compute amplification factors over a range of engine counts.

    Parameters
    ----------
    N_range : tuple[int, int]
        (N_min, N_max) inclusive.
    params : DampingParameters
        For damping margin computation.

    Returns
    -------
    AmplificationResult
        Arrays of coherent, incoherent, ratio, and damping margin.
    """
    N = np.arange(N_range[0], N_range[1] + 1, dtype=np.float64)

    return AmplificationResult(
        n_engines=N,
        coherent=coherent_amplification(N),
        incoherent=incoherent_amplification(N),
        ratio=amplification_ratio(N),
        damping_margin_ratio=damping_margin_ratio(N, params),
    )

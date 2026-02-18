"""Coupled oscillator array — normal mode eigenfrequencies (Eqs 5–6).

Eq 5: m x_ddot_i + c x_dot_i + k_0 x_i + kappa * sum_j(x_i - x_j) = F_i(t)
Eq 6: omega_n^2 = (k_0/m) + (2*kappa/m) * [1 - cos(2*pi*n/N)]

For N engines in a ring with nearest-neighbor coupling coefficient kappa.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def normal_mode_frequencies_squared(
    k0: float, m: float, kappa: float, N: int
) -> NDArray[np.floating]:
    """Compute squared eigenfrequencies for all N coupled modes (Eq 6).

    Parameters
    ----------
    k0 : float
        Individual engine stiffness [N/m].
    m : float
        Effective oscillating mass [kg].
    kappa : float
        Inter-engine coupling coefficient [N/m].
    N : int
        Number of engines in the ring.

    Returns
    -------
    omega_sq : ndarray of shape (N,)
        omega_n^2 for n = 0, 1, ..., N-1 [rad^2/s^2].
    """
    n = np.arange(N)
    omega_sq = (k0 / m) + (2.0 * kappa / m) * (1.0 - np.cos(2.0 * np.pi * n / N))
    return omega_sq


def normal_mode_frequencies(
    k0: float, m: float, kappa: float, N: int
) -> NDArray[np.floating]:
    """Compute eigenfrequencies for all N coupled modes.

    Returns
    -------
    omega : ndarray of shape (N,)
        omega_n for n = 0, 1, ..., N-1 [rad/s].
    """
    omega_sq = normal_mode_frequencies_squared(k0, m, kappa, N)
    return np.sqrt(omega_sq)


def mode_frequency_ratios(
    kappa: float, k0: float, N: int
) -> NDArray[np.floating]:
    """Compute omega_n / omega_0 for all modes.

    Parameters
    ----------
    kappa : float
        Inter-engine coupling coefficient [N/m].
    k0 : float
        Individual engine stiffness [N/m].
    N : int
        Number of engines.

    Returns
    -------
    ratios : ndarray of shape (N,)
        Frequency ratio omega_n / omega_0 for each mode.
    """
    # omega_0 = sqrt(k0/m), omega_n = sqrt(k0/m + 2*kappa/m * [1-cos(2*pi*n/N)])
    # ratio = omega_n / omega_0 = sqrt(1 + 2*kappa/k0 * [1-cos(2*pi*n/N)])
    n = np.arange(N)
    coupling_term = (2.0 * kappa / k0) * (1.0 - np.cos(2.0 * np.pi * n / N))
    return np.sqrt(1.0 + coupling_term)

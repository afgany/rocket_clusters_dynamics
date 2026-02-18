"""Base cavity acoustics (Eqs 7–8 from white paper).

Eq 7: H_ij(omega) = sum_mn [g_i^(mn) * g_j^(mn)] / [omega_mn^2 - omega^2 + i*omega*omega_mn/Q_mn]
Eq 8: f_1T = 1.841 * c / (2 * pi * R)

Base cavity modes follow standard cylindrical acoustics with Bessel function zeros.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Bessel function derivative zeros alpha'_mn for cylindrical cavity modes
# Determines resonance frequencies for mode (m, n)
_ALPHA_PRIME = {
    (1, 1): 1.8412,  # First tangential (1T)
    (2, 1): 3.0542,  # Second tangential (2T)
    (0, 1): 3.8317,  # First radial (1R)
    (3, 1): 4.2012,  # Third tangential (3T)
    (1, 2): 5.3314,  # Mixed mode
}


def cavity_mode_frequency(
    c: float, R: float, mode: tuple[int, int] = (1, 1)
) -> float:
    """Compute base cavity resonance frequency (Eq 8).

    Parameters
    ----------
    c : float
        Speed of sound in the cavity medium [m/s].
        For atmospheric: ~343 m/s. For rocket base recirculation: ~500-1000 m/s.
    R : float
        Base cavity radius [m] (base_diameter / 2).
    mode : tuple[int, int]
        Cylindrical mode indices (m, n). Default (1,1) = first tangential.

    Returns
    -------
    f : float
        Resonance frequency [Hz].
    """
    alpha_mn = _ALPHA_PRIME.get(mode)
    if alpha_mn is None:
        available = list(_ALPHA_PRIME.keys())
        raise ValueError(f"Mode {mode} not tabulated. Available: {available}")
    return alpha_mn * c / (2.0 * np.pi * R)


def acoustic_transfer_function(
    omega: ArrayLike,
    g_i: float,
    g_j: float,
    omega_mn: float,
    Q_mn: float,
) -> NDArray[np.complexfloating]:
    """Compute the acoustic transfer function H_ij(omega) for a single cavity mode (Eq 7).

    Parameters
    ----------
    omega : array_like
        Angular frequency [rad/s].
    g_i : float
        Coupling coefficient of engine i to mode (m,n).
    g_j : float
        Coupling coefficient of engine j to mode (m,n).
    omega_mn : float
        Resonance angular frequency of mode (m,n) [rad/s].
    Q_mn : float
        Quality factor of mode (m,n). Typically 5–50 for rocket base cavities.

    Returns
    -------
    H : ndarray (complex)
        Transfer function at each frequency.
    """
    omega = np.asarray(omega, dtype=np.float64)
    numerator = g_i * g_j
    denominator = omega_mn**2 - omega**2 + 1j * omega * omega_mn / Q_mn
    return numerator / denominator

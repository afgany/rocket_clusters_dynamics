"""Crocco n-tau combustion response model (Eq 2 from white paper).

R(omega) = n * [1 - exp(-i * omega * tau)]

The interaction index n captures sensitivity of heat release to pressure
perturbations. The sensitive time lag tau sets the phase relationship.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def crocco_response(
    omega: ArrayLike, n: float, tau: float
) -> NDArray[np.complexfloating]:
    """Compute the Crocco combustion response function R(omega).

    Parameters
    ----------
    omega : array_like
        Angular frequency [rad/s]. Scalar or array.
    n : float
        Crocco interaction index (dimensionless, typically 0.3–8).
    tau : float
        Sensitive time lag [s].

    Returns
    -------
    R : ndarray (complex)
        Complex combustion response R(omega) = n * [1 - exp(-i*omega*tau)].
    """
    omega = np.asarray(omega, dtype=np.float64)
    return n * (1.0 - np.exp(-1j * omega * tau))


def crocco_magnitude(omega: ArrayLike, n: float, tau: float) -> NDArray[np.floating]:
    """Compute |R(omega)| — magnitude of the Crocco response.

    |R| = 2*n*|sin(omega*tau/2)|
    """
    return np.abs(crocco_response(omega, n, tau))


def crocco_phase(omega: ArrayLike, n: float, tau: float) -> NDArray[np.floating]:
    """Compute the phase angle of R(omega) in radians."""
    return np.angle(crocco_response(omega, n, tau))

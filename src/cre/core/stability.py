"""Stability boundary computation (Eqs 10–11 from white paper).

Eq 10: n_crit = [alpha_acoustic + alpha_nozzle + alpha_viscous] / [omega * |sin(omega*tau)| * G_coupling]
Eq 11: zeta_min = (n * omega * sin(omega*tau)) / (2*omega_n^2) * (gamma-1)/gamma * (p_bar / rho*c^2)

Stability boundaries in the (n, tau) parameter space.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from cre.models.results import DISCLAIMER, StabilitySweepResult


def n_critical(
    tau: ArrayLike,
    alpha_total: float,
    omega: float,
    G_coupling: float = 1.0,
) -> NDArray[np.floating]:
    """Compute the critical interaction index n_crit (Eq 10).

    Parameters
    ----------
    tau : array_like
        Sensitive time lag [s]. Scalar or array.
    alpha_total : float
        Total absorption coefficient [Np/m].
        Sum of acoustic, nozzle, and viscous absorption.
    omega : float
        Angular frequency [rad/s].
    G_coupling : float
        Coupling gain factor (default 1.0).

    Returns
    -------
    n_crit : ndarray
        Critical interaction index. Clipped to [0, max_n] to avoid singularities.
    """
    tau = np.asarray(tau, dtype=np.float64)
    sin_term = np.abs(np.sin(omega * tau))

    # Avoid division by zero at sin=0 points
    with np.errstate(divide="ignore", invalid="ignore"):
        n_crit = alpha_total / (omega * sin_term * G_coupling)

    # Clip to physical range
    max_n = 20.0  # Upper bound for plot display
    n_crit = np.clip(n_crit, 0.0, max_n)
    return n_crit


def stability_boundary_sweep(
    tau_range: tuple[float, float],
    frequencies: ArrayLike,
    alpha_earth: float,
    alpha_vacuum: float,
    n_tau: int = 500,
    G_coupling: float = 1.0,
) -> StabilitySweepResult:
    """Sweep stability boundaries over tau range for multiple frequencies and environments.

    Parameters
    ----------
    tau_range : tuple[float, float]
        (tau_min, tau_max) in seconds.
    frequencies : array_like
        Frequencies to evaluate [Hz].
    alpha_earth : float
        Total absorption for Earth environment.
    alpha_vacuum : float
        Total absorption for vacuum environment.
    n_tau : int
        Number of tau points.
    G_coupling : float
        Coupling gain factor.

    Returns
    -------
    StabilitySweepResult
        Contains tau, n_crit arrays, frequencies, environment labels.
    """
    frequencies = np.asarray(frequencies, dtype=np.float64)
    tau = np.linspace(tau_range[0], tau_range[1], n_tau)

    # Shape: (n_envs, n_freqs, n_tau)
    n_envs = 2
    n_freqs = len(frequencies)
    n_crit = np.zeros((n_envs, n_freqs, n_tau))

    alphas = [alpha_earth, alpha_vacuum]
    for i_env, alpha in enumerate(alphas):
        for i_freq, f in enumerate(frequencies):
            omega = 2.0 * np.pi * f
            n_crit[i_env, i_freq, :] = n_critical(tau, alpha, omega, G_coupling)

    return StabilitySweepResult(
        tau=tau,
        n_crit=n_crit,
        frequencies=frequencies,
        environments=["earth_sl", "lunar_vacuum"],
    )


def is_stable(
    n: float,
    tau: float,
    alpha_total: float,
    omega: float,
    G_coupling: float = 1.0,
) -> bool:
    """Check if operating point (n, tau) is stable.

    Returns True if n < n_critical.
    """
    n_c = float(n_critical(np.array([tau]), alpha_total, omega, G_coupling)[0])
    return n < n_c


def stability_margin(
    n: float,
    tau: float,
    alpha_total: float,
    omega: float,
    G_coupling: float = 1.0,
) -> float:
    """Compute distance from operating point to stability boundary.

    Returns
    -------
    margin : float
        n_critical - n. Positive = stable, negative = unstable.
    """
    n_c = float(n_critical(np.array([tau]), alpha_total, omega, G_coupling)[0])
    return n_c - n


def zeta_minimum(
    n: float,
    omega: float,
    tau: float,
    omega_n: float,
    gamma: float = 1.25,
    p_bar_over_rho_c2: float = 0.5,
) -> float:
    """Compute minimum damping ratio for stability (Eq 11).

    Parameters
    ----------
    n : float
        Crocco interaction index.
    omega : float
        Driving angular frequency [rad/s].
    tau : float
        Sensitive time lag [s].
    omega_n : float
        Natural frequency of coupled mode [rad/s].
    gamma : float
        Ratio of specific heats.
    p_bar_over_rho_c2 : float
        Normalized mean pressure ratio (p_bar / rho * c^2).

    Returns
    -------
    zeta_min : float
        Minimum damping ratio required for stability.
    """
    sin_term = np.abs(np.sin(omega * tau))
    zeta_min = (n * omega * sin_term) / (2.0 * omega_n**2) * ((gamma - 1.0) / gamma) * p_bar_over_rho_c2
    return float(zeta_min)

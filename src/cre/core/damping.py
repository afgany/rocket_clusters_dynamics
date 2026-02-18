"""Damping spectrum analysis (Eqs 12–13 from white paper).

Eq 12: zeta_total = zeta_internal + zeta_nozzle + zeta_atm + zeta_struct_dissip + zeta_feed_dissip
Eq 13: zeta_internal + zeta_nozzle > n * omega * |sin(omega*tau)| / (2*omega_0^2) * (gamma-1)/gamma * (p_bar/rho*c^2)

The breathing mode (n=0) receives NO inter-engine coupling damping.
Higher modes receive additional damping proportional to [1 - cos(2*pi*n/N)].
"""

import numpy as np
from numpy.typing import NDArray

from cre.models.environment import DampingParameters, Environment
from cre.models.results import DampingSpectrumResult


def damping_spectrum(
    N: int,
    params: DampingParameters,
    environment: Environment,
) -> NDArray[np.floating]:
    """Compute per-mode total damping ratio for an N-engine ring (Eq 12).

    Parameters
    ----------
    N : int
        Number of engines in the ring.
    params : DampingParameters
        Damping coefficients.
    environment : Environment
        Operating environment (Earth or vacuum).

    Returns
    -------
    zeta : ndarray of shape (N,)
        Total damping ratio for each coupled mode n = 0, 1, ..., N-1.
    """
    n = np.arange(N)

    # Base damping (same for all modes)
    zeta_base = params.zeta_internal + params.zeta_nozzle + params.zeta_feed

    # Atmospheric damping (zero in vacuum)
    zeta_atm = environment.zeta_atmospheric

    # Inter-engine coupling damping: proportional to [1 - cos(2*pi*n/N)]
    # This is zero for the breathing mode (n=0) — the central result
    coupling_damping = params.zeta_coupling_max * (1.0 - np.cos(2.0 * np.pi * n / N))

    zeta_total = zeta_base + zeta_atm + coupling_damping
    return zeta_total


def damping_spectrum_multi_env(
    N: int,
    params: DampingParameters,
    environments: list[Environment],
) -> DampingSpectrumResult:
    """Compute damping spectrum for multiple environments.

    Returns
    -------
    DampingSpectrumResult
        With zeta_total shape (n_envs, N).
    """
    mode_indices = np.arange(N)
    zeta_all = np.zeros((len(environments), N))

    for i, env in enumerate(environments):
        zeta_all[i, :] = damping_spectrum(N, params, env)

    return DampingSpectrumResult(
        mode_indices=mode_indices,
        zeta_total=zeta_all,
        n_engines=N,
        environments=[env.name for env in environments],
    )


def breathing_mode_damping(
    params: DampingParameters,
    environment: Environment,
) -> float:
    """Compute damping ratio for the breathing mode (n=0) specifically.

    The breathing mode receives NO coupling damping — only internal terms.
    """
    return params.zeta_internal + params.zeta_nozzle + params.zeta_feed + environment.zeta_atmospheric


def critical_damping_threshold(
    n_crocco: float,
    tau: float,
    omega_n: float,
    gamma: float = 1.25,
    p_bar_over_rho_c2: float = 0.5,
) -> float:
    """Compute the critical damping threshold zeta_crit.

    This is the minimum damping required for stability of a given mode.
    Equivalent to Eq 11 zeta_minimum evaluated for specific parameters.
    """
    omega = omega_n  # Evaluate at mode's own frequency
    sin_term = np.abs(np.sin(omega * tau))
    zeta_crit = (n_crocco * omega * sin_term) / (2.0 * omega_n**2) * ((gamma - 1.0) / gamma) * p_bar_over_rho_c2
    return float(zeta_crit)


def is_mode_stable(
    mode_n: int,
    N: int,
    params: DampingParameters,
    environment: Environment,
    n_crocco: float,
    tau: float,
    omega_n: float,
    gamma: float = 1.25,
    p_bar_over_rho_c2: float = 0.5,
) -> bool:
    """Check if a specific coupled mode is stable.

    Returns True if zeta_total(mode) > zeta_critical.
    """
    zeta_all = damping_spectrum(N, params, environment)
    zeta_mode = zeta_all[mode_n]
    zeta_crit = critical_damping_threshold(n_crocco, tau, omega_n, gamma, p_bar_over_rho_c2)
    return bool(zeta_mode > zeta_crit)

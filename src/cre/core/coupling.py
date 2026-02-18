"""Three coupling pathways (Eqs 9, 14 from white paper).

Eq 9:  kappa_total = kappa_atm(P_a) + kappa_struct + kappa_feed
Eq 14: Kn_p(theta) = (1/2) * Kn_0 * A_pl^0 * (D / 2*r_n) * [1/sin^2(theta)] * [1/f(theta)]

The three coupling pathways have fundamentally different dependencies on ambient pressure:
- Atmospheric: proportional to ambient density, zero in vacuum
- Structural: independent of ambient pressure, dominant in vacuum
- Feed system: independent of ambient pressure, pogo mechanism
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from cre.models.engine import Engine
from cre.models.environment import Environment


def coupling_atmospheric(
    environment: Environment,
    engine: Engine,
    ring_radius: float,
    n_engines: int,
) -> float:
    """Compute atmospheric acoustic coupling coefficient kappa_atm.

    Proportional to ambient pressure. Zero in vacuum.

    Parameters
    ----------
    environment : Environment
        Operating environment (provides ambient_pressure, acoustic_impedance).
    engine : Engine
        Engine spec (provides nozzle_exit_diameter).
    ring_radius : float
        Ring radius [m] — sets inter-engine spacing.
    n_engines : int
        Number of engines in ring.

    Returns
    -------
    kappa_atm : float
        Atmospheric coupling coefficient [N/m].
    """
    if environment.ambient_pressure <= 0:
        return 0.0

    # Coupling proportional to: acoustic_impedance * nozzle_area / spacing
    # Normalized to match white paper default zeta_atmospheric contribution
    Z = environment.acoustic_impedance  # rayl
    De = engine.nozzle_exit_diameter
    A_nozzle = np.pi * (De / 2.0) ** 2

    # Inter-engine spacing from ring geometry
    if n_engines <= 1:
        return 0.0
    D_spacing = 2.0 * ring_radius * np.sin(np.pi / n_engines)

    # Acoustic coupling scales as Z * A / D (dimensional analysis)
    # Normalized coefficient — calibrated so atmospheric contribution matches paper
    efficiency = 0.005  # ~0.5% acoustic efficiency (NASA SP-8072)
    kappa_atm = efficiency * Z * A_nozzle / max(D_spacing, 0.01)
    return kappa_atm


def coupling_structural(
    engine: Engine,
    ring_radius: float,
    n_engines: int,
) -> float:
    """Compute structural coupling coefficient kappa_struct.

    Independent of ambient pressure. Transmits through thrust frame.

    Parameters
    ----------
    engine : Engine
        Engine spec.
    ring_radius : float
        Ring radius [m].
    n_engines : int
        Number of engines.

    Returns
    -------
    kappa_struct : float
        Structural coupling coefficient [N/m].
    """
    if n_engines <= 1:
        return 0.0

    # Structural coupling proportional to engine stiffness and inversely to spacing
    # Use engine mass * omega_0^2 as stiffness proxy
    c = engine.sound_speed
    D = engine.chamber_diameter
    omega_0 = 1.8412 * c / D  # ~omega_1T (from chamber acoustics)
    k_engine = engine.mass * omega_0 ** 2

    D_spacing = 2.0 * ring_radius * np.sin(np.pi / n_engines)

    # Structural coupling is a fraction of engine stiffness
    # Typical structural transmission: 1-5% of engine stiffness
    coupling_fraction = 0.02
    kappa_struct = coupling_fraction * k_engine * (engine.nozzle_exit_diameter / max(D_spacing, 0.01))
    return kappa_struct


def coupling_feed(
    engine: Engine,
    n_engines: int,
) -> float:
    """Compute feed-system coupling coefficient kappa_feed.

    Independent of ambient pressure. Pogo mechanism through shared manifolds.

    Parameters
    ----------
    engine : Engine
        Engine spec.
    n_engines : int
        Number of engines sharing the feed system.

    Returns
    -------
    kappa_feed : float
        Feed system coupling coefficient [N/m].
    """
    if n_engines <= 1:
        return 0.0

    c = engine.sound_speed
    D = engine.chamber_diameter
    omega_0 = 1.8412 * c / D
    k_engine = engine.mass * omega_0 ** 2

    # FFSCC has tighter coupling than gas-generator
    if engine.cycle == "ffscc":
        coupling_fraction = 0.015
    else:
        coupling_fraction = 0.008

    # Feed coupling increases with number of engines sharing manifold
    kappa_feed = coupling_fraction * k_engine * np.log(n_engines) / np.log(33)
    return kappa_feed


def total_coupling(
    environment: Environment,
    engine: Engine,
    ring_radius: float,
    n_engines: int,
) -> float:
    """Compute total inter-engine coupling coefficient kappa_total (Eq 9).

    kappa_total = kappa_atm + kappa_struct + kappa_feed
    """
    k_atm = coupling_atmospheric(environment, engine, ring_radius, n_engines)
    k_struct = coupling_structural(engine, ring_radius, n_engines)
    k_feed = coupling_feed(engine, n_engines)
    return k_atm + k_struct + k_feed


def penetration_knudsen(
    Kn_0: float,
    A_pl: float,
    D: float,
    r_n: float,
    theta: ArrayLike,
) -> NDArray[np.floating]:
    """Compute plume penetration Knudsen number (Eq 14).

    Kn_p(theta) = (1/2) * Kn_0 * A_pl * (D / (2*r_n)) * (1/sin^2(theta)) * (1/f(theta))

    Parameters
    ----------
    Kn_0 : float
        Reference Knudsen number at nozzle exit.
    A_pl : float
        Plume area coefficient.
    D : float
        Half-distance between nozzle centres [m].
    r_n : float
        Nozzle exit radius [m].
    theta : array_like
        Angle from plume axis [rad]. Must be > 0.

    Returns
    -------
    Kn_p : ndarray
        Penetration Knudsen number at each angle.
    """
    theta = np.asarray(theta, dtype=np.float64)
    sin_theta = np.sin(theta)
    # f(theta) approximated as 1 + cos(theta) for Maxwellian distribution
    f_theta = 1.0 + np.cos(theta)
    Kn_p = 0.5 * Kn_0 * A_pl * (D / (2.0 * r_n)) / (sin_theta**2 * f_theta)
    return Kn_p

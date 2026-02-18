"""Single-engine oscillator model (Eqs 1, 3, 4 from white paper).

Eq 1: m_0 x_ddot + c_0 x_dot + k_0 x = F(t)  — damped harmonic oscillator
Eq 3: Rayleigh criterion ∮ p'·Q' dV dt > 0
Eq 4: Y_nozzle ≈ [(gamma+1)/2] · M_entrance · (1/(rho_bar * c_bar))

Chamber acoustic modes estimated from speed of sound and chamber geometry.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from cre.models.engine import Engine

# Bessel function first zero for transverse modes: j'_mn
# j'_11 = 1.8412 (first tangential)
# j'_21 = 3.0542 (second tangential)
# j'_01 = 3.8317 (first radial)
_BESSEL_ZEROS = {
    "1T": 1.8412,
    "2T": 3.0542,
    "1R": 3.8317,
}


@dataclass(frozen=True)
class AcousticModes:
    """Chamber acoustic mode frequencies [Hz]."""

    f_1T: float  # First tangential
    f_1L: float  # First longitudinal
    f_2T: float  # Second tangential


def chamber_acoustic_modes(engine: Engine) -> AcousticModes:
    """Compute chamber acoustic mode frequencies for a given engine.

    Parameters
    ----------
    engine : Engine
        Engine specification with chamber_diameter and sound_speed.

    Returns
    -------
    AcousticModes
        First tangential, first longitudinal, and second tangential frequencies [Hz].

    Notes
    -----
    Transverse modes: f_mn = j'_mn * c / (pi * D)
    Longitudinal modes: f_1L ≈ c / (2 * L_chamber), where L_chamber ≈ D (approximation).
    """
    c = engine.sound_speed
    D = engine.chamber_diameter
    R = D / 2.0

    # Transverse modes: f = j'_mn * c / (2 * pi * R)
    f_1T = _BESSEL_ZEROS["1T"] * c / (2.0 * np.pi * R)
    f_2T = _BESSEL_ZEROS["2T"] * c / (2.0 * np.pi * R)

    # Longitudinal mode: approximate chamber length ≈ diameter
    L_chamber = D
    f_1L = c / (2.0 * L_chamber)

    return AcousticModes(f_1T=f_1T, f_1L=f_1L, f_2T=f_2T)


def engine_natural_frequency(engine: Engine) -> float:
    """Compute the engine natural frequency omega_0 from the 1T mode.

    Returns
    -------
    omega_0 : float
        Natural angular frequency [rad/s].
    """
    modes = chamber_acoustic_modes(engine)
    return 2.0 * np.pi * modes.f_1T


def nozzle_admittance(engine: Engine) -> float:
    """Compute the nozzle acoustic admittance Y_nozzle (Eq 4).

    Y_nozzle ≈ [(gamma+1)/2] · M_entrance · (1/(rho_bar · c_bar))

    Parameters
    ----------
    engine : Engine
        Engine specification.

    Returns
    -------
    Y : float
        Nozzle admittance [s/m]. Always positive (energy sink).

    Notes
    -----
    M_entrance (Mach number at nozzle entrance) estimated from expansion ratio.
    rho_bar · c_bar approximated from chamber conditions.
    """
    gamma = engine.gamma
    c = engine.sound_speed
    Pc = engine.chamber_pressure

    # Estimate density from ideal gas: rho = Pc / (c^2 / gamma)
    # since c^2 = gamma * R_gas * T, and Pc = rho * R_gas * T
    # => rho = gamma * Pc / c^2
    rho_bar = gamma * Pc / (c ** 2)

    # Nozzle entrance Mach estimated as subsonic throat approach (~0.3-0.5)
    M_entrance = 0.4

    Y = ((gamma + 1.0) / 2.0) * M_entrance / (rho_bar * c)
    return Y


def rayleigh_criterion(
    p_prime: NDArray[np.floating], Q_prime: NDArray[np.floating]
) -> float:
    """Evaluate the Rayleigh criterion integral (Eq 3).

    Parameters
    ----------
    p_prime : ndarray
        Pressure perturbation time series [Pa].
    Q_prime : ndarray
        Heat release rate perturbation time series [W/m^3].

    Returns
    -------
    integral : float
        Rayleigh integral value. Positive → instability driving.
    """
    return float(np.trapezoid(p_prime * Q_prime))

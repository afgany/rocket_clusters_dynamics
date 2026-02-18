"""Environment and damping parameter models."""

from pydantic import BaseModel, ConfigDict


class Environment(BaseModel):
    """Operating environment for coupled resonance analysis.

    Defines ambient conditions that affect atmospheric acoustic coupling.
    Instances are frozen (immutable).
    """

    model_config = ConfigDict(frozen=True)

    name: str  # e.g. "earth_sl", "lunar_vacuum"
    ambient_pressure: float  # [Pa]
    acoustic_impedance: float  # [rayl]
    zeta_atmospheric: float  # Atmospheric damping contribution


class DampingParameters(BaseModel):
    """Damping coefficients for stability analysis.

    Default values from white paper Section IV.
    Mutable — users may override for parametric studies.
    """

    zeta_internal: float = 0.015  # Internal combustion damping
    zeta_nozzle: float = 0.020  # Nozzle admittance damping
    zeta_feed: float = 0.005  # Feed system dissipation
    zeta_coupling_max: float = 0.022  # Max inter-engine coupling damping
    zeta_atmospheric: float = 0.028  # Atmospheric damping (Earth only)

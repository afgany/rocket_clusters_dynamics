"""Default configurations for damping parameters and environments."""

from cre.models.environment import DampingParameters, Environment

# --- Pre-loaded environments ---

EARTH_SL = Environment(
    name="earth_sl",
    ambient_pressure=101_325.0,  # 1 atm [Pa]
    acoustic_impedance=420.0,    # ~420 rayl
    zeta_atmospheric=0.028,
)

LUNAR_VACUUM = Environment(
    name="lunar_vacuum",
    ambient_pressure=0.0,
    acoustic_impedance=0.0,
    zeta_atmospheric=0.0,
)

# --- Default damping (white paper Section IV) ---

DEFAULT_DAMPING = DampingParameters()

# --- Registry ---

_ENVIRONMENT_REGISTRY: dict[str, Environment] = {
    "earth_sl": EARTH_SL,
    "lunar_vacuum": LUNAR_VACUUM,
}


def get_environment(name: str) -> Environment:
    """Look up a pre-loaded environment by name."""
    key = name.lower().replace(" ", "_")
    if key not in _ENVIRONMENT_REGISTRY:
        available = ", ".join(sorted(_ENVIRONMENT_REGISTRY.keys()))
        raise KeyError(f"Unknown environment '{name}'. Available: {available}")
    return _ENVIRONMENT_REGISTRY[key]


def list_environments() -> list[str]:
    """Return sorted list of available environment names."""
    return sorted(_ENVIRONMENT_REGISTRY.keys())

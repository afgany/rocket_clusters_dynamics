"""Engine data model — frozen Pydantic model for pre-loaded rocket engine specifications."""

from pydantic import BaseModel, ConfigDict


class Engine(BaseModel):
    """Rocket engine specification for coupled resonance analysis.

    All values are from publicly available SpaceX data (white paper Table 1).
    Instances are frozen (immutable) — pre-loaded configurations only in v1.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    thrust_sl: float | None = None  # Sea-level thrust [N]
    thrust_vac: float | None = None  # Vacuum thrust [N]
    chamber_pressure: float  # Pc [Pa]
    chamber_diameter: float  # [m] — for acoustic mode calculation
    nozzle_exit_diameter: float  # De [m]
    expansion_ratio: float  # Ae/At
    mass: float  # Engine dry mass [kg]
    isp_sl: float | None = None  # Sea-level specific impulse [s]
    isp_vac: float | None = None  # Vacuum specific impulse [s]
    gamma: float = 1.25  # Ratio of specific heats (combustion products)
    sound_speed: float  # Speed of sound in chamber [m/s]
    injector_type: str  # "pintle" | "coaxial_swirl"
    cycle: str  # "gas_generator" | "ffscc"
    n_range: tuple[float, float]  # Crocco interaction index range
    tau_range: tuple[float, float]  # Sensitive time lag range [s]

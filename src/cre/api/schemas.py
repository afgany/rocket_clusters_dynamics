"""Pydantic request/response schemas for the CRE REST API."""

from pydantic import BaseModel, Field

from cre.models.results import DISCLAIMER


class StabilitySweepRequest(BaseModel):
    tau_min: float = Field(0.1e-3, gt=0, description="Min tau [s]")
    tau_max: float = Field(5.0e-3, gt=0, description="Max tau [s]")
    frequencies: list[float] = Field(
        default=[50.0, 135.0, 56.0],
        min_length=1,
        max_length=50,
        description="Frequencies to evaluate [Hz]",
    )
    alpha_earth: float = Field(0.12, ge=0)
    alpha_vacuum: float = Field(0.06, ge=0)
    n_tau: int = Field(500, ge=10, le=10_000)


class DampingSpectrumRequest(BaseModel):
    cluster_name: str = "super_heavy"
    ring_index: int = Field(2, ge=0, description="Ring index (0-based)")


class AmplificationSweepRequest(BaseModel):
    n_min: int = Field(1, ge=1)
    n_max: int = Field(40, ge=1, le=1000)


class DisclaimerMixin(BaseModel):
    disclaimer: str = DISCLAIMER


class EngineResponse(DisclaimerMixin):
    name: str
    thrust_sl: float | None
    thrust_vac: float | None
    chamber_pressure: float
    chamber_diameter: float
    nozzle_exit_diameter: float
    expansion_ratio: float
    mass: float
    isp_sl: float | None
    isp_vac: float | None
    cycle: str
    injector_type: str


class ClusterResponse(DisclaimerMixin):
    name: str
    engine_name: str
    total_engines: int
    base_diameter: float
    rings: list[dict]


class StabilitySweepResponse(DisclaimerMixin):
    tau: list[float]
    n_crit: list[list[list[float]]]  # [env][freq][tau]
    frequencies: list[float]
    environments: list[str]


class DampingSpectrumResponse(DisclaimerMixin):
    mode_indices: list[int]
    zeta_total: list[list[float]]  # [env][mode]
    n_engines: int
    environments: list[str]


class AmplificationSweepResponse(DisclaimerMixin):
    n_engines: list[int]
    coherent: list[float]
    incoherent: list[float]
    ratio: list[float]
    damping_margin_ratio: list[float] | None

"""Pydantic request/response schemas for the CRE REST API."""

from pydantic import BaseModel

from cre.models.results import DISCLAIMER


class StabilitySweepRequest(BaseModel):
    tau_min: float = 0.1e-3
    tau_max: float = 5.0e-3
    frequencies: list[float] = [50.0, 135.0, 56.0]
    alpha_earth: float = 0.12
    alpha_vacuum: float = 0.06
    n_tau: int = 500


class DampingSpectrumRequest(BaseModel):
    cluster_name: str = "super_heavy"
    ring_index: int = 2  # Default: outer ring (20 engines)


class AmplificationSweepRequest(BaseModel):
    n_min: int = 1
    n_max: int = 40


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

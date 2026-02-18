"""Stability computation endpoints."""

from fastapi import APIRouter

from cre.api.schemas import StabilitySweepRequest, StabilitySweepResponse
from cre.core.stability import stability_boundary_sweep

router = APIRouter(prefix="/stability", tags=["stability"])


@router.post("/sweep", response_model=StabilitySweepResponse)
def run_stability_sweep(req: StabilitySweepRequest):
    result = stability_boundary_sweep(
        tau_range=(req.tau_min, req.tau_max),
        frequencies=req.frequencies,
        alpha_earth=req.alpha_earth,
        alpha_vacuum=req.alpha_vacuum,
        n_tau=req.n_tau,
    )
    return StabilitySweepResponse(
        tau=result.tau.tolist(),
        n_crit=result.n_crit.tolist(),
        frequencies=result.frequencies.tolist(),
        environments=result.environments,
    )

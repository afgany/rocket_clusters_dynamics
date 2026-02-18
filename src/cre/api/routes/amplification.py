"""Amplification computation endpoints."""

from fastapi import APIRouter

from cre.api.schemas import AmplificationSweepRequest, AmplificationSweepResponse
from cre.configs.defaults import DEFAULT_DAMPING
from cre.core.amplification import amplification_sweep

router = APIRouter(prefix="/amplification", tags=["amplification"])


@router.post("/sweep", response_model=AmplificationSweepResponse)
def run_amplification_sweep(req: AmplificationSweepRequest):
    result = amplification_sweep(N_range=(req.n_min, req.n_max), params=DEFAULT_DAMPING)
    return AmplificationSweepResponse(
        n_engines=result.n_engines.astype(int).tolist(),
        coherent=result.coherent.tolist(),
        incoherent=result.incoherent.tolist(),
        ratio=result.ratio.tolist(),
        damping_margin_ratio=result.damping_margin_ratio.tolist() if result.damping_margin_ratio is not None else None,
    )

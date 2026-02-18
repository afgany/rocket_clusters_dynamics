"""Damping computation endpoints."""

from fastapi import APIRouter

from cre.api.schemas import DampingSpectrumRequest, DampingSpectrumResponse
from cre.configs.clusters import get_cluster
from cre.configs.defaults import DEFAULT_DAMPING, EARTH_SL, LUNAR_VACUUM
from cre.core.damping import damping_spectrum_multi_env

router = APIRouter(prefix="/damping", tags=["damping"])


@router.post("/spectrum", response_model=DampingSpectrumResponse)
def run_damping_spectrum(req: DampingSpectrumRequest):
    cluster = get_cluster(req.cluster_name)
    ring = cluster.rings[req.ring_index]
    N = ring.n_engines

    result = damping_spectrum_multi_env(N, DEFAULT_DAMPING, [EARTH_SL, LUNAR_VACUUM])
    return DampingSpectrumResponse(
        mode_indices=result.mode_indices.tolist(),
        zeta_total=result.zeta_total.tolist(),
        n_engines=result.n_engines,
        environments=result.environments,
    )

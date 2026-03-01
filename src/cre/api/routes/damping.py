"""Damping computation endpoints."""

from fastapi import APIRouter, HTTPException

from cre.api.schemas import DampingSpectrumRequest, DampingSpectrumResponse
from cre.configs.clusters import get_cluster
from cre.configs.defaults import DEFAULT_DAMPING, EARTH_SL, LUNAR_VACUUM
from cre.core.damping import damping_spectrum_multi_env

router = APIRouter(prefix="/damping", tags=["damping"])


@router.post("/spectrum", response_model=DampingSpectrumResponse)
def run_damping_spectrum(req: DampingSpectrumRequest):
    try:
        cluster = get_cluster(req.cluster_name)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    if req.ring_index >= len(cluster.rings):
        raise HTTPException(
            status_code=422,
            detail=f"ring_index {req.ring_index} out of range, cluster has {len(cluster.rings)} rings (0-{len(cluster.rings) - 1})",
        )
    ring = cluster.rings[req.ring_index]
    N = ring.n_engines

    result = damping_spectrum_multi_env(N, DEFAULT_DAMPING, [EARTH_SL, LUNAR_VACUUM])
    return DampingSpectrumResponse(
        mode_indices=result.mode_indices.tolist(),
        zeta_total=result.zeta_total.tolist(),
        n_engines=result.n_engines,
        environments=result.environments,
    )

"""Plot generation endpoints — return PNG images."""

import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from cre.api.schemas import (
    AmplificationSweepRequest,
    DampingSpectrumRequest,
    StabilitySweepRequest,
)
from cre.configs.clusters import get_cluster
from cre.configs.defaults import DEFAULT_DAMPING, EARTH_SL, LUNAR_VACUUM
from cre.core.amplification import amplification_sweep
from cre.core.damping import damping_spectrum_multi_env
from cre.core.stability import stability_boundary_sweep
from cre.plotting.amplification import plot_amplification
from cre.plotting.damping_spectrum import plot_damping_spectrum
from cre.plotting.stability_map import plot_stability_map

router = APIRouter(prefix="/plots", tags=["plots"])


def _fig_to_png(fig: plt.Figure) -> StreamingResponse:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@router.post("/stability")
def plot_stability(req: StabilitySweepRequest):
    result = stability_boundary_sweep(
        tau_range=(req.tau_min, req.tau_max),
        frequencies=req.frequencies,
        alpha_earth=req.alpha_earth,
        alpha_vacuum=req.alpha_vacuum,
        n_tau=req.n_tau,
    )
    fig = plot_stability_map(result)
    return _fig_to_png(fig)


@router.post("/damping")
def plot_damping(req: DampingSpectrumRequest):
    cluster = get_cluster(req.cluster_name)
    ring = cluster.rings[req.ring_index]
    result = damping_spectrum_multi_env(ring.n_engines, DEFAULT_DAMPING, [EARTH_SL, LUNAR_VACUUM])
    fig = plot_damping_spectrum(result, zeta_crit=0.035)
    return _fig_to_png(fig)


@router.post("/amplification")
def plot_amp(req: AmplificationSweepRequest):
    result = amplification_sweep(N_range=(req.n_min, req.n_max), params=DEFAULT_DAMPING)
    fig = plot_amplification(result)
    return _fig_to_png(fig)

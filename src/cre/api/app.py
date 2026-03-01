"""FastAPI application for the Coupled Resonance Engine."""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from cre.api.routes import amplification, clusters, damping, engines, plots, stability
from cre.models.results import DISCLAIMER

app = FastAPI(
    title="Coupled Resonance Engine API",
    description=(
        "REST API for multi-engine rocket thrust oscillation analysis. "
        f"**{DISCLAIMER}**"
    ),
    version="1.0.0",
)

_cors_origins = os.environ.get("CRE_CORS_ORIGINS", "").split(",")
_cors_origins = [o.strip() for o in _cors_origins if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(engines.router)
app.include_router(clusters.router)
app.include_router(stability.router)
app.include_router(damping.router)
app.include_router(amplification.router)
app.include_router(plots.router)


@app.get("/")
def root():
    return {"name": "Coupled Resonance Engine", "version": "1.0.0", "disclaimer": DISCLAIMER}

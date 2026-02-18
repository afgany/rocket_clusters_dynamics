"""Cluster geometry models — rings, symmetry groups, and multi-engine layouts."""

from pydantic import BaseModel, ConfigDict


class Ring(BaseModel):
    """A single concentric ring of engines within a cluster.

    Each ring is analyzed independently in v1 (inter-ring coupling deferred to v2).
    """

    model_config = ConfigDict(frozen=True)

    n_engines: int  # N for this ring
    radius: float  # Ring radius from vehicle centerline [m]
    symmetry_group: str  # e.g. "C20", "D8", "C3"
    gimbaling: bool  # Whether engines in this ring gimbal


class ClusterGeometry(BaseModel):
    """Multi-engine cluster layout for a vehicle stage.

    Pre-loaded only in v1. Each ring is analyzed independently.
    """

    model_config = ConfigDict(frozen=True)

    name: str  # e.g. "Super Heavy"
    engine_name: str  # Reference to Engine.name (lookup key)
    total_engines: int  # N total (informational for multi-ring)
    rings: list[Ring]  # Concentric ring definitions
    base_diameter: float  # Vehicle base diameter [m]

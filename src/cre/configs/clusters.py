"""Pre-loaded cluster geometries from white paper Section II.C.

All values are READ-ONLY frozen instances derived from publicly available data.
"""

from cre.models.cluster import ClusterGeometry, Ring

FALCON_9 = ClusterGeometry(
    name="Falcon 9",
    engine_name="merlin_1d",
    total_engines=9,
    rings=[
        Ring(n_engines=1, radius=0.0, symmetry_group="C1", gimbaling=True),
        Ring(n_engines=8, radius=1.35, symmetry_group="D8", gimbaling=True),
    ],
    base_diameter=3.66,
)

FALCON_HEAVY = ClusterGeometry(
    name="Falcon Heavy",
    engine_name="merlin_1d",
    total_engines=27,
    rings=[
        # Modeled as 3 separate Falcon 9 cores
        # Center core: 1 center + 8 ring
        Ring(n_engines=1, radius=0.0, symmetry_group="C1", gimbaling=True),
        Ring(n_engines=8, radius=1.35, symmetry_group="D8", gimbaling=True),
        # Side boosters (each 9 engines in identical layout)
        # For v1, treat each booster's outer ring independently
        Ring(n_engines=8, radius=1.35, symmetry_group="D8", gimbaling=True),
        Ring(n_engines=8, radius=1.35, symmetry_group="D8", gimbaling=True),
    ],
    base_diameter=12.2,
)

SUPER_HEAVY = ClusterGeometry(
    name="Super Heavy",
    engine_name="raptor_2",
    total_engines=33,
    rings=[
        Ring(n_engines=3, radius=1.0, symmetry_group="C3", gimbaling=True),
        Ring(n_engines=10, radius=2.8, symmetry_group="C10", gimbaling=True),
        Ring(n_engines=20, radius=4.0, symmetry_group="C20", gimbaling=False),
    ],
    base_diameter=9.0,
)

STARSHIP = ClusterGeometry(
    name="Starship",
    engine_name="raptor_2",  # Mix of Raptor SL + RVac, simplified to Raptor 2
    total_engines=6,
    rings=[
        Ring(n_engines=3, radius=1.5, symmetry_group="C3", gimbaling=True),   # SL Raptors
        Ring(n_engines=3, radius=3.5, symmetry_group="C3", gimbaling=False),  # RVac
    ],
    base_diameter=9.0,
)

# Registry for lookup by name
_CLUSTER_REGISTRY: dict[str, ClusterGeometry] = {
    "falcon_9": FALCON_9,
    "falcon_heavy": FALCON_HEAVY,
    "super_heavy": SUPER_HEAVY,
    "starship": STARSHIP,
}


def get_cluster(name: str) -> ClusterGeometry:
    """Look up a pre-loaded cluster by name (case-insensitive, underscore-separated)."""
    key = name.lower().replace(" ", "_")
    if key not in _CLUSTER_REGISTRY:
        available = ", ".join(sorted(_CLUSTER_REGISTRY.keys()))
        raise KeyError(f"Unknown cluster '{name}'. Available: {available}")
    return _CLUSTER_REGISTRY[key]


def list_clusters() -> list[str]:
    """Return sorted list of available cluster names."""
    return sorted(_CLUSTER_REGISTRY.keys())

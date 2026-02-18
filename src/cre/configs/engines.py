"""Pre-loaded engine specifications from white paper Table 1.

All values are READ-ONLY frozen instances derived from publicly available data.
"""

from cre.models.engine import Engine

MERLIN_1D = Engine(
    name="Merlin 1D",
    thrust_sl=845_000.0,       # 845 kN
    thrust_vac=914_000.0,      # 914 kN
    chamber_pressure=97e5,     # 97 bar
    chamber_diameter=0.36,     # ~0.36 m estimated
    nozzle_exit_diameter=0.92, # ~0.92 m
    expansion_ratio=16.0,      # 16:1
    mass=470.0,                # ~470 kg
    isp_sl=282.0,              # 282 s
    isp_vac=311.0,             # 311 s
    gamma=1.25,
    sound_speed=1240.0,        # c ≈ 1,240 m/s (RP-1/LOX at ~3400 K)
    injector_type="pintle",
    cycle="gas_generator",
    n_range=(0.5, 3.0),        # Crocco interaction index range
    tau_range=(0.5e-3, 5.0e-3),  # Sensitive time lag 0.5–5 ms
)

RAPTOR_2 = Engine(
    name="Raptor 2",
    thrust_sl=2_256_000.0,     # 2,256 kN
    thrust_vac=None,           # SL variant — no published vac value
    chamber_pressure=300e5,    # 300 bar
    chamber_diameter=0.42,     # ~0.42 m estimated
    nozzle_exit_diameter=1.3,  # 1.3 m
    expansion_ratio=33.0,      # ~33–36:1 (using lower bound)
    mass=1630.0,               # 1,630 kg
    isp_sl=327.0,              # ~327 s
    isp_vac=347.0,             # ~347 s
    gamma=1.25,
    sound_speed=1310.0,        # c ≈ 1,310 m/s (CH4/LOX at ~3600 K)
    injector_type="coaxial_swirl",
    cycle="ffscc",
    n_range=(0.3, 2.0),
    tau_range=(0.2e-3, 2.0e-3),  # 0.2–2 ms
)

RAPTOR_3 = Engine(
    name="Raptor 3",
    thrust_sl=2_747_000.0,     # 2,747 kN
    thrust_vac=None,
    chamber_pressure=350e5,    # 350 bar
    chamber_diameter=0.42,     # ~0.42 m (same chamber design)
    nozzle_exit_diameter=1.3,  # ~1.3 m
    expansion_ratio=36.0,      # TBD — using upper Raptor 2 estimate
    mass=1525.0,               # 1,525 kg
    isp_sl=330.0,              # ~330 s
    isp_vac=350.0,             # ~350 s
    gamma=1.25,
    sound_speed=1310.0,
    injector_type="coaxial_swirl",
    cycle="ffscc",
    n_range=(0.3, 2.0),
    tau_range=(0.2e-3, 2.0e-3),
)

RVAC_2 = Engine(
    name="RVac 2",
    thrust_sl=None,            # Vacuum-only variant
    thrust_vac=2_530_000.0,    # 2,530 kN
    chamber_pressure=300e5,    # ~300 bar
    chamber_diameter=0.42,     # Same Raptor chamber
    nozzle_exit_diameter=2.4,  # ~2.4 m
    expansion_ratio=85.0,      # ~80–90:1 (midpoint)
    mass=1700.0,               # ~1,700 kg estimated
    isp_sl=None,
    isp_vac=363.0,             # ~363 s
    gamma=1.25,
    sound_speed=1310.0,
    injector_type="coaxial_swirl",
    cycle="ffscc",
    n_range=(0.3, 2.0),
    tau_range=(0.2e-3, 2.0e-3),
)

# Registry for lookup by name
_ENGINE_REGISTRY: dict[str, Engine] = {
    "merlin_1d": MERLIN_1D,
    "raptor_2": RAPTOR_2,
    "raptor_3": RAPTOR_3,
    "rvac_2": RVAC_2,
}


def get_engine(name: str) -> Engine:
    """Look up a pre-loaded engine by name (case-insensitive, underscore-separated)."""
    key = name.lower().replace(" ", "_")
    if key not in _ENGINE_REGISTRY:
        available = ", ".join(sorted(_ENGINE_REGISTRY.keys()))
        raise KeyError(f"Unknown engine '{name}'. Available: {available}")
    return _ENGINE_REGISTRY[key]


def list_engines() -> list[str]:
    """Return sorted list of available engine names."""
    return sorted(_ENGINE_REGISTRY.keys())

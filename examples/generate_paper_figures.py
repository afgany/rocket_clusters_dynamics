"""Generate all three figures from the white paper.

Produces: fig1_stability.png, fig2_damping.png, fig3_amplification.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from cre.configs.clusters import get_cluster
from cre.configs.defaults import DEFAULT_DAMPING, EARTH_SL, LUNAR_VACUUM
from cre.core.amplification import amplification_sweep
from cre.core.damping import damping_spectrum_multi_env
from cre.core.stability import stability_boundary_sweep
from cre.models.results import DISCLAIMER
from cre.plotting.amplification import plot_amplification
from cre.plotting.damping_spectrum import plot_damping_spectrum
from cre.plotting.stability_map import plot_stability_map

output_dir = Path("figures")
output_dir.mkdir(exist_ok=True)

# --- Figure 1: Stability boundaries ---
print("Generating Figure 1: Stability boundaries...")
result_stab = stability_boundary_sweep(
    tau_range=(0.1e-3, 5.0e-3),
    frequencies=[50.0, 135.0, 56.0],
    alpha_earth=0.12,
    alpha_vacuum=0.06,
)
fig1 = plot_stability_map(result_stab, save_path=output_dir / "fig1_stability.png")
print(f"  Saved to {output_dir / 'fig1_stability.png'}")

# --- Figure 2: Damping spectrum ---
print("Generating Figure 2: Damping spectrum...")
sh = get_cluster("super_heavy")
ring = sh.rings[2]  # 20-engine outer ring
result_damp = damping_spectrum_multi_env(ring.n_engines, DEFAULT_DAMPING, [EARTH_SL, LUNAR_VACUUM])
fig2 = plot_damping_spectrum(result_damp, zeta_crit=0.035, save_path=output_dir / "fig2_damping.png")
print(f"  Saved to {output_dir / 'fig2_damping.png'}")

# --- Figure 3: Amplification ---
print("Generating Figure 3: Amplification factors...")
result_amp = amplification_sweep(N_range=(1, 40), params=DEFAULT_DAMPING)
fig3 = plot_amplification(result_amp, save_path=output_dir / "fig3_amplification.png")
print(f"  Saved to {output_dir / 'fig3_amplification.png'}")

print(f"\nAll figures saved to {output_dir}/")
print(DISCLAIMER)

"""Parametric sweep exploring stability boundary sensitivity.

Sweeps absorption coefficient and frequency to map stability margins.
"""

import numpy as np

from cre.core.stability import stability_boundary_sweep, stability_margin
from cre.models.results import DISCLAIMER

# --- Sweep alpha at 135 Hz ---
print("Stability boundary sensitivity to alpha_total at f=135 Hz")
print(f"{'alpha':>8s} {'n_crit @ tau=1ms':>18s} {'n_crit @ tau=2ms':>18s}")
print("-" * 48)

for alpha in [0.04, 0.08, 0.12, 0.16, 0.20]:
    result = stability_boundary_sweep(
        tau_range=(0.5e-3, 3.0e-3),
        frequencies=[135.0],
        alpha_earth=alpha,
        alpha_vacuum=alpha * 0.5,
        n_tau=300,
    )
    # Find values near tau=1ms and tau=2ms
    tau = result.tau
    idx_1ms = np.argmin(np.abs(tau - 1e-3))
    idx_2ms = np.argmin(np.abs(tau - 2e-3))
    n1 = result.n_crit[0, 0, idx_1ms]
    n2 = result.n_crit[0, 0, idx_2ms]
    print(f"{alpha:8.2f} {n1:18.4f} {n2:18.4f}")

# --- Stability margin for specific operating point ---
print("\nStability margins at n=1.0, tau=1.5ms, f=135Hz:")
omega = 2 * np.pi * 135
for alpha, env in [(0.12, "Earth"), (0.06, "Vacuum")]:
    margin = stability_margin(n=1.0, tau=1.5e-3, alpha_total=alpha, omega=omega)
    status = "STABLE" if margin > 0 else "UNSTABLE"
    print(f"  {env}: margin = {margin:+.4f} ({status})")

print(f"\n{DISCLAIMER}")

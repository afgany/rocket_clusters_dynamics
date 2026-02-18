"""Basic usage of the Coupled Resonance Engine library.

Demonstrates: engine/cluster lookup, damping spectrum, and amplification factors.
"""

from cre.configs.clusters import get_cluster
from cre.configs.defaults import DEFAULT_DAMPING, EARTH_SL, LUNAR_VACUUM
from cre.configs.engines import get_engine
from cre.core.amplification import coherent_amplification, incoherent_amplification
from cre.core.damping import breathing_mode_damping, damping_spectrum
from cre.core.oscillator import chamber_acoustic_modes
from cre.models.results import DISCLAIMER

# --- 1. Engine info ---
raptor = get_engine("raptor_2")
modes = chamber_acoustic_modes(raptor)
print(f"Raptor 2 chamber 1T mode: {modes.f_1T:.0f} Hz")

# --- 2. Cluster info ---
sh = get_cluster("super_heavy")
print(f"\nSuper Heavy: {sh.total_engines} engines in {len(sh.rings)} rings")
for i, ring in enumerate(sh.rings):
    print(f"  Ring {i}: N={ring.n_engines}, R={ring.radius}m ({ring.symmetry_group})")

# --- 3. Damping spectrum for outer ring ---
N = sh.rings[2].n_engines  # 20-engine outer ring
zeta_earth = damping_spectrum(N, DEFAULT_DAMPING, EARTH_SL)
zeta_vac = damping_spectrum(N, DEFAULT_DAMPING, LUNAR_VACUUM)

print(f"\nDamping spectrum (N={N}):")
print(f"  Breathing mode (n=0): Earth={zeta_earth[0]:.4f}, Vacuum={zeta_vac[0]:.4f}")
print(f"  Highest mode (n={N//2}): Earth={zeta_earth[N//2]:.4f}, Vacuum={zeta_vac[N//2]:.4f}")

# --- 4. Breathing mode danger ---
zeta_b_earth = breathing_mode_damping(DEFAULT_DAMPING, EARTH_SL)
zeta_b_vac = breathing_mode_damping(DEFAULT_DAMPING, LUNAR_VACUUM)
print(f"\nBreathing mode margin lost in vacuum: {(1 - zeta_b_vac/zeta_b_earth)*100:.1f}%")

# --- 5. Amplification ---
for N_val in [9, 33]:
    coh = float(coherent_amplification(N_val))
    incoh = float(incoherent_amplification(N_val))
    print(f"\nN={N_val}: coherent={coh:.0f}x, incoherent={incoh:.2f}x, ratio={coh/incoh:.2f}x")

print(f"\n{DISCLAIMER}")

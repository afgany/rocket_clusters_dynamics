# Coupled Resonance Engine (CRE)

Analytical model for coupled thrust oscillation dynamics in multi-engine rocket clusters.

Implements the framework from:

> **"Coupled Resonance Dynamics of Multi-Engine Rocket Clusters: A Cross-Scale Analytical Framework"** — G. Aharon, 2026

**Analytical model — not experimentally validated.**

## Features

- **14 equations** from the white paper: Crocco n-tau response, Rayleigh criterion, nozzle admittance, coupled oscillator eigenfrequencies, cavity acoustics, three coupling pathways, stability boundaries, damping spectrum, and amplification factors
- **Pre-loaded SpaceX configurations**: Merlin 1D, Raptor 2/3, RVac 2 engines; Falcon 9, Falcon Heavy, Super Heavy, Starship clusters
- **Three publication-ready figures**: stability map, damping spectrum, amplification plot
- **REST API** (FastAPI) for programmatic access
- **CLI** (Typer) for quick analysis from the terminal
- **Jupyter notebook** for interactive exploration

## Install

```bash
# Core library
pip install .

# With all extras (API, plots, CLI, notebook, dev)
pip install ".[all]"
```

## Quickstart

### Python

```python
from cre.configs.clusters import get_cluster
from cre.configs.defaults import DEFAULT_DAMPING, EARTH_SL, LUNAR_VACUUM
from cre.core.damping import damping_spectrum

sh = get_cluster("super_heavy")
ring = sh.rings[2]  # 20-engine outer ring

zeta = damping_spectrum(ring.n_engines, DEFAULT_DAMPING, EARTH_SL)
print(f"Breathing mode damping: {zeta[0]:.4f}")
```

### CLI

```bash
# Engine/cluster info
cre info raptor_2
cre info super_heavy

# Generate figures
cre stability --output fig1.png
cre damping --cluster-name super_heavy --ring-index 2 --output fig2.png
cre amplify --n-min 1 --n-max 40 --output fig3.png

# Start API server
cre serve
```

### API

```bash
# Start the server
cre serve

# Query endpoints
curl http://localhost:8000/engines/
curl http://localhost:8000/clusters/super_heavy
curl -X POST http://localhost:8000/stability/sweep -H "Content-Type: application/json" -d '{}'
curl -X POST http://localhost:8000/plots/stability -H "Content-Type: application/json" -d '{}' --output fig1.png
```

### Docker

```bash
docker build -t cre .
docker run -p 8000:8000 cre
```

## Project Structure

```
src/cre/
  models/       # Pydantic data models (Engine, Cluster, Environment, Results)
  configs/      # Pre-loaded engine/cluster/environment configurations
  core/         # Physics computations (14 equations)
  plotting/     # Publication-quality figure generators
  api/          # FastAPI REST endpoints
  cli.py        # Typer CLI
notebooks/      # Jupyter demo notebook
examples/       # Standalone example scripts
tests/          # pytest test suite (~180 tests)
```

## Key Results

1. **Breathing mode (n=0)** receives no inter-engine coupling damping — most dangerous mode
2. **Vacuum** eliminates atmospheric absorption, reducing stability margins by ~40%
3. **Coherent amplification** scales as N (vs. sqrt(N) for random phase)
4. **Super Heavy** (N=33) has the largest breathing-mode risk among current vehicles

## Tests

```bash
pip install ".[dev]"
pytest
```

## License

MIT — see [LICENSE](LICENSE).

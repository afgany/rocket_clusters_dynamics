"""Typer CLI for the Coupled Resonance Engine."""

from __future__ import annotations

from pathlib import Path

import typer

from cre.models.results import DISCLAIMER

app = typer.Typer(
    name="cre",
    help=f"Coupled Resonance Engine — multi-engine thrust oscillation analysis.\n\n{DISCLAIMER}",
)


@app.command()
def info(
    name: str = typer.Argument(help="Engine or cluster name (e.g. merlin_1d, super_heavy)"),
):
    """Print specifications for an engine or cluster."""
    from cre.configs.clusters import get_cluster
    from cre.configs.engines import get_engine

    # Try engine first, then cluster
    try:
        engine = get_engine(name)
        typer.echo(f"Engine: {engine.name}")
        typer.echo(f"  Thrust SL: {engine.thrust_sl or 'N/A'} N")
        typer.echo(f"  Thrust vac: {engine.thrust_vac or 'N/A'} N")
        typer.echo(f"  Chamber pressure: {engine.chamber_pressure / 1e5:.0f} bar")
        typer.echo(f"  Chamber diameter: {engine.chamber_diameter} m")
        typer.echo(f"  Isp SL/vac: {engine.isp_sl or 'N/A'}/{engine.isp_vac or 'N/A'} s")
        typer.echo(f"  Cycle: {engine.cycle}")
        typer.echo(f"  Injector: {engine.injector_type}")
        typer.echo(f"  n range: {engine.n_range}")
        typer.echo(f"  tau range: {engine.tau_range}")
        typer.echo(f"\n{DISCLAIMER}")
        return
    except KeyError:
        pass

    try:
        cluster = get_cluster(name)
        typer.echo(f"Cluster: {cluster.name}")
        typer.echo(f"  Engine: {cluster.engine_name}")
        typer.echo(f"  Total engines: {cluster.total_engines}")
        typer.echo(f"  Base diameter: {cluster.base_diameter} m")
        typer.echo(f"  Rings:")
        for i, ring in enumerate(cluster.rings):
            typer.echo(f"    [{i}] N={ring.n_engines}, R={ring.radius}m, "
                       f"{ring.symmetry_group}, gimbal={ring.gimbaling}")
        typer.echo(f"\n{DISCLAIMER}")
        return
    except KeyError:
        pass

    from cre.configs.engines import list_engines
    from cre.configs.clusters import list_clusters
    typer.echo(f"Unknown name '{name}'.")
    typer.echo(f"Engines: {', '.join(list_engines())}")
    typer.echo(f"Clusters: {', '.join(list_clusters())}")
    raise typer.Exit(1)


@app.command()
def stability(
    tau_min: float = typer.Option(0.1e-3, help="Min tau [s]"),
    tau_max: float = typer.Option(5.0e-3, help="Max tau [s]"),
    freq: str = typer.Option("50,135,56", help="Comma-separated frequencies [Hz]"),
    output: Path = typer.Option(Path("stability.png"), help="Output file path"),
):
    """Run stability boundary sweep and generate Fig 1."""
    import matplotlib
    matplotlib.use("Agg")

    from cre.core.stability import stability_boundary_sweep
    from cre.plotting.stability_map import plot_stability_map

    frequencies = [float(f) for f in freq.split(",")]
    result = stability_boundary_sweep(
        tau_range=(tau_min, tau_max),
        frequencies=frequencies,
        alpha_earth=0.12,
        alpha_vacuum=0.06,
    )
    fig = plot_stability_map(result, save_path=output)
    typer.echo(f"Stability map saved to {output}")
    typer.echo(DISCLAIMER)


@app.command()
def damping(
    cluster_name: str = typer.Option("super_heavy", help="Cluster name"),
    ring_index: int = typer.Option(2, help="Ring index (0-based)"),
    output: Path = typer.Option(Path("damping.png"), help="Output file path"),
):
    """Run per-mode damping analysis and generate Fig 2."""
    import matplotlib
    matplotlib.use("Agg")

    from cre.configs.clusters import get_cluster
    from cre.configs.defaults import DEFAULT_DAMPING, EARTH_SL, LUNAR_VACUUM
    from cre.core.damping import damping_spectrum_multi_env
    from cre.plotting.damping_spectrum import plot_damping_spectrum

    cluster = get_cluster(cluster_name)
    ring = cluster.rings[ring_index]
    result = damping_spectrum_multi_env(ring.n_engines, DEFAULT_DAMPING, [EARTH_SL, LUNAR_VACUUM])
    fig = plot_damping_spectrum(result, zeta_crit=0.035, save_path=output)
    typer.echo(f"Damping spectrum ({cluster.name}, ring {ring_index}, N={ring.n_engines}) saved to {output}")
    typer.echo(DISCLAIMER)


@app.command()
def amplify(
    n_min: int = typer.Option(1, help="Min engine count"),
    n_max: int = typer.Option(40, help="Max engine count"),
    output: Path = typer.Option(Path("amplification.png"), help="Output file path"),
):
    """Run amplification analysis and generate Fig 3."""
    import matplotlib
    matplotlib.use("Agg")

    from cre.configs.defaults import DEFAULT_DAMPING
    from cre.core.amplification import amplification_sweep
    from cre.plotting.amplification import plot_amplification

    result = amplification_sweep(N_range=(n_min, n_max), params=DEFAULT_DAMPING)
    fig = plot_amplification(result, save_path=output)
    typer.echo(f"Amplification plot saved to {output}")
    typer.echo(DISCLAIMER)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Host address"),
    port: int = typer.Option(8000, help="Port"),
):
    """Start the FastAPI server."""
    import uvicorn
    typer.echo(f"Starting CRE API server on {host}:{port}")
    typer.echo(f"Docs at http://{host}:{port}/docs")
    typer.echo(DISCLAIMER)
    uvicorn.run("cre.api.app:app", host=host, port=port, reload=False)

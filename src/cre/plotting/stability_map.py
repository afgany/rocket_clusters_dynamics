"""Stability map plot generator (Fig 1 from white paper)."""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cre.models.results import DISCLAIMER, StabilitySweepResult
from cre.plotting.style import PlotStyle, apply_style


def plot_stability_map(
    result: StabilitySweepResult,
    style: PlotStyle | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Generate stability boundary map in (n, tau) space (Fig 1).

    Parameters
    ----------
    result : StabilitySweepResult
        Output from stability_boundary_sweep().
    style : PlotStyle, optional
        Plot style overrides.
    save_path : str or Path, optional
        Save figure to this path if provided.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    s = apply_style(style)

    fig, ax = plt.subplots(figsize=(s.figure_width, s.figure_height), dpi=s.dpi)
    plt.rcParams["font.family"] = s.font_family
    plt.rcParams["font.size"] = s.font_size

    tau_ms = result.tau * 1000.0  # Convert to ms

    env_colors = [s.color_earth, s.color_vacuum]
    env_labels = ["Earth (1 atm)", "Lunar vacuum"]
    line_styles = ["-", "--"]

    for i_env in range(result.n_crit.shape[0]):
        for i_freq in range(result.n_crit.shape[1]):
            freq = result.frequencies[i_freq]
            n_c = result.n_crit[i_env, i_freq, :]

            label = f"{freq:.0f} Hz, {env_labels[i_env]}"
            ax.plot(
                tau_ms, n_c,
                color=env_colors[i_env],
                linestyle=line_styles[i_env] if i_freq == 0 else ("-." if i_freq == 1 else ":"),
                linewidth=1.5,
                label=label,
            )

    # Shaded region between Earth and vacuum for first frequency
    if result.n_crit.shape[0] >= 2 and result.n_crit.shape[1] >= 1:
        earth_0 = result.n_crit[0, 0, :]
        vacuum_0 = result.n_crit[1, 0, :]
        ax.fill_between(
            tau_ms, vacuum_0, earth_0,
            alpha=0.15, color=s.color_vacuum,
            label="Stability margin lost in vacuum",
        )

    ax.set_xlabel(r"Sensitive time lag, $\tau$ (ms)")
    ax.set_ylabel(r"Critical interaction index, $n_{\mathrm{crit}}$")
    ax.set_title(
        r"Stability boundaries in ($n$, $\tau$) parameter space: Earth vs. Vacuum",
        fontsize=s.font_size + 1,
    )
    ax.legend(fontsize=s.font_size - 2, loc="upper right")
    ax.set_xlim(tau_ms[0], tau_ms[-1])
    ax.set_ylim(0, 6)

    if s.grid:
        ax.grid(True, alpha=s.grid_alpha)

    # Disclaimer
    fig.text(0.5, 0.01, DISCLAIMER, ha="center", fontsize=7, style="italic", color="gray")

    plt.tight_layout(rect=[0, 0.03, 1, 1])

    if save_path:
        fig.savefig(save_path, dpi=s.dpi, bbox_inches="tight")

    return fig

"""Damping spectrum plot generator (Fig 2 from white paper)."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cre.models.results import DISCLAIMER, DampingSpectrumResult
from cre.plotting.style import PlotStyle, apply_style


def plot_damping_spectrum(
    result: DampingSpectrumResult,
    zeta_crit: float | None = None,
    style: PlotStyle | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Generate per-mode damping ratio bar chart (Fig 2).

    Parameters
    ----------
    result : DampingSpectrumResult
        Output from damping_spectrum_multi_env().
    zeta_crit : float, optional
        Representative stability threshold to draw as dashed line.
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

    n = result.mode_indices
    N = result.n_engines
    env_colors = [s.color_earth, s.color_vacuum]
    env_markers = ["o", "s"]
    env_labels = {"earth_sl": "Earth (1 atm)", "lunar_vacuum": "Lunar vacuum"}

    width = 0.35
    for i_env in range(result.zeta_total.shape[0]):
        env_name = result.environments[i_env]
        label = env_labels.get(env_name, env_name)
        offset = -width / 2 + i_env * width if result.zeta_total.shape[0] > 1 else 0

        ax.bar(
            n + offset, result.zeta_total[i_env],
            width=width, label=label,
            color=env_colors[i_env % 2], alpha=0.7,
            edgecolor="white", linewidth=0.5,
        )

    if zeta_crit is not None:
        ax.axhline(
            y=zeta_crit, color="black", linestyle="--", linewidth=1.5,
            label=r"$\zeta_{\mathrm{crit}}$ (representative threshold)",
        )

    # Annotate breathing mode
    ax.annotate(
        "Breathing mode (n = 0):\nNO inter-engine\ncoupling damping",
        xy=(0, result.zeta_total[-1, 0] if result.zeta_total.shape[0] > 1 else result.zeta_total[0, 0]),
        xytext=(3, 0.02),
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="orange"),
        arrowprops=dict(arrowstyle="->", color="orange"),
    )

    ax.set_xlabel("Coupled mode index, $n$")
    ax.set_ylabel(r"Total damping ratio, $\zeta_{\mathrm{total}}$")
    ax.set_title(
        f"Damping ratio per coupled mode, N = {N}: Earth vs. Vacuum",
        fontsize=s.font_size + 1,
    )
    ax.legend(fontsize=s.font_size - 2)
    ax.set_xlim(-1, N)

    if s.grid:
        ax.grid(True, alpha=s.grid_alpha, axis="y")

    fig.text(0.5, 0.01, DISCLAIMER, ha="center", fontsize=7, style="italic", color="gray")
    plt.tight_layout(rect=[0, 0.03, 1, 1])

    if save_path:
        fig.savefig(save_path, dpi=s.dpi, bbox_inches="tight")

    return fig

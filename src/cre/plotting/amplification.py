"""Amplification plot generator (Fig 3 from white paper)."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cre.models.results import DISCLAIMER, AmplificationResult
from cre.plotting.style import PlotStyle, apply_style

# SpaceX vehicle markers for Fig 3
_VEHICLE_MARKERS = {
    6: "Starship\nupper",
    9: "Falcon 9",
    27: "Falcon\nHeavy",
    33: "Super\nHeavy",
}


def plot_amplification(
    result: AmplificationResult,
    style: PlotStyle | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Generate amplification vs. engine count plot (Fig 3).

    Parameters
    ----------
    result : AmplificationResult
        Output from amplification_sweep().
    style : PlotStyle, optional
        Plot style overrides.
    save_path : str or Path, optional
        Save figure to this path if provided.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    s = apply_style(style)

    fig, ax1 = plt.subplots(figsize=(s.figure_width, s.figure_height), dpi=s.dpi)
    plt.rcParams["font.family"] = s.font_family
    plt.rcParams["font.size"] = s.font_size

    N = result.n_engines

    # Left axis: amplification
    ax1.plot(N, result.coherent, color=s.color_vacuum, linewidth=2,
             label=r"Coherent: $N \times \Delta F_{\mathrm{single}}$")
    ax1.plot(N, result.incoherent, color=s.color_incoherent, linewidth=2,
             label=r"Incoherent: $\sqrt{N} \times \Delta F_{\mathrm{single}}$")

    # Phase-locking risk zone (shaded between coherent and incoherent)
    ax1.fill_between(N, result.incoherent, result.coherent,
                     alpha=0.1, color=s.color_vacuum, label="Phase-locking risk zone")

    # Vehicle markers
    for n_eng, label in _VEHICLE_MARKERS.items():
        idx = np.where(N == n_eng)[0]
        if len(idx) > 0:
            i = idx[0]
            ax1.plot(n_eng, result.coherent[i], "D", color=s.color_vacuum,
                     markersize=8, zorder=5)
            ax1.plot(n_eng, result.incoherent[i], "D", color=s.color_incoherent,
                     markersize=8, zorder=5)
            ax1.annotate(label, xy=(n_eng, result.coherent[i]),
                        xytext=(n_eng + 0.5, result.coherent[i] + 1),
                        fontsize=8, fontweight="bold")

    ax1.set_xlabel("Number of engines, $N$")
    ax1.set_ylabel(r"Normalised total oscillation, $\Delta F_{\mathrm{total}} / \Delta F_{\mathrm{single}}$")
    ax1.set_title(
        "Thrust oscillation amplification and vacuum damping margin vs. engine count",
        fontsize=s.font_size + 1,
    )

    # Right axis: damping margin
    if result.damping_margin_ratio is not None:
        ax2 = ax1.twinx()
        ax2.plot(N, result.damping_margin_ratio, color=s.color_margin,
                 linewidth=2, linestyle=":",
                 label="Vacuum/Earth breathing-mode\ndamping margin (%)")
        ax2.set_ylabel("Vacuum-to-Earth damping margin (%)", color=s.color_margin)
        ax2.tick_params(axis="y", labelcolor=s.color_margin)

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2,
                   fontsize=s.font_size - 2, loc="upper left")
    else:
        ax1.legend(fontsize=s.font_size - 2, loc="upper left")

    if s.grid:
        ax1.grid(True, alpha=s.grid_alpha)

    fig.text(0.5, 0.01, DISCLAIMER, ha="center", fontsize=7, style="italic", color="gray")
    plt.tight_layout(rect=[0, 0.03, 1, 1])

    if save_path:
        fig.savefig(save_path, dpi=s.dpi, bbox_inches="tight")

    return fig

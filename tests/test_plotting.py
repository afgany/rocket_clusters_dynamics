"""Tests for plot generation (Figs 1, 2, 3)."""

import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from cre.configs.defaults import DEFAULT_DAMPING, EARTH_SL, LUNAR_VACUUM
from cre.core.amplification import amplification_sweep
from cre.core.damping import damping_spectrum_multi_env
from cre.core.stability import stability_boundary_sweep
from cre.plotting.amplification import plot_amplification
from cre.plotting.damping_spectrum import plot_damping_spectrum
from cre.plotting.stability_map import plot_stability_map
from cre.plotting.style import DEFAULT_STYLE, PlotStyle


class TestPlotStyle:
    def test_defaults(self):
        assert DEFAULT_STYLE.font_family == "serif"
        assert DEFAULT_STYLE.dpi == 300

    def test_override(self):
        custom = PlotStyle(dpi=150, color_earth="#000000")
        assert custom.dpi == 150
        assert custom.color_earth == "#000000"
        assert custom.font_family == "serif"  # Others preserved


class TestPlotStabilityMap:
    def test_generates_figure(self):
        result = stability_boundary_sweep(
            tau_range=(0.1e-3, 5e-3),
            frequencies=[50.0, 135.0, 56.0],
            alpha_earth=0.12,
            alpha_vacuum=0.06,
        )
        fig = plot_stability_map(result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_to_png(self):
        result = stability_boundary_sweep(
            tau_range=(0.1e-3, 5e-3),
            frequencies=[50.0, 135.0],
            alpha_earth=0.12,
            alpha_vacuum=0.06,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "fig1.png"
            fig = plot_stability_map(result, save_path=path)
            assert path.exists()
            assert path.stat().st_size > 0
            plt.close(fig)

    def test_custom_style(self):
        result = stability_boundary_sweep(
            tau_range=(0.1e-3, 5e-3),
            frequencies=[50.0],
            alpha_earth=0.10,
            alpha_vacuum=0.05,
        )
        custom = PlotStyle(dpi=72, figure_width=8, figure_height=5)
        fig = plot_stability_map(result, style=custom)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotDampingSpectrum:
    def test_generates_figure(self):
        result = damping_spectrum_multi_env(33, DEFAULT_DAMPING, [EARTH_SL, LUNAR_VACUUM])
        fig = plot_damping_spectrum(result, zeta_crit=0.035)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_to_png(self):
        result = damping_spectrum_multi_env(33, DEFAULT_DAMPING, [EARTH_SL, LUNAR_VACUUM])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "fig2.png"
            fig = plot_damping_spectrum(result, save_path=path)
            assert path.exists()
            plt.close(fig)


class TestPlotAmplification:
    def test_generates_figure(self):
        result = amplification_sweep(N_range=(1, 40), params=DEFAULT_DAMPING)
        fig = plot_amplification(result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_to_png(self):
        result = amplification_sweep(N_range=(1, 40), params=DEFAULT_DAMPING)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "fig3.png"
            fig = plot_amplification(result, save_path=path)
            assert path.exists()
            plt.close(fig)

    def test_saves_to_pdf(self):
        result = amplification_sweep(N_range=(1, 40), params=DEFAULT_DAMPING)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "fig3.pdf"
            fig = plot_amplification(result, save_path=path)
            assert path.exists()
            plt.close(fig)

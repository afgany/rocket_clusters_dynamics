"""Tests for Typer CLI."""

import tempfile
from pathlib import Path

from typer.testing import CliRunner

from cre.cli import app

runner = CliRunner()


class TestCLIInfo:
    def test_engine_info(self):
        result = runner.invoke(app, ["info", "merlin_1d"])
        assert result.exit_code == 0
        assert "Merlin 1D" in result.stdout
        assert "845000" in result.stdout

    def test_cluster_info(self):
        result = runner.invoke(app, ["info", "super_heavy"])
        assert result.exit_code == 0
        assert "Super Heavy" in result.stdout
        assert "33" in result.stdout

    def test_unknown_name(self):
        result = runner.invoke(app, ["info", "saturn_v"])
        assert result.exit_code == 1


class TestCLIStability:
    def test_generates_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = Path(tmpdir) / "test_stability.png"
            result = runner.invoke(app, [
                "stability",
                "--tau-min", "0.5e-3",
                "--tau-max", "3e-3",
                "--freq", "50,135",
                "--output", str(outpath),
            ])
            assert result.exit_code == 0
            assert outpath.exists()


class TestCLIDamping:
    def test_generates_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = Path(tmpdir) / "test_damping.png"
            result = runner.invoke(app, [
                "damping",
                "--cluster-name", "super_heavy",
                "--ring-index", "2",
                "--output", str(outpath),
            ])
            assert result.exit_code == 0
            assert outpath.exists()


class TestCLIAmplify:
    def test_generates_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = Path(tmpdir) / "test_amplify.png"
            result = runner.invoke(app, [
                "amplify",
                "--n-min", "1",
                "--n-max", "40",
                "--output", str(outpath),
            ])
            assert result.exit_code == 0
            assert outpath.exists()


class TestCLIHelp:
    def test_main_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Coupled Resonance Engine" in result.stdout

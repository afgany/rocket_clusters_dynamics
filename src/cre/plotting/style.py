"""Plot style configuration — white paper defaults with optional overrides."""

from pydantic import BaseModel


class PlotStyle(BaseModel):
    """Publication-quality plot style matching white paper figures.

    Pass an instance to any plot function to override defaults.
    """

    font_family: str = "serif"
    font_size: int = 11
    figure_width: float = 10.0
    figure_height: float = 6.0
    dpi: int = 300
    color_earth: str = "#2196F3"      # Blue
    color_vacuum: str = "#F44336"     # Red
    color_coherent: str = "#1565C0"   # Dark blue
    color_incoherent: str = "#E65100" # Dark orange
    color_margin: str = "#2E7D32"     # Green
    grid: bool = True
    grid_alpha: float = 0.3


DEFAULT_STYLE = PlotStyle()


def apply_style(style: PlotStyle | None = None) -> PlotStyle:
    """Return the given style or the default."""
    return style if style is not None else DEFAULT_STYLE

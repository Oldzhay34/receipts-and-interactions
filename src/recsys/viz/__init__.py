"""recsys.viz subpackage: visualization style and plotting utilities (stage 1.6)."""

from recsys.viz.plots import (
    bar_counts,
    box_by_group,
    corr_heatmap,
    hist_box,
    long_tail_hist,
    scatter_sample,
)
from recsys.viz.style import PALETTE, apply_style

__all__ = [
    "PALETTE",
    "apply_style",
    "hist_box",
    "bar_counts",
    "scatter_sample",
    "corr_heatmap",
    "box_by_group",
    "long_tail_hist",
]

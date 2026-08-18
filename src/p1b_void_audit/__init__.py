"""Standalone, read-only GT Void auditing.

This package is deliberately outside :mod:`vggt_bev_method1`.  P1B training
does not import it and never receives its masks or measurements.
"""

from .coverage import VoidCoverageIndex, fill_enclosed_scene_domain

__all__ = ["VoidCoverageIndex", "fill_enclosed_scene_domain"]

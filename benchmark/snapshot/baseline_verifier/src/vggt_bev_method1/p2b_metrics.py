"""Legacy import aliases; new code must import :mod:`p1b_metrics`."""

from .p1b_metrics import finalize_p1b_metrics, p1b_metric_totals

finalize_p2b_metrics = finalize_p1b_metrics
p2b_metric_totals = p1b_metric_totals

__all__ = ["finalize_p2b_metrics", "p2b_metric_totals"]

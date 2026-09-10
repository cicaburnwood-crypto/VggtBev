"""Planner support boundary contract; compatible with NumPy and torch masks.

Input is predicted support and its two-pixel eroded interior. A border without
older trusted evidence remains unknown, never declared free. Interior updates
are latest-wins for all numeric fields, not just the displayed gate.
"""
POLICY = 'eroded_predicted_support_unknown_border_v6'


def trusted_latest_mask(support, interior):
    if support.shape != interior.shape:
        raise ValueError('FOV support/interior shape mismatch')
    return (support >= .5) & (interior >= .5)

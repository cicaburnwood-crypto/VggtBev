"""Legacy model aliases for loading pre-rename P1B checkpoints."""

from .p1b import DenseMetricQueryDecoder, P1BHead, P1BSystem, PixelRoutedBEVDecoder

P2BHead = P1BHead
P2BSystem = P1BSystem

__all__ = [
    "DenseMetricQueryDecoder",
    "P2BHead",
    "P2BSystem",
    "PixelRoutedBEVDecoder",
]

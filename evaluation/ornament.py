"""Ornament metrics are token-set metrics, never surface-string metrics."""
from .fingering import rates, set_counts


def ornament_metrics(events):
    return rates(set_counts(events, "ornaments"), include_accuracy=False)


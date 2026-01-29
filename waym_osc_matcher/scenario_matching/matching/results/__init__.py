# scenario_matching/matching/results/__init__.py
from .types import (
    CallKey, BindingKey,
    Interval, PerCallSignal, BlockSignal,
)
from .store import ResultsStore
from .ops import (
    intervals_from_mask, intervals_to_mask,
    mask_from_intervals, coalesce_intervals,
)

__all__ = [
    "CallKey", "BindingKey",
    "Interval", "PerCallSignal", "BlockSignal",
    "ResultsStore",
    "intervals_from_mask", "intervals_to_mask",
    "mask_from_intervals", "coalesce_intervals",
]

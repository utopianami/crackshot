"""Differentiable heads for cross-sectional portfolio construction."""

from src.training.heads.long_short_topk import LongShortResult, long_short_soft_topk
from src.training.heads.soft_topk import SoftTopKResult, soft_topk

__all__ = ["SoftTopKResult", "soft_topk", "LongShortResult", "long_short_soft_topk"]

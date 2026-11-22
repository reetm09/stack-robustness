"""Dynamic range fraction (DRF), as computed in Miller et al.'s calibration analysis.

DRF is the share of the gap between the negative control and a perfect score that
the positive control closes. It keeps its sign (negative when the positive control
does worse than the negative control) and is clipped to [-1, 1].
"""

import math

EPS = 1e-6


def drf(pos: float, neg: float, perfect: float, higher_is_better: bool) -> float:
    if math.isnan(pos) or math.isnan(neg):
        return float("nan")
    if higher_is_better:
        value = (pos - neg) / (perfect - neg + EPS)
    else:
        value = (neg - pos) / (neg - perfect + EPS)
    return max(-1.0, min(1.0, value))

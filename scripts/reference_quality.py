"""Causal continuity screening for 1 Hz pedestrian position references.

These thresholds flag suspect transitions; they cannot certify absolute GT
accuracy or repair a reference-frame reset. Coordinates are never modified.
"""
import numpy as np


DEFAULT_MAX_GAP_S = 1.5
DEFAULT_MAX_SPEED_MPS = 3.
DEFAULT_SETTLING_S = 10.


def reference_continuity(t, ref, max_gap_s=DEFAULT_MAX_GAP_S,
                         max_speed_mps=DEFAULT_MAX_SPEED_MPS,
                         settling_s=DEFAULT_SETTLING_S):
    """Return per-sample diagnostics; edge quantities end at the named sample.

    t is elapsed seconds, ref is (N, 2) metres. A suspect transition starts a
    new segment. Following each such transition, wait settling_s before using
    a new position anchor. No future sample or predicted position is consulted.
    """
    t, ref = np.asarray(t, dtype=float), np.asarray(ref, dtype=float)
    if t.ndim != 1 or len(t) < 2 or ref.shape != (len(t), 2):
        raise ValueError("Expected at least two timestamps and (N, 2) positions")
    if not np.isfinite(t).all() or not np.isfinite(ref).all():
        raise ValueError("Nonfinite reference samples")
    if not np.isfinite([max_gap_s, max_speed_mps, settling_s]).all():
        raise ValueError("Continuity thresholds must be finite")
    if max_gap_s <= 0 or max_speed_mps <= 0 or settling_s < 0:
        raise ValueError("Gap/speed must be positive; settling must be nonnegative")
    dt = np.diff(t)
    if np.any(dt <= 0):
        raise ValueError("Reference timestamps must be strictly increasing")
    step = np.linalg.norm(np.diff(ref, axis=0), axis=1)
    gap = np.r_[False, dt > max_gap_s]
    speed = np.r_[0., step / dt]
    jump = speed > max_speed_mps
    broken = gap | jump
    ready = np.ones(len(t), dtype=bool)
    last_break = None
    for k in range(len(t)):
        if broken[k]:
            last_break = t[k]
        if last_break is not None:
            # 1 ms tolerance absorbs sub-millisecond reference clock drift.
            ready[k] = t[k] - last_break >= settling_s - .001
    return {
        "gt_dt_s": np.r_[0., dt], "gt_step_m": np.r_[0., step],
        "gt_speed_mps": speed, "gt_gap_flag": gap, "gt_speed_flag": jump,
        "gt_discontinuity": broken, "gt_segment": np.cumsum(broken),
        "gt_anchor_ready": ready,
    }


def continuous_distance_interval(t, ref, distance_m=10., **quality_options):
    """First continuous, post-settling interval accumulating distance_m.

    Reset both anchor and accumulated distance at a discontinuity. This chooses
    the same interval for all methods and never searches for a favourable fit.
    """
    if not np.isfinite(distance_m) or distance_m <= 0:
        raise ValueError("Initialization distance must be finite and positive")
    quality = reference_continuity(t, ref, **quality_options)
    start, distance = None, 0.
    for k in range(len(t)):
        if quality["gt_discontinuity"][k] or not quality["gt_anchor_ready"][k]:
            start, distance = None, 0.
        if not quality["gt_anchor_ready"][k]:
            continue
        if start is None:
            start = k
            continue
        distance += quality["gt_step_m"][k]
        if distance >= distance_m:
            return start, k
    raise ValueError(f"No continuous reference interval reaches {distance_m:g} metres "
                     "after continuity screening and settling")

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import savgol_filter

def smooth_time_series(
    normalized_pose3d: dict[str, Any],
    *,
    method: str = "savgol",
    window: int = 11,
    polyorder: int = 2,
    missing_max_gap: int = 5,
) -> dict[str, Any]:
    """
    Smooth normalized 3D joint trajectories and handle missing/noisy keypoints.

    Output contract: same dict keys as input, but coordinates are temporally smoothed.
    """
    if method not in {"savgol", "moving_average"}:
        raise ValueError("method must be 'savgol' or 'moving_average'")

    xyz = np.asarray(normalized_pose3d["keypoints_3d"], dtype=np.float32)  # (T,J,3)
    out = xyz.copy()

    T, J, D = out.shape
    if T < 3:
        return {**normalized_pose3d, "keypoints_3d": out}

    for j in range(J):
        for d in range(D):
            s = out[:, j, d]
            s = _interp_short_gaps_1d(s, max_gap=missing_max_gap)

            if method == "savgol":
                win = min(window, T if T % 2 == 1 else T - 1)
                if win < 3:
                    out[:, j, d] = s
                    continue
                poly = min(polyorder, win - 1)
                valid = ~np.isnan(s)
                if np.count_nonzero(valid) < max(5, poly + 2):
                    out[:, j, d] = s
                    continue
                # Fill remaining NaNs with nearest-neighbor for filter stability, then restore NaNs.
                s_filled = _fill_remaining_nans(s)
                sm = savgol_filter(s_filled, window_length=win, polyorder=poly, mode="interp")
                sm[~valid] = np.nan
                out[:, j, d] = sm.astype(np.float32)
            else:
                out[:, j, d] = _moving_average_1d(s, window=max(3, window))

    return {**normalized_pose3d, "keypoints_3d": out.astype(np.float32)}


def _interp_short_gaps_1d(x: np.ndarray, max_gap: int) -> np.ndarray:
    y = x.astype(np.float32).copy()
    n = len(y)
    isn = np.isnan(y)
    if np.all(isn):
        return y

    i = 0
    while i < n:
        if not isn[i]:
            i += 1
            continue
        s = i
        while i < n and isn[i]:
            i += 1
        e = i
        gap = e - s
        left = s - 1
        right = e
        if gap <= max_gap and left >= 0 and right < n and not np.isnan(y[left]) and not np.isnan(y[right]):
            y[s:e] = np.interp(np.arange(s, e), [left, right], [y[left], y[right]])
    return y


def _fill_remaining_nans(x: np.ndarray) -> np.ndarray:
    y = x.copy()
    n = len(y)
    valid = ~np.isnan(y)
    if not np.any(valid):
        return np.zeros_like(y)

    first = int(np.where(valid)[0][0])
    last = int(np.where(valid)[0][-1])
    y[:first] = y[first]
    y[last + 1 :] = y[last]
    for i in range(first + 1, last + 1):
        if np.isnan(y[i]):
            y[i] = y[i - 1]
    return y


def _moving_average_1d(x: np.ndarray, window: int) -> np.ndarray:
    y = x.copy()
    if window % 2 == 0:
        window += 1
    pad = window // 2
    valid = ~np.isnan(y)
    vv = np.where(valid, y, 0.0)
    ww = valid.astype(np.float32)
    vv_pad = np.pad(vv, (pad, pad), mode="edge")
    ww_pad = np.pad(ww, (pad, pad), mode="edge")
    num = np.convolve(vv_pad, np.ones(window, dtype=np.float32), mode="valid")
    den = np.convolve(ww_pad, np.ones(window, dtype=np.float32), mode="valid")
    sm = num / np.clip(den, 1e-6, None)
    sm[~valid] = np.nan
    return sm.astype(np.float32)


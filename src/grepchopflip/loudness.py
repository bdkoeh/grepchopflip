"""Level measurements for one clip: peak, RMS, crest factor and integrated LUFS (ITU-R BS.1770 via pyloudnorm).

BS.1770 gates on 400 ms blocks, so a clip shorter than that cannot be metered directly. Such clips are
zero-padded to 400 ms and reported with `lufs_measure: "short_term_padded"`: the value is the clip's energy
averaged over one block, which keeps short hits comparable to each other (it is lower than the loudness of
the hit's own duration). Clips of 400 ms or more are measured as they are (`"integrated"`).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np

BLOCK_S = 0.4


@dataclass
class Levels:
    peak_dbfs: float | None
    rms_dbfs: float | None
    crest_db: float | None
    lufs_i: float | None
    lufs_measure: str | None      # "integrated" | "short_term_padded" | None

    def to_dict(self) -> dict:
        return {"peak_dbfs": self.peak_dbfs, "rms_dbfs": self.rms_dbfs, "crest_db": self.crest_db,
                "lufs_i": self.lufs_i, "lufs_measure": self.lufs_measure}


def _db(x: float) -> float | None:
    return round(20 * math.log10(x), 2) if x > 1e-9 else None


def measure(y: np.ndarray, sr: int) -> Levels:
    """`y` is float, shape (n,) or (n, channels)."""
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    if y.size == 0:
        return Levels(None, None, None, None, None)
    peak = _db(float(np.max(np.abs(y))))
    rms = _db(float(np.sqrt(np.mean(np.square(y)))))
    crest = round(peak - rms, 2) if peak is not None and rms is not None else None
    lufs, how = _lufs(y, sr)
    return Levels(peak, rms, crest, lufs, how)


def _lufs(y: np.ndarray, sr: int) -> tuple[float | None, str | None]:
    import pyloudnorm as pyln

    need = int(math.ceil(BLOCK_S * sr))
    how = "integrated"
    if len(y) < need:
        y = np.vstack([y, np.zeros((need - len(y), y.shape[1]))])
        how = "short_term_padded"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            v = float(pyln.Meter(sr).integrated_loudness(y))
    except Exception:
        return None, None
    if not math.isfinite(v):
        return None, None
    return round(v, 2), how

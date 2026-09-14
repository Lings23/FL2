"""R2 label-free, leave-one-out pre-clipping trainable norm cap."""
import hashlib
import json
from pathlib import Path
import numpy as np

VERSION = 'rtc_r2_raw_norm_loo.v1'


def load_calibration(path):
    c = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if (not isinstance(c, dict) or set(c) != {'version', 'f', 'ratio_threshold', 'cap_multiplier', 'provenance'} or
            c['version'] != VERSION or type(c['f']) is not int or c['f'] != 3 or
            type(c['ratio_threshold']) not in (int, float) or not np.isfinite(c['ratio_threshold']) or
            c['ratio_threshold'] < 3 or type(c['cap_multiplier']) not in (int, float) or c['cap_multiplier'] != 0):
        raise ValueError('invalid raw norm calibration')
    return c


def calibration_hash(c):
    return hashlib.sha256(json.dumps(c, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def trainable_norms(deltas, positions):
    if (not deltas or not positions or len(set(positions)) != len(positions) or
            any(type(i) is not int or i < 0 or any(i >= len(d) for d in deltas) for i in positions)):
        raise ValueError('raw norm requires explicit trainable positions')
    return [float(np.sqrt(sum(float(np.dot(np.asarray(d[i], dtype=np.float64).ravel(),
                                           np.asarray(d[i], dtype=np.float64).ravel())) for i in positions))) for d in deltas]


def score(norms, calibration, mode):
    if mode not in ('observe', 'cap'):
        raise ValueError('invalid raw norm mode')
    values = np.asarray(norms, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('nonfinite/negative raw norms')
    rows = []
    for i, value in enumerate(values):
        others = [float(v) for j, v in enumerate(values) if j != i and v > 1e-12]
        valid = len(others) >= 2 * calibration['f'] + 3
        median = float(np.median(others)) if valid else None
        ratio = float(value / median) if valid else None
        if ratio is not None and not np.isfinite(ratio):
            raise ValueError('nonfinite raw norm ratio')
        flagged = bool(valid and ratio > calibration['ratio_threshold'])
        rows.append({'rtc_r2_raw_norm': float(value), 'rtc_r2_reference_count': len(others),
                     'rtc_r2_reference_median': median, 'rtc_r2_ratio': ratio, 'rtc_r2_valid': valid,
                     'rtc_r2_flagged': flagged, 'rtc_r2_q': 0. if mode == 'cap' and flagged else 1.})
    return rows


def summary(calibration, mode, seconds):
    return {'rtc_r2_version': VERSION, 'rtc_r2_calibration_hash': calibration_hash(calibration),
            'rtc_r2_mode': mode, 'rtc_r2_scoring_seconds': seconds}

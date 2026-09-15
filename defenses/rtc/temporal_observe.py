"""Inert trainable residual sketches for R3 temporal diagnosis; no decisions/RNG."""
import json
import numpy as np
from defenses.rtc.sketch import signed_count_sketch, SPLITMIX64_V1

VERSION = 'rtc_r3_trainable_residual_countsketch.v1'
DIMENSION = 512
SEED = 20260915


def observe(residuals, positions):
    positions = list(positions)
    if not positions or len(set(positions)) != len(positions):
        raise ValueError('Explicit unique trainable positions required')
    records = []
    for residual in residuals:
        arrays = [np.asarray(residual[p], dtype=np.float64) for p in positions]
        if not all(np.isfinite(a).all() for a in arrays):
            raise ValueError('Nonfinite temporal residual')
        norm = float(np.sqrt(sum(float(a.reshape(-1) @ a.reshape(-1)) for a in arrays)))
        sketch = signed_count_sketch(arrays, dimension=DIMENSION, seed=SEED, algorithm=SPLITMIX64_V1)
        if not np.isfinite(norm) or not np.isfinite(sketch).all():
            raise ValueError('Nonfinite temporal observation')
        records.append({'rtc_r3t_residual_norm': norm,
                        'rtc_r3t_sketch_json': json.dumps(np.asarray(sketch, dtype=float).tolist()),
                        'rtc_r3t_trainable_dimension': sum(a.size for a in arrays)})
    return records


def metadata():
    return {'rtc_r3t_version': VERSION, 'rtc_r3t_mode': 'observe',
            'rtc_r3t_sketch_dimension': DIMENSION, 'rtc_r3t_sketch_seed': SEED,
            'rtc_r3t_sketch_algorithm': SPLITMIX64_V1,
            'rtc_r3t_sketch_normalized': True,
            'rtc_r3t_reference': 'existing_nominal_coordinate_median',
            'rtc_r3t_scope': 'trainable_clipped_delta_minus_existing_anchor'}

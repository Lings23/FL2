"""R1c pairwise corroboration, calibrated only on independent clean records."""
import numpy as np

VERSION = 'rtc_r1c_pairwise_corroboration.v1'


def validate(config):
    if not isinstance(config, dict) or set(config) != {'version', 'pairwise_threshold', 'quorum_numerator', 'quorum_denominator', 'provenance'}:
        raise ValueError('invalid corroboration calibration keys')
    threshold = config['pairwise_threshold']
    if (config['version'] != VERSION or type(threshold) not in (float, int) or
            not np.isfinite(threshold) or not -1 <= threshold < 0):
        raise ValueError('invalid corroboration threshold/version')
    if (type(config['quorum_numerator']) is not int or config['quorum_numerator'] != 2 or
            type(config['quorum_denominator']) is not int or config['quorum_denominator'] != 3):
        raise ValueError('corroboration quorum must be fixed two-thirds')


def corroboration(gram, index, references, threshold):
    if not references:
        return {'votes': 0, 'required': 0, 'passes': False}
    g = np.asarray(gram, dtype=float)
    if (g.ndim != 2 or g.shape[0] != g.shape[1] or not np.isfinite(g).all() or
            type(index) is not int or not 0 <= index < len(g) or
            len(set(references)) != len(references) or index in references or
            any(type(j) is not int or not 0 <= j < len(g) for j in references) or
            not np.isfinite(threshold) or not -1 <= threshold < 0):
        raise ValueError('invalid corroboration inputs')
    norm = np.sqrt(np.maximum(np.diag(g), 0))
    if norm[index] <= 1e-12 or np.any(norm[references] <= 1e-12):
        return {'votes': 0, 'required': (2 * len(references) + 2) // 3, 'passes': False}
    values = g[index, references] / (norm[index] * norm[references])
    votes = int(np.count_nonzero(values < threshold))
    required = (2 * len(references) + 2) // 3
    return {'votes': votes, 'required': required, 'passes': votes >= required}

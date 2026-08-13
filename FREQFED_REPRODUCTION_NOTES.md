# FreqFed reproduction notes

This repository implements FreqFed as a malicious-client defense following
“FreqFed: A Frequency Analysis-Based Approach for Mitigating Poisoning Attacks
in Federated Learning”.

## Paper-aligned path

For each server round, `FreqFedDefense`:

1. receives each participating client's local model parameters through the
   standard defense aggregation interface;
2. applies DCT/DCTN to floating-point model tensors;
3. extracts low-frequency DCT coefficients using the paper-style triangular
   region, controlled by `low_frequency_ratio` and selected with
   `low_frequency_shape: triangular`;
4. normalizes per-layer low-frequency vectors and concatenates them into a
   client fingerprint;
5. optionally applies deterministic random projection when `projection_dim` is
   set to a positive value and the fingerprint is larger than that limit;
6. computes a pairwise cosine-distance matrix between client fingerprints;
7. runs HDBSCAN with `metric: precomputed`;
8. selects the largest non-noise cluster as the trusted client set;
9. aggregates only the selected cluster and assigns zero aggregation weight to
   non-selected clients and HDBSCAN noise points.

## Intentional engineering choices

- The paper's pseudocode averages the selected cluster, while this project keeps
  compatibility with the existing FedAvg-style interface and uses client sample
  counts inside the selected cluster.
- Multi-layer neural networks are handled by extracting a low-frequency DCT
  vector per tensor, L2-normalizing each tensor's vector, and concatenating the
  result. This avoids large layers dominating the cosine distance.
- Deterministic random projection is available for high-dimensional models to
  keep HDBSCAN tractable, but is disabled by default (`projection_dim: 0`) to
  stay closer to the paper's direct low-frequency DCT distance computation.
- `low_frequency_shape: rectangle` remains available only as a compatibility
  option for older experiments; `triangular` is the paper-aligned default.
- Fallback aggregation modes (`median`, `fedavg`, `keep_global`) are explicitly
  logged. Median and keep-global fallback do not emit fake client weights,
  because their per-client scalar influence is not well-defined.

## Recommended optimization protocol

Start with clean smoke/pilot runs and inspect:

- `freqfed_fallback`
- `freqfed_fallback_reason`
- `freqfed_cluster_count`
- `freqfed_selected_ratio`
- clean accuracy

If fallback is too frequent, tune only clean-pilot parameters before attack
comparisons:

1. lower `min_cluster_size`;
2. lower `min_samples`;
3. enable `allow_single_cluster`;
4. adjust `low_frequency_ratio`;
5. adjust or disable `projection_dim`.

Do not tune FreqFed parameters using attacked defense results, otherwise the
baseline would leak information from the evaluated attacks.

## Current pilot result

After switching to the paper-aligned triangular DCT region and disabling random
projection by default, the local 6-round clean precheck at
`logs/freqfed_paper_clean_precheck_noproj` observed:

- `freqfed_fallback = 0.0`;
- final clean accuracy drop versus clean FedAvg: `0.014`;
- distance metric: `precomputed_cosine`;
- low-frequency shape: `triangular`;
- low-frequency ratio: `0.5`.

The full 6-round smoke run at `logs/freqfed_paper_smoke` also passed both
FreqFed fallback gates:

- attacked model-replacement smoke fallback: `0.0`;
- clean smoke fallback: `0.0`.

Clean selected ratios can still be low in short smoke runs, so formal results
should report `freqfed_selected_ratio`, `freqfed_rejected_clients`, and
`freqfed_noise_clients` alongside security metrics.

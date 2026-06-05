# FedSec — Federated Security Research Framework

A modular, extensible federated learning framework built on [Flower (flwr)](https://flower.ai/)
for conducting security experiments: evaluating attacks, comparing defenses,
and plugging in your own defense methods.

---

## Project Structure

```
federated_security/
│
├── config/
│   ├── config.yaml             # Master experiment configuration
│   └── config_loader.py        # Typed dataclass config + YAML loader
│
├── data/
│   └── dataset.py              # CIFAR-10 / MNIST / FEMNIST loaders + partitioners
│                               #   Partitions: IID | Non-IID (shards) | Dirichlet
│
├── models/
│   └── model_factory.py        # ResNet-18, LightCNN, MLP + registry
│
├── client/
│   ├── fl_client.py            # FedSecClient (Flower Client + extension hooks)
│   └── fedprox_client.py       # FedProx client (proximal regularisation)
│
├── server/
│   └── fl_server.py            # Server builder + checkpointing
│
├── strategies/
│   └── fed_strategy.py         # FedSecStrategy: FedAvg / FedProx / FedYogi
│                               #   wraps any BaseDefense for robust aggregation
│
├── attacks/
│   └── attack_client.py        # Label flip | Backdoor | Gaussian | Byzantine
│                               #   | Model replacement
│
├── defenses/
│   ├── defense_base.py         # BaseDefense + Krum | TrimmedMean | Median
│   │                           #   | FLTrust | FoolsGold | FedAvg (baseline)
│   └── your_defense.py         # ← Template for YOUR defense
│
├── utils/
│   ├── metrics.py              # Per-round metric tracker + CSV/JSON export
│   └── logger.py               # Centralised logging setup
│
├── experiments/
│   └── sweep.py                # Grid sweep: attacks × defenses
│
├── tests/
│   └── test_framework.py       # pytest suite (partitioner, models, defenses)
│
├── main.py                     # Entry point — Flower simulation runner
└── requirements.txt
```

---

## Quick Start

### 1. Install

```bash
cd federated_security
pip install -r requirements.txt
```

### 2. Run a basic experiment

```bash
# Default: CIFAR-10, ResNet-18, FedAvg, no attack, no defense
python main.py

# With CLI overrides
python main.py \
  --override federation.num_rounds=20 \
  --override dataset.name=mnist \
  --override dataset.partition=non_iid
```

### 3. Run with an attack + defense

```bash
python main.py \
  --override security.attack.enabled=true \
  --override security.attack.type=label_flip \
  --override security.defense.enabled=true \
  --override security.defense.type=krum \
  --experiment label_flip_vs_krum
```

### 4. Run a full sweep

```bash
# Quick smoke test (5 rounds, 2×2 grid)
python experiments/sweep.py --quick

# Full sweep (50 rounds, all attack/defense combinations)
python experiments/sweep.py --rounds 50
```

### 5. Run tests

```bash
pytest tests/ -v
```

---

## Supported Datasets

| Dataset  | Classes | Input Shape | Notes                            |
|----------|---------|-------------|----------------------------------|
| CIFAR-10 | 10      | 3 × 32 × 32 | Auto-downloaded via torchvision  |
| MNIST    | 10      | 1 × 28 × 28 | Auto-downloaded via torchvision  |
| FEMNIST  | 62      | 1 × 28 × 28 | Falls back to EMNIST 'byclass'   |

Set `dataset.name` in `config.yaml` or via `--override dataset.name=mnist`.

---

## Supported Models

| Key        | Architecture           | Best for      |
|------------|------------------------|---------------|
| `resnet18` | ResNet-18 (adapted)    | CIFAR-10      |
| `cnn`      | LightCNN (Conv×2+FC×2) | MNIST/FEMNIST |
| `mlp`      | Fully-connected MLP    | MNIST (fast)  |

---

## Data Partitioning

| Strategy    | Description                                      |
|-------------|--------------------------------------------------|
| `iid`       | Random uniform split (McMahan et al. baseline)   |
| `non_iid`   | Shard-based, 2 classes per client                |
| `dirichlet` | LDA split — set `dirichlet_alpha` (lower=harder) |

---

## Aggregation Strategies

| Key        | Description                              |
|------------|------------------------------------------|
| `fedavg`   | Standard weighted average                |
| `fedprox`  | FedAvg + proximal term (client-side μ)   |
| `fedyogi`  | Server-side Yogi adaptive optimizer      |
| `fedadam`  | Server-side Adam adaptive optimizer      |

---

## Attacks

| Key                | Type          | Description                         |
|--------------------|---------------|-------------------------------------|
| `label_flip`       | Data poison   | Flip source → target label          |
| `backdoor`         | Data poison   | Pixel trigger + target label        |
| `gaussian_noise`   | Model poison  | Add noise to uploaded weights       |
| `byzantine`        | Model poison  | Send random weights                 |
| `model_replacement`| Model poison  | Scale update to hijack global model |

Configure in `config.yaml` under `security.attack`.

---

## Defenses

| Key            | Description                                   |
|----------------|-----------------------------------------------|
| `none`         | No defense (FedAvg baseline)                  |
| `krum`         | Krum / Multi-Krum (Blanchard et al. 2017)     |
| `trimmed_mean` | Coordinate-wise trimmed mean (Yin et al. 2018)|
| `median`       | Coordinate-wise median                        |
| `fltrust`      | FLTrust cosine trust scoring (Cao et al. 2020)|
| `foolsgold`    | Sybil detection via history similarity        |

---

## Adding YOUR Defense

1. Open `defenses/your_defense.py` — fill in `aggregate()`
2. Register it in `defenses/defense_base.py`:
   ```python
   from defenses.your_defense import YourDefense
   DEFENSE_REGISTRY["your_defense"] = YourDefense
   ```
3. Set in config:
   ```yaml
   security:
     defense:
       type: your_defense
       custom_params:
         your_param: 0.5
   ```
4. Run: `python main.py --override security.defense.type=your_defense`

---

## Configuration Reference

All parameters live in `config/config.yaml`. Key sections:

```yaml
federation:
  num_rounds: 50
  num_clients: 10
  clients_per_round: 5

dataset:
  name: cifar10          # cifar10 | mnist | femnist
  partition: iid         # iid | non_iid | dirichlet
  dirichlet_alpha: 0.5

model:
  architecture: resnet18

security:
  attack:
    enabled: false
    type: none            # label_flip | backdoor | gaussian_noise | byzantine
    malicious_fraction: 0.2
  defense:
    enabled: false
    type: none            # krum | trimmed_mean | median | fltrust | foolsgold
```

---

## Citation

If you use this framework in your research, please cite the relevant
baseline papers and the Flower framework:

```
@article{beutel2020flower,
  title={Flower: A Friendly Federated Learning Research Framework},
  author={Beutel, Daniel J and others},
  journal={arXiv:2007.14390}, year={2020}
}
```

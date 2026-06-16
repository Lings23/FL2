# 🛡️ FedSec — Federated Security Research Framework

A highly modular and extensible federated learning research framework built on [Flower (flwr)](https://flower.ai/) and PyTorch. 

**FedSec** is specifically designed for conducting security and privacy experiments in federated environments: simulating adversarial attacks, evaluating robust aggregation defenses, integrating Differential Privacy, and providing a streamlined API to plug in your own defense mechanisms.

---

## ✨ Key Features

- **Extensive Attack Models:** Simulates malicious clients with Data Poisoning (Label Flip, Backdoor, DBA) and Model Poisoning (Byzantine, Gaussian Noise, Model Replacement).
- **Robust Defenses:** Built-in aggregation mechanisms to defend against attacks, including Krum, Trimmed Mean, Median, FLTrust, FoolsGold, and Time Consistency.
- **Advanced FL Strategies:** Out-of-the-box support for `FedAvg`, `FedProx` (with proximal regularization), `FedYogi`, and `FedAdam` (server-side adaptive optimizers).
- **Data Partitioning:** Supports various distribution setups like IID, non-IID (sharded), and Dirichlet (LDA) heterogeneous data splitting.
- **Differential Privacy (DP):** Integrated DP-SGD wrapper for local client training, allowing custom noise multipliers and gradient clipping.
- **Experiment Sweeps:** Comprehensive scripts for automated grid sweeps across multiple attacks, defenses, and configurations.

---

## 📁 Project Structure

```text
FedSec/
├── config/
│   ├── config.yaml             # Master experiment configuration
│   └── config_loader.py        # Typed dataclass config + YAML loader
├── data/
│   └── dataset.py              # CIFAR-10 / MNIST / FEMNIST loaders + partitioners
│                               # (IID | Non-IID | Dirichlet)
├── models/
│   └── model_factory.py        # ResNet-18, LightCNN, MLP + Model registry
├── client/
│   ├── fl_client.py            # FedSecClient (Flower Client wrapper & DP support)
│   └── fedprox_client.py       # FedProx client handling proximal regulation
├── server/
│   └── fl_server.py            # Server evaluation, checkpointing & early stopping
├── strategies/
│   └── fed_strategy.py         # FedSecStrategy: Wraps BaseDefense for robust aggregation
├── attacks/
│   └── attack_client.py        # Attack clients hook implementations (Label flip, Backdoor, etc.)
├── defenses/
│   ├── defense_base.py         # Abstract BaseDefense + Native defense implementations
│   └── your_defense.py         # ← Sandbox Template for YOUR custom defense
├── utils/
│   ├── metrics.py              # Per-round metric tracker + CSV/JSON export
│   └── logger.py               # Centralised logging setup
├── experiments/
│   └── sweep.py                # Grid sweep runner: attacks × defenses
├── tests/
│   └── test_framework.py       # Pytest suite protecting partitioner, models, and defenses
├── main.py                     # Entry point — Flower simulation orchestrator
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Installation

Python 3.8+ is recommended.
```bash
git clone https://github.com/your-username/FedSec.git
cd FedSec
pip install -r requirements.txt
```

### 2. Run a Baseline Experiment

Run a standard federated session (CIFAR-10, ResNet-18, FedAvg, no attacks, no defenses):
```bash
python main.py
```

Provide runtime overrides overriding `config.yaml`:
```bash
python main.py \
  --override federation.num_rounds=20 \
  --override dataset.name=mnist \
  --override dataset.partition=dirichlet \
  --override dataset.dirichlet_alpha=0.5
```

### 3. Simulate Attack & Defense Scenarios

Launch an experiment with **Label-Flipping Attack** targeting 20% of clients, defended by the **Krum** aggregation algorithm.

```bash
python main.py \
  --override security.attack.enabled=true \
  --override security.attack.type=label_flip \
  --override security.defense.enabled=true \
  --override security.defense.type=krum \
  --experiment label_flip_vs_krum
```

Launch the S1-S7 time-consistency defense, which maintains per-client temporal
trust histories and performs soft trust-weighted aggregation:

```bash
python main.py \
  --override security.defense.enabled=true \
  --override security.defense.type=time_consistency \
  --experiment time_consistency_baseline
```

### 4. Automated Grid Sweeps

Easily iterate over multiple configurations to compare defenses against different attacks.
```bash
# Quick smoke test (5 rounds, limited grid)
python experiments/sweep.py --quick

# Full scale sweep (default rounds, exhaustive attack x defense match-ups)
python experiments/sweep.py --rounds 50
```

### 5. Running Tests
```bash
pytest tests/ -v
```

---

## ⚙️ Configuration Reference (`config.yaml`)

Your experiments revolve around `config/config.yaml`. The schema is typed and validated. Here's a brief look at key knobs:

```yaml
federation:
  num_rounds: 50
  num_clients: 10
  clients_per_round: 5

dataset:
  name: cifar10          # cifar10 | mnist | femnist
  partition: dirichlet   # iid | non_iid | dirichlet
  dirichlet_alpha: 0.5   # degree of heterogeneity

model:
  architecture: resnet18 # resnet18 | cnn | mlp

differential_privacy:
  enabled: false
  noise_multiplier: 1.0  # DP noise
  max_grad_norm: 1.0     # Gradient clipping threshold

security:
  attack:
    enabled: true
    type: label_flip     # label_flip | backdoor | dba | gaussian_noise | byzantine | model_replacement
    malicious_fraction: 0.2
  defense:
    enabled: true
    type: krum           # krum | trimmed_mean | median | fltrust | foolsgold | time_consistency | none
```

*(You can override ANY key directly from the CLI via `--override key.subkey=value`)*

---

## 🛠 Adding YOUR Custom Defense

FedSec is designed to make trying out new defense ideas frictionless.

1. Implement your defense logic in `defenses/your_defense.py` by overriding `aggregate()`:
   ```python
   from .defense_base import BaseDefense, UpdateList

   class YourDefense(BaseDefense):
       def aggregate(self, updates: UpdateList):
           # Custom robust aggregation logic over `updates`
           # updates: List of (List[np.ndarray], num_samples)
           # Return: Aggregated global parameters as List[np.ndarray]
           pass
   ```
2. Register your defense in `defenses/defense_base.py`:
   ```python
   from defenses.your_defense import YourDefense
   
   DEFENSE_REGISTRY = {
       "fedavg": FedAvgDefense,
       "krum": KrumDefense,
       # ...
       "your_defense": YourDefense  # <-- Add here
   }
   ```
3. Update configuration and run:
   ```bash
   python main.py --override security.defense.type=your_defense
   ```

---

## 📚 Acknowledgements & Citations

If you utilize this framework in your academic work, please consider citing the underlying **Flower** framework and the specific defense algorithms you analyze:

```bibtex
@article{beutel2020flower,
  title={Flower: A Friendly Federated Learning Research Framework},
  author={Beutel, Daniel J and others},
  journal={arXiv preprint arXiv:2007.14390},
  year={2020}
}
```
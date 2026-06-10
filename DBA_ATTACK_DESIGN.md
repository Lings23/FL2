# DBA Attack Notes And Integration Plan

## 1. DBA Source Reading Notes

Reference repository: `/home/lc/.openclaw/python-workspace/DBA`

Paper/code target: `DBA: Distributed Backdoor Attacks against Federated Learning`.

### Core Idea

DBA is a federated backdoor attack that splits one global trigger into several local sub-triggers. Each malicious client trains only on its assigned local sub-trigger, but after aggregation the global model is expected to respond to the combined global trigger.

This differs from a centralized backdoor attack:

- Centralized backdoor: one attacker stamps the full trigger.
- DBA: multiple attackers stamp disjoint trigger fragments.

The source repository uses this split to make malicious updates less similar to each other and more difficult for robust aggregation defenses to detect.

### Trigger Definition

For image tasks, trigger fragments are configured in YAML files.

MNIST example from `utils/mnist_params.yaml`:

```yaml
trigger_num: 4
0_poison_pattern: [[0, 0], [0, 1], [0, 2], [0, 3]]
1_poison_pattern: [[0, 6], [0, 7], [0, 8], [0, 9]]
2_poison_pattern: [[3, 0], [3, 1], [3, 2], [3, 3]]
3_poison_pattern: [[3, 6], [3, 7], [3, 8], [3, 9]]
```

CIFAR example from `utils/cifar_params.yaml`:

```yaml
trigger_num: 4
0_poison_pattern: [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5]]
1_poison_pattern: [[0, 9], [0, 10], [0, 11], [0, 12], [0, 13], [0, 14]]
2_poison_pattern: [[4, 0], [4, 1], [4, 2], [4, 3], [4, 4], [4, 5]]
3_poison_pattern: [[4, 9], [4, 10], [4, 11], [4, 12], [4, 13], [4, 14]]
```

`trigger_num` is the number of trigger fragments. `N_poison_pattern` is the pixel coordinate list used by adversary index `N`.

### Poisoning Batch Logic

Implemented in `image_helper.py`:

- `get_poison_batch(...)` copies a normal batch into poisoned tensors.
- During training, only the first `poisoning_per_batch` samples in a batch are poisoned.
- During evaluation, all samples are poisoned.
- Poisoned samples are relabeled to `poison_label_swap`.
- `add_pixel_pattern(...)` stamps either:
  - one local fragment when `adversarial_index >= 0`;
  - the full combined trigger when `adversarial_index == -1`.

For CIFAR/TinyImageNet, the trigger value is written into all three channels. For MNIST, it is written into the single channel.

### Malicious Client Assignment

Configured by `adversary_list` in YAML. Example:

```yaml
adversary_list: [41, 73, 51, 74]
```

In `image_train.py`, each selected client checks whether its `agent_name_key` is in `adversary_list`. If yes, the client is mapped to an `adversarial_index` according to its position in `adversary_list`.

Example:

- client `41` -> trigger fragment `0`
- client `73` -> trigger fragment `1`
- client `51` -> trigger fragment `2`
- client `74` -> trigger fragment `3`

If there is only one attacker, the implementation sets `adversarial_index = -1`, which means centralized full-trigger attack rather than distributed DBA.

### Attack Round Scheduling

Each fragment can have its own poisoning rounds:

```yaml
0_poison_epochs: [12]
1_poison_epochs: [14]
2_poison_epochs: [16]
3_poison_epochs: [18]
```

The main loop in `main.py` ensures attackers whose poison round is active are selected into the current training round when `is_random_adversary` is false.

This means DBA can be single-shot and staggered across attackers, or multi-shot by configuring long epoch lists.

### Malicious Local Training

Implemented in `image_train.py`:

1. Copy the current global model into the local model.
2. If the client is malicious and the current round is in its poison schedule:
   - use `poison_lr`;
   - train for `internal_poison_epochs`;
   - call `helper.get_poison_batch(...)`;
   - optimize cross entropy on relabeled poisoned samples.
3. Optionally include a distance regularization term:
   - `loss = alpha_loss * class_loss + (1 - alpha_loss) * distance_loss`
   - in provided configs, `alpha_loss: 1`, so the distance term is disabled.

Benign clients use the normal training path with `internal_epochs`.

### Model Update Scaling

After poisoned local training, if `baseline` is false, the malicious model is scaled:

```python
new_value = target_value + (value - target_value) * scale_weights_poison
```

This is the same update-amplification idea used in model replacement attacks. In the reference configs, `scale_weights_poison` is often `100`.

The submitted local update is then:

```python
local_update = local_model_after_training - last_global_model
```

### Aggregation And Evaluation

The reference implementation supports mean, geometric median, and FoolsGold aggregation.

For mean aggregation, updates are accumulated and applied to the global model by `average_shrink_models(...)`.

Backdoor evaluation has three levels:

- `Mytest_poison`: evaluate the combined global trigger, using `adversarial_index = -1`.
- `Mytest_poison_trigger`: evaluate a trigger fragment by index.
- `Mytest_poison_agent_trigger`: evaluate the fragment assigned to a specific adversarial client.

This is useful because DBA wants both:

- high ASR on the combined trigger;
- visibility into whether individual local fragments are learned.

## 2. Integration Plan For This Project

Current project root: `/home/lc/.openclaw/python-workspace/federated_security`

The current project already has a clean attack extension point:

- `attacks/attack_client.py`
  - attack datasets;
  - malicious client subclasses;
  - `ATTACK_REGISTRY`.
- `client/fl_client.py`
  - calls `on_before_fit(...)` before local training;
  - calls `on_after_fit(...)` before uploading model parameters.
- `config/config_loader.py`
  - typed `AttackConfig`.
- `config/config.yaml`
  - attack configuration.

The request says `/federated_security/attack`, but the actual project directory is `attacks/`. The implementation should follow the existing `attacks/` package unless the project is intentionally renamed.

### Minimal DBA Implementation

Add DBA as a new attack type: `dba`.

Recommended additions:

- `DBADataset`
  - wraps a base dataset;
  - selects a configurable fraction of samples to poison;
  - stamps a client-specific local trigger fragment;
  - relabels poisoned samples to `backdoor_target_label`.
- `DBAClient`
  - subclass of `FedSecClient`;
  - in `on_before_fit(...)`, replace `train_loader` with a `DBADataset`;
  - cache global parameters if update scaling is enabled;
  - in `on_after_fit(...)`, optionally scale the update like model replacement.
- Register in `ATTACK_REGISTRY`:

```python
"dba": DBAClient
```

### Trigger Fragment Assignment

This project currently passes only `client_id` and `attack_cfg` into each malicious client. It also sets `client.is_malicious`.

A simple deterministic assignment can avoid adding extra plumbing:

```python
fragment_index = client_id % dba_trigger_num
```

This gives every malicious client a stable local fragment. It does not exactly preserve the reference repository's `adversary_list` ordering, but it fits this project's randomized malicious-client selection.

If exact reference behavior is required, add `malicious_ids` ordering into the client factory or attack config and compute:

```python
fragment_index = sorted_malicious_ids.index(client_id) % dba_trigger_num
```

The minimal version should use `client_id % dba_trigger_num` because it is small, deterministic, and compatible with Flower simulation serialization.

### Trigger Pattern Defaults

DBA needs configurable local trigger fragments. Add these `AttackConfig` fields:

- `poison_fraction: float = 0.1`
- `trigger_size: int = 3`
- `trigger_value: float = 1.0`
- `dba_trigger_num: int = 4`
- `dba_gap: int = 3`
- `dba_base_row: int = 0`
- `dba_base_col: int = 0`
- `dba_scale_update: bool = true`
- `dba_boost_factor: float = 10.0`

For a first version, generate fragments procedurally rather than storing long coordinate lists in YAML.

Example fragment layout:

- fragment 0: top-left horizontal block
- fragment 1: top-right horizontal block with gap
- fragment 2: lower-left horizontal block
- fragment 3: lower-right horizontal block with gap

The helper should clamp generated coordinates to image height/width so it works for MNIST `1x28x28` and CIFAR `3x32x32`.

### Update Scaling

The current `ModelReplacementClient` already does:

```python
global + boost_factor * (local - global)
```

DBA can reuse the same logic in `on_after_fit(...)`.

For minimal implementation:

- cache `server_params` in `on_before_fit(...)`;
- after local poisoned training, return scaled parameters when `dba_scale_update` is true;
- otherwise return normal trained parameters.

The default boost factor should probably be lower than the reference repo's `100` because this project has robust defenses and a smaller/default `clients_per_round`. Start with `10.0`, matching the current `ModelReplacementClient`.

### Round Scheduling

The reference DBA has per-fragment poison rounds. This project currently does not pass `server_round` into attack clients except through Flower `config`.

`FedSecStrategy._default_fit_config(...)` likely controls fit config. If it includes `server_round`, DBA can use it directly. If not, add `server_round` to fit config.

Minimal version:

- poison every selected malicious-client round.

Optional enhancement:

- add `dba_poison_rounds: List[int] | None`;
- add `dba_stagger_rounds: bool`;
- only poison if `server_round` matches the configured schedule.

### Evaluation Plan

The current server evaluator only measures clean accuracy. For DBA experiments, add optional ASR evaluation later:

- Build a poisoned test loader or apply trigger transformation on the fly.
- Evaluate full combined trigger ASR.
- Optionally evaluate each local fragment ASR.

Minimal implementation can postpone ASR integration and rely on clean metrics plus follow-up tests. A complete DBA experiment should add:

- `backdoor_asr`
- `dba_fragment_0_asr`
- `dba_fragment_1_asr`
- `dba_fragment_2_asr`
- `dba_fragment_3_asr`

### Files To Change

Minimal code changes:

- `attacks/attack_client.py`
  - add `DBADataset`;
  - add trigger-coordinate helper;
  - add `DBAClient`;
  - register `dba`.
- `config/config_loader.py`
  - extend `AttackConfig` with DBA parameters.
- `config/config.yaml`
  - document `dba` as a valid attack type;
  - add DBA-specific defaults.
- `README.md`
  - include DBA in attack list and example command.
- `tests/test_framework.py`
  - add test for local fragment stamping;
  - add test that different client/fragment indices stamp different pixels.

Optional later changes:

- `server/fl_server.py`
  - add full-trigger ASR and fragment ASR evaluation.
- `strategies/fed_strategy.py`
  - ensure `server_round` is passed into fit config for scheduled DBA.
- `experiments/sweep.py`
  - add `dba` to default attack sweep.

### Recommended Implementation Order

1. Add typed config fields and YAML defaults.
2. Implement trigger-fragment generation as a small pure function.
3. Implement `DBADataset`.
4. Implement `DBAClient` with optional update scaling.
5. Register attack type `dba`.
6. Add focused unit tests for trigger placement and registry lookup.
7. Run tests.
8. Add ASR evaluation only after the client-side attack path is stable.

### Example Usage After Implementation

```bash
python main.py \
  --override security.attack.enabled=true \
  --override security.attack.type=dba \
  --override security.attack.malicious_fraction=0.4 \
  --override security.attack.backdoor_target_label=2 \
  --override security.attack.dba_trigger_num=4 \
  --override security.attack.dba_scale_update=true \
  --experiment dba_fedavg
```

### Design Decision

Do not copy the full DBA reference repository into this project. The reference code owns its own training loop, model wrappers, Visdom logging, CSV state, and dataset helpers. This project already has Flower-based client/server abstractions, so DBA should be implemented as a compact attack client that preserves the research behavior while fitting the local architecture.

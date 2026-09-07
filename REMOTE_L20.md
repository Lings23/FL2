# L20 experiment environment

The deployment uses `/root/FL2`, branch `codex/periodic-attack-defenses`,
and a Python 3.11 virtual environment at `/root/FL2/.venv`.

## Start a session

```bash
cd /root/FL2
source .venv/bin/activate
git pull --ff-only origin codex/periodic-attack-defenses
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Recreate the environment

Install Python 3.11 and Git using the operating system package manager first.
Keep the system Python installation unchanged.

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip uv
.venv/bin/uv pip install --python .venv/bin/python \
  -r requirements.txt -c requirements-l20.txt --torch-backend cu126
.venv/bin/python -m pip check
```

If PyPI downloads are slow, add
`--index-url https://mirrors.aliyun.com/pypi/simple/` to the `uv pip install`
command. `--torch-backend cu126` selects the CUDA PyTorch distribution.
The version constraints match the main packages in the local RTX 4060
environment; this is not a complete transitive dependency lock.

## Experiment commands

Use the Python CLI on Linux. The existing `.ps1` wrappers use Windows paths
and are not Linux launch scripts.

For an existing experiment command, keep its protocol, attack, defense, seed,
and training arguments. Start resource tuning with:

```text
--batch-size 48 --ray-client-num-cpus 1 --ray-client-num-gpus 0.2
```

The deployed server has 8 vCPUs and one L20. A GPU quota of 0.2 allows up to
five client actors by GPU quota, subject to other resource constraints.
It does not impose a GPU memory limit. Compare actual round times before
increasing concurrency. Changing batch size changes the training protocol
and may require new RTC calibration.

CIFAR-10 files belong in `data/huggingface_cifar10/`. They are intentionally
outside Git, as are `.venv`, training logs, and checkpoints.

## Short installation check

This uses capped data and one local epoch only to validate the installation;
it is not a formal experiment or a speed benchmark.

```bash
python main.py --experiment l20_environment_smoke \
  --override federation.num_rounds=1 \
  --override federation.num_clients=4 \
  --override federation.clients_per_round=2 \
  --override federation.min_fit_clients=2 \
  --override federation.min_available_clients=4 \
  --override federation.min_evaluate_clients=2 \
  --override federation.max_client_samples=48 \
  --override client.local_epochs=1 \
  --override evaluation.max_test_samples=100 \
  --override ray.client_num_gpus=0.2 \
  --override ray.object_store_memory_mb=3072
```

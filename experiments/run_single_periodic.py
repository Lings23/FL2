import argparse
from pathlib import Path

from periodic_attack import build_matrix, _run_specs


def main():
    spec = next(
        x for x in build_matrix("untargeted")
        if x["attack"] == "gaussian_noise"
        and x["period"] == "short_1_1"
        and x["malicious_fraction"] == 0.2
        and x["seed"] == 42
        and x["defense"] == "fedavg"
    )

    output = Path("logs/periodic_attack_formal_retry")
    rounds_dir = output / "rounds"
    rounds_dir.mkdir(parents=True, exist_ok=True)
    args = argparse.Namespace(
        config="config/config.yaml",
        output=str(output),
        mode="untargeted",
        smoke=False,
        rerun=True,
    )

    _run_specs([spec], args, output, rounds_dir)


if __name__ == "__main__":
    main()

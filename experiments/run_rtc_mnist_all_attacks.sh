#!/usr/bin/env bash
set -euo pipefail
workspace="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_path="${RTC_PYTHON:-python}"
args=()
while (( $# > 0 )); do
    case "$1" in
        --python)
            (( $# >= 2 )) || { echo 'Missing --python value' >&2; exit 2; }
            python_path="$2"; shift 2 ;;
        *) args+=("$1"); shift ;;
    esac
done
cd -- "$workspace"
exec "$python_path" -B -m experiments.rtc_mnist_all_attacks "${args[@]}"

#!/usr/bin/env bash
set -euo pipefail
workspace="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_path="${RTC_PYTHON:-}"
if [[ -z "$python_path" ]]; then
    if [[ -x "$workspace/.venv/bin/python" ]]; then python_path="$workspace/.venv/bin/python";
    else python_path="$(command -v python)"; fi
fi
args=()
mode=""
data_set=""
while (( $# > 0 )); do
    case "$1" in
        --execute|--analyze)
            [[ -z "$mode" ]] || { printf '%s\n' 'Choose one execution mode.' >&2; exit 2; }
            mode="$1"; args+=("$1"); shift ;;
        --python|--output|--data-dir)
            (( $# >= 2 )) || { printf 'Missing value for %s\n' "$1" >&2; exit 2; }
            if [[ "$1" == --python ]]; then python_path="$2";
            else args+=("$1" "$2"); fi
            if [[ "$1" == --data-dir ]]; then data_set=1; fi
            shift 2 ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done
[[ -z "$data_set" || -z "$mode" ]] || { printf '%s\n' '--data-dir is preparation-only.' >&2; exit 2; }
[[ -x "$python_path" ]] || { printf 'Python not executable: %s\n' "$python_path" >&2; exit 2; }
cd -- "$workspace"
exec "$python_path" -B -m experiments.rtc_g2_integration "${args[@]}"

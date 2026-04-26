#!/bin/bash
#SBATCH --job-name=prag_run
#SBATCH --nodelist=server2
#SBATCH --output=logs/prag_%j.out
#SBATCH --error=logs/prag_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

mkdir -p logs

CONFIG_PATH="${1:-${CONFIG_PATH:-}}"
if [[ -z "${CONFIG_PATH}" ]]; then
    echo "Usage: sbatch run_prag_config.sh <config-script>" >&2
    echo "Example: sbatch run_prag_config.sh configs/hotpotqa_llama3.2-1b-instruct.sh" >&2
    echo "" >&2
    echo "Available config scripts:" >&2
    find configs -maxdepth 1 -type f -name '*.sh' | sort >&2
    exit 1
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
    if [[ -f "${PROJECT_DIR}/${CONFIG_PATH}" ]]; then
        CONFIG_PATH="${PROJECT_DIR}/${CONFIG_PATH}"
    else
        echo "Config script not found: ${CONFIG_PATH}" >&2
        exit 1
    fi
fi

if [[ ! -f ".venv/bin/activate" ]]; then
    echo "Virtual environment not found: ${PROJECT_DIR}/.venv" >&2
    exit 1
fi

source .venv/bin/activate
export PYTHONUNBUFFERED=1

echo "Project dir: ${PROJECT_DIR}"
echo "Config: ${CONFIG_PATH}"
echo "Python path: $(which python)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "Started at: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo
echo "===== Config Contents ====="
sed -n '1,200p' "${CONFIG_PATH}"
echo "==========================="
echo

bash "${CONFIG_PATH}"

echo
echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S %Z')"

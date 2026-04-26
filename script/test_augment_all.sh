#!/bin/bash
#SBATCH --job-name=augment_all
#SBATCH --nodelist=server2
#SBATCH --output=logs/augment_all_%j.out
#SBATCH --error=logs/augment_all_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

mkdir -p logs
source .venv/bin/activate
export PYTHONUNBUFFERED=1

SAMPLE="${SAMPLE:-300}"
TOPK="${TOPK:-3}"

MODELS=(
    "llama3.2-1b-instruct"
    "qwen2.5-1.5b-instruct"
    "llama3-8b-instruct"
)

DATASETS=(
    "2wikimultihopqa"
    "hotpotqa"
    "popqa"
    "complexwebquestions"
)

get_data_path() {
    case "$1" in
        2wikimultihopqa) echo "data/2wikimultihopqa/" ;;
        hotpotqa) echo "data/hotpotqa/" ;;
        popqa) echo "data/popqa/" ;;
        complexwebquestions) echo "data/complexwebquestions/" ;;
        *)
            echo "Unknown dataset: $1" >&2
            return 1
            ;;
    esac
}

print_header() {
    echo "Project dir: ${PROJECT_DIR}"
    echo "Python path: $(which python)"
    echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
    echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"
    echo "Sample: ${SAMPLE}"
    echo "Topk: ${TOPK}"
    if command -v hf >/dev/null 2>&1; then
        echo "HF auth:"
        hf auth whoami || true
    fi
    echo "Checking Elasticsearch..."
    curl -fsS http://localhost:9200 >/dev/null
    echo "Elasticsearch is reachable at http://localhost:9200"
}

run_augment() {
    local model="$1"
    local dataset="$2"
    local data_path

    data_path="$(get_data_path "${dataset}")"

    if [[ ! -d "${data_path}" ]]; then
        echo "Missing data path: ${data_path}" >&2
        return 1
    fi

    echo "============================================================"
    echo "Starting augmentation"
    echo "Model: ${model}"
    echo "Dataset: ${dataset}"
    echo "Data path: ${data_path}"
    echo "============================================================"

    if python src/augment.py \
        --model_name "${model}" \
        --dataset "${dataset}" \
        --data_path "${data_path}" \
        --sample "${SAMPLE}" \
        --topk "${TOPK}"; then
        echo "Completed ${model}:${dataset}"
        return 0
    fi

    echo "Failed ${model}:${dataset}" >&2
    return 1
}

print_header

declare -a SUCCEEDED=()
declare -a FAILED=()

for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        if run_augment "${model}" "${dataset}"; then
            SUCCEEDED+=("${model}:${dataset}")
        else
            FAILED+=("${model}:${dataset}")
        fi
    done
done

echo "============================================================"
echo "Augmentation summary"
echo "Succeeded: ${#SUCCEEDED[@]}"
for item in "${SUCCEEDED[@]}"; do
    echo "  ${item}"
done
echo "Failed: ${#FAILED[@]}"
for item in "${FAILED[@]}"; do
    echo "  ${item}"
done

if ((${#FAILED[@]} > 0)); then
    exit 1
fi

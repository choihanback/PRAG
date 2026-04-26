#!/bin/bash
#SBATCH --job-name=prag_l8b_v2
#SBATCH --nodelist=server2
#SBATCH --output=logs/prag_l8b_v2_%j.out
#SBATCH --error=logs/prag_l8b_v2_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=72:00:00

set -euo pipefail

mkdir -p logs

run_inference() {
    local dataset="$1"
    local data_type="$2"
    local num_train_epochs="$3"
    local max_new_tokens="$4"
    shift 4

    echo "============================================================"
    echo "Running llama3-8b vanilla inference: dataset=${dataset} data_type=${data_type} epochs=${num_train_epochs}"
    echo "Started at: $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "============================================================"

    python3 src/inference.py \
        --model_name=llama3-8b-instruct \
        --dataset="${dataset}" \
        --data_type="${data_type}" \
        --sample=300 \
        --num_train_epochs="${num_train_epochs}" \
        --learning_rate=0.0003 \
        --lora_rank=2 \
        --lora_alpha=32 \
        --max_new_tokens="${max_new_tokens}" \
        --inference_method=vanilla \
        "$@"
}

# Balanced bucket: 2 CoT shards + 1 light shard
run_inference 2wikimultihopqa comparison 1 128 --with_cot
run_inference hotpotqa comparison 1 128 --with_cot
run_inference popqa total 2 20

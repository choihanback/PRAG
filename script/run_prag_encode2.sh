#!/bin/bash
#SBATCH --job-name=prag_encode2
#SBATCH --nodelist=server2
#SBATCH --output=logs/prag_encode2_%j.out
#SBATCH --error=logs/prag_encode2_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=72:00:00


run_encode() {
    local model_name="$1"
    local dataset="$2"
    local num_train_epochs="$3"
    shift 3

    echo "============================================================"
    echo "Running encode: model=${model_name} dataset=${dataset} epochs=${num_train_epochs}"
    echo "Started at: $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "============================================================"

    python3 src/encode.py \
        --model_name="${model_name}" \
        --dataset="${dataset}" \
        --sample=300 \
        --per_device_train_batch_size=1 \
        --num_train_epochs="${num_train_epochs}" \
        --learning_rate=0.0003 \
        --lora_rank=2 \
        --lora_alpha=32 \
        "$@"
}

# Estimated runtime: ~13.6 hours
run_encode qwen2.5-1.5b-instruct 2wikimultihopqa 1 --with_cot
run_encode qwen2.5-1.5b-instruct popqa 2

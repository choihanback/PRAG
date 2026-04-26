#!/bin/bash
#SBATCH --job-name=prag_inference_2
#SBATCH --nodelist=server2
#SBATCH --output=logs/prag_inference_2_%j.out
#SBATCH --error=logs/prag_inference_2_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=72:00:00


run_inference() {
    local model_name="$1"
    local dataset="$2"
    local num_train_epochs="$3"
    local inference_method="$4"
    local max_new_tokens="$5"
    local data_type="$6"
    shift 6

    local cmd=(
        python3 src/inference.py
        --model_name="${model_name}"
        --dataset="${dataset}"
        --sample=300
        --num_train_epochs="${num_train_epochs}"
        --learning_rate=0.0003
        --lora_rank=2
        --lora_alpha=32
        --max_new_tokens="${max_new_tokens}"
        --inference_method="${inference_method}"
    )
    if [[ -n "${data_type}" ]]; then
        cmd+=(--data_type="${data_type}")
    fi
    cmd+=("$@")

    echo "============================================================"
    echo "Running inference: model=${model_name} dataset=${dataset} data_type=${data_type:-all} method=${inference_method} epochs=${num_train_epochs}"
    echo "Started at: $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "============================================================"

    "${cmd[@]}"
}

# Extra llama3-8b-instruct modes: 2wikimultihopqa heavy bucket
run_inference llama3-8b-instruct 2wikimultihopqa 1 icl 128 "" --with_cot
run_inference llama3-8b-instruct 2wikimultihopqa 1 prag 128 "" --with_cot
run_inference llama3-8b-instruct 2wikimultihopqa 1 combine 128 "" --with_cot

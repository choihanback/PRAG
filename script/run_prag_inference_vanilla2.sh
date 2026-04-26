#!/bin/bash
#SBATCH --job-name=prag_vanilla2
#SBATCH --nodelist=server2
#SBATCH --output=logs/prag_vanilla2_%j.out
#SBATCH --error=logs/prag_vanilla2_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=72:00:00


run_inference() {
    local model_name="$1"
    local dataset="$2"
    local num_train_epochs="$3"
    local max_new_tokens="$4"
    shift 4

    echo "============================================================"
    echo "Running vanilla inference: model=${model_name} dataset=${dataset} epochs=${num_train_epochs}"
    echo "Started at: $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "============================================================"

    python3 src/inference.py \
        --model_name="${model_name}" \
        --dataset="${dataset}" \
        --sample=300 \
        --num_train_epochs="${num_train_epochs}" \
        --learning_rate=0.0003 \
        --lora_rank=2 \
        --lora_alpha=32 \
        --max_new_tokens="${max_new_tokens}" \
        --inference_method=vanilla \
        "$@"
}

run_inference qwen2.5-1.5b-instruct hotpotqa 2 128 --with_cot
run_inference llama3.2-1b-instruct 2wikimultihopqa 1 128 --with_cot
run_inference qwen2.5-1.5b-instruct complexwebquestions 1 20
run_inference llama3.2-1b-instruct popqa 2 20

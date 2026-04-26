#!/bin/bash
#SBATCH --job-name=prag_inference2
#SBATCH --nodelist=server2
#SBATCH --output=logs/prag_inference2_%j.out
#SBATCH --error=logs/prag_inference2_%j.err
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
    shift 5

    echo "============================================================"
    echo "Running inference: model=${model_name} dataset=${dataset} method=${inference_method} epochs=${num_train_epochs}"
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
        --inference_method="${inference_method}" \
        "$@"
}

# Estimated runtime: ~4.0-4.8 hours
run_inference qwen2.5-1.5b-instruct hotpotqa 2 icl 128 --with_cot
run_inference qwen2.5-1.5b-instruct hotpotqa 2 prag 128 --with_cot
run_inference qwen2.5-1.5b-instruct hotpotqa 2 combine 128 --with_cot

run_inference llama3.2-1b-instruct 2wikimultihopqa 1 icl 128 --with_cot
run_inference llama3.2-1b-instruct 2wikimultihopqa 1 prag 128 --with_cot
run_inference llama3.2-1b-instruct 2wikimultihopqa 1 combine 128 --with_cot

run_inference qwen2.5-1.5b-instruct complexwebquestions 1 icl 20
run_inference qwen2.5-1.5b-instruct complexwebquestions 1 prag 20
run_inference qwen2.5-1.5b-instruct complexwebquestions 1 combine 20

run_inference llama3.2-1b-instruct popqa 2 icl 20
run_inference llama3.2-1b-instruct popqa 2 prag 20
run_inference llama3.2-1b-instruct popqa 2 combine 20

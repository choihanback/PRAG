#!/bin/bash
#SBATCH --job-name=prag_encode_8b_1
#SBATCH --nodelist=server2
#SBATCH --output=logs/prag_encode_8b_1_%j.out
#SBATCH --error=logs/prag_encode_8b_1_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=72:00:00


echo "Allocated GPUs from Slurm: ${CUDA_VISIBLE_DEVICES:-${SLURM_JOB_GPUS:-unknown}}"

run_encode() {
    local model_name="$1"
    local dataset="$2"
    local num_train_epochs="$3"
    shift 3

    echo "============================================================"
    echo "Running encode: model=${model_name} dataset=${dataset} epochs=${num_train_epochs}"
    echo "Started at: $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "============================================================"

    PRAG_LLAMA8B_ENCODE_MODE=single_gpu \
    .venv/bin/python src/encode.py \
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

run_encode llama3-8b-instruct 2wikimultihopqa 1 --data_type inference --with_cot
run_encode llama3-8b-instruct 2wikimultihopqa 1 --data_type comparison --with_cot
run_encode llama3-8b-instruct hotpotqa 1 --data_type bridge --with_cot

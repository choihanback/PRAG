#!/bin/bash
#SBATCH --job-name=prag_encode8b
#SBATCH --nodelist=server2
#SBATCH --output=logs/prag_encode8b_%j.out
#SBATCH --error=logs/prag_encode8b_%j.err
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=72:00:00


ALLOCATED_GPU_CSV="${CUDA_VISIBLE_DEVICES:-${SLURM_JOB_GPUS:-}}"
if [[ -z "${ALLOCATED_GPU_CSV}" ]]; then
    echo "Could not determine allocated GPUs from Slurm."
    exit 1
fi

IFS=',' read -r -a ALLOCATED_GPUS <<< "${ALLOCATED_GPU_CSV}"
if [[ "${#ALLOCATED_GPUS[@]}" -lt 3 ]]; then
    echo "Expected 3 allocated GPUs, got: ${ALLOCATED_GPU_CSV}"
    exit 1
fi

echo "Allocated GPUs from Slurm: ${ALLOCATED_GPU_CSV}"

run_encode() {
    local gpu_id="$1"
    local worker_name="$2"
    local model_name="$3"
    local dataset="$4"
    local num_train_epochs="$5"
    shift 5

    echo "============================================================"
    echo "[${worker_name}] Running encode: gpu=${gpu_id} model=${model_name} dataset=${dataset} epochs=${num_train_epochs}"
    echo "[${worker_name}] Started at: $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES="${gpu_id}" \
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

worker0() {
    run_encode "${ALLOCATED_GPUS[0]}" worker0 llama3-8b-instruct 2wikimultihopqa 1 --data_type inference --with_cot
    run_encode "${ALLOCATED_GPUS[0]}" worker0 llama3-8b-instruct 2wikimultihopqa 1 --data_type comparison --with_cot
    run_encode "${ALLOCATED_GPUS[0]}" worker0 llama3-8b-instruct hotpotqa 1 --data_type bridge --with_cot
}

worker1() {
    run_encode "${ALLOCATED_GPUS[1]}" worker1 llama3-8b-instruct 2wikimultihopqa 1 --data_type bridge_comparison --with_cot
    run_encode "${ALLOCATED_GPUS[1]}" worker1 llama3-8b-instruct 2wikimultihopqa 1 --data_type compositional --with_cot
    run_encode "${ALLOCATED_GPUS[1]}" worker1 llama3-8b-instruct hotpotqa 1 --data_type comparison --with_cot
}

worker2() {
    run_encode "${ALLOCATED_GPUS[2]}" worker2 llama3-8b-instruct complexwebquestions 1
    run_encode "${ALLOCATED_GPUS[2]}" worker2 llama3-8b-instruct popqa 2
}

trap 'jobs -pr | xargs -r kill || true' INT TERM

worker0 >"logs/prag_encode8b_worker0_${SLURM_JOB_ID}.out" 2>"logs/prag_encode8b_worker0_${SLURM_JOB_ID}.err" &
pid0=$!
worker1 >"logs/prag_encode8b_worker1_${SLURM_JOB_ID}.out" 2>"logs/prag_encode8b_worker1_${SLURM_JOB_ID}.err" &
pid1=$!
worker2 >"logs/prag_encode8b_worker2_${SLURM_JOB_ID}.out" 2>"logs/prag_encode8b_worker2_${SLURM_JOB_ID}.err" &
pid2=$!

wait "${pid0}" "${pid1}" "${pid2}"

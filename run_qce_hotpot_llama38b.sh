#!/bin/bash
#SBATCH --job-name=qce_hotpot_l8b
#SBATCH --nodelist=server2
#SBATCH --output=logs/qce_hotpot_l8b_%j.out
#SBATCH --error=logs/qce_hotpot_l8b_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=72:00:00

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

bash run_qce_config.sh configs/hotpotqa_qce_param_llama3-8b-instruct.sh

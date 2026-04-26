#!/bin/bash
#SBATCH --job-name=wiki_parameterization
#SBATCH --nodelist=server2
#SBATCH --output=logs/training_%j.out
#SBATCH --error=logs/training_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00

set -euo pipefail

# 프로젝트 디렉토리로 이동
cd /mnt/raid5/choihb/PRAG
# UV 가상환경 활성화
source .venv/bin/activate
export ES_HOST="${ES_HOST:-${ELASTICSEARCH_URL:-http://localhost:9200}}"
export ELASTICSEARCH_URL="${ELASTICSEARCH_URL:-${ES_HOST}}"

# 환경 확인
echo "Python path: $(which python)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "Elasticsearch host: ${ES_HOST}"

source ./ensure_elasticsearch.sh
ensure_elasticsearch

# 실제 작업 실행
python3 prep_elastic.py --data_path data/dpr/psgs_w100.tsv --index_name wiki --es_host "${ES_HOST}" # prep_elastic.py 실행할 때만 활성화

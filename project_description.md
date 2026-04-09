# Project Description

## 한 줄 요약

이 프로젝트는 논문 **Parametric RAG (PRAG)** 의 공식 구현체로, 검색된 문서를 프롬프트에 길게 붙이는 대신 **문서별 LoRA 어댑터**로 바꿔 두고, 추론 시 필요한 문서들의 LoRA를 합쳐 답을 생성하는 실험 코드베이스입니다.

가장 중요한 핵심 흐름은 아래 3단계입니다.

1. `src/augment.py`: 질문과 관련된 문서를 BM25로 찾고, 문서 재작성문과 QA를 생성한다. `src/augment.py`가 질문마다 top-k passage를 검색해서, 문서 재작성문과 QA 생성 후, data["augment"]에 저장
2. `src/encode.py`: 생성된 augmentation 데이터를 사용해 문서별 LoRA를 학습한다.
3. `src/inference.py`: 질문에 해당하는 여러 문서 LoRA를 merge해서 최종 답을 생성하고 평가한다.

즉, 이 저장소는 단순한 추론 코드가 아니라 아래 전체 실험 파이프라인을 포함합니다.

```text
원본 QA 데이터 + Wikipedia BM25 인덱스
    -> Augmentation
    -> 문서별 LoRA 학습
    -> LoRA merge 기반 추론
    -> EM / F1 평가
```

---

## 1. 이 프로젝트를 어떻게 이해하면 좋은가

처음 보면 파일 수가 많아 보이지만, 실제로는 아래 구조로 이해하면 거의 다 풀립니다.

- `data/`: 원본 QA 데이터와 BM25 검색용 위키 문서
- `data_aug/`: 질문별 검색 결과 + 문서 재작성 + 문서 기반 QA
- `offline/`: 문서별로 학습된 LoRA 어댑터
- `output/`: 최종 추론 결과와 평가 점수
- `src/`: 위 과정을 수행하는 핵심 코드

즉, **데이터가 어떻게 이동하는지**만 보면 전체 구조가 매우 명확합니다.

```text
data/
  -> src/augment.py
  -> data_aug/
  -> src/encode.py
  -> offline/
  -> src/inference.py
  -> output/
```

---

## 2. 프로젝트의 핵심 아이디어

일반적인 RAG는 검색한 문서를 추론 시점에 프롬프트에 넣습니다. 문서가 많아질수록 컨텍스트가 길어지고 비용과 복잡도가 커집니다.

이 프로젝트의 아이디어는 다릅니다.

- 검색된 문서를 미리 **LoRA 파라미터 형태로 저장**합니다.
- 추론 시점에는 문서 본문 전체를 다 넣는 대신, 관련 문서 LoRA를 **합쳐서 모델 내부에 주입**합니다.
- 필요에 따라 문서 텍스트를 같이 넣는 `combine` 모드도 지원합니다.

정리하면:

- `icl`: 일반적인 문맥 기반 RAG
- `prag`: 문맥 없이 LoRA만 사용
- `combine`: 문맥 + LoRA를 함께 사용

---

## 3. 현재 루트 디렉터리 구조

현재 작업 트리에서 중요한 항목은 아래와 같습니다.

```text
PRAG/
├── README.md
├── project_description.md
├── all_prompt.md
├── pyproject.toml
├── requirements.txt
├── uv.lock
├── .python-version
├── assets/
├── configs/
├── data/
├── data_aug/
├── logs/
├── src/
├── prep_elastic.py
├── prep_elastic.sh
├── ensure_elasticsearch.sh
├── test.sh
├── test2.sh
├── test3.sh
└── test_augment_all.sh
```

중요한 점은, `offline/`, `output/`, `warmup/` 같은 디렉터리는 **실험 실행 후 생성되는 산출물 폴더**라서 현재 루트에 없을 수도 있다는 것입니다.

---

## 4. 루트 파일/폴더별 역할

| 경로 | 역할 |
| --- | --- |
| `README.md` | 공식 사용 문서. 환경 구성, 데이터 준비, augmentation/encoding/inference 실행법이 들어 있음 |
| `project_description.md` | 현재 문서. 구조와 흐름을 빠르게 이해하기 위한 온보딩 문서 |
| `all_prompt.md` | 실험에 사용된 프롬프트들을 모아둔 참고 문서 |
| `pyproject.toml` | 프로젝트 메타데이터와 의존성 정의 |
| `requirements.txt` | `pip install -r requirements.txt` 용 의존성 목록 |
| `uv.lock` | `uv` 기반 환경 관리 흔적. 현재 프로젝트는 `pip`와 `uv` 흔적이 함께 존재 |
| `.python-version` | Python 버전 힌트 파일 |
| `assets/` | README에 들어가는 그림, GIF 등 시각 자료 |
| `configs/` | 논문 실험 재현용 명령 모음. 주로 `encode.py` + `inference.py` 조합 |
| `data/` | 원본 데이터셋과 Wikipedia/BM25 인덱싱 관련 리소스 |
| `data_aug/` | augmentation 결과 저장 폴더 |
| `logs/` | SLURM 배치 실행 로그 저장 폴더 |
| `src/` | 핵심 연구 코드 |
| `prep_elastic.py` | DPR Wikipedia TSV를 Elasticsearch 인덱스로 구축 |
| `prep_elastic.sh` | 위 인덱싱 작업용 SLURM 스크립트 |
| `ensure_elasticsearch.sh` | 로컬 Elasticsearch가 없으면 자동으로 띄우는 보조 스크립트 |
| `test.sh`, `test2.sh`, `test3.sh`, `test_augment_all.sh` | 이름은 `test`지만 실제로는 실험/증강용 배치 실행 스크립트 |

---

## 5. `src/` 내부 구조

```text
src/
├── augment.py
├── encode.py
├── inference.py
├── get_warmup_data.py
├── warmup_lora.py
├── utils.py
├── prompt_template.py
├── root_dir_path.py
├── fewshot/
│   ├── 2wikimultihopqa.json
│   └── hotpotqa.json
└── retrieve/
    ├── retriever.py
    ├── readme.md
    └── beir/
```

### `src/augment.py`

역할:

- 데이터셋 로드
- 질문별 BM25 검색
- passage rewrite 생성
- passage 기반 QA 생성
- 결과를 `data_aug/` 에 저장

핵심 포인트:

- 데이터셋별 로더가 이 파일에 함께 들어 있습니다.
  - `load_2wikimultihopqa`
  - `load_hotpotqa`
  - `load_popqa`
  - `load_complexwebquestions`
  - `load_default_format_data`
- 검색은 `retrieve.retriever.bm25_retrieve()` 를 호출합니다.
- 실제로는 `topk + 10` 개를 더 넉넉히 가져온 뒤, QA 생성에 실패한 passage를 건너뛰고 최종 `topk` 개를 채웁니다.
- 출력 JSON에는 단순 passage뿐 아니라 rewrite와 synthetic QA까지 함께 들어갑니다.

### `src/encode.py`

역할:

- augmentation 데이터를 읽어서
- 질문별 passage마다
- 별도의 LoRA adapter를 학습하고 저장

핵심 포인트:

- LoRA target module은 `down_proj`, `gate_proj`, `up_proj`
- 각 passage가 **독립된 adapter 디렉터리**로 저장됩니다.
- 동일한 `lora_rank`, `lora_alpha` 조합으로 처음 실행할 때 `base_weight/` 를 자동 생성합니다.
- `with_cot` 옵션이 켜지면 few-shot CoT 프롬프트를 사용합니다.

### `src/inference.py`

역할:

- augmentation 데이터를 다시 읽고
- 각 질문의 관련 passage 수만큼 LoRA adapter를 불러오고
- adapter들을 merge한 뒤
- 답변을 생성하고 평가

핵심 포인트:

- `icl`: passage 텍스트만 넣는 방식
- `prag`: merge된 LoRA만 사용
- `combine`: passage 텍스트와 merge된 LoRA를 함께 사용
- 결과는 `predict.json`, `result.txt`, `config.json` 으로 저장됩니다.
- `read_complete()` 를 이용해 중간에 끊긴 실행도 이어서 쓸 수 있게 구성되어 있습니다.

### `src/utils.py`

역할:

- 모델 로딩
- augmentation 결과 로딩
- generation
- 평가

핵심 함수:

- `get_model()`: Hugging Face 모델 로드
- `load_data()`: `data_aug/` 결과 파일 읽기
- `model_generate()`: augmentation용 일반 생성
- `predict()`: inference용 답변 생성
- `evaluate()`: EM/F1/Precision/Recall 계산

### `src/prompt_template.py`

역할:

- 질문, passage, answer를 모델 입력 포맷으로 변환
- CoT용 few-shot 예시 주입

핵심 포인트:

- `get_fewshot(dataset)` 이 `src/fewshot/` 의 예시를 로드합니다.
- `get_prompt()` 가 실제 chat template 기반 입력 토큰을 만듭니다.

### `src/root_dir_path.py`

역할:

- 프로젝트 루트 경로를 하드코딩합니다.

현재 값:

```python
ROOT_DIR = "/mnt/raid5/choihb/PRAG"
```

즉, 다른 머신이나 다른 경로로 옮기면 이 파일을 가장 먼저 확인해야 합니다.

### `src/get_warmup_data.py`

역할:

- warm-up LoRA 학습용 데이터를 만듭니다.
- 테스트 샘플과 겹치지 않도록 뒤쪽 데이터에서 학습용 샘플을 뽑는 로직이 들어 있습니다.

### `src/warmup_lora.py`

역할:

- warm-up용 초기 LoRA를 학습합니다.

모드:

- `direct`
- `cot`

출력:

- `warmup/lora_base_weight/...`

### `src/retrieve/retriever.py`

역할:

- Elasticsearch + BEIR 기반 BM25 검색 래퍼

핵심 포인트:

- 기본 index name은 `wiki`
- 기본 ES endpoint는 `http://localhost:9200`
- `BM25Search` 와 `ElasticSearch` 일부 메서드를 monkey patch 해서 사용합니다.
- 프로젝트 내부에 `src/retrieve/beir/` 를 vendoring 해 둔 구조입니다.

### `src/retrieve/beir/`

역할:

- 외부 라이브러리 BEIR 코드 사본

`src/retrieve/readme.md` 에 따르면:

- `beir2.0.0` 기반
- 서버 제약 때문에 `pytrec_eval` 의존 부분은 비활성화된 상태

---

## 6. 실제 데이터 흐름

이 프로젝트를 가장 쉽게 이해하는 방법은 "어떤 파일이 어떤 파일을 만든다"를 보는 것입니다.

### 6-1. 입력 데이터: `data/`

현재 실제 데이터는 아래와 같이 보입니다.

```text
data/
├── 2wikimultihopqa/
│   ├── dev.json
│   ├── train.json
│   ├── test.json
│   └── id_aliases.json
├── hotpotqa/
│   └── hotpot_dev_distractor_v1.json
├── popqa/
│   └── popQA.tsv
├── complexwebquestions/
│   └── ComplexWebQuestions_dev.json
├── dpr/
│   └── psgs_w100.tsv
└── elasticsearch-8.15.0/
```

의미는 아래와 같습니다.

- QA 데이터셋: 질문/정답 평가용
- `psgs_w100.tsv`: BM25 검색 대상 Wikipedia passage
- `elasticsearch-8.15.0/`: 로컬 검색 엔진 실행 파일 및 데이터

### 6-2. augmentation 결과: `data_aug/`

현재 실제로 생성되어 있는 구조 예시는 아래와 같습니다.

```text
data_aug/
├── 2wikimultihopqa/
│   ├── llama3-8b-instruct/
│   ├── llama3.2-1b-instruct/
│   └── qwen2.5-1.5b-instruct/
├── hotpotqa/
├── popqa/
└── complexwebquestions/
```

예를 들면:

```text
data_aug/2wikimultihopqa/llama3.2-1b-instruct/
├── total.json
├── compositional.json
├── comparison.json
├── inference.json
└── bridge_comparison.json
```

실제 샘플 키를 보면 각 항목은 대략 아래 구조를 가집니다.

```text
question
answer
passages
augment
test_id
qid / type / golden_passages   # 데이터셋에 따라 다름
```

`augment` 내부 원소는 대략 아래 구조입니다.

```text
pid
passage
{model_name}_rewrite
{model_name}_qa
```

즉 `data_aug/` 는 단순 검색 결과가 아니라, 이후 LoRA 학습에 직접 쓰이는 **중간 실험 데이터셋**입니다.

### 6-3. 문서별 LoRA 결과: `offline/`

`encode.py` 실행 후 생성되는 구조는 아래와 같습니다.

```text
offline/
└── {model_name}/
    └── rank={lora_rank}_alpha={lora_alpha}/
        ├── base_weight/
        └── {dataset}/
            └── lr={learning_rate}_epoch={num_train_epochs}_{direct_or_cot}/
                └── aug_model={augment_model}/
                    └── {data_type}/
                        └── data_{did}/
                            └── passage_{pid}/
                                ├── adapter_config.json
                                └── adapter_model.safetensors
```

의미:

- `base_weight/`: 특정 LoRA 설정의 공통 초기 가중치
- `data_{did}/passage_{pid}/`: 질문 `did` 의 `pid` 번째 passage에 대한 adapter

즉, 이 프로젝트는 passage 하나마다 adapter 하나를 저장합니다.

### 6-4. 최종 추론 결과: `output/`

`inference.py` 실행 후 생성되는 구조는 아래와 같습니다.

```text
output/
└── {model_name}/
    └── rank={lora_rank}_alpha={lora_alpha}/
        └── {dataset}/
            └── lr={learning_rate}_epoch={num_train_epochs}_{direct_or_cot}/
                └── aug_model={augment_model}/
                    └── {inference_method}/
                        └── {data_type}/
                            ├── config.json
                            ├── predict.json
                            └── result.txt
```

각 파일의 의미:

- `config.json`: 실행 인자 기록
- `predict.json`: 샘플별 질문, 정답, 생성 결과, 평가값
- `result.txt`: 평균 EM/F1/Precision/Recall 요약

### 6-5. warm-up 산출물: `warmup/`

warm-up 관련 경로는 아래와 같습니다.

```text
warmup/
├── data/
│   ├── direct/
│   └── cot/
└── lora_base_weight/
```

이 경로는 본 실험의 핵심 메인라인은 아니지만, 초기 LoRA를 따로 준비하고 싶을 때 사용됩니다.

---

## 7. 데이터셋별 동작 차이

이 프로젝트는 데이터셋마다 저장 구조와 처리 방식이 조금 다릅니다.

### `2wikimultihopqa`

- 원본 타입 정보가 있음
- augmentation 후 `compositional`, `comparison`, `inference`, `bridge_comparison`, `total` 로 나뉠 수 있음
- `golden_passages` 와 `type` 메타데이터를 함께 유지

### `hotpotqa`

- 원본 타입 정보가 있음
- augmentation 후 `bridge`, `comparison`, `total` 로 나뉠 수 있음
- supporting facts 기반 `golden_passages` 를 유지

### `popqa`

- TSV 입력
- alias를 포함한 answer 리스트를 구성
- 현재 augmentation 결과는 보통 `total.json`

### `complexwebquestions`

- JSON 입력
- 여러 alias를 answer 후보로 합침
- 현재 augmentation 결과는 보통 `total.json`

### 사용자 정의 데이터셋

`augment.py` 의 기본 포맷 로더를 사용하면 JSON 배열 형태로 확장 가능합니다.

필수 필드:

```json
[
    {
        "question": "string",
        "answer": "string or list[string]"
    }
]
```

---

## 8. 주요 실행 경로

### A. 검색 인프라 준비

1. `data/dpr/psgs_w100.tsv` 준비
2. Elasticsearch 실행
3. `prep_elastic.py` 로 `wiki` 인덱스 생성

관련 파일:

- `prep_elastic.py`
- `prep_elastic.sh`
- `ensure_elasticsearch.sh`
- `src/retrieve/retriever.py`

### B. augmentation

대표 명령:

```bash
python src/augment.py \
    --model_name llama3.2-1b-instruct \
    --dataset 2wikimultihopqa \
    --data_path data/2wikimultihopqa/ \
    --sample 300 \
    --topk 3
```

입력:

- QA 데이터셋
- BM25 index
- 생성 모델

출력:

- `data_aug/...`

### C. encoding

대표 명령:

```bash
python src/encode.py \
    --model_name llama3.2-1b-instruct \
    --dataset 2wikimultihopqa \
    --sample 300 \
    --per_device_train_batch_size 1 \
    --num_train_epochs 1 \
    --learning_rate 0.0003 \
    --lora_rank 2 \
    --lora_alpha 32 \
    --with_cot
```

입력:

- `data_aug/...`

출력:

- `offline/...`

### D. inference

대표 명령:

```bash
python src/inference.py \
    --model_name llama3.2-1b-instruct \
    --dataset 2wikimultihopqa \
    --sample 300 \
    --num_train_epochs 1 \
    --learning_rate 0.0003 \
    --lora_rank 2 \
    --lora_alpha 32 \
    --max_new_tokens 128 \
    --inference_method combine \
    --with_cot
```

입력:

- `data_aug/...`
- `offline/...`

출력:

- `output/...`

### E. warm-up

1. `python src/get_warmup_data.py`
2. `python src/warmup_lora.py ...`

출력:

- `warmup/data/...`
- `warmup/lora_base_weight/...`

---

## 9. `configs/` 와 배치 스크립트는 무엇인가

### `configs/`

이 폴더는 논문 실험 재현용 커맨드 모음입니다.

예:

- `configs/2wikimultihopqa_llama3-8b-instruct.sh`
- `configs/hotpotqa_qwen2.5-1.5b-instruct.sh`
- `configs/popqa_llama3.2-1b-instruct.sh`

공통 특징:

- 먼저 `src/encode.py`
- 다음에 `src/inference.py`
- 데이터셋과 모델별 추천 파라미터가 이미 적혀 있음

즉, 이 폴더는 "실험 레시피 모음"에 가깝습니다.

### `test.sh`, `test2.sh`, `test3.sh`, `test_augment_all.sh`

이름 때문에 테스트 파일처럼 보일 수 있지만, 실제로는 다음 목적입니다.

- SLURM 배치 제출용 스크립트
- augmentation 일괄 실행
- GPU, 메모리, 로그 경로 설정
- `.venv` 활성화
- Elasticsearch 확인 및 자동 시작

현재 `logs/` 에 있는 `training_*.out`, `training_*.err`, `augment_all_*.out`, `augment_all_*.err` 는 이 스크립트들의 실행 흔적입니다.

---

## 10. 코드 간 의존 관계

코드를 읽을 때 아래 연결만 기억하면 훨씬 쉽습니다.

```text
augment.py
  -> retriever.py
  -> utils.py
  -> root_dir_path.py

encode.py
  -> utils.py
  -> prompt_template.py
  -> root_dir_path.py

inference.py
  -> utils.py
  -> prompt_template.py
  -> root_dir_path.py
```

조금 더 풀어서 보면:

- `augment.py` 는 passage를 찾고 augmentation 데이터를 만든다.
- `encode.py` 는 augmentation 데이터를 학습 데이터로 바꿔 passage별 LoRA를 만든다.
- `inference.py` 는 augmentation 데이터의 passage 순서와 `offline/` 의 adapter 경로를 맞춰서 불러온다.

즉, **augmentation 결과의 passage 순서가 이후 전체 파이프라인의 기준축** 입니다.

---

## 11. 처음 보는 사람이 꼭 알아야 하는 주의점

### 1. `ROOT_DIR` 가 하드코딩되어 있음

`src/root_dir_path.py` 값을 바꾸지 않으면 다른 환경에서 경로 오류가 날 수 있습니다.

### 2. Elasticsearch가 사실상 필수임

augmentation 단계는 BM25 retrieval에 의존하므로, `wiki` 인덱스와 ES 서버가 준비되어 있어야 합니다.

### 3. `data/`, `data_aug/`, `logs/` 등은 `.gitignore` 대상

즉, 실험 산출물과 로컬 실행 흔적은 Git으로 관리하지 않는 구조입니다.

### 4. `README.md` 와 실제 의존성 파일 사이에 약간의 차이가 있음

예를 들어:

- `README.md` 에는 `torch==2.1.0` 설치 예시가 있음
- `requirements.txt` 와 `pyproject.toml` 에는 `torch==1.13.1` 이 적혀 있음

환경을 새로 만들 때는 어느 기준을 따를지 먼저 맞춰 보는 것이 좋습니다.

### 5. `test*.sh` 는 단위 테스트가 아님

이 프로젝트에는 전통적인 의미의 테스트 스위트보다는 **실험 실행 스크립트**가 중심입니다.

### 6. `offline/`, `output/`, `warmup/` 는 실행 후 생성됨

처음 clone 직후 안 보이는 것이 정상일 수 있습니다.

---

## 12. 새로운 사람이 코드를 읽는 추천 순서

가장 효율적인 읽기 순서는 아래와 같습니다.

1. `README.md`
2. `project_description.md`
3. `src/augment.py`
4. `src/retrieve/retriever.py`
5. `src/encode.py`
6. `src/inference.py`
7. `src/utils.py`
8. `src/prompt_template.py`
9. `configs/` 의 원하는 실험 파일

이 순서의 장점:

- 먼저 전체 목적을 잡고
- 검색 단계부터 이해한 뒤
- 문서별 LoRA 학습과 merge 추론으로 넘어갈 수 있습니다.

---

## 13. 빠르게 길을 잃지 않는 법

처음 이 프로젝트를 볼 때는 아래 세 질문만 계속 붙잡으면 됩니다.

### 질문 1. 입력 데이터는 어디 있나

- QA 데이터: `data/...`
- retrieval corpus: `data/dpr/psgs_w100.tsv`

### 질문 2. 중간 산출물은 어디에 쌓이나

- augmentation 결과: `data_aug/...`
- 문서별 LoRA: `offline/...`

### 질문 3. 최종 결과는 어디서 보나

- 샘플별 예측: `output/.../predict.json`
- 평균 성능: `output/.../result.txt`
- 배치 로그: `logs/...`

이 세 가지만 잡으면 프로젝트 전체가 금방 연결됩니다.

---

## 14. 이 프로젝트를 한 문장으로 다시 정리하면

> **PRAG는 검색된 문서를 프롬프트에 넣는 대신 문서별 LoRA로 바꿔 저장해 두고, 추론 시 관련 문서들의 LoRA를 합쳐 답을 생성하는 Parametric RAG 실험 코드베이스다.**

처음 보는 사람에게 가장 중요한 파일은 결국 아래 다섯 개입니다.

```text
src/augment.py
src/retrieve/retriever.py
src/encode.py
src/inference.py
src/utils.py
```

이 다섯 파일을 이해하면, 이 프로젝트의 구조와 실행 원리를 거의 다 이해한 것입니다.

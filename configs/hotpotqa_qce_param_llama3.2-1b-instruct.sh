python3 src/prepare_hotpot_qce.py \
    --model_name=llama3.2-1b-instruct \
    --topk=3 \
    --test_samples=300 \
    --train_samples_per_type=300 \
    --valid_samples_per_type=100

python3 src/train_hotpot_qce_param.py \
    --model_name=llama3.2-1b-instruct \
    --topk=3 \
    --test_samples=300 \
    --train_samples_per_type=300 \
    --valid_samples_per_type=100 \
    --selected_sentences=4 \
    --lora_rank=2 \
    --lora_alpha=32 \
    --max_new_tokens=128 \
    --with_cot

python3 src/infer_hotpot_qce_param.py \
    --model_name=llama3.2-1b-instruct \
    --topk=3 \
    --test_samples=300 \
    --train_samples_per_type=300 \
    --valid_samples_per_type=100 \
    --selected_sentences=4 \
    --lora_rank=2 \
    --lora_alpha=32 \
    --max_new_tokens=128 \
    --with_cot

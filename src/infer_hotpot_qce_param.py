import argparse
import json
import os

import torch
from tqdm import tqdm

import prompt_template
from qce_utils import (
    BilinearSentenceSelector,
    DirectLoRAHyperNetwork,
    QueryEvidenceCompressor,
    build_result_text,
    clear_dynamic_lora_state,
    collect_dynamic_lora_specs,
    ensure_pad_token,
    get_embedding_device,
    get_qce_ckpt_dir,
    get_qce_data_dir,
    get_qce_output_dir,
    load_json,
    mean_pool_text_embeddings,
    save_json,
    select_top_indices,
    set_dynamic_lora_state,
)
from train_hotpot_qce_param import TARGET_MODULES
from utils import evaluate, get_model, predict


def load_test_sets(data_dir, data_type):
    if data_type is None:
        return [
            ("bridge", load_json(os.path.join(data_dir, "bridge_test.json"))),
            ("comparison", load_json(os.path.join(data_dir, "comparison_test.json"))),
        ]
    return [(data_type, load_json(os.path.join(data_dir, f"{data_type}_test.json")))]


def main(args):
    data_dir = get_qce_data_dir(
        model_name=args.model_name,
        topk=args.topk,
        test_samples=args.test_samples,
        train_samples=args.train_samples_per_type,
        valid_samples=args.valid_samples_per_type,
    )
    checkpoint_dir = get_qce_ckpt_dir(
        model_name=args.model_name,
        topk=args.topk,
        selected_sentences=args.selected_sentences,
        test_samples=args.test_samples,
        train_samples=args.train_samples_per_type,
        valid_samples=args.valid_samples_per_type,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        with_cot=args.with_cot,
    )
    checkpoint_path = os.path.join(checkpoint_dir, "qce_param_best.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model, tokenizer, generation_config = get_model(args.model_name, max_new_tokens=args.max_new_tokens)
    ensure_pad_token(tokenizer)
    if args.with_cot:
        prompt_template.get_fewshot("hotpotqa")
    model.eval()

    specs = collect_dynamic_lora_specs(model, TARGET_MODULES, rank=args.lora_rank)
    hidden_size = model.get_input_embeddings().embedding_dim
    selector = BilinearSentenceSelector(hidden_size).to(get_embedding_device(model))
    compressor = QueryEvidenceCompressor(hidden_size, args.latent_size).to(get_embedding_device(model))
    hypernetwork = DirectLoRAHyperNetwork(
        latent_size=args.latent_size,
        hidden_size=args.hypernet_hidden_size,
        specs=specs,
        output_scale=args.hypernet_output_scale,
    ).to(get_embedding_device(model))

    selector.load_state_dict(checkpoint["selector"])
    compressor.load_state_dict(checkpoint["compressor"])
    hypernetwork.load_state_dict(checkpoint["hypernetwork"])
    selector.eval()
    compressor.eval()
    hypernetwork.eval()

    scaling = args.lora_alpha / args.lora_rank
    output_root = get_qce_output_dir(
        model_name=args.model_name,
        topk=args.topk,
        selected_sentences=args.selected_sentences,
        test_samples=args.test_samples,
        train_samples=args.train_samples_per_type,
        valid_samples=args.valid_samples_per_type,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        with_cot=args.with_cot,
    )
    os.makedirs(output_root, exist_ok=True)

    for split_name, episodes in load_test_sets(data_dir, args.data_type):
        split_output_dir = os.path.join(output_root, split_name)
        os.makedirs(split_output_dir, exist_ok=True)
        save_json(os.path.join(split_output_dir, "config.json"), vars(args))

        predictions = []
        for episode in tqdm(episodes, desc=f"infer {split_name}"):
            texts = [episode["question"]] + [record["text"] for record in episode["sentence_records"]]
            embeddings = mean_pool_text_embeddings(model, tokenizer, texts, max_length=args.max_text_length)
            query_embed = embeddings[0]
            sentence_embeds = embeddings[1:]
            with torch.no_grad():
                scores = selector(query_embed, sentence_embeds)
                top_indices = select_top_indices(scores, args.selected_sentences)
                selected_embeds = sentence_embeds[top_indices] if top_indices else sentence_embeds[:0]
                latent = compressor(query_embed, selected_embeds)
                generated = hypernetwork(latent)
                set_dynamic_lora_state(model, specs, generated, scaling=scaling)
                text = predict(
                    model=model,
                    tokenizer=tokenizer,
                    generation_config=generation_config,
                    question=episode["question"],
                    with_cot=args.with_cot,
                    passages=None,
                )
                clear_dynamic_lora_state(model, specs)

            pred = {
                "test_id": episode["test_id"],
                "qid": episode["qid"],
                "question": episode["question"],
                "answer": episode["answer"],
                "text": text,
                "selected_evidence": [episode["sentence_records"][idx]["text"] for idx in top_indices],
            }
            pred.update(evaluate(text, episode["answer"], args.with_cot))
            predictions.append(pred)

        save_json(os.path.join(split_output_dir, "predict.json"), predictions)
        result_text = build_result_text(vars(args), predictions)
        with open(os.path.join(split_output_dir, "result.txt"), "w") as fout:
            fout.write(result_text)
        print(f"{split_name}: wrote {len(predictions)} predictions to {split_output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--test_samples", type=int, default=300)
    parser.add_argument("--train_samples_per_type", type=int, default=300)
    parser.add_argument("--valid_samples_per_type", type=int, default=100)
    parser.add_argument("--selected_sentences", type=int, default=4)
    parser.add_argument("--latent_size", type=int, default=256)
    parser.add_argument("--hypernet_hidden_size", type=int, default=128)
    parser.add_argument("--hypernet_output_scale", type=float, default=0.01)
    parser.add_argument("--max_text_length", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--with_cot", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=2)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--data_type", type=str, choices=["bridge", "comparison"])
    args = parser.parse_args()
    main(args)

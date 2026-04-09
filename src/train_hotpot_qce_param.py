import argparse
import json
import os
from typing import List

import torch
import torch.nn.functional as F
from tqdm import tqdm

import prompt_template
from qce_utils import (
    BilinearSentenceSelector,
    DirectLoRAHyperNetwork,
    QueryEvidenceCompressor,
    build_lm_batch,
    clear_dynamic_lora_state,
    collect_dynamic_lora_specs,
    compute_selector_metrics,
    ensure_pad_token,
    get_embedding_device,
    get_qce_ckpt_dir,
    get_qce_data_dir,
    load_json,
    mean_pool_text_embeddings,
    set_dynamic_lora_state,
    save_json,
    select_top_indices,
)
from utils import get_model


TARGET_MODULES = ["down_proj", "gate_proj", "up_proj"]


def to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def encode_episode(model, tokenizer, episode, max_text_length):
    texts = [episode["question"]] + [record["text"] for record in episode["sentence_records"]]
    embeddings = mean_pool_text_embeddings(model, tokenizer, texts, max_length=max_text_length)
    query_embed = embeddings[0]
    sentence_embeds = embeddings[1:]
    labels = torch.tensor([record["label"] for record in episode["sentence_records"]], dtype=torch.float32, device=query_embed.device)
    return query_embed, sentence_embeds, labels


def evaluate_selector(model, tokenizer, selector, episodes, max_text_length, topk):
    selector.eval()
    records = []
    with torch.no_grad():
        for episode in episodes:
            query_embed, sentence_embeds, labels = encode_episode(model, tokenizer, episode, max_text_length)
            scores = selector(query_embed, sentence_embeds)
            records.append({"labels": labels.tolist(), "indices": select_top_indices(scores, topk)})
    return compute_selector_metrics(records, topk=topk)


def pretrain_selector(args, model, tokenizer, selector, train_episodes, valid_episodes, checkpoint_dir):
    optimizer = torch.optim.AdamW(selector.parameters(), lr=args.selector_learning_rate)
    best_recall = -1.0
    selector_path = os.path.join(checkpoint_dir, "selector_best.pt")

    for epoch in range(args.selector_epochs):
        selector.train()
        epoch_loss = 0.0
        for episode in tqdm(train_episodes, desc=f"selector epoch {epoch+1}/{args.selector_epochs}"):
            query_embed, sentence_embeds, labels = encode_episode(model, tokenizer, episode, args.max_text_length)
            scores = selector(query_embed, sentence_embeds)
            loss = F.binary_cross_entropy_with_logits(scores, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        metrics = evaluate_selector(
            model=model,
            tokenizer=tokenizer,
            selector=selector,
            episodes=valid_episodes,
            max_text_length=args.max_text_length,
            topk=args.selected_sentences,
        )
        avg_loss = epoch_loss / max(len(train_episodes), 1)
        print(
            f"selector epoch={epoch+1} loss={avg_loss:.4f} "
            f"valid_support_recall@{args.selected_sentences}={metrics['support_recall']:.4f}"
        )
        if metrics["support_recall"] > best_recall:
            best_recall = metrics["support_recall"]
            torch.save({"selector": selector.state_dict(), "metrics": metrics}, selector_path)

    ckpt = torch.load(selector_path, map_location=get_embedding_device(model))
    selector.load_state_dict(ckpt["selector"])
    return selector


def run_qce_epoch(
    *,
    args,
    model,
    tokenizer,
    selector,
    compressor,
    hypernetwork,
    specs,
    episodes,
    optimizer=None,
):
    train_mode = optimizer is not None
    compressor.train(train_mode)
    hypernetwork.train(train_mode)
    selector.eval()
    total_loss = 0.0
    lm_device = next(model.parameters()).device
    scaling = args.lora_alpha / args.lora_rank

    for episode in tqdm(episodes, desc="qce train" if train_mode else "qce valid"):
        query_embed, sentence_embeds, _labels = encode_episode(model, tokenizer, episode, args.max_text_length)
        with torch.no_grad():
            selector_scores = selector(query_embed, sentence_embeds)
            top_indices = select_top_indices(selector_scores, args.selected_sentences)
        selected_embeds = sentence_embeds[top_indices] if top_indices else sentence_embeds[:0]

        latent = compressor(query_embed, selected_embeds)
        generated = hypernetwork(latent)
        set_dynamic_lora_state(model, specs, generated, scaling=scaling)

        batch = build_lm_batch(
            tokenizer=tokenizer,
            question=episode["question"],
            answer=episode["answer"],
            with_cot=args.with_cot,
            max_length=args.max_prompt_length,
        )
        batch = to_device(batch, lm_device)
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.item()

        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(compressor.parameters()) + list(hypernetwork.parameters()),
                max_norm=1.0,
            )
            optimizer.step()

        clear_dynamic_lora_state(model, specs)

    return total_loss / max(len(episodes), 1)


def main(args):
    data_dir = get_qce_data_dir(
        model_name=args.model_name,
        topk=args.topk,
        test_samples=args.test_samples,
        train_samples=args.train_samples_per_type,
        valid_samples=args.valid_samples_per_type,
    )
    train_episodes = load_json(os.path.join(data_dir, "train.json"))
    valid_episodes = load_json(os.path.join(data_dir, "valid.json"))

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
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_json(os.path.join(checkpoint_dir, "train_config.json"), vars(args))

    model, tokenizer, _generation_config = get_model(args.model_name, max_new_tokens=args.max_new_tokens)
    ensure_pad_token(tokenizer)
    if args.with_cot:
        prompt_template.get_fewshot("hotpotqa")
    for param in model.parameters():
        param.requires_grad = False
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

    selector = pretrain_selector(args, model, tokenizer, selector, train_episodes, valid_episodes, checkpoint_dir)

    optimizer = torch.optim.AdamW(
        list(compressor.parameters()) + list(hypernetwork.parameters()),
        lr=args.learning_rate,
    )
    best_valid_loss = float("inf")
    best_ckpt_path = os.path.join(checkpoint_dir, "qce_param_best.pt")

    for epoch in range(args.train_epochs):
        train_loss = run_qce_epoch(
            args=args,
            model=model,
            tokenizer=tokenizer,
            selector=selector,
            compressor=compressor,
            hypernetwork=hypernetwork,
            specs=specs,
            episodes=train_episodes,
            optimizer=optimizer,
        )
        with torch.no_grad():
            valid_loss = run_qce_epoch(
                args=args,
                model=model,
                tokenizer=tokenizer,
                selector=selector,
                compressor=compressor,
                hypernetwork=hypernetwork,
                specs=specs,
                episodes=valid_episodes,
                optimizer=None,
            )
        print(f"epoch={epoch+1} train_loss={train_loss:.4f} valid_loss={valid_loss:.4f}")
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(
                {
                    "selector": selector.state_dict(),
                    "compressor": compressor.state_dict(),
                    "hypernetwork": hypernetwork.state_dict(),
                    "specs": [spec.__dict__ for spec in specs],
                    "args": vars(args),
                    "best_valid_loss": best_valid_loss,
                },
                best_ckpt_path,
            )

    print(f"saved best checkpoint to {best_ckpt_path}")


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
    parser.add_argument("--selector_epochs", type=int, default=3)
    parser.add_argument("--train_epochs", type=int, default=3)
    parser.add_argument("--selector_learning_rate", type=float, default=1e-3)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_text_length", type=int, default=128)
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--with_cot", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=2)
    parser.add_argument("--lora_alpha", type=int, default=32)
    args = parser.parse_args()
    main(args)

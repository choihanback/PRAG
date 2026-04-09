import json
import math
import os
import re
import string
import types
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import prompt_template
from root_dir_path import ROOT_DIR
from utils import BaseDataset


QCE_DATA_ROOT = os.path.join(ROOT_DIR, "data_qce")
QCE_CKPT_ROOT = os.path.join(ROOT_DIR, "checkpoints_qce")
QCE_OUTPUT_ROOT = os.path.join(ROOT_DIR, "output_qce")


def ensure_pad_token(tokenizer):
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token


def normalize_text_for_match(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())


def split_into_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    parts = [part.strip() for part in parts if part and part.strip()]
    return parts or [text]


def load_hotpot_raw(data_path: str) -> List[dict]:
    with open(os.path.join(data_path, "hotpot_dev_distractor_v1.json"), "r") as fin:
        return json.load(fin)


def group_hotpot_by_type(raw_data: List[dict]) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = {"bridge": [], "comparison": []}
    for item in raw_data:
        if item["type"] in grouped:
            grouped[item["type"]].append(item)
    return grouped


def build_hotpot_support_sentences(raw_item: dict) -> List[str]:
    title_to_sentences = {title: sentences for title, sentences in raw_item["context"]}
    support_sentences = []
    for title, sent_id in raw_item["supporting_facts"]:
        if title not in title_to_sentences:
            continue
        if sent_id >= len(title_to_sentences[title]):
            continue
        sent = title_to_sentences[title][sent_id].strip()
        if sent:
            support_sentences.append(sent)
    dedup = []
    seen = set()
    for sent in support_sentences:
        norm = normalize_text_for_match(sent)
        if norm and norm not in seen:
            dedup.append(sent)
            seen.add(norm)
    return dedup


def label_passages_with_support(passages: List[str], support_sentences: List[str]) -> List[dict]:
    normalized_support = [normalize_text_for_match(sent) for sent in support_sentences if sent.strip()]
    records = []
    for passage_id, passage in enumerate(passages):
        for sentence_id, sentence in enumerate(split_into_sentences(passage)):
            norm_sentence = normalize_text_for_match(sentence)
            label = 0
            if norm_sentence:
                for norm_support in normalized_support:
                    if not norm_support:
                        continue
                    if norm_support in norm_sentence or norm_sentence in norm_support:
                        label = 1
                        break
            records.append(
                {
                    "passage_id": passage_id,
                    "sentence_id": sentence_id,
                    "text": sentence,
                    "label": label,
                }
            )
    return records


def build_qce_example(
    *,
    qid: str,
    question: str,
    answer,
    data_type: str,
    raw_test_id: int,
    passages: List[str],
    support_sentences: List[str],
    baseline_test_id: int = None,
) -> dict:
    sentence_records = label_passages_with_support(passages, support_sentences)
    return {
        "qid": qid,
        "question": question,
        "answer": answer,
        "type": data_type,
        "raw_test_id": raw_test_id,
        "test_id": raw_test_id if baseline_test_id is None else baseline_test_id,
        "passages": passages,
        "support_sentences": support_sentences,
        "sentence_records": sentence_records,
        "has_positive_sentence": any(record["label"] == 1 for record in sentence_records),
    }


def save_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fout:
        json.dump(data, fout, indent=4)


def load_json(path: str):
    with open(path, "r") as fin:
        return json.load(fin)


def get_qce_data_dir(model_name: str, topk: int, test_samples: int, train_samples: int, valid_samples: int) -> str:
    return os.path.join(
        QCE_DATA_ROOT,
        "hotpotqa",
        model_name,
        f"topk={topk}_test={test_samples}_train={train_samples}_valid={valid_samples}",
    )


def get_qce_ckpt_dir(
    model_name: str,
    topk: int,
    selected_sentences: int,
    test_samples: int,
    train_samples: int,
    valid_samples: int,
    lora_rank: int,
    lora_alpha: int,
    with_cot: bool,
) -> str:
    cot_name = "cot" if with_cot else "direct"
    return os.path.join(
        QCE_CKPT_ROOT,
        model_name,
        f"topk={topk}_select={selected_sentences}_test={test_samples}_train={train_samples}_valid={valid_samples}",
        f"rank={lora_rank}_alpha={lora_alpha}_{cot_name}",
    )


def get_qce_output_dir(
    model_name: str,
    topk: int,
    selected_sentences: int,
    test_samples: int,
    train_samples: int,
    valid_samples: int,
    lora_rank: int,
    lora_alpha: int,
    with_cot: bool,
) -> str:
    cot_name = "cot" if with_cot else "direct"
    return os.path.join(
        QCE_OUTPUT_ROOT,
        model_name,
        "hotpotqa",
        f"topk={topk}_select={selected_sentences}_test={test_samples}_train={train_samples}_valid={valid_samples}",
        f"rank={lora_rank}_alpha={lora_alpha}_{cot_name}",
        "qce_param",
    )


def get_embedding_device(model: nn.Module) -> torch.device:
    return model.get_input_embeddings().weight.device


@torch.no_grad()
def mean_pool_text_embeddings(
    model: nn.Module,
    tokenizer,
    texts: List[str],
    max_length: int,
) -> torch.Tensor:
    ensure_pad_token(tokenizer)
    device = get_embedding_device(model)
    tokenized = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    embeds = outputs.hidden_states[-1]
    masked = embeds * attention_mask.unsqueeze(-1)
    lengths = attention_mask.sum(dim=-1, keepdim=True).clamp_min(1)
    pooled = masked.sum(dim=1) / lengths
    return pooled.to(torch.float32)


def build_lm_batch(tokenizer, question: str, answer: str, with_cot: bool, max_length: int) -> Dict[str, torch.Tensor]:
    input_ids = prompt_template.get_prompt(
        tokenizer,
        question,
        passages=None,
        answer=answer,
        with_cot=with_cot,
    )
    labels = input_ids.copy()
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    attention_mask = [1] * len(input_ids) + [0] * (max_length - len(input_ids))
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    labels += [-100] * (max_length - len(labels))
    return {
        "input_ids": torch.tensor(input_ids).unsqueeze(0),
        "labels": torch.tensor(labels).unsqueeze(0),
        "attention_mask": torch.tensor(attention_mask).unsqueeze(0),
    }


@dataclass
class DynamicLoraSpec:
    name: str
    in_features: int
    out_features: int
    rank: int

    @property
    def a_shape(self) -> Tuple[int, int]:
        return (self.rank, self.in_features)

    @property
    def b_shape(self) -> Tuple[int, int]:
        return (self.out_features, self.rank)

    @property
    def a_numel(self) -> int:
        return self.rank * self.in_features

    @property
    def b_numel(self) -> int:
        return self.out_features * self.rank


def _dynamic_linear_forward(self, x):
    result = F.linear(x, self.weight, self.bias)
    state = getattr(self, "_qce_dynamic_lora", None)
    if state is None:
        return result
    delta = F.linear(F.linear(x, state["A"]), state["B"])
    return result + delta * state["scale"]


def patch_dynamic_lora_module(module: nn.Linear) -> None:
    if getattr(module, "_qce_is_patched", False):
        return
    module.forward = types.MethodType(_dynamic_linear_forward, module)
    module._qce_is_patched = True


def collect_dynamic_lora_specs(model: nn.Module, target_modules: Iterable[str], rank: int) -> List[DynamicLoraSpec]:
    specs = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not any(name.endswith(target_name) for target_name in target_modules):
            continue
        patch_dynamic_lora_module(module)
        specs.append(
            DynamicLoraSpec(
                name=name,
                in_features=module.in_features,
                out_features=module.out_features,
                rank=rank,
            )
        )
    if not specs:
        raise ValueError(f"No target modules found for {target_modules}")
    return specs


def _get_module_by_name(root: nn.Module, name: str) -> nn.Module:
    module = root
    for attr in name.split("."):
        module = getattr(module, attr)
    return module


def set_dynamic_lora_state(
    model: nn.Module,
    specs: List[DynamicLoraSpec],
    generated: Dict[str, Dict[str, torch.Tensor]],
    scaling: float,
) -> None:
    for spec in specs:
        module = _get_module_by_name(model, spec.name)
        target_device = module.weight.device
        module._qce_dynamic_lora = {
            "A": generated[spec.name]["A"].to(target_device),
            "B": generated[spec.name]["B"].to(target_device),
            "scale": scaling,
        }


def clear_dynamic_lora_state(model: nn.Module, specs: List[DynamicLoraSpec]) -> None:
    for spec in specs:
        module = _get_module_by_name(model, spec.name)
        if hasattr(module, "_qce_dynamic_lora"):
            delattr(module, "_qce_dynamic_lora")


class BilinearSentenceSelector(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.query_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.sent_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, query_embed: torch.Tensor, sentence_embeds: torch.Tensor) -> torch.Tensor:
        query = self.query_proj(query_embed)
        sentence = self.sent_proj(sentence_embeds)
        return (sentence * query.unsqueeze(0)).sum(dim=-1)


class QueryEvidenceCompressor(nn.Module):
    def __init__(self, hidden_size: int, latent_size: int):
        super().__init__()
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.sent_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_size * 2, latent_size),
            nn.Tanh(),
            nn.Linear(latent_size, latent_size),
        )

    def forward(self, query_embed: torch.Tensor, sentence_embeds: torch.Tensor) -> torch.Tensor:
        if sentence_embeds.shape[0] == 0:
            pooled = torch.zeros_like(query_embed)
        else:
            query = self.query_proj(query_embed)
            sentence = self.sent_proj(sentence_embeds)
            attn = torch.softmax((sentence * query.unsqueeze(0)).sum(dim=-1) / math.sqrt(query.shape[-1]), dim=0)
            pooled = (attn.unsqueeze(-1) * sentence_embeds).sum(dim=0)
        combined = torch.cat([query_embed, pooled], dim=-1)
        return self.out_proj(combined)


class DirectLoRAHyperNetwork(nn.Module):
    def __init__(self, latent_size: int, hidden_size: int, specs: List[DynamicLoraSpec], output_scale: float = 0.01):
        super().__init__()
        self.specs = specs
        self.output_scale = output_scale
        self.total_params = sum(spec.a_numel + spec.b_numel for spec in specs)
        self.trunk = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.head = nn.Linear(hidden_size, self.total_params)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.002)
        nn.init.zeros_(self.head.bias)

    def forward(self, latent: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        hidden = self.trunk(latent)
        flat = self.head(hidden) * self.output_scale
        outputs: Dict[str, Dict[str, torch.Tensor]] = {}
        start = 0
        for spec in self.specs:
            end = start + spec.a_numel
            a = flat[start:end].view(spec.a_shape)
            start = end
            end = start + spec.b_numel
            b = flat[start:end].view(spec.b_shape)
            start = end
            outputs[spec.name] = {"A": a, "B": b}
        return outputs


def compute_selector_metrics(pred_records: List[dict], topk: int) -> Dict[str, float]:
    total = len(pred_records)
    if total == 0:
        return {"support_recall": 0.0, "avg_positive": 0.0}
    support_hits = 0
    positive_count = 0
    for record in pred_records:
        labels = record["labels"]
        indices = record["indices"][:topk]
        if any(labels[idx] == 1 for idx in indices):
            support_hits += 1
        positive_count += sum(labels)
    return {
        "support_recall": support_hits / total,
        "avg_positive": positive_count / total,
    }


def select_top_indices(scores: torch.Tensor, topk: int) -> List[int]:
    if scores.numel() == 0:
        return []
    topk = min(topk, scores.shape[0])
    return torch.topk(scores, k=topk).indices.tolist()


def average_metric(examples: List[dict], key: str) -> float:
    if not examples:
        return 0.0
    return sum(float(example[key]) for example in examples) / len(examples)


def build_result_text(args: dict, predictions: List[dict]) -> str:
    metrics = ["em", "f1", "prec", "recall"]
    lines = []
    for metric in metrics:
        lines.append(f"{metric}\t{round(average_metric(predictions, metric), 4)}")
    lines.append("")
    lines.append(json.dumps(args, indent=4))
    return "\n".join(lines)

import argparse
import os

from qce_utils import (
    build_hotpot_support_sentences,
    build_qce_example,
    get_qce_data_dir,
    group_hotpot_by_type,
    load_hotpot_raw,
    load_json,
    save_json,
)
from retrieve.retriever import bm25_retrieve
from root_dir_path import ROOT_DIR


def build_train_valid_examples(raw_grouped, topk, test_samples, train_samples_per_type, valid_samples_per_type):
    train_examples = []
    valid_examples = []
    metadata = {"train_counts": {}, "valid_counts": {}}

    for data_type in ("bridge", "comparison"):
        type_data = raw_grouped[data_type]
        train_slice = type_data[test_samples:test_samples + train_samples_per_type]
        valid_start = test_samples + train_samples_per_type
        valid_slice = type_data[valid_start:valid_start + valid_samples_per_type]

        metadata["train_counts"][data_type] = len(train_slice)
        metadata["valid_counts"][data_type] = len(valid_slice)

        for split_name, split_examples, target in (
            ("train", train_slice, train_examples),
            ("valid", valid_slice, valid_examples),
        ):
            for raw_item in split_examples:
                passages = bm25_retrieve(raw_item["question"], topk=topk)
                target.append(
                    build_qce_example(
                        qid=raw_item["_id"],
                        question=raw_item["question"],
                        answer=raw_item["answer"],
                        data_type=data_type,
                        raw_test_id=raw_item["test_id"],
                        passages=passages,
                        support_sentences=build_hotpot_support_sentences(raw_item),
                    )
                )
            print(f"{split_name}: {data_type} -> {len(split_examples)} examples")

    return train_examples, valid_examples, metadata


def build_test_examples(raw_by_qid, eval_examples, data_type):
    test_examples = []
    for eval_item in eval_examples:
        raw_item = raw_by_qid[eval_item["qid"]]
        test_examples.append(
            build_qce_example(
                qid=eval_item["qid"],
                question=eval_item["question"],
                answer=eval_item["answer"],
                data_type=data_type,
                raw_test_id=raw_item["test_id"],
                baseline_test_id=eval_item["test_id"],
                passages=eval_item["passages"],
                support_sentences=build_hotpot_support_sentences(raw_item),
            )
        )
    return test_examples


def main(args):
    raw_data = load_hotpot_raw(args.data_path)
    for idx, item in enumerate(raw_data):
        item["test_id"] = idx
    raw_grouped = group_hotpot_by_type(raw_data)
    raw_by_qid = {item["_id"]: item for item in raw_data}

    baseline_dir = os.path.join(ROOT_DIR, "data_aug", "hotpotqa", args.model_name)
    bridge_eval = load_json(os.path.join(baseline_dir, "bridge.json"))[:args.test_samples]
    comparison_eval = load_json(os.path.join(baseline_dir, "comparison.json"))[:args.test_samples]

    output_dir = get_qce_data_dir(
        model_name=args.model_name,
        topk=args.topk,
        test_samples=args.test_samples,
        train_samples=args.train_samples_per_type,
        valid_samples=args.valid_samples_per_type,
    )
    os.makedirs(output_dir, exist_ok=True)

    train_examples, valid_examples, metadata = build_train_valid_examples(
        raw_grouped=raw_grouped,
        topk=args.topk,
        test_samples=args.test_samples,
        train_samples_per_type=args.train_samples_per_type,
        valid_samples_per_type=args.valid_samples_per_type,
    )

    bridge_test = build_test_examples(raw_by_qid, bridge_eval, "bridge")
    comparison_test = build_test_examples(raw_by_qid, comparison_eval, "comparison")

    metadata.update(
        {
            "model_name": args.model_name,
            "topk": args.topk,
            "test_samples": args.test_samples,
            "train_samples_per_type": args.train_samples_per_type,
            "valid_samples_per_type": args.valid_samples_per_type,
            "bridge_test": len(bridge_test),
            "comparison_test": len(comparison_test),
            "baseline_eval_dir": baseline_dir,
            "raw_data_path": args.data_path,
        }
    )

    save_json(os.path.join(output_dir, "train.json"), train_examples)
    save_json(os.path.join(output_dir, "valid.json"), valid_examples)
    save_json(os.path.join(output_dir, "bridge_test.json"), bridge_test)
    save_json(os.path.join(output_dir, "comparison_test.json"), comparison_test)
    save_json(os.path.join(output_dir, "metadata.json"), metadata)

    print(f"Saved QCE data to {output_dir}")
    print(
        f"train={len(train_examples)} valid={len(valid_examples)} "
        f"bridge_test={len(bridge_test)} comparison_test={len(comparison_test)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--data_path", type=str, default=os.path.join(ROOT_DIR, "data", "hotpotqa"))
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--test_samples", type=int, default=300)
    parser.add_argument("--train_samples_per_type", type=int, default=300)
    parser.add_argument("--valid_samples_per_type", type=int, default=100)
    args = parser.parse_args()
    main(args)

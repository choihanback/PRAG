import argparse
import json
from pathlib import Path

from root_dir_path import ROOT_DIR


def load_json(path: Path):
    with path.open("r") as fin:
        return json.load(fin)


def infer_data_type(config: dict, predict_path: Path) -> str:
    data_type = config.get("data_type")
    if data_type:
        return data_type
    return predict_path.parent.name


def export_predict_with_passages(predict_path: Path) -> Path:
    predict_path = Path(predict_path)
    config_path = predict_path.with_name("config.json")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json next to {predict_path}")

    config = load_json(config_path)
    predictions = load_json(predict_path)
    inference_method = config.get("inference_method")
    if inference_method == "vanila":
        inference_method = "vanilla"

    dataset = config["dataset"]
    augment_model = config["augment_model"]
    data_type = infer_data_type(config, predict_path)
    source_path = Path(ROOT_DIR) / "data_aug" / dataset / augment_model / f"{data_type}.json"
    if not source_path.exists():
        raise FileNotFoundError(f"Missing source data file: {source_path}")

    source_rows = load_json(source_path)

    merged = []
    for idx, pred in enumerate(predictions):
        test_id = pred.get("test_id", idx)
        if not (0 <= test_id < len(source_rows)):
            raise KeyError(f"Could not find source row for test_id={test_id} in {source_path}")
        source_row = source_rows[test_id]

        merged.append({
            "test_id": test_id,
            "question": pred.get("question", source_row.get("question")),
            "answer": pred.get("answer", source_row.get("answer")),
            "text": pred.get("text"),
            "eval_predict": pred.get("eval_predict"),
            "retrieved_passages": [] if inference_method == "vanilla" else source_row.get("passages", []),
        })

    output_path = predict_path.with_name("predict_with_passages.json")
    with output_path.open("w") as fout:
        json.dump(merged, fout, indent=4, ensure_ascii=False)
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict-file", action="append", default=[])
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    predict_paths = []
    if args.all:
        predict_paths.extend(sorted(Path(ROOT_DIR).glob("output/**/predict.json")))
    predict_paths.extend(Path(path) for path in args.predict_file)

    deduped_paths = []
    seen = set()
    for path in predict_paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped_paths.append(resolved)

    if not deduped_paths:
        raise ValueError("No predict.json files specified. Use --all or --predict-file.")

    for predict_path in deduped_paths:
        output_path = export_predict_with_passages(predict_path)
        print(output_path)


if __name__ == "__main__":
    main()

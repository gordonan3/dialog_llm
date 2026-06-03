import json
from pathlib import Path

import click
import evaluate
import pandas as pd
import yaml


def compute_metrics(df, pred_col, ref_col="summary"):
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")

    predictions = df[pred_col].fillna("").astype(str).tolist()
    references = df[ref_col].fillna("").astype(str).tolist()

    rouge_result = rouge.compute(
        predictions=predictions,
        references=references,
    )

    bleu_result = bleu.compute(
        predictions=predictions,
        references=[[x] for x in references],
    )

    pred_len = sum(len(x.split()) for x in predictions)
    ref_len = sum(len(x.split()) for x in references)

    return {
        "rouge1": rouge_result["rouge1"],
        "rouge2": rouge_result["rouge2"],
        "rougeL": rouge_result["rougeL"],
        "rougeLsum": rouge_result["rougeLsum"],
        "bleu": bleu_result["bleu"],
        "length_ratio": pred_len / ref_len if ref_len > 0 else None,
        "prediction_tokens": pred_len,
        "reference_tokens": ref_len,
    }


@click.command()
@click.option("--config", required=True, help="Path to YAML config")
def main(config):
    with open(config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    input_path = Path(cfg.get("input_path", "results/df_full.csv"))
    output_csv_path = Path(cfg.get("output_csv_path", "results/metrics.csv"))
    output_json_path = Path(cfg.get("output_json_path", "results/metrics.json"))

    ref_col = cfg.get("reference_column", "summary")
    prediction_columns = cfg.get(
        "prediction_columns",
        ["null_baseline", "qwen_zero_shot", "lora"],
    )

    df = pd.read_csv(input_path)

    rows = []

    for pred_col in prediction_columns:
        if pred_col not in df.columns:
            print(f"Skip column: {pred_col} — not found")
            continue

        metrics = compute_metrics(df, pred_col=pred_col, ref_col=ref_col)
        metrics["method"] = pred_col
        rows.append(metrics)

    metrics_df = pd.DataFrame(rows)

    cols = [
        "method",
        "rouge1",
        "rouge2",
        "rougeL",
        "rougeLsum",
        "bleu",
        "length_ratio",
        "prediction_tokens",
        "reference_tokens",
    ]

    metrics_df = metrics_df[cols]

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    metrics_df.to_csv(output_csv_path, index=False)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print(metrics_df)


if __name__ == "__main__":
    main()
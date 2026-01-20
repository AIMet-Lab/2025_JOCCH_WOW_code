#!/usr/bin/env python3
import argparse
import os
import json
import pandas
import numpy
import matplotlib.pyplot as plt


def read_datasets_map(datasets_csv_path: str) -> dict:
    datasets_df = pandas.read_csv(datasets_csv_path)
    dataset_map = {}
    for _, row in datasets_df.iterrows():
        dataset_map[str(row["ID"])] = {
            "language": str(row["LANGUAGE"]).upper(),
            "path": str(row["PATH"])
        }
    return dataset_map


def infer_dataset_id_from_filename(filename: str, dataset_ids: list[str]) -> str:
    """
    Match the LONGEST dataset_id that is a prefix of filename.
    Example: quotes_c_en_nmt_M7_T1.csv -> dataset_id = quotes_c_en_nmt
    """
    candidates = [d for d in dataset_ids if filename.startswith(d + "_")]
    if not candidates:
        return ""
    return sorted(candidates, key=len, reverse=True)[0]


def normalise_binary_frame(df: pandas.DataFrame, label_cols: list[str]) -> pandas.DataFrame:
    """
    Force label columns to 0/1 integers when possible.
    Accepts 0/1, True/False, "0"/"1", "true"/"false".
    """
    out = df.copy()
    for col in label_cols:
        series = out[col]

        if series.dtype == bool:
            out[col] = series.astype(int)
            continue

        coerced = pandas.to_numeric(series, errors="coerce")
        if coerced.notna().mean() > 0.95:
            out[col] = (coerced.fillna(0) > 0).astype(int)
            continue

        s = series.astype(str).str.strip().str.lower()
        out[col] = s.isin(["1", "true", "t", "yes", "y"]).astype(int)

    return out


def compute_label_metrics(y_true: numpy.ndarray, y_pred: numpy.ndarray) -> dict:
    tp = (y_true & y_pred).sum(axis=0)
    fp = ((1 - y_true) & y_pred).sum(axis=0)
    fn = (y_true & (1 - y_pred)).sum(axis=0)

    support = y_true.sum(axis=0)

    precision = numpy.divide(tp, tp + fp, out=numpy.zeros_like(tp, dtype=float), where=(tp + fp) > 0)
    recall = numpy.divide(tp, tp + fn, out=numpy.zeros_like(tp, dtype=float), where=(tp + fn) > 0)
    f1 = numpy.divide(2 * precision * recall, precision + recall, out=numpy.zeros_like(precision, dtype=float), where=(precision + recall) > 0)

    macro_f1 = float(numpy.mean(f1))

    tp_micro = int(tp.sum())
    fp_micro = int(fp.sum())
    fn_micro = int(fn.sum())
    micro_precision = tp_micro / (tp_micro + fp_micro) if (tp_micro + fp_micro) > 0 else 0.0
    micro_recall = tp_micro / (tp_micro + fn_micro) if (tp_micro + fn_micro) > 0 else 0.0
    micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)) if (micro_precision + micro_recall) > 0 else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "support": support,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1
    }


def compute_text_level_stats(y_true: numpy.ndarray, y_pred: numpy.ndarray) -> dict:
    tp = (y_true & y_pred).sum(axis=1)
    fp = ((1 - y_true) & y_pred).sum(axis=1)
    fn = (y_true & (1 - y_pred)).sum(axis=1)

    true_count = y_true.sum(axis=1)
    pred_count = y_pred.sum(axis=1)

    precision = numpy.divide(tp, tp + fp, out=numpy.zeros_like(tp, dtype=float), where=(tp + fp) > 0)
    recall = numpy.divide(tp, tp + fn, out=numpy.zeros_like(tp, dtype=float), where=(tp + fn) > 0)
    f1 = numpy.divide(2 * precision * recall, precision + recall, out=numpy.zeros_like(precision, dtype=float), where=(precision + recall) > 0)

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "true_count": true_count, "pred_count": pred_count,
        "precision": precision, "recall": recall, "f1": f1
    }


def spearman_corr(x: numpy.ndarray, y: numpy.ndarray) -> float:
    def rank(a):
        return pandas.Series(a).rank(method="average").to_numpy()

    rx = rank(x)
    ry = rank(y)

    rxm = rx - rx.mean()
    rym = ry - ry.mean()
    denom = (numpy.sqrt((rxm ** 2).sum()) * numpy.sqrt((rym ** 2).sum()))
    if denom == 0:
        return 0.0
    return float((rxm * rym).sum() / denom)


def save_plots(output_dir: str, run_id: str, label_df: pandas.DataFrame, text_df: pandas.DataFrame):
    os.makedirs(output_dir, exist_ok=True)

    plt.figure()
    plt.scatter(label_df["support"], label_df["f1"])
    plt.xlabel("Label support (#positives)")
    plt.ylabel("Label F1")
    plt.title(f"Support vs F1 ({run_id})")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{run_id}__support_vs_f1.pdf"))
    plt.close()

    top = label_df.sort_values("f1", ascending=False).head(15)
    bottom = label_df.sort_values("f1", ascending=True).head(15)

    plt.figure()
    plt.barh(top["label"][::-1], top["f1"][::-1])
    plt.xlabel("F1")
    plt.title(f"Top-15 labels by F1 ({run_id})")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{run_id}__top15_labels_f1.pdf"))
    plt.close()

    plt.figure()
    plt.barh(bottom["label"][::-1], bottom["f1"][::-1])
    plt.xlabel("F1")
    plt.title(f"Bottom-15 labels by F1 ({run_id})")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{run_id}__bottom15_labels_f1.pdf"))
    plt.close()

    plt.figure()
    plt.hist(text_df["f1"], bins=30)
    plt.xlabel("Per-text F1")
    plt.ylabel("Count")
    plt.title(f"Per-text F1 distribution ({run_id})")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{run_id}__text_f1_hist.pdf"))
    plt.close()


def build_examples_report(
    output_path: str,
    merged_texts: list[str],
    label_cols: list[str],
    y_true: numpy.ndarray,
    y_pred: numpy.ndarray,
    per_text: dict,
    top_k: int
):
    text_df = pandas.DataFrame({
        "text": merged_texts,
        "true_count": per_text["true_count"],
        "pred_count": per_text["pred_count"],
        "tp": per_text["tp"],
        "fp": per_text["fp"],
        "fn": per_text["fn"],
        "f1": per_text["f1"]
    })

    easiest = text_df.sort_values(["f1", "true_count"], ascending=[False, False]).head(top_k)
    hardest = text_df.sort_values(["f1", "true_count"], ascending=[True, False]).head(top_k)

    high_true = text_df.sort_values("true_count", ascending=False).head(max(top_k, 20))
    complex_hard = high_true.sort_values("f1", ascending=True).head(top_k)

    def labels_for_row(idx: int):
        true_labels = [label_cols[j] for j in range(len(label_cols)) if y_true[idx, j] == 1]
        pred_labels = [label_cols[j] for j in range(len(label_cols)) if y_pred[idx, j] == 1]
        missed = [l for l in true_labels if l not in pred_labels]
        spurious = [l for l in pred_labels if l not in true_labels]
        return true_labels, pred_labels, missed, spurious

    def format_block(row):
        idx = int(row.name)
        true_labels, pred_labels, missed, spurious = labels_for_row(idx)

        snippet = str(row["text"])
        if len(snippet) > 900:
            snippet = snippet[:900] + " …"

        return (
            f"- true_count={int(row['true_count'])} pred_count={int(row['pred_count'])} "
            f"tp={int(row['tp'])} fp={int(row['fp'])} fn={int(row['fn'])} f1={row['f1']:.3f}\n"
            f"  - text: {snippet}\n"
            f"  - true labels: {', '.join(true_labels) if true_labels else '(none)'}\n"
            f"  - predicted: {', '.join(pred_labels) if pred_labels else '(none)'}\n"
            f"  - missed (FN): {', '.join(missed) if missed else '(none)'}\n"
            f"  - spurious (FP): {', '.join(spurious) if spurious else '(none)'}\n"
        )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Qualitative examples report\n\n")

        f.write("## Easiest texts (high per-text F1)\n\n")
        for _, r in easiest.iterrows():
            f.write(format_block(r) + "\n")

        f.write("\n## Hardest texts (low per-text F1)\n\n")
        for _, r in hardest.iterrows():
            f.write(format_block(r) + "\n")

        f.write("\n## Complex-hard texts (many true labels + low F1)\n\n")
        for _, r in complex_hard.iterrows():
            f.write(format_block(r) + "\n")


def analyze_one_run(
    pred_path: str,
    dataset_id: str,
    dataset_lang: str,
    dataset_path: str,
    output_dir: str,
    threshold: float,
    top_k: int
):
    pred_df = pandas.read_csv(pred_path)
    gt_df = pandas.read_csv(dataset_path)

    pred_label_cols = [c for c in pred_df.columns if c not in ["text", "time"]]
    gt_label_cols = [c for c in gt_df.columns if c != "text"]
    label_cols = [c for c in pred_label_cols if c in gt_label_cols]

    if len(label_cols) == 0:
        raise ValueError(f"No overlapping label columns between prediction and ground truth for {pred_path}")

    gt_df = normalise_binary_frame(gt_df, label_cols)

    merged = gt_df[["text"] + label_cols].merge(
        pred_df[["text"] + label_cols],
        on="text",
        how="inner",
        suffixes=("_true", "_score")
    )

    y_true = merged[[c + "_true" for c in label_cols]].to_numpy(dtype=int)
    y_score = merged[[c + "_score" for c in label_cols]].to_numpy(dtype=float)
    y_pred = (y_score >= threshold).astype(int)

    metrics = compute_label_metrics(y_true, y_pred)
    per_text = compute_text_level_stats(y_true, y_pred)

    label_df = pandas.DataFrame({
        "label": label_cols,
        "support": metrics["support"].astype(int),
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"]
    })

    rho = spearman_corr(label_df["support"].to_numpy(), label_df["f1"].to_numpy())

    f1_q90 = label_df["f1"].quantile(0.90)
    f1_q10 = label_df["f1"].quantile(0.10)
    support_med = label_df["support"].median()

    label_df["high_f1_low_support"] = (label_df["f1"] >= f1_q90) & (label_df["support"] < support_med)
    label_df["low_f1_high_support"] = (label_df["f1"] <= f1_q10) & (label_df["support"] > support_med)

    text_df = pandas.DataFrame({
        "text": merged["text"].tolist(),
        "true_count": per_text["true_count"].astype(int),
        "pred_count": per_text["pred_count"].astype(int),
        "tp": per_text["tp"].astype(int),
        "fp": per_text["fp"].astype(int),
        "fn": per_text["fn"].astype(int),
        "f1": per_text["f1"]
    })

    run_id = os.path.splitext(os.path.basename(pred_path))[0]
    run_dir = os.path.join(output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    pct_labels_support_lt_5 = float((label_df["support"] < 5).mean())
    pct_labels_f1_ge_0_5 = float((label_df["f1"] >= 0.5).mean())

    summary = {
        "run_id": run_id,
        "dataset_id": dataset_id,
        "dataset_lang": dataset_lang,
        "threshold": threshold,
        "n_texts_merged": int(len(merged)),
        "n_labels": int(len(label_cols)),
        "macro_f1": metrics["macro_f1"],
        "micro_f1": metrics["micro_f1"],
        "support_f1_spearman_rho": rho,
        "pct_labels_support_lt_5": pct_labels_support_lt_5,
        "pct_labels_f1_ge_0_5": pct_labels_f1_ge_0_5
    }

    with open(os.path.join(run_dir, f"{run_id}__summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    label_df.sort_values(["f1", "support"], ascending=[False, False]).to_csv(
        os.path.join(run_dir, f"{run_id}__labels_metrics.csv"), index=False
    )
    text_df.sort_values(["f1", "true_count"], ascending=[True, False]).to_csv(
        os.path.join(run_dir, f"{run_id}__texts_metrics.csv"), index=False
    )

    save_plots(run_dir, run_id, label_df, text_df)

    build_examples_report(
        output_path=os.path.join(run_dir, f"{run_id}__examples.md"),
        merged_texts=merged["text"].tolist(),
        label_cols=label_cols,
        y_true=y_true,
        y_pred=y_pred,
        per_text=per_text,
        top_k=top_k
    )

    top_labels = label_df.sort_values(["f1", "support"], ascending=[False, False]).head(top_k).copy()
    top_labels["rank_group"] = "top"
    bottom_labels = label_df.sort_values(["f1", "support"], ascending=[True, False]).head(top_k).copy()
    bottom_labels["rank_group"] = "bottom"
    top_bottom_df = pandas.concat([top_labels, bottom_labels], ignore_index=True)
    top_bottom_df.insert(0, "run_id", run_id)
    top_bottom_df.insert(1, "dataset_id", dataset_id)
    top_bottom_df.insert(2, "dataset_lang", dataset_lang)
    top_bottom_df.insert(3, "threshold", threshold)

    return summary, top_bottom_df


def main():
    parser = argparse.ArgumentParser(
        description="Analyze raw model outputs vs ground truth, produce per-run artifacts and aggregated CSVs."
    )
    parser.add_argument("--datasets_csv", required=True, type=str,
                        help="Path to datasets.csv (ID,LANGUAGE,PATH)")
    parser.add_argument("--predictions_dir", required=True, type=str,
                        help="Folder containing model output CSVs")
    parser.add_argument("--output_dir", required=True, type=str,
                        help="Folder to save analysis artifacts")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold to binarize scores (modifiable on-the-fly)")
    parser.add_argument("--top_k", type=int, default=10,
                        help="How many best/worst labels/texts to report")
    args = parser.parse_args()

    dataset_map = read_datasets_map(args.datasets_csv)
    dataset_ids = list(dataset_map.keys())

    os.makedirs(args.output_dir, exist_ok=True)

    all_summaries = []
    all_top_bottom = []

    for filename in os.listdir(args.predictions_dir):
        if not filename.endswith(".csv"):
            continue

        dataset_id = infer_dataset_id_from_filename(filename, dataset_ids)
        if dataset_id == "":
            continue

        pred_path = os.path.join(args.predictions_dir, filename)
        gt_path = dataset_map[dataset_id]["path"]
        dataset_lang = dataset_map[dataset_id]["language"]

        summary, top_bottom_df = analyze_one_run(
            pred_path=pred_path,
            dataset_id=dataset_id,
            dataset_lang=dataset_lang,
            dataset_path=gt_path,
            output_dir=args.output_dir,
            threshold=args.threshold,
            top_k=args.top_k
        )

        all_summaries.append(summary)
        all_top_bottom.append(top_bottom_df)

    if len(all_summaries) > 0:
        summaries_df = pandas.DataFrame(all_summaries)
        summaries_df.to_csv(os.path.join(args.output_dir, "all_runs_summary.csv"), index=False)

    if len(all_top_bottom) > 0:
        top_bottom_all_df = pandas.concat(all_top_bottom, ignore_index=True)
        top_bottom_all_df.to_csv(os.path.join(args.output_dir, "all_runs_top_bottom_labels.csv"), index=False)

    print(f"Done. Analysis saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

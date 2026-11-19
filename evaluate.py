"""evaluate.py - run Champion (A) and Challenger (B) on the SAME inputs and log everything to MLflow.

Run structure (one experiment, one parent summary run, one nested run per model):
    compare_sentiment                      <- summary run: decision, delta, comparison plots, README
      |- A_sentiment_distilbert-sst2       <- Champion candidate: params, metrics, tags, model, artifacts
      |- B_sentiment_roberta-large         <- Challenger candidate: params, metrics, tags, model, artifacts

Usage:
    python -m mlflow server --host 127.0.0.1 --port 5000      # terminal 1
    python evaluate.py                                        # terminal 2
    python evaluate.py --samples 10                           # quick debug run
"""
import argparse
import json
import os
import platform
import random
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import sklearn
import torch
import transformers
from mlflow.models import infer_signature
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

import config as C
from hf_sentiment_model import HFSentimentModel, normalise_label

LABELS = ["negative", "positive"]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_eval_set(n: int, seed: int) -> pd.DataFrame:
    """Sample n rows from the SST-2 validation split and save them to data/eval_inputs.csv.

    If data/eval_inputs.csv already exists with the right size it is reused, so both
    models (and every re-run) see exactly the same inputs in the same order.
    """
    path = os.path.join(C.DATA_DIR, "eval_inputs.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        if len(df) == n:
            print(f"Reusing saved evaluation set: {path} ({n} rows)")
            return df

    from datasets import load_dataset

    ds = load_dataset("glue", "sst2", split="validation")
    ds = ds.shuffle(seed=seed).select(range(n))
    df = pd.DataFrame(
        {
            "id": list(range(n)),
            "source_idx": ds["idx"],
            "text": [s.strip() for s in ds["sentence"]],
            "reference": [C.LABEL_MAP[l] for l in ds["label"]],
        }
    )
    os.makedirs(C.DATA_DIR, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved evaluation set: {path} ({n} rows)")
    return df


def compute_metrics(refs, preds) -> dict:
    return {
        "accuracy": accuracy_score(refs, preds),
        "macro_f1": f1_score(refs, preds, average="macro", labels=LABELS),
        "macro_precision": precision_score(refs, preds, average="macro", labels=LABELS, zero_division=0),
        "macro_recall": recall_score(refs, preds, average="macro", labels=LABELS, zero_division=0),
        "f1_positive": f1_score(refs, preds, pos_label="positive", average="binary"),
        "f1_negative": f1_score(refs, preds, pos_label="negative", average="binary"),
        "n_errors": int(sum(r != p for r, p in zip(refs, preds))),
    }


def plot_confusion(refs, preds, title, path):
    cm = confusion_matrix(refs, preds, labels=LABELS)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    return fig


def run_model(key: str, eval_df: pd.DataFrame, args) -> dict:
    m = C.MODELS[key]
    run_name = f"{key}_{C.TASK}_{m['short_name']}"
    out_dir = C.OUTPUT_DIR

    with mlflow.start_run(run_name=run_name, nested=True) as run:
        # ---------- parameters ----------
        mlflow.log_params(
            {
                "hf_model_id": m["hf_model_id"],
                "task": C.TASK,
                "role": m["role"],
                "batch_size": C.BATCH_SIZE,
                "max_length": C.MAX_LENGTH,
                "truncation": True,
                "eval_samples": len(eval_df),
                "seed": C.SEED,
                "device": "cpu",
            }
        )
        # ---------- tags ----------
        mlflow.set_tags(
            {
                "student_usn": C.STUDENT_USN,
                "student_name": C.STUDENT_NAME,
                "dataset_name": C.DATASET_NAME,
                "role": m["role"],
                "model_key": key,
                "transformers_version": transformers.__version__,
                "torch_version": torch.__version__,
                "mlflow_version": mlflow.__version__,
                "sklearn_version": sklearn.__version__,
                "python_version": platform.python_version(),
            }
        )

        # ---------- inference through the same wrapper that gets logged ----------
        wrapper = HFSentimentModel()

        class _Ctx:  # minimal context so we can reuse load_context outside MLflow
            model_config = {
                "hf_model_id": m["hf_model_id"],
                "max_length": C.MAX_LENGTH,
                "batch_size": C.BATCH_SIZE,
            }

        print(f"\n[{run_name}] loading {m['hf_model_id']} ...")
        wrapper.load_context(_Ctx())
        wrapper.predict(None, eval_df[["text"]].head(2))  # warm-up, not timed

        t0 = time.perf_counter()
        pred_df = wrapper.predict(None, eval_df[["text"]])
        total_s = time.perf_counter() - t0

        preds = pred_df["prediction"].map(normalise_label).tolist()
        refs = eval_df["reference"].tolist()
        metrics = compute_metrics(refs, preds)
        metrics["total_inference_s"] = total_s
        metrics["avg_latency_ms"] = 1000 * total_s / len(eval_df)
        mlflow.log_metrics(metrics)
        print(f"[{run_name}] " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

        # ---------- artifacts ----------
        preds_out = pd.DataFrame(
            {
                "id": eval_df["id"],
                "text": eval_df["text"],
                "reference": refs,
                "prediction": preds,
                "score": pred_df["score"],
                "correct": [int(r == p) for r, p in zip(refs, preds)],
            }
        )
        pred_path = os.path.join(out_dir, f"predictions_{m['short_name']}.csv")
        preds_out.to_csv(pred_path, index=False)
        mlflow.log_artifact(pred_path, artifact_path="predictions")
        mlflow.log_table(preds_out, artifact_file="tables/predictions.json")

        mpath = os.path.join(out_dir, f"metrics_{m['short_name']}.json")
        with open(mpath, "w") as f:
            json.dump({"run_name": run_name, "hf_model_id": m["hf_model_id"], **metrics}, f, indent=2)
        mlflow.log_artifact(mpath, artifact_path="metrics")

        cm_path = os.path.join(out_dir, f"confusion_matrix_{m['short_name']}.png")
        fig = plot_confusion(refs, preds, f"{run_name}\nmacro-F1={metrics['macro_f1']:.3f}", cm_path)
        mlflow.log_figure(fig, "plots/" + os.path.basename(cm_path))
        plt.close(fig)

        mlflow.log_artifact(os.path.join(C.DATA_DIR, "eval_inputs.csv"), artifact_path="data")

        # ---------- model (custom pyfunc, loadable via runs:/ or models:/) ----------
        example = eval_df[["text"]].head(3)
        signature = infer_signature(example, wrapper.predict(None, example))
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=HFSentimentModel(),
            code_paths=["hf_sentiment_model.py"],
            model_config={
                "hf_model_id": m["hf_model_id"],
                "max_length": C.MAX_LENGTH,
                "batch_size": C.BATCH_SIZE,
            },
            input_example=example,
            signature=signature,
            pip_requirements=[
                f"mlflow=={mlflow.__version__}",
                f"transformers=={transformers.__version__}",
                f"torch=={torch.__version__.split('+')[0]}",
                "pandas",
            ],
        )
        mlflow.set_tag("model_uri", f"runs:/{run.info.run_id}/model")
        print(f"[{run_name}] logged model: {info.model_uri}")

        return {
            "key": key,
            "run_id": run.info.run_id,
            "run_name": run_name,
            "hf_model_id": m["hf_model_id"],
            "short_name": m["short_name"],
            "metrics": metrics,
            "preds": preds,
            "scores": pred_df["score"].tolist(),
        }


def decide(a: dict, b: dict) -> dict:
    """Apply the Champion rule stated in README section 3 (and config.py)."""
    pa, pb = a["metrics"][C.PRIMARY_METRIC], b["metrics"][C.PRIMARY_METRIC]
    delta = pb - pa
    ratio = b["metrics"]["avg_latency_ms"] / a["metrics"]["avg_latency_ms"]

    if ratio > C.MAX_LATENCY_RATIO:
        winner, reason = "A", (
            f"Challenger rejected by guardrail: latency {ratio:.1f}x Champion "
            f"(> {C.MAX_LATENCY_RATIO:.0f}x allowed)."
        )
    elif delta >= C.MARGIN:
        winner, reason = "B", (
            f"Challenger beats Champion on {C.PRIMARY_METRIC} by {delta*100:.2f} pp "
            f"(>= {C.MARGIN*100:.0f} pp margin) and passes the latency guardrail ({ratio:.1f}x)."
        )
    elif abs(delta) < C.MARGIN:
        faster = "A" if a["metrics"]["avg_latency_ms"] <= b["metrics"]["avg_latency_ms"] else "B"
        winner, reason = faster, (
            f"Difference {delta*100:.2f} pp is within the {C.MARGIN*100:.0f} pp margin; "
            f"tie-breaker picks the faster model ({faster})."
        )
    else:
        winner, reason = "A", f"Challenger is worse by {abs(delta)*100:.2f} pp; Champion stays."
    return {"winner": winner, "delta": delta, "latency_ratio": ratio, "reason": reason}


def comparison_plots(a, b, eval_df):
    out = C.OUTPUT_DIR
    names = [a["run_name"], b["run_name"]]
    figs = {}

    # 1) bar chart, every quality metric
    qm = ["accuracy", "macro_f1", "macro_precision", "macro_recall", "f1_positive", "f1_negative"]
    x = np.arange(len(qm))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - 0.2, [a["metrics"][k] for k in qm], 0.4, label=f"A (Champion) {a['short_name']}")
    ax.bar(x + 0.2, [b["metrics"][k] for k in qm], 0.4, label=f"B (Challenger) {b['short_name']}")
    for i, k in enumerate(qm):
        ax.text(i - 0.2, a["metrics"][k] + 0.005, f"{a['metrics'][k]:.3f}", ha="center", fontsize=7)
        ax.text(i + 0.2, b["metrics"][k] + 0.005, f"{b['metrics'][k]:.3f}", ha="center", fontsize=7)
    lo = min(min(a["metrics"][k], b["metrics"][k]) for k in qm)
    ax.set_ylim(max(0, lo - 0.08), 1.0)
    ax.set_xticks(x, qm, rotation=15)
    ax.set_ylabel("score (higher is better)")
    ax.set_title("Champion vs Challenger - quality metrics")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    p = os.path.join(out, "comparison_metrics_bar.png")
    fig.savefig(p, dpi=130)
    figs[p] = fig

    # 2) latency comparison
    fig, ax = plt.subplots(figsize=(5.5, 4))
    vals = [a["metrics"]["avg_latency_ms"], b["metrics"]["avg_latency_ms"]]
    ax.bar(["A " + a["short_name"], "B " + b["short_name"]], vals, color=["tab:blue", "tab:orange"])
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.1f} ms", ha="center", va="bottom")
    ax.set_ylabel("avg latency per example (ms, lower is better)")
    ax.set_title("Inference latency on CPU")
    fig.tight_layout()
    p = os.path.join(out, "comparison_latency_bar.png")
    fig.savefig(p, dpi=130)
    figs[p] = fig

    # 3) confidence distribution, split by correct / wrong
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    refs = eval_df["reference"].tolist()
    for ax, r in zip(axes, [a, b]):
        sc = np.array(r["scores"])
        ok = np.array([p == t for p, t in zip(r["preds"], refs)])
        ax.hist(sc[ok], bins=20, range=(0.5, 1.0), alpha=0.7, label="correct")
        ax.hist(sc[~ok], bins=20, range=(0.5, 1.0), alpha=0.9, label="wrong")
        ax.set_title(r["run_name"], fontsize=9)
        ax.set_xlabel("model confidence")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("examples")
    fig.suptitle("Per-example confidence distribution (correct vs wrong)")
    fig.tight_layout()
    p = os.path.join(out, "confidence_distribution.png")
    fig.savefig(p, dpi=130)
    figs[p] = fig
    return figs


def disagreements(a, b, eval_df, k=5) -> pd.DataFrame:
    rows = []
    for i, (pa, pb) in enumerate(zip(a["preds"], b["preds"])):
        if pa != pb:
            ref = eval_df["reference"].iloc[i]
            right = "A" if pa == ref else "B"
            rows.append(
                {
                    "id": int(eval_df["id"].iloc[i]),
                    "text": eval_df["text"].iloc[i],
                    "reference": ref,
                    f"pred_A_{a['short_name']}": pa,
                    f"score_A": a["scores"][i],
                    f"pred_B_{b['short_name']}": pb,
                    f"score_B": b["scores"][i],
                    "correct_model": right,
                    "comment": "",
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(C.OUTPUT_DIR, "disagreements_all.csv"), index=False)
    return df.head(k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=C.EVAL_SAMPLES)
    args = ap.parse_args()

    set_seed(C.SEED)
    os.makedirs(C.OUTPUT_DIR, exist_ok=True)
    mlflow.set_tracking_uri(C.TRACKING_URI)
    mlflow.set_experiment(C.EXPERIMENT_NAME)

    eval_df = build_eval_set(args.samples, C.SEED)

    with mlflow.start_run(run_name=f"compare_{C.TASK}") as parent:
        mlflow.set_tags(
            {
                "student_usn": C.STUDENT_USN,
                "student_name": C.STUDENT_NAME,
                "dataset_name": C.DATASET_NAME,
                "run_type": "summary",
                "transformers_version": transformers.__version__,
                "torch_version": torch.__version__,
                "mlflow_version": mlflow.__version__,
            }
        )
        mlflow.log_params(
            {
                "task": C.TASK,
                "eval_samples": len(eval_df),
                "seed": C.SEED,
                "champion_candidate": C.MODELS["A"]["hf_model_id"],
                "challenger_candidate": C.MODELS["B"]["hf_model_id"],
                "primary_metric": C.PRIMARY_METRIC,
                "metric_direction": "higher_is_better",
                "margin": C.MARGIN,
                "max_latency_ratio": C.MAX_LATENCY_RATIO,
                "tie_breaker": "faster_model_within_margin",
            }
        )

        a = run_model("A", eval_df, args)
        b = run_model("B", eval_df, args)
        d = decide(a, b)
        winner = a if d["winner"] == "A" else b
        loser = b if d["winner"] == "A" else a

        mlflow.log_metrics(
            {
                "delta_primary_metric": d["delta"],
                "delta_accuracy": b["metrics"]["accuracy"] - a["metrics"]["accuracy"],
                "latency_ratio_B_over_A": d["latency_ratio"],
                f"A_{C.PRIMARY_METRIC}": a["metrics"][C.PRIMARY_METRIC],
                f"B_{C.PRIMARY_METRIC}": b["metrics"][C.PRIMARY_METRIC],
                "A_avg_latency_ms": a["metrics"]["avg_latency_ms"],
                "B_avg_latency_ms": b["metrics"]["avg_latency_ms"],
            }
        )
        mlflow.set_tags(
            {
                "winner": winner["run_name"],
                "winner_hf_model_id": winner["hf_model_id"],
                "winner_run_id": winner["run_id"],
                "loser_run_id": loser["run_id"],
                "champion_run_id": winner["run_id"],
                "challenger_run_id": loser["run_id"],
                "decision_reason": d["reason"],
            }
        )

        figs = comparison_plots(a, b, eval_df)
        for path, fig in figs.items():
            mlflow.log_figure(fig, "plots/" + os.path.basename(path))
            plt.close(fig)

        dis = disagreements(a, b, eval_df)
        dis_path = os.path.join(C.OUTPUT_DIR, "disagreements_top5.csv")
        # keep hand-written comments if the file already exists for the same ids
        if os.path.exists(dis_path):
            old = pd.read_csv(dis_path)
            if "comment" in old and list(old["id"]) == list(dis["id"]):
                dis["comment"] = old["comment"].fillna("")
        dis.to_csv(dis_path, index=False)
        mlflow.log_artifact(dis_path, artifact_path="qualitative")

        summary = {
            "experiment": C.EXPERIMENT_NAME,
            "summary_run_id": parent.info.run_id,
            "dataset": C.DATASET_NAME,
            "eval_samples": len(eval_df),
            "seed": C.SEED,
            "rule": {
                "primary_metric": C.PRIMARY_METRIC,
                "direction": "higher is better",
                "margin": C.MARGIN,
                "tie_breaker": "within margin -> prefer faster model",
                "guardrail": f"reject Challenger if avg latency > {C.MAX_LATENCY_RATIO}x Champion",
            },
            "A_champion_candidate": {k: a[k] for k in ("run_id", "run_name", "hf_model_id", "metrics")},
            "B_challenger_candidate": {k: b[k] for k in ("run_id", "run_name", "hf_model_id", "metrics")},
            "delta_primary_metric": d["delta"],
            "latency_ratio_B_over_A": d["latency_ratio"],
            "winner": winner["run_name"],
            "winner_hf_model_id": winner["hf_model_id"],
            "decision_reason": d["reason"],
            "n_disagreements": int(sum(x != y for x, y in zip(a["preds"], b["preds"]))),
        }
        spath = os.path.join(C.OUTPUT_DIR, "metrics_summary.json")
        with open(spath, "w") as f:
            json.dump(summary, f, indent=2)
        mlflow.log_artifact(spath)
        mlflow.log_artifact(os.path.join(C.DATA_DIR, "eval_inputs.csv"), artifact_path="data")
        if os.path.exists("README.md"):
            mlflow.log_artifact("README.md")

        print("\n================ DECISION ================")
        print(f"A {a['hf_model_id']}: {C.PRIMARY_METRIC}={a['metrics'][C.PRIMARY_METRIC]:.4f} "
              f"latency={a['metrics']['avg_latency_ms']:.1f} ms")
        print(f"B {b['hf_model_id']}: {C.PRIMARY_METRIC}={b['metrics'][C.PRIMARY_METRIC]:.4f} "
              f"latency={b['metrics']['avg_latency_ms']:.1f} ms")
        print(f"delta (B - A) = {d['delta']*100:+.2f} pp, latency ratio = {d['latency_ratio']:.2f}x")
        print(f"WINNER: {winner['run_name']}  -> {d['reason']}")
        print(f"Summary run id: {parent.info.run_id}")
        print("Next: python register_champion.py")


if __name__ == "__main__":
    main()

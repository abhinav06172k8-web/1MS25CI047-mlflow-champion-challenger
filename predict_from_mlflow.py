"""predict_from_mlflow.py - load the Champion by its registry alias and predict on NEW data.

* Loads models:/sentiment-model@champion (no run ID, no version number in the code).
* Never imports or calls Hugging Face directly: every prediction goes through MLflow.
* Prints which registry version and Hugging Face model ID were loaded (read from the registry
  and the logged model's metadata), then saves outputs/new_data_predictions.csv.

Usage:
    python predict_from_mlflow.py                    # Champion only
    python predict_from_mlflow.py --with-challenger  # also show the Challenger side by side
"""
import argparse
import os

import mlflow
import pandas as pd
from mlflow import MlflowClient

import config as C


def load_alias(client: MlflowClient, alias: str):
    uri = f"models:/{C.REGISTERED_MODEL_NAME}@{alias}"
    mv = client.get_model_version_by_alias(C.REGISTERED_MODEL_NAME, alias)
    model = mlflow.pyfunc.load_model(uri)
    # model_config is stored inside the logged MLmodel file -> proves which HF checkpoint is inside
    cfg = model.model_config or {}
    hf_id = cfg.get("hf_model_id") or mv.tags.get("hf_model_id", "unknown")
    print(f"Loaded {uri}")
    print(f"   registry version : {mv.version}")
    print(f"   source run       : {mv.run_id} ({mv.tags.get('run_name', '')})")
    print(f"   hf_model_id      : {hf_id}")
    return model, mv, hf_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-challenger", action="store_true")
    args = ap.parse_args()

    mlflow.set_tracking_uri(C.TRACKING_URI)
    client = MlflowClient()

    new_df = pd.read_csv(os.path.join(C.DATA_DIR, "new_data.csv"))
    eval_texts = set(pd.read_csv(os.path.join(C.DATA_DIR, "eval_inputs.csv"))["text"].str.lower())
    overlap = [t for t in new_df["text"] if t.lower() in eval_texts]
    assert not overlap, f"new_data.csv overlaps the evaluation set: {overlap}"

    model, mv, hf_id = load_alias(client, "champion")
    out = model.predict(new_df[["text"]])
    result = pd.DataFrame(
        {
            "id": new_df["id"],
            "text": new_df["text"],
            "expected": new_df.get("expected", ""),
            "champion_prediction": out["prediction"],
            "champion_score": out["score"],
        }
    )
    result["alias"] = "champion"
    result["model_version"] = mv.version
    result["hf_model_id"] = hf_id

    if args.with_challenger:
        print()
        cmodel, cmv, chf = load_alias(client, "challenger")
        cout = cmodel.predict(new_df[["text"]])
        result["challenger_prediction"] = cout["prediction"]
        result["challenger_score"] = cout["score"]
        result["challenger_version"] = cmv.version
        result["challenger_hf_model_id"] = chf

    os.makedirs(C.OUTPUT_DIR, exist_ok=True)
    path = os.path.join(C.OUTPUT_DIR, "new_data_predictions.csv")
    result.to_csv(path, index=False)

    show = ["id", "expected", "champion_prediction", "champion_score"]
    if args.with_challenger:
        show += ["challenger_prediction", "challenger_score"]
    view = result[show].copy()
    view.insert(1, "text", result["text"].str.slice(0, 55))
    print("\nPredictions on new data (not in the evaluation set):")
    print(view.to_string(index=False))
    if "expected" in new_df:
        acc = (result["champion_prediction"] == result["expected"]).mean()
        print(f"\nChampion agreed with my expected label on {acc:.0%} of the new examples.")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()

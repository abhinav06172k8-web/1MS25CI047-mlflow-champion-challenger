"""register_champion.py - register both logged models and set the champion / challenger aliases.

It finds the most recent `compare_sentiment` summary run in the experiment, reads the
decision that evaluate.py logged there (tags champion_run_id / challenger_run_id), registers
both models under one registered model name and points the aliases at the right versions.
No run IDs are hard-coded, so it works on any machine.

Usage:
    python register_champion.py                 # follow the logged decision
    python register_champion.py --swap          # stretch goal: flip the aliases
"""
import argparse

import mlflow
from mlflow import MlflowClient

import config as C


def latest_summary_run(client: MlflowClient):
    exp = client.get_experiment_by_name(C.EXPERIMENT_NAME)
    if exp is None:
        raise SystemExit(f"Experiment '{C.EXPERIMENT_NAME}' not found. Run evaluate.py first.")
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string=f"attributes.run_name = 'compare_{C.TASK}' and attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise SystemExit("No finished compare_sentiment run found. Run evaluate.py first.")
    return runs[0]


def find_version(client: MlflowClient, run_id: str):
    for mv in client.search_model_versions(f"name = '{C.REGISTERED_MODEL_NAME}'"):
        if mv.run_id == run_id:
            return mv
    return None


def register(client: MlflowClient, run_id: str):
    """Register the model logged in run_id (re-uses an existing version for the same run)."""
    mv = find_version(client, run_id)
    if mv is not None:
        return mv
    mv = mlflow.register_model(f"runs:/{run_id}/model", C.REGISTERED_MODEL_NAME)
    run = client.get_run(run_id)
    for k in ("hf_model_id", "role", "task"):
        client.set_model_version_tag(C.REGISTERED_MODEL_NAME, mv.version, k, run.data.params.get(k, ""))
    client.set_model_version_tag(C.REGISTERED_MODEL_NAME, mv.version, "run_name", run.info.run_name)
    client.set_model_version_tag(C.REGISTERED_MODEL_NAME, mv.version, "macro_f1",
                                 f"{run.data.metrics.get('macro_f1', float('nan')):.4f}")
    client.set_model_version_tag(C.REGISTERED_MODEL_NAME, mv.version, "student_usn", C.STUDENT_USN)
    return mv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--swap", action="store_true", help="flip champion/challenger aliases")
    args = ap.parse_args()

    mlflow.set_tracking_uri(C.TRACKING_URI)
    client = MlflowClient()

    summary = latest_summary_run(client)
    champ_run = summary.data.tags["champion_run_id"]
    chall_run = summary.data.tags["challenger_run_id"]
    print(f"Summary run  : {summary.info.run_id}")
    print(f"Logged winner: {summary.data.tags['winner']}")

    try:
        client.create_registered_model(
            C.REGISTERED_MODEL_NAME,
            description="HF sentiment Champion-Challenger (SST-2) - USN " + C.STUDENT_USN,
        )
    except Exception:
        pass  # already exists

    # register Model A first then Model B so version numbers follow A=1, B=2 on a fresh registry
    run_ids = {}
    for key in ("A", "B"):
        rid = champ_run if client.get_run(champ_run).data.tags.get("model_key") == key else chall_run
        run_ids[key] = rid
    versions = {k: register(client, rid) for k, rid in run_ids.items()}

    champ_v = find_version(client, champ_run).version
    chall_v = find_version(client, chall_run).version
    if args.swap:
        champ_v, chall_v = chall_v, champ_v
        print("--swap given: flipping aliases (stretch goal)")

    client.set_registered_model_alias(C.REGISTERED_MODEL_NAME, "champion", champ_v)
    client.set_registered_model_alias(C.REGISTERED_MODEL_NAME, "challenger", chall_v)

    print(f"\nRegistered model: {C.REGISTERED_MODEL_NAME}")
    print(f"{'alias':<12}{'version':<9}{'hf_model_id':<52}{'macro_f1'}")
    for alias in ("champion", "challenger"):
        mv = client.get_model_version_by_alias(C.REGISTERED_MODEL_NAME, alias)
        print(f"@{alias:<11}{mv.version:<9}{mv.tags.get('hf_model_id',''):<52}{mv.tags.get('macro_f1','')}")
    print("\nNext: python predict_from_mlflow.py")


if __name__ == "__main__":
    main()

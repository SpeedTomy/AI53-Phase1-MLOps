"""Verify real MLflow records and export validation evidence without test inference."""
from __future__ import annotations

import json
from pathlib import Path

import mlflow
import yaml
from mlflow.tracking import MlflowClient


def main() -> None:
    params = yaml.safe_load(Path("params.yaml").read_text(encoding="utf-8"))
    mlflow.set_tracking_uri(params["tracking"]["uri"])
    client = MlflowClient()
    experiment = client.get_experiment_by_name(params["tracking"]["experiment"])
    if experiment is None:
        raise RuntimeError("The course experiment does not exist")
    runs = client.search_runs([experiment.experiment_id], order_by=["attributes.start_time ASC"])
    records = []
    required_parameters = {
        "image_size", "batch_size", "epochs", "learning_rate", "dropout", "random_state",
        "augmentation_enabled", "horizontal_flip", "rotation_degrees", "device",
        "training_data_identity",
    }
    for run in runs:
        run_id = run.info.run_id
        if run.info.status != "FINISHED":
            raise RuntimeError(f"Run {run_id} is not FINISHED")
        if not required_parameters.issubset(run.data.params):
            raise RuntimeError(f"Missing parameters in {run_id}")
        epochs = int(run.data.params["epochs"])
        history = {}
        for name in ("train_loss", "val_loss", "val_accuracy", "val_macro_f1"):
            points = sorted(client.get_metric_history(run_id, name), key=lambda item: item.step)
            if [point.step for point in points] != list(range(1, epochs + 1)):
                raise RuntimeError(f"Missing epoch steps for {name} in {run_id}")
            history[name] = [{"step": item.step, "value": item.value} for item in points]
            if points[-1].value != run.data.metrics[name]:
                raise RuntimeError(f"Final metric does not match history in {run_id}")
        artifacts = sorted(item.path for item in client.list_artifacts(run_id, "reports"))
        expected_artifacts = {f"reports/{name}" for name in (
            "training_curves.png", "train_summary.json", "run_manifest.json"
        )}
        if not expected_artifacts.issubset(artifacts):
            raise RuntimeError(f"Missing report artifacts in {run_id}")
        datasets = run.inputs.dataset_inputs
        matching = [item for item in datasets if item.dataset.name == "bean-training"
                    and any(tag.key == "mlflow.data.context" and tag.value == "training"
                            for tag in item.tags)]
        if not matching:
            raise RuntimeError(f"Missing training Dataset in {run_id}")
        model_uri = f"runs:/{run_id}/bean-cnn"
        info = mlflow.models.get_model_info(model_uri)
        if info.signature is None or info.saved_input_example_info is None:
            raise RuntimeError(f"Missing model input contract in {run_id}")
        if info.metadata != {"class_names": ["angular_leaf_spot", "bean_rust", "healthy"],
                             "image_size": int(run.data.params["image_size"])}:
            raise RuntimeError(f"Unexpected model metadata in {run_id}")
        records.append({
            "label": chr(ord("A") + len(records)), "run_id": run_id,
            "status": run.info.status, "parameters": run.data.params,
            "metrics": run.data.metrics, "history": history, "artifacts": artifacts,
            "dataset": matching[0].dataset.to_dictionary(),
            "model_uri": model_uri, "signature": info.signature.to_dict(),
            "input_example": info.saved_input_example_info, "metadata": info.metadata,
        })
    output = Path("reports/experiment_comparison.json")
    output.write_text(json.dumps(records, indent=2, sort_keys=True), encoding="utf-8")
    for record in records:
        metrics = record["metrics"]
        print(f"{record['label']} {record['run_id']} "
              f"accuracy={metrics['val_accuracy']:.6f} macro_f1={metrics['val_macro_f1']:.6f} "
              f"seconds={metrics['training_seconds']:.3f} | all logging checks PASS")


if __name__ == "__main__":
    main()

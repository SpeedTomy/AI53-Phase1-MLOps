from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from importlib import metadata
from pathlib import Path

import torch


def training_data_identity(
    train_manifest: Path = Path("data/processed/manifests/train.csv"),
    validation_manifest: Path = Path("data/processed/manifests/validation.csv"),
) -> str:
    """Return a deterministic identity for the training + validation manifests.

    This helper is provided because the final evaluator uses the same identity
    to verify that the selected MLflow Run matches the current prepared data.
    Do not hard-code the value.
    """
    paths = (train_manifest, validation_manifest)
    if any(not path.exists() for path in paths):
        return "unavailable"

    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def build_run_manifest(params: dict, device: str, mlflow_run_id: str) -> dict:
    """Capture code, data and execution state before training writes outputs."""
    git = {"commit_sha": None, "dirty": None, "status": "unavailable"}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=False
        )
        if commit.returncode == 0 and status.returncode == 0:
            git.update(commit_sha=commit.stdout.strip(), dirty=bool(status.stdout.strip()),
                       status="available")
    except OSError:
        # A missing Git executable must not discard the other provenance fields.
        pass

    environment = {"python": platform.python_version(), "platform": platform.platform(),
                   "machine": platform.machine(), "cuda_runtime": torch.version.cuda,
                   "torch_num_threads": torch.get_num_threads()}
    for package in ("torch", "torchvision", "dvc", "mlflow", "numpy", "pillow"):
        try:
            environment[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            environment[package] = "unavailable"
    return {
        "git": git,
        "data": {"training_data_identity": training_data_identity()},
        "environment": environment,
        "configuration": {
            "image_size": int(params["data"]["image_size"]),
            **params["train"],
            "augmentation": dict(params["augmentation"]),
        },
        "execution": {"device": str(device),
                      "random_state": int(params["train"]["random_state"]),
                      "mlflow_run_id": mlflow_run_id},
    }


def write_run_manifest(params: dict, device: str, mlflow_run_id: str, path: Path) -> dict:
    manifest = build_run_manifest(params, device, mlflow_run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest

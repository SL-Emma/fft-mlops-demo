import json
import pickle
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
    }

    return model, metrics


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
        ).strip()
    except subprocess.CalledProcessError:
        return "unknown"


if __name__ == "__main__":
    df = pd.read_csv("data/processed/bearing_features.csv")

    feature_columns = [
        col for col in df.columns
        if col != "Label"
    ]

    X = df[feature_columns]
    y = df["Label"]

    model, metrics = train_model(X, y)

    # -----------------------------
    # Version information
    # -----------------------------

    model_version = "v1"
    dataset_version = "v1"

    # -----------------------------
    # Create directories
    # -----------------------------

    model_dir = Path("models")
    metrics_dir = Path("metrics")
    registry_dir = Path("model_registry") / model_version

    model_dir.mkdir(exist_ok=True)
    metrics_dir.mkdir(exist_ok=True)
    registry_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Save trained model
    # -----------------------------

    model_path = model_dir / f"bearing_model_{model_version}.pkl"

    with open(model_path, "wb") as file:
        pickle.dump(model, file)

    # -----------------------------
    # Save metrics
    # -----------------------------

    metrics_path = metrics_dir / "metrics.json"

    with open(metrics_path, "w") as file:
        json.dump(metrics, file, indent=2)

    # -----------------------------
    # Create metadata
    # -----------------------------

    metadata = {
        "model_version": model_version,
        "dataset_version": dataset_version,
        "git_commit": get_git_commit(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_type": "RandomForestClassifier",
        "n_estimators": 100,
        "random_state": 42,
        "accuracy": metrics["accuracy"],
        "artifact": str(model_path),
        "status": "validated",
    }

    metadata_path = registry_dir / "metadata.json"

    with open(metadata_path, "w") as file:
        json.dump(metadata, file, indent=2)

    # Also place metrics inside registry entry
    registry_metrics_path = registry_dir / "metrics.json"

    with open(registry_metrics_path, "w") as file:
        json.dump(metrics, file, indent=2)

    print("Training completed.")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Model: {model_path}")
    print(f"Registry entry: {registry_dir}")

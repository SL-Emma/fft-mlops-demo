import json
import pickle

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
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "train_samples": len(X_train),
        "test_samples": len(X_test)
    }
    return model, metrics


if __name__ == "__main__":
    X = pd.read_csv("data/processed/bearing_features.csv")
    y = pd.read_csv("data/raw/bearing_features_N15_M07_F10.csv")["Label"]

    model, metrics = train_model(X, y)

    with open("models/model.pkl", "wb") as file:
        pickle.dump(model, file)

    with open("metrics/metrics.json", "w") as file:
        json.dump(metrics, file, indent=2)

    print(metrics)

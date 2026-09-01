import pandas as pd
from src.train import train_model


def test_training_smoke():
    data = pd.read_csv("data/raw/bearing_features_N15_M07_F10.csv")
    sample = pd.concat([
        data[data["Label"] == 0].head(7),
        data[data["Label"] == 1].head(7),
        data[data["Label"] == 2].head(6)
    ])

    X = sample.drop(columns=["Bearing", "File", "Label"])
    y = sample["Label"]
    model, _ = train_model(X, y)

    assert model is not None

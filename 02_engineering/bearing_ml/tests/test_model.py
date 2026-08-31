import pickle

import pandas as pd


def test_prediction():
    with open("models/model.pkl", "rb") as file:
        model = pickle.load(file)

    features = pd.read_csv("data/processed/bearing_features.csv").head(1)
    prediction = model.predict(features)[0]

    assert prediction in {0, 1, 2}


def test_model_save_and_load(tmp_path):
    with open("models/model.pkl", "rb") as file:
        model = pickle.load(file)

    features = pd.read_csv("data/processed/bearing_features.csv").head(1)
    prediction_before = model.predict(features)[0]

    model_file = tmp_path / "model.pkl"
    with open(model_file, "wb") as file:
        pickle.dump(model, file)
    with open(model_file, "rb") as file:
        loaded_model = pickle.load(file)

    prediction_after = loaded_model.predict(features)[0]

    assert prediction_before == prediction_after

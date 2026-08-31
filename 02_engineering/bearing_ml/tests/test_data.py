import pandas as pd


EXPECTED_FEATURES = [
    "I1_RMS",
    "I1_Variance",
    "I1_PeakToPeak",
    "I1_CrestFactor",
    "I1_Skewness",
    "I1_Kurtosis",
    "I1_LineIntegral",
    "I1_FFTPeak",
    "I1_FFTEnergy",
    "I2_RMS",
    "I2_Variance",
    "I2_PeakToPeak",
    "I2_CrestFactor",
    "I2_Skewness",
    "I2_Kurtosis",
    "I2_LineIntegral",
    "I2_FFTPeak",
    "I2_FFTEnergy"
]


def test_data():
    data = pd.read_csv("data/raw/bearing_features_N15_M07_F10.csv")
    feature_columns = [
        column for column in data.columns
        if column not in ["Bearing", "File", "Label"]
    ]

    for column in EXPECTED_FEATURES:
        assert column in data.columns, f"Expected column: {column}"

    assert len(feature_columns) == 18
    assert "Label" in data.columns
    assert set(data["Label"].unique()) <= {0, 1, 2}
    assert not data.isna().any().any()

import pandas as pd

df = pd.read_csv(
    "data/raw/bearing_features_N15_M07_F10.csv",
    on_bad_lines="skip"
)

feature_columns = [
    col for col in df.columns
    if col not in ["Bearing", "File", "Label"]
]

if len(feature_columns) != 18:
    raise ValueError("Expected exactly 18 features.")

columns = feature_columns + ["Label"]

df[columns] = df[columns].apply(
    pd.to_numeric,
    errors="coerce"
)

print("NaN values:", df[columns].isna().sum().sum())

df = df.dropna(subset=columns)

df = df[df["Label"].isin([0, 1, 2])]
df["Label"] = df["Label"].astype(int)

df[columns].to_csv(
    "data/processed/bearing_features.csv",
    index=False
)

print(f"Prepared {len(df)} samples.")

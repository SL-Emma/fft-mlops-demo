from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import ks_2samp


# ============================================================
# Pfade
# ============================================================

MONITORING_DIR = Path(__file__).resolve().parent
REPO_ROOT = MONITORING_DIR.parent

DATA_DIR = (
    REPO_ROOT
    / "02_engineering"
    / "bearing_ml"
    / "data"
    / "raw"
)

REFERENCE_FILE = (
    DATA_DIR
    / "bearing_features_N15_M07_F10.csv"
)

CURRENT_FILE = (
    DATA_DIR
    / "bearing_features_N15_M07_F10.csv"
)

MODEL_FILE = (
    REPO_ROOT
    / "02_engineering"
    / "bearing_ml"
    / "models"
    / "model.pkl"
)


# ============================================================
# Streamlit Setup
# ============================================================

st.set_page_config(
    page_title="Bearing ML Monitoring",
    layout="wide"
)

st.title("⚙️ Bearing ML Monitoring")

st.caption(
    "Feature drift, predictions and model confidence"
)


# ============================================================
# Dateien prüfen
# ============================================================

if not REFERENCE_FILE.exists():
    st.error(
        f"Reference file not found:\n{REFERENCE_FILE}"
    )
    st.stop()

if not CURRENT_FILE.exists():
    st.error(
        f"Current file not found:\n{CURRENT_FILE}"
    )
    st.stop()

if not MODEL_FILE.exists():
    st.error(
        f"Model file not found:\n{MODEL_FILE}"
    )
    st.stop()


# ============================================================
# Daten laden
# ============================================================

try:
    reference = pd.read_csv(REFERENCE_FILE)
except Exception as e:
    st.error(
        f"Could not load reference CSV:\n{e}"
    )
    st.stop()

try:
    current = pd.read_csv(CURRENT_FILE)
except Exception as e:
    st.error(
        f"Could not load current CSV:\n{e}"
    )
    st.stop()


# ============================================================
# Leere Dateien prüfen
# ============================================================

if reference.empty:
    st.error("Reference CSV is empty.")
    st.stop()

if current.empty:
    st.error("Current CSV is empty.")
    st.stop()


# ============================================================
# Modell laden
# ============================================================

try:
    model = joblib.load(MODEL_FILE)
except Exception as e:
    st.error(
        f"Could not load model:\n{e}"
    )
    st.stop()


# ============================================================
# Feature-Spalten bestimmen
# ============================================================

ignore_columns = [
    "Label",
    "Bearing",
    "File",
    "Prediction",
    "Confidence"
]

feature_columns = [
    col
    for col in reference.columns
    if col not in ignore_columns
    and col in current.columns
    and pd.api.types.is_numeric_dtype(reference[col])
]


if len(feature_columns) == 0:
    st.error(
        "No common numeric feature columns found."
    )
    st.stop()


# ============================================================
# Nur Feature-Spalten fürs Modell
# ============================================================

X_current = current[feature_columns].copy()


# ============================================================
# NaN prüfen
# ============================================================

if X_current.isna().any().any():

    st.warning(
        "NaN values detected in current data. "
        "Affected rows are removed."
    )

    valid_rows = ~X_current.isna().any(axis=1)

    X_current = X_current.loc[valid_rows]
    current = current.loc[valid_rows].copy()


if len(X_current) == 0:
    st.error("No valid current samples available.")
    st.stop()


# ============================================================
# Prediction
# ============================================================

try:
    predictions = model.predict(X_current)
except Exception as e:
    st.error(
        "Prediction failed.\n\n"
        f"{e}\n\n"
        "Possible cause: feature columns do not match "
        "the features used during training."
    )
    st.stop()


current["Prediction"] = predictions


class_names = {
    0: "Healthy",
    1: "Inner Ring",
    2: "Outer Ring"
}


# ============================================================
# Confidence
# ============================================================

confidence_available = hasattr(
    model,
    "predict_proba"
)

if confidence_available:

    try:
        probabilities = model.predict_proba(
            X_current
        )

        confidence = np.max(
            probabilities,
            axis=1
        )

        current["Confidence"] = confidence

        average_confidence = (
            confidence.mean()
        )

    except Exception:
        confidence_available = False
        average_confidence = None

else:
    average_confidence = None


# ============================================================
# Data Drift mit KS-Test
# ============================================================

drift_results = []

for feature in feature_columns:

    ref_values = (
        reference[feature]
        .dropna()
    )

    current_values = (
        current[feature]
        .dropna()
    )

    if (
        len(ref_values) == 0
        or len(current_values) == 0
    ):
        continue

    statistic, p_value = ks_2samp(
        ref_values,
        current_values
    )

    drift_detected = (
        p_value < 0.05
    )

    drift_results.append(
        {
            "Feature": feature,
            "KS Statistic": statistic,
            "p-value": p_value,
            "Drift": drift_detected
        }
    )


drift_df = pd.DataFrame(
    drift_results
)


if drift_df.empty:
    st.error(
        "Drift analysis could not be performed."
    )
    st.stop()


number_drifted = int(
    drift_df["Drift"].sum()
)


# ============================================================
# Header / Status
# ============================================================

st.subheader("Current Status")

col1, col2, col3, col4 = st.columns(4)


col1.metric(
    "Current Samples",
    len(current)
)


col2.metric(
    "Features with Drift",
    f"{number_drifted} / "
    f"{len(feature_columns)}"
)


healthy_percentage = (
    np.mean(predictions == 0)
    * 100
)

col3.metric(
    "Predicted Healthy",
    f"{healthy_percentage:.1f} %"
)


if average_confidence is not None:

    col4.metric(
        "Average Confidence",
        f"{average_confidence * 100:.1f} %"
    )

else:

    col4.metric(
        "Average Confidence",
        "N/A"
    )


# ============================================================
# Gesamtstatus
# ============================================================

if number_drifted == 0:

    st.success(
        "✅ No significant feature drift detected."
    )

elif number_drifted <= 2:

    st.warning(
        f"⚠️ Drift detected in "
        f"{number_drifted} feature(s)."
    )

else:

    st.error(
        f"🚨 Significant drift detected in "
        f"{number_drifted} features."
    )


# ============================================================
# Feature Distribution
# ============================================================

st.divider()

st.subheader("Feature Drift")

selected_feature = st.selectbox(
    "Select feature",
    feature_columns
)


fig, ax = plt.subplots(
    figsize=(10, 4)
)

ax.hist(
    reference[
        selected_feature
    ].dropna(),
    bins=30,
    alpha=0.5,
    density=True,
    label="Training / Reference"
)

ax.hist(
    current[
        selected_feature
    ].dropna(),
    bins=30,
    alpha=0.5,
    density=True,
    label="Current / Production"
)

ax.set_xlabel(
    selected_feature
)

ax.set_ylabel(
    "Density"
)

ax.set_title(
    f"Distribution of {selected_feature}"
)

ax.legend()

ax.grid(
    alpha=0.3
)

st.pyplot(fig)


# ============================================================
# Selected Feature Metrics
# ============================================================

selected_result = drift_df[
    drift_df["Feature"]
    == selected_feature
].iloc[0]


c1, c2, c3 = st.columns(3)


c1.metric(
    "Reference Mean",
    f"{reference[selected_feature].mean():.4f}"
)


c2.metric(
    "Current Mean",
    f"{current[selected_feature].mean():.4f}"
)


c3.metric(
    "KS Statistic",
    f"{selected_result['KS Statistic']:.3f}"
)


if selected_result["Drift"]:

    st.warning(
        f"⚠️ Drift detected for "
        f"{selected_feature} "
        f"(p = "
        f"{selected_result['p-value']:.4g})"
    )

else:

    st.success(
        f"✅ No significant drift for "
        f"{selected_feature}"
    )


# ============================================================
# Drift Übersicht
# ============================================================

st.subheader(
    "Drift Overview"
)

display_drift = (
    drift_df.copy()
)

display_drift["Status"] = np.where(
    display_drift["Drift"],
    "⚠️ Drift",
    "✅ OK"
)

display_drift = display_drift[
    [
        "Feature",
        "KS Statistic",
        "p-value",
        "Status"
    ]
]

display_drift = (
    display_drift
    .sort_values(
        "KS Statistic",
        ascending=False
    )
)


st.dataframe(
    display_drift,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# Prediction Distribution
# ============================================================

st.divider()

st.subheader(
    "Prediction Distribution"
)


prediction_counts = (
    pd.Series(predictions)
    .value_counts()
    .reindex(
        [0, 1, 2],
        fill_value=0
    )
)


prediction_df = pd.DataFrame(
    {
        "Condition": [
            class_names[0],
            class_names[1],
            class_names[2]
        ],
        "Count": [
            prediction_counts[0],
            prediction_counts[1],
            prediction_counts[2]
        ]
    }
)


st.bar_chart(
    prediction_df,
    x="Condition",
    y="Count"
)


# ============================================================
# Confidence Distribution
# ============================================================

if confidence_available:

    st.subheader(
        "Prediction Confidence"
    )

    fig2, ax2 = plt.subplots(
        figsize=(10, 4)
    )

    ax2.hist(
        current["Confidence"],
        bins=20
    )

    ax2.set_xlabel(
        "Confidence"
    )

    ax2.set_ylabel(
        "Number of predictions"
    )

    ax2.set_xlim(
        0,
        1
    )

    ax2.grid(
        alpha=0.3
    )

    st.pyplot(fig2)


# ============================================================
# Einzelne Predictions
# ============================================================

st.divider()

st.subheader(
    "Current Predictions"
)


prediction_table = pd.DataFrame()


prediction_table["Prediction"] = [
    class_names.get(
        int(p),
        str(p)
    )
    for p in predictions
]


if confidence_available:

    prediction_table[
        "Confidence [%]"
    ] = (
        current["Confidence"]
        * 100
    ).round(1)


st.dataframe(
    prediction_table,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# Model Performance
# ============================================================

st.divider()

st.subheader(
    "Model Performance"
)

st.info(
    "Ground truth is currently unavailable. "
    "Accuracy, precision and recall cannot "
    "be calculated for production data."
)


# ============================================================
# Debug / Dateipfade
# ============================================================

with st.expander(
    "Show data sources"
):

    st.write(
        "Reference:"
    )

    st.code(
        str(REFERENCE_FILE)
    )

    st.write(
        "Current:"
    )

    st.code(
        str(CURRENT_FILE)
    )

    st.write(
        "Model:"
    )

    st.code(
        str(MODEL_FILE)
    )


# ============================================================
# Footer
# ============================================================

st.caption(
    "Demo monitoring dashboard. "
    "Feature drift is detected using a "
    "Kolmogorov-Smirnov test."
)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

# ------------------------------------------------------------
# Page setup
# ------------------------------------------------------------

st.set_page_config(
    page_title="Bearing ML Monitoring",
    layout="wide"
)

st.title("⚙️ Bearing ML Monitoring Dashboard")

st.write(
    "Comparison of training/reference data with current production data."
)

# ------------------------------------------------------------
# Upload data
# ------------------------------------------------------------

st.sidebar.header("Data")

reference_file = st.sidebar.file_uploader(
    "Reference / Training Data",
    type="csv"
)

current_file = st.sidebar.file_uploader(
    "Current / Production Data",
    type="csv"
)

if reference_file is None or current_file is None:
    st.info("Upload a reference CSV and a current CSV to start monitoring.")
    st.stop()

reference = pd.read_csv(reference_file)
current = pd.read_csv(current_file)

# ------------------------------------------------------------
# Determine feature columns
# ------------------------------------------------------------

ignore_columns = [
    "Label",
    "Bearing",
    "File",
    "Prediction",
    "prediction",
    "Confidence",
    "confidence"
]

feature_columns = [
    col for col in reference.columns
    if col not in ignore_columns
    and col in current.columns
    and pd.api.types.is_numeric_dtype(reference[col])
]

if len(feature_columns) == 0:
    st.error("No common numeric feature columns found.")
    st.stop()

# ------------------------------------------------------------
# Drift calculation
# Kolmogorov-Smirnov test
# ------------------------------------------------------------

drift_results = []

for feature in feature_columns:

    ref_values = reference[feature].dropna()
    current_values = current[feature].dropna()

    if len(ref_values) == 0 or len(current_values) == 0:
        continue

    statistic, p_value = ks_2samp(
        ref_values,
        current_values
    )

    drift = p_value < 0.05

    drift_results.append({
        "Feature": feature,
        "KS Statistic": statistic,
        "p-value": p_value,
        "Drift": drift
    })

drift_df = pd.DataFrame(drift_results)

number_drifted = drift_df["Drift"].sum()

# ------------------------------------------------------------
# Overview metrics
# ------------------------------------------------------------

st.subheader("Monitoring Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Reference Samples",
    len(reference)
)

col2.metric(
    "Current Samples",
    len(current)
)

col3.metric(
    "Features monitored",
    len(feature_columns)
)

col4.metric(
    "Features with Drift",
    f"{number_drifted} / {len(feature_columns)}"
)

# ------------------------------------------------------------
# Global status
# ------------------------------------------------------------

if number_drifted == 0:
    st.success("✅ No significant data drift detected.")

elif number_drifted <= 2:
    st.warning(
        f"⚠️ Drift detected in {number_drifted} feature(s)."
    )

else:
    st.error(
        f"🚨 Significant drift detected in {number_drifted} features."
    )

# ------------------------------------------------------------
# Feature distribution
# ------------------------------------------------------------

st.divider()

st.subheader("Feature Distribution")

selected_feature = st.selectbox(
    "Select feature",
    feature_columns
)

fig, ax = plt.subplots(figsize=(10, 4))

ax.hist(
    reference[selected_feature].dropna(),
    bins=30,
    alpha=0.5,
    density=True,
    label="Reference"
)

ax.hist(
    current[selected_feature].dropna(),
    bins=30,
    alpha=0.5,
    density=True,
    label="Current"
)

ax.set_xlabel(selected_feature)
ax.set_ylabel("Density")
ax.legend()
ax.grid(True, alpha=0.3)

st.pyplot(fig)

# ------------------------------------------------------------
# Selected feature statistics
# ------------------------------------------------------------

feature_result = drift_df[
    drift_df["Feature"] == selected_feature
].iloc[0]

c1, c2, c3 = st.columns(3)

c1.metric(
    "Reference Mean",
    f"{reference[selected_feature].mean():.3f}"
)

c2.metric(
    "Current Mean",
    f"{current[selected_feature].mean():.3f}"
)

c3.metric(
    "KS Drift Score",
    f"{feature_result['KS Statistic']:.3f}"
)

if feature_result["Drift"]:
    st.warning(
        f"⚠️ Drift detected for {selected_feature} "
        f"(p = {feature_result['p-value']:.4f})"
    )
else:
    st.success(
        f"✅ No significant drift for {selected_feature}"
    )

# ------------------------------------------------------------
# Drift overview
# ------------------------------------------------------------

st.divider()

st.subheader("Drift Detection")

display_df = drift_df.copy()

display_df["Status"] = np.where(
    display_df["Drift"],
    "⚠️ Drift",
    "✅ OK"
)

display_df = display_df[
    ["Feature", "KS Statistic", "p-value", "Status"]
]

display_df = display_df.sort_values(
    by="KS Statistic",
    ascending=False
)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)

# ------------------------------------------------------------
# Prediction distribution
# ------------------------------------------------------------

prediction_column = None

if "Prediction" in current.columns:
    prediction_column = "Prediction"

elif "prediction" in current.columns:
    prediction_column = "prediction"

elif "Label" in current.columns:
    prediction_column = "Label"

if prediction_column is not None:

    st.divider()

    st.subheader("Prediction Distribution")

    prediction_counts = (
        current[prediction_column]
        .value_counts()
        .sort_index()
    )

    prediction_names = {
        0: "Healthy",
        1: "Inner Ring",
        2: "Outer Ring"
    }

    prediction_df = pd.DataFrame({
        "Class": [
            prediction_names.get(x, str(x))
            for x in prediction_counts.index
        ],
        "Count": prediction_counts.values
    })

    st.bar_chart(
        prediction_df,
        x="Class",
        y="Count"
    )

# ------------------------------------------------------------
# Confidence
# ------------------------------------------------------------

confidence_column = None

if "Confidence" in current.columns:
    confidence_column = "Confidence"

elif "confidence" in current.columns:
    confidence_column = "confidence"

if confidence_column is not None:

    st.divider()

    st.subheader("Model Confidence")

    average_confidence = current[
        confidence_column
    ].mean()

    st.metric(
        "Average Confidence",
        f"{average_confidence:.1%}"
    )

# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------

st.divider()

st.caption(
    "Demo monitoring dashboard – KS test used for simple feature drift detection."
)
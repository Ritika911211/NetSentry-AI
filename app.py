import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from joblib import load

st.set_page_config(
    page_title="NetSentry AI",
    page_icon="🛡️",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "random_forest.joblib"
FEATURE_PATH = BASE_DIR / "models" / "feature_names.joblib"

st.title("🛡️ NetSentry AI")
st.subheader("Intelligent Network Intrusion Detection")
st.write("Machine Learning powered network traffic analysis.")

if not MODEL_PATH.exists() or not FEATURE_PATH.exists():
    st.error("Model files not found. Please train the model first.")
    st.stop()

@st.cache_resource
def load_model():
    model = load(MODEL_PATH)
    features = load(FEATURE_PATH)
    return model, features

model, feature_names = load_model()

st.success("Intrusion detection model loaded successfully!")

st.markdown("---")
st.header("📂 Analyze Network Traffic CSV")
st.write(
    "Upload a CSV containing the same network traffic features "
    "used during model training."
)

uploaded_file = st.file_uploader(
    "Choose a network traffic CSV",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file, low_memory=False)
        st.write("Uploaded dataset preview")
        st.dataframe(data.head())

        missing = [col for col in feature_names if col not in data.columns]

        if missing:
            st.error(
                f"CSV is missing {len(missing)} required model features. "
                "Please upload a compatible CIC-IDS2017 CSV."
            )
        else:
            X = data[feature_names].apply(
                pd.to_numeric, errors="coerce"
            )
            X.replace([np.inf, -np.inf], np.nan, inplace=True)
            X = X.fillna(X.median()).fillna(0)

            predictions = model.predict(X)

            results = data.copy()
            results["Prediction"] = np.where(
                predictions == 1,
                "BOT / Suspicious",
                "BENIGN / Normal"
            )

            st.header("📊 Analysis Results")

            total = len(results)
            suspicious = int((predictions == 1).sum())
            normal = total - suspicious

            c1, c2, c3 = st.columns(3)
            c1.metric("Total flows", f"{total:,}")
            c2.metric("Normal flows", f"{normal:,}")
            c3.metric("Suspicious flows", f"{suspicious:,}")

            st.subheader("Prediction distribution")
            st.bar_chart(
                results["Prediction"].value_counts()
            )

            st.subheader("Classified traffic")
            st.dataframe(results, use_container_width=True)

            csv_output = results.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download analysis results",
                data=csv_output,
                file_name="netsentry_analysis.csv",
                mime="text/csv"
            )

            st.caption(
                "Predictions are model estimates, not proof of an attack. "
                "Validate findings before taking security action."
            )

    except Exception as e:
        st.error(f"Could not analyze this file: {e}")
else:
    st.info("Upload a compatible CSV to start traffic analysis.")
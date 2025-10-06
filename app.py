import streamlit as st
import joblib
import pandas as pd
import os

# Page setup
st.set_page_config(page_title="Fraud Detection (CTGAN)", layout="wide")
st.title("💳 Fraud Detection using Synthetic Data (CTGAN)")
st.write("Enter transaction details to predict if it's Fraud or Not:")

# === Load trained model safely ===
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
if not os.path.exists(MODEL_PATH):
    st.error("model.pkl not found! Put your trained model.pkl in the same folder as app.py")
    st.stop()

model = joblib.load(MODEL_PATH)

# === Feature list ===
all_features = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']

# === Default values for all features ===
defaults = {f: 0.0 for f in all_features}
defaults['Amount'] = 100.0
defaults['Time'] = 50000.0

# Attempt to load medians from CSV if available
CSV_PATH = os.path.join(os.path.dirname(__file__), "creditcard.csv")
if os.path.exists(CSV_PATH):
    try:
        df = pd.read_csv(CSV_PATH)
        if set(all_features).issubset(df.columns):
            defaults.update(df[all_features].median().to_dict())
            st.caption("Using medians from creditcard.csv for default values.")
        else:
            st.caption("creditcard.csv loaded but columns mismatch; using safe defaults.")
    except Exception as e:
        st.caption(f"Failed to read creditcard.csv, using safe defaults. Error: {e}")
else:
    st.caption("No creditcard.csv found — using built-in defaults.")

# === Main Inputs ===
col1, col2, col3 = st.columns(3)
with col1:
    amount = st.number_input("Transaction Amount", value=float(defaults['Amount']), format="%.2f")
    time_val = st.number_input("Transaction Time", value=float(defaults['Time']), format="%.2f")
with col2:
    v1 = st.number_input("V1", value=float(defaults['V1']), format="%.6f")
    v2 = st.number_input("V2", value=float(defaults['V2']), format="%.6f")
with col3:
    v3 = st.number_input("V3", value=float(defaults['V3']), format="%.6f")
    v4 = st.number_input("V4", value=float(defaults['V4']), format="%.6f")

# === Advanced feature inputs ===
with st.expander("Advanced: Edit V5–V28 features", expanded=False):
    advanced_cols = st.columns(4)
    advanced_inputs = {}
    idx = 5
    for i, col in enumerate(advanced_cols):
        for _ in range(6):  # 6 rows per column approx
            if idx <= 28:
                key = f"V{idx}"
                advanced_inputs[key] = col.number_input(key, value=float(defaults[key]), format="%.6f", key=key)
                idx += 1
            else:
                break

# === Prepare input feature vector in exact model order ===
input_features = defaults.copy()
input_features.update({
    'Amount': float(amount),
    'Time': float(time_val),
    'V1': float(v1),
    'V2': float(v2),
    'V3': float(v3),
    'V4': float(v4)
})
# Add advanced features if any
input_features.update(advanced_inputs)

# Convert to DataFrame
input_df = pd.DataFrame([[input_features[f] for f in all_features]], columns=all_features)

# Show preview
st.write("### Input Preview (first 10 features)")
st.write(input_df.iloc[0, :10])

# === Prediction ===
if st.button("Predict Fraud"):
    try:
        pred = model.predict(input_df)
        prob = None
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(input_df)[0][1]
        elif hasattr(model, "decision_function"):
            score = model.decision_function(input_df)[0]
            prob = 1 / (1 + 2.71828 ** (-score))  # sigmoid approximation

        if pred[0] == 1:
            if prob is not None:
                st.error(f"⚠️ Fraud Detected! Probability: {prob:.4f}")
            else:
                st.error("⚠️ Fraud Detected!")
        else:
            if prob is not None:
                st.success(f"✅ Legitimate Transaction. Probability of fraud: {prob:.4f}")
            else:
                st.success("✅ Legitimate Transaction.")

    except ValueError as ve:
        st.exception(ve)
        st.error("Feature mismatch! Ensure model.pkl was trained with the features: " + ", ".join(all_features))
    except Exception as e:
        st.exception(e)
        st.error("Prediction failed — check model.pkl compatibility.")

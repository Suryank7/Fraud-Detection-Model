import streamlit as st
import joblib
import pandas as pd
import os
import requests

st.set_page_config(page_title="Fraud Detection (CTGAN)", layout="wide")

st.title("💳 Fraud Detection using Synthetic Data (CTGAN)")
st.write("Enter transaction details (main features shown). Advanced features can be adjusted in the expander.")

# === Load model safely ===
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
if not os.path.exists(MODEL_PATH):
    st.error("model.pkl not found in the app folder. Put your trained model.pkl in the same folder as app.py")
    st.stop()

model = joblib.load(MODEL_PATH)

# === Feature order ===
all_features = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']

# === Download CSV if not exists ===
CSV_PATH = os.path.join(os.path.dirname(__file__), "creditcard.csv")
if not os.path.exists(CSV_PATH):
    st.info("Downloading dataset for default values...")
    # Replace this link with your Google Drive shareable link
    file_url = "https://drive.google.com/file/d/1OtYUiue6RMqZTuvkc7woEQGwvxWVgNPt/view?usp=sharing"  
    r = requests.get(file_url)
    with open(CSV_PATH, "wb") as f:
        f.write(r.content)
    st.success("Dataset downloaded!")

# Load dataset to get medians for sensible defaults
try:
    df = pd.read_csv(CSV_PATH)
    if set(all_features).issubset(set(df.columns)):
        defaults = df[all_features].median().to_dict()
        source_note = "Using medians from creditcard.csv for defaults."
    else:
        defaults = {f: 0.0 for f in all_features}
        defaults['Amount'] = float(df['Amount'].median()) if 'Amount' in df.columns else 100.0
        source_note = "Column mismatch; using safe defaults (V features ≈ 0)."
except Exception as e:
    defaults = {f: 0.0 for f in all_features}
    defaults['Amount'] = 100.0
    source_note = f"Failed to read creditcard.csv — using safe defaults. Error: {e}"

st.caption(source_note)

# === Main inputs ===
col1, col2, col3 = st.columns(3)
with col1:
    amount = st.number_input("Transaction Amount", value=float(defaults.get('Amount', 100.0)), format="%.2f")
    time_val = st.number_input("Transaction Time", value=float(defaults.get('Time', 50000.0)), format="%.2f")
with col2:
    v1 = st.number_input("V1", value=float(defaults.get('V1', 0.0)), format="%.6f")
    v2 = st.number_input("V2", value=float(defaults.get('V2', 0.0)), format="%.6f")
with col3:
    v3 = st.number_input("V3", value=float(defaults.get('V3', 0.0)), format="%.6f")
    v4 = st.number_input("V4", value=float(defaults.get('V4', 0.0)), format="%.6f")

# Advanced features
with st.expander("Advanced: Edit all features (V5–V28) and defaults", expanded=False):
    st.write("Change defaults for any feature used by your trained model.")
    advanced_cols = st.columns(4)
    advanced_inputs = {}
    idx = 5
    for i, col in enumerate(advanced_cols):
        for _ in range(7):
            if idx <= 28:
                key = f"V{idx}"
                advanced_inputs[key] = col.number_input(key, value=float(defaults.get(key, 0.0)), format="%.6f", key=key)
                idx += 1
            else:
                break
    # Optional overrides
    adv_time = st.number_input("Advanced Time override", value=float(defaults.get('Time', time_val)))
    adv_amount = st.number_input("Advanced Amount override", value=float(defaults.get('Amount', amount)))
    if adv_time != defaults.get('Time', time_val):
        time_val = adv_time
    if adv_amount != defaults.get('Amount', amount):
        amount = adv_amount

# === Prepare input DataFrame ===
input_features = defaults.copy()
input_features.update({
    'Amount': float(amount),
    'Time': float(time_val),
    'V1': float(v1),
    'V2': float(v2),
    'V3': float(v3),
    'V4': float(v4)
})
for k, v in advanced_inputs.items():
    input_features[k] = float(v)

input_df = pd.DataFrame([[input_features[f] for f in all_features]], columns=all_features)

st.write("### Input Preview (first 10 features)")
st.write(input_df.iloc[0, :10])

# === Prediction ===
if st.button("Predict Fraud"):
    try:
        pred = model.predict(input_df)
        prob = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None

        if pred[0] == 1:
            st.error(f"⚠️ Fraud Detected! Probability: {prob:.4f}" if prob else "⚠️ Fraud Detected!")
        else:
            st.success(f"✅ Legitimate Transaction. Probability of fraud: {prob:.4f}" if prob else "✅ Legitimate Transaction.")
    except ValueError as ve:
        st.exception(ve)
        st.error("Feature mismatch: make sure model.pkl matches features: " + ", ".join(all_features))
    except Exception as e:
        st.exception(e)
        st.error("Prediction failed — check model compatibility.")

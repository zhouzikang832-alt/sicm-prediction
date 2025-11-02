import streamlit as st
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="SICM Mortality Prediction", layout="centered")

st.title("Sepsis-Induced Cardiomyopathy Mortality Risk Prediction")
st.write("输入患者指标，计算死亡风险概率")

# ---------------- load model ----------------
model = joblib.load("best_model_CatBoost.pkl")
pipeline = model["pipeline"]
features = [
    "Percentage of lymphocytes-MEAN",
    "LAC-MEAN",
    "BUN-MIN",
    "age",
    "Total protein-MEAN",
    "Percentage of lymphocytes-MIN",
    "PTT-MEAN",
    "SPO2-MEAN",
    "PLT-MAX",
    "LAC-MIN",
    "BUN-MEAN",
    "NA-MAX",
    "T-MEAN",
    "Percentage of lymphocytes-MAX",
    "ALB-MAX",
    "AG-MEAN",
    "LAC-MAX",
    "PH-MIN",
    "SPO2-MIN",
    "PLT-MIN"
]

# units mapping
units = {
    "Percentage of lymphocytes-MEAN": "%",
    "LAC-MEAN": "mmol/L",
    "BUN-MIN": "mg/dL",
    "Total protein-MEAN": "g/dL",
    "Percentage of lymphocytes-MIN": "%",
    "PTT-MEAN": "s",
    "SPO2-MEAN": "%",
    "PLT-MAX": "*10^9/L",
    "LAC-MIN": "mmol/L",
    "BUN-MEAN": "mg/dL",
    "NA-MAX": "mmol/L",
    "T-MEAN": "℃",
    "Percentage of lymphocytes-MAX": "%",
    "ALB-MAX": "g/dL",
    "AG-MEAN": "mmol/L",
    "LAC-MAX": "mmol/L",
    "PH-MIN": "",
    "SPO2-MIN": "%",
    "PLT-MIN": "*10^9/L"
}

# ---------------- input UI ----------------
values = []
for f in features:
    if f == "age":
        lab = f + " (years)"
        v = st.number_input(lab, step=1, format="%d")
    else:
        lab = f + " (入院后第一天) [" + units.get(f, "") + "]"
        v = st.number_input(lab, value=0.0, format="%.3f")
    values.append(v)


# ---------------- prediction ----------------
if st.button("Predict"):
    X = np.array(values).reshape(1, -1)
    prob = pipeline.predict_proba(X)[0][1]

    st.markdown(f"### Predicted mortality probability = **{prob:.3f}**")

    if prob >= 0.5:
        st.error("High Risk")
    else:
        st.success("Low Risk")

    # ---------------- SHAP ----------------
    st.subheader("Feature contribution (SHAP)")
    explainer = shap.TreeExplainer(pipeline["model"])
    shap_values = explainer.shap_values(X)

    fig, ax = plt.subplots()
    shap.bar_plot(shap_values[0], features=features, max_display=20, show=False)
    st.pyplot(fig)

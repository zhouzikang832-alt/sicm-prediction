import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="SICM Mortality Prediction", layout="centered")

st.title("Sepsis-Induced Cardiomyopathy Mortality Risk Prediction")
st.write("输入患者指标，计算死亡风险概率")

# load model
model = joblib.load("best_model_CatBoost.pkl")
pipeline = model["pipeline"]
features = model["features"]

# 输入框
values = []
for f in features:
    v = st.number_input(f, value=0.0, format="%.3f")
    values.append(v)

if st.button("Predict"):
    X = np.array(values).reshape(1, -1)
    prob = pipeline.predict_proba(X)[0][1]
    st.markdown(f"### Predicted mortality probability = **{prob:.3f}**")

    if prob >= 0.5:
        st.error("High Risk")
    else:
        st.success("Low Risk")

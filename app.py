import streamlit as st
import pickle
import pandas as pd

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Customer Churn App", layout="centered")

st.title("📊 Customer Churn Prediction App")
st.write("Predict whether a customer will stay or leave using ML model 🔥")

# -----------------------------
# LOAD MODEL
# -----------------------------
model = pickle.load(open("model.pkl", "rb"))

# -----------------------------
# INPUT UI
# -----------------------------
st.header("Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (Months)", 0, 72, 12)

with col2:
    monthly_charges = st.slider("Monthly Charges", 0, 150, 50)

# -----------------------------
# PREDICTION BUTTON
# -----------------------------
if st.button("Predict Churn"):

    # input dataframe (must match training features)
    input_data = pd.DataFrame({
        "tenure": [tenure],
        "MonthlyCharges": [monthly_charges]
    })

    # prediction
    prediction = model.predict(input_data)

    # probability (if supported)
    try:
        prob = model.predict_proba(input_data)
        churn_prob = prob[0][1] * 100
        stay_prob = prob[0][0] * 100
    except:
        churn_prob = None
        stay_prob = None

    # -----------------------------
    # OUTPUT
    # -----------------------------
    st.subheader("Result:")

    if prediction[0] == 1:
        st.error("⚠️ High Risk: Customer will CHURN")
    else:
        st.success("✅ Low Risk: Customer will STAY")

    # probabilities
    if churn_prob is not None:
        st.write(f"📊 Churn Probability: {churn_prob:.2f}%")
        st.write(f"📊 Stay Probability: {stay_prob:.2f}%")
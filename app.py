
import streamlit as st
import pandas as pd
import joblib
import json

# Load model and top features
model = joblib.load("model.pkl")
with open("feature_columns.json", "r") as f:
    feature_columns = json.load(f)

st.set_page_config(page_title="Attrition Predictor", layout="centered")
st.title("💼 Employee Attrition Prediction App")
st.write("Enter the key employee details below to predict attrition.")

# Collect user input for important features
user_input = {}
for col in feature_columns:
    if "OverTime" in col:
        user_input[col] = st.selectbox(f"{col}", ["Yes", "No"])
    elif "Travel" in col:
        user_input[col] = st.selectbox(f"{col}", ["0", "1"], help="1 = Travels frequently, 0 = Rarely/Never")
    else:
        user_input[col] = st.number_input(f"{col}", min_value=0.0)

# Predict button
if st.button("Predict"):
    try:
        input_df = pd.DataFrame([user_input])

        if "OverTime" in input_df.columns:
            input_df["OverTime"] = input_df["OverTime"].map({"Yes": 1, "No": 0}).astype(int)

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None

        st.success(f"Prediction: {'Employee likely to leave' if prediction == 1 else 'Employee likely to stay'}")
        if probability is not None:
            st.info(f"Confidence: {probability:.2%}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

st.caption("Built with ❤️ using Streamlit & Scikit-learn")

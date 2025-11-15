import streamlit as st
import pandas as pd
import joblib
import json

# -----------------------------
# Load the model and full feature list
# -----------------------------
model = joblib.load("model.pkl")

with open("feature_columns.json", "r") as f:
    full_feature_list = json.load(f)

st.set_page_config(page_title="Employee Attrition Predictor", layout="centered")

st.title("💼 Employee Attrition Prediction App")
st.write(
    """
    This tool predicts whether an employee is likely to **leave** or **stay** 
    based on key HR attributes.  
    Enter the details below and click **Predict**.
    """
)

# -----------------------------
# User Input Section
# -----------------------------
st.subheader("Enter Employee Details")

# Choose only 8 main features for user input
key_features = [
    "Age",
    "JobLevel",
    "MonthlyIncome",
    "TotalWorkingYears",
    "YearsAtCompany",
    "JobSatisfaction",
    "WorkLifeBalance",
    "OverTime",
]

user_input = {}
for col in key_features:
    if col == "OverTime":
        user_input[col] = st.selectbox("OverTime", ["Yes", "No"])
    else:
        user_input[col] = st.number_input(f"{col}", min_value=0.0)

# -----------------------------
# Prediction Logic
# -----------------------------
if st.button("Predict"):
    try:
        # Create a dataframe for user input
        input_df = pd.DataFrame([user_input])

        # Encode categorical variable
        input_df["OverTime"] = input_df["OverTime"].map({"Yes": 1, "No": 0}).astype(int)

        # Create the full feature dataframe (to match model’s expectations)
        full_input = pd.DataFrame(columns=full_feature_list)
        for col in input_df.columns:
            if col in full_input.columns:
                full_input[col] = input_df[col]
        full_input = full_input.fillna(0)

        # Make prediction
        prediction = model.predict(full_input)[0]
        probability = (
            model.predict_proba(full_input)[0][1]
            if hasattr(model, "predict_proba")
            else None
        )

        st.success(
            f"Prediction: {'Employee likely to leave' if prediction == 1 else 'Employee likely to stay'}"
        )
        if probability is not None:
            st.info(f"Confidence: {probability:.2%}")

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit and Scikit-learn")

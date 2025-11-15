import streamlit as st
import pandas as pd
import joblib
import json

# -----------------------------
# Load the saved model + metadata
# -----------------------------
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    with open("feature_columns.json", "r") as f:
        feature_columns = json.load(f)
    return model, feature_columns

model, feature_columns = load_model()

# -----------------------------
# App Title and Description
# -----------------------------
st.set_page_config(page_title="Employee Attrition Predictor", layout="centered")

st.title("👩‍💼 Employee Attrition Prediction App")
st.write(
    """
    This tool predicts whether an employee is likely to **leave (Attrition = 1)** 
    or **stay (Attrition = 0)** based on input features.  
    Fill in the details below and click **Predict**.
    """
)

# -----------------------------
# User Input Form
# -----------------------------
st.subheader("Enter Employee Details")

user_input = {}
for col in feature_columns:
    user_input[col] = st.text_input(f"{col}")

# -----------------------------
# Prediction Logic
# -----------------------------
if st.button("Predict"):
    try:
        input_df = pd.DataFrame([user_input])
        prediction = model.predict(input_df)[0]
        probability = (
            model.predict_proba(input_df)[0][1]
            if hasattr(model, "predict_proba")
            else None
        )

        st.success(
            f"✅ Prediction: {'Employee will leave' if prediction == 1 else 'Employee will stay'}"
        )
        if probability is not None:
            st.info(f"Confidence (Attrition Probability): {probability:.2%}")

    except Exception as e:
        st.error(f"⚠️ Error while predicting: {e}")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit, Pandas, and Scikit-learn")

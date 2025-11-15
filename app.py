# Predict button
if st.button("Predict"):
    try:
        input_df = pd.DataFrame([user_input])

        # Convert categorical fields
        if "OverTime" in input_df.columns:
            input_df["OverTime"] = input_df["OverTime"].map({"Yes": 1, "No": 0}).astype(int)

        # Load the full feature list that the model was trained on
        full_feature_list = json.load(open("feature_columns.json", "r"))

        # Create a DataFrame with all columns expected by the model
        full_input = pd.DataFrame(columns=full_feature_list)
        for col in input_df.columns:
            if col in full_input.columns:
                full_input[col] = input_df[col]
        full_input = full_input.fillna(0)  # fill missing columns with 0

        # Predict using the full column structure
        prediction = model.predict(full_input)[0]
        probability = (
            model.predict_proba(full_input)[0][1]
            if hasattr(model, "predict_proba")
            else None
        )

        st.success(f"Prediction: {'Employee likely to leave' if prediction == 1 else 'Employee likely to stay'}")
        if probability is not None:
            st.info(f"Confidence: {probability:.2%}")

    except Exception as e:
        st.error(f"Error during prediction: {e}")

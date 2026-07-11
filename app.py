import streamlit as st
import pandas as pd
import tensorflow as tf
import pickle

# -------------------------------
# Load Model
# -------------------------------
model = tf.keras.models.load_model("model.h5")

# Load Encoders and Scaler
with open("label_encoder_gender.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)

with open("onehot_encoder_geo.pkl", "rb") as file:
    onehot_encoder_geo = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🏦 Customer Churn Prediction")

st.write("Enter customer details and click Predict.")

# User Inputs
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600)

geography = st.selectbox(
    "Geography",
    onehot_encoder_geo.categories_[0]
)

gender = st.selectbox(
    "Gender",
    label_encoder_gender.classes_
)

age = st.slider("Age", 18, 92, 35)

tenure = st.slider("Tenure", 0, 10, 5)

balance = st.number_input("Balance", value=0.0)

num_of_products = st.slider("Number of Products", 1, 4, 1)

has_cr_card = st.selectbox(
    "Has Credit Card",
    [0, 1]
)

is_active_member = st.selectbox(
    "Is Active Member",
    [0, 1]
)

estimated_salary = st.number_input("Estimated Salary", value=50000.0)

# -------------------------------
# Predict Button
# -------------------------------
if st.button("Predict"):

    # Label Encoding
    gender_encoded = label_encoder_gender.transform([gender])[0]

    # One Hot Encoding
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()

    geo_encoded_df = pd.DataFrame(
        geo_encoded,
        columns=onehot_encoder_geo.get_feature_names_out(["Geography"])
    )

    # Create Input DataFrame
    input_data = pd.DataFrame({
        "CreditScore": [credit_score],
        "Gender": [gender_encoded],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_of_products],
        "HasCrCard": [has_cr_card],
        "IsActiveMember": [is_active_member],
        "EstimatedSalary": [estimated_salary]
    })

    # Combine Encoded Geography
    input_data = pd.concat(
        [input_data.reset_index(drop=True), geo_encoded_df],
        axis=1
    )

    # Scale Input
    input_data_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_data_scaled)

    prediction_proba = prediction[0][0]

    st.subheader("Prediction Result")

    st.write(f"**Churn Probability : {prediction_proba:.2%}**")

    if prediction_proba > 0.5:
        st.error("Customer is likely to churn.")
    else:
        st.success("Customer is not likely to churn.")
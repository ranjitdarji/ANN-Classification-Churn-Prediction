import streamlit as st
import pandas as pd
import tensorflow as tf
import pickle

# Load trained model
model = tf.keras.models.load_model("regression_model.h5")

# Load encoders
with open("label_encoder_gender.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)

with open("onehot_encoder_geo.pkl", "rb") as file:
    onehot_encoder_geo = pickle.load(file)

# Load scaler
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

#  Streamlit UI
st.title("Estimated Salary Prediction")

credit_score = st.number_input("Credit Score", value=600)

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

exited = st.selectbox(
    "Exited",
    [0, 1]
)

# Prediction

if st.button("Predict Salary"):

    # Label Encode Gender
    gender_encoded = label_encoder_gender.transform([gender])[0]

    # One-Hot Encode Geography
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
        "Exited": [exited]
    })

    # Combine Geography Columns
    input_data = pd.concat(
        [input_data.reset_index(drop=True), geo_encoded_df],
        axis=1
    )

    # Scale Data
    input_data_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_data_scaled)

    predicted_salary = prediction[0][0]

    st.success(f"Estimated Salary : ${predicted_salary:.2f}")
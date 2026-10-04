import streamlit as st
import pickle
import pandas as pd

# Load model
with open("best_housing_price_model (3).pkl", "rb") as file:
    data = pickle.load(file)

model = data["model"]

# Page title
st.title("🏠 House Price Prediction")

st.write("Enter the details of your house to predict its price.")

# Inputs
area = st.number_input("Area", min_value=0)
bedrooms = st.number_input("Bedrooms", min_value=0, step=1)
bathrooms = st.number_input("Bathrooms", min_value=0, step=1)
stories = st.number_input("Stories", min_value=0, step=1)
parking = st.number_input("Parking", min_value=0, step=1)

# Prediction button
if st.button("Predict Price"):

    input_data = pd.DataFrame({
        "area": [area],
        "bedrooms": [bedrooms],
        "bathrooms": [bathrooms],
        "stories": [stories],
        "parking": [parking]
    })

    prediction = model.predict(input_data)

    st.success(f"Predicted House Price: {prediction[0]:,.2f}")
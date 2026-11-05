import streamlit as st
import joblib

# Load the trained model
model = joblib.load('house_model.pkl')

#title of the web app
st.title("House Price Prediction App")
st.write("Enter house details to predict the price.")

#User input for house features
size = st.number_input("House size in square feet:")
bedrooms = st.number_input("Number of bedrooms:")
age = st.number_input("Age of the house in years:")
# Prediction button
if st.button("Predict Price"):
    # Prepare the input data for prediction
    input_data = [[size, bedrooms, age]]
    # Make prediction using the trained model
    predicted_price = model.predict(input_data)
    # Display the predicted price
    st.success(f"The predicted price of the house is: ${predicted_price[0]:,.2f}")
import streamlit as st
import pickle
import pandas as pd

# Load model
model = pickle.load(open('models/model.pkl', 'rb'))

# Load dataset to get correct columns
df = pd.read_csv('data/final.csv')
X = df.drop('price', axis=1)

st.title("🏠 Real Estate Price Prediction")

st.write("Enter property details:")

# Create inputs dynamically
input_data = {}

for col in X.columns:
    input_data[col] = st.number_input(f"{col}", value=0.0)

# Predict
if st.button("Predict Price"):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)
    price = max(0, prediction[0])
    st.success(f"Estimated Price: ${price:,.2f}")
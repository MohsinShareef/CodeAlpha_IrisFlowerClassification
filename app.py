import streamlit as st
import numpy as np
import joblib

# Load model and label encoder
model = joblib.load('logistic_regression_iris.pkl')
le = joblib.load('label_encoder_iris.pkl')

st.set_page_config(page_title="Iris Species Predictor", layout="centered")

st.title("🌸 Iris Flower Species Predictor")
st.markdown("""
This app uses a **Logistic Regression** model trained on the classic Iris dataset
to predict the species based on sepal and petal measurements.
""")

# Input sliders
st.sidebar.header("Input Features")

sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.8, 0.1)
sepal_width  = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.0, 0.1)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 4.0, 0.1)
petal_width  = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 1.3, 0.1)

# Predict button
if st.button("Predict Species"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)
    species = le.inverse_transform(prediction)[0]
    st.success(f"**Predicted Species: {species}**")
    st.balloons()
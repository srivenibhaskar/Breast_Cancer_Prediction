import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

# Set page configuration with a title and icon
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🎗️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Display the background image at the top of the app
st.image(
    "Breast_Cancer_Analysis.png",
    use_container_width=True,  # Updated parameter to use the full container width
    caption="Breast Cancer Awareness",
)

# Title with color
st.markdown(
    """
    <h1 style='text-align: center; color: #C08081;'>
        🎗️ Breast Cancer Prediction 🎗️
    </h1>
    """,
    unsafe_allow_html=True,
)

# Description with color
st.markdown(
    """
    <p style='text-align: center; font-size: 16px; color: #C08081;'>
        This app predicts whether a breast tumor is <b>Malignant</b> or <b>Benign</b> based on input values for selected features.
    </p>
    """,
    unsafe_allow_html=True,
)

# Feature descriptions with color
st.markdown(
    """
    <h3 style='color: #C08081;'>Feature Descriptions</h3>
    <p style='color: #333;'>
    - <b>Mean Radius</b>: Average radius of the tumor.<br>
    - <b>Mean Perimeter</b>: Average perimeter of the tumor.<br>
    - <b>Mean Area</b>: Average area of the tumor.<br>
    - <b>Mean Concavity</b>: Average severity of concave portions of the tumor contour.<br>
    - <b>Mean Concave Points</b>: Average number of concave points on the tumor contour.<br>
    - <b>Worst Radius</b>: Largest radius value for the tumor.<br>
    - <b>Worst Perimeter</b>: Largest perimeter value for the tumor.<br>
    - <b>Worst Area</b>: Largest area value for the tumor.<br>
    - <b>Worst Concavity</b>: Largest severity of concave portions of the tumor contour.<br>
    - <b>Worst Concave Points</b>: Largest number of concave points on the tumor contour.<br>
    </p>
    """,
    unsafe_allow_html=True,
)

# User input section heading with color
st.markdown(
    """
    <h3 style='color: #C08081;'>Enter Feature Values Below</h3>
    """,
    unsafe_allow_html=True,
)

features = ['mean radius', 'mean perimeter', 'mean area', 'mean concavity',
            'mean concave points', 'worst radius', 'worst perimeter',
            'worst area', 'worst concavity', 'worst concave points']

user_input = {}
for feature in features:
    user_input[feature] = st.number_input(f"{feature.capitalize()}", value=0.0)

# Add example inputs
if st.button("Use Example Inputs"):
    example_values = {
        'mean radius': 17.99,
        'mean perimeter': 122.8,
        'mean area': 1001.0,
        'mean concavity': 0.3001,
        'mean concave points': 0.1471,
        'worst radius': 25.38,
        'worst perimeter': 184.6,
        'worst area': 2019.0,
        'worst concavity': 0.7119,
        'worst concave points': 0.2654,
    }
    for feature, value in example_values.items():
        user_input[feature] = value
        st.write(f"Suggested Value for {feature}: {value}")

# Convert user input to DataFrame
user_data = pd.DataFrame([user_input])

# Load the trained model
model = load("best_ann_model.pkl")

# Predict
if st.button("Predict"):
    prediction = model.predict(user_data)
    if prediction[0] == 1:
        st.success("🎗️ The prediction is: *Malignant* 🎗️")
    else:
        st.info("🎗️ The prediction is: *Benign* 🎗️")

# Model performance section with color
st.markdown(
    """
    <h3 style='color: #FF69B4;'>Model Performance</h3>
    <p style='color: #333;'>
    - <b>Accuracy</b>: **96.74%**<br>
    - <b>Classification Report</b>:<br>
      - <b>Benign</b>:<br>
        - Precision: 0.97<br>
        - Recall: 0.96<br>
        - F1-Score: 0.97<br>
      - <b>Malignant</b>:<br>
        - Precision: 0.96<br>
        - Recall: 0.97<br>
        - F1-Score: 0.97<br>
    </p>
    """,
    unsafe_allow_html=True,
)
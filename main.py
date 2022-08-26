import streamlit as st
import numpy as np
import pickle
from array import array
import sklearn

st.image("20522070.jpg")

st.title("Breast Cancer Prediction Model")
radius_worst = st.number_input(label="Enter radius_worst", min_value=0.0, max_value=5.0, step=0.01)
perimeter_worst = st.number_input(label="Enter perimeter_worst", min_value=0.0, max_value=5.0, step=0.01)
concave_points_worst = st.number_input(label="Enter concave_points_worst", min_value=0.0, max_value=3.0, step=0.01)
area_worst = st.number_input(label="Enter area_worst", min_value=0.0, max_value=6.0, step=0.01)
concave_points_mean = st.number_input(label="Enter concave_points_mean", min_value=0.0, max_value=4.0, step=0.01)
radius_mean = st.number_input(label="Enter radius_mean", min_value=0.0, max_value=4.0, step=0.01)
concavity_mean = st.number_input(label="Enter concavity_mean", min_value=0.0, max_value=5.0, step=0.01)
area_mean = st.number_input(label="Enter area_mean", min_value=0.0, max_value=6.0, step=0.01)

inputs = np.array([[radius_worst, perimeter_worst, concave_points_worst, area_worst, concave_points_mean, radius_mean, concavity_mean, area_mean]])
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

prediction = model.predict(inputs)

st.button('Submit')
if prediction == 1:
    st.write("Cancer is Malignant")
else:
    st.write("Cancer is Benign")






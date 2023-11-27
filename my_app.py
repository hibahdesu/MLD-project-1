# Streamlit Documentation: https://docs.streamlit.io/

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image 
from textwrap import fill
import pickle
import requests
import json
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OrdinalEncoder

st.sidebar.markdown("# Main page 🎈")

html_style0 = """
<div style="background: linear-gradient(to bottom, #000099 0%, #ffffff 100%);padding:5px;margin-bottom:32px;border-radius:50px">
<h1 style="color:white;text-align:center;">Cars Prediction App</h1>
</div>"""
st.markdown(html_style0,unsafe_allow_html=True)

# Add image
# img = Image.open("car.jpg")
# st.image(img, caption="car")



import streamlit as st
from PIL import Image
import base64

# Define the CSS style
css = """
<style>
    .rounded-image {
        border-radius: 40px;
        overflow: hidden;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 50px 30px 50px;
        max-width: 100%;
    }
</style>
"""

# Display the CSS style
st.markdown(css, unsafe_allow_html=True)

# Display the image with rounded corners
image_path = 'car.jpg'
image_html = f'<div class="rounded-image"><img src="data:image/jpeg;base64,{base64.b64encode(open(image_path, "rb").read()).decode()}"></div>'
st.markdown(image_html, unsafe_allow_html=True)



st.header("About")
st.markdown('Welcome to the Cars Prediction App! This application is designed to assist you in predicting various aspects related to cars.')
st.markdown("With the Cars Prediction App, you can leverage the power of machine learning algorithms to make accurate predictions based on historical data and relevant features of different car models. Whether you're an automobile enthusiast, a car buyer, or a car dealer, this app provides valuable insights to help you make informed decisions.")

st.header('How it works')
st.markdown("Using a user-friendly interface, you can input specific details about a car, such as its make, model, age, km, and other relevant features.")
st.markdown("The app will then process this information and apply advanced prediction models to provide estimated price.")

# Add image
img = Image.open("al.png")
st.image(img, caption="ML")


st.header('Features')
st.markdown("With the Cars Prediction App, you no longer need to rely solely on guesswork or outdated information. Our cutting-edge algorithms analyze vast amounts of data to deliver accurate predictions tailored to your specific car-related inquiries.")
st.markdown("Whether you're looking to buy a car, sell one, this app empowers you with valuable insights.")


# Add image
img = Image.open("car2.jpg")
st.image(img, caption="car")

st.header('Still Waiting! 🎈')
st.markdown("Harry Up, and get ready to embark on a journey of car prediction and exploration. Start using the Cars Prediction App today and unlock a world of data-driven predictions to enhance your car-related decisions.")


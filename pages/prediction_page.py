# Streamlit Documentation: https://docs.streamlit.io/

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image 
from textwrap import fill
import pickle
import requests
import base64
import json
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OrdinalEncoder




st.markdown("# Prediction page ")
st.sidebar.markdown("# Prediction page 🎉")


html_style = """
<div style="background: linear-gradient(to bottom, #33ccff 0%, #ff99cc 100%);padding:8px;border-radius:40px;margin-bottom:24px">
<h2 style="color:white;text-align:center;font-size:24px">Cars Prediction</h2>
</div>"""
st.sidebar.markdown(html_style,unsafe_allow_html=True)

html_style2 = """
<div style="background: linear-gradient(to bottom, #33ccff 0%, #ff99cc 100%);padding:8px; border-radius:40px;margin-bottom:24px">
<h2 style="color:white;text-align:center;font-size:32px">Now, you can predict with ML 🎉</h2>
</div>"""
st.markdown(html_style2,unsafe_allow_html=True)


# Define the CSS style
css = """
<style>
    .rounded-image {
        border-radius: 25px 10px ;
        overflow: hidden;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 50px;
        max-width: 91%;
        margin-left: 30px;
    }
</style>
"""

# Display the CSS style
st.markdown(css, unsafe_allow_html=True)

# Display the image with rounded corners
image_path = 'c3.jpg'
image_html = f'<div class="rounded-image"><img src="data:image/jpeg;base64,{base64.b64encode(open(image_path, "rb").read()).decode()}"></div>'
st.markdown(image_html, unsafe_allow_html=True)



car_model=st.sidebar.selectbox("Select car's model", ('Audi A3', 'Opel Insignia', 'Audi A1', 'Opel Astra', 'Opel Corsa', 'Renault Clio', 'Renault Espace'))
# car_gear=st.sidebar.slider("What's car's Gear", 1.0,8.0, step=1)
car_gear=st.sidebar.selectbox("Select car's Gear:",(1.0,2.0,3.0, 4.0, 5.0, 6.0, 7.0, 8.0))
car_age=st.sidebar.selectbox("Select car's age:",(0,1,2,3))
car_gearing_type=st.sidebar.selectbox("Select car's Gearing_Type", ('Automatic', 'Manual', 'Semi-automatic'))
car_km=st.sidebar.slider("What's car's km", 0,350000, step=1000)
car_hp_kW = st.sidebar.slider("Select car's hp_kw", 40, 300, step=5)
car_wg = st.sidebar.slider("Select car's Weight in kg", 800, 2500, step=100)


filename = "final_rf"
model=pickle.load(open(filename, "rb"))


my_dict = {
    "make_model": car_model,
    "km": car_km,
    "Gears": car_gear,
    "age": car_age,
    "hp_kW": car_hp_kW,
    "Gearing_Type": car_gearing_type,
    "Weight_kg": car_wg,
}


df = pd.DataFrame.from_dict([my_dict])

st.header("The values you have chosen: ")
st.table(df)



# Prediction with user inputs
predict = st.button("Predict")
result = model.predict(df)
try: 
    if predict :
        st.balloons()
        st.success(f'The predicted price is: {result[0]}')
    else: 
        st.warning('Choose values first')
except:
    st.warning('Something went wrong, please try again.')

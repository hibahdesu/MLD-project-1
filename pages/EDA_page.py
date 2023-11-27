# Streamlit Documentation: https://docs.streamlit.io/

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image 
from textwrap import fill
import matplotlib.pyplot as plt
import pickle
import requests
from io import StringIO
# import seaborn as sns
import json
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OrdinalEncoder

st.markdown("# EDA page ❄️")
st.sidebar.markdown("# EDA page ❄️")

st.header('Reading the data')



st.code("import pandas as pd\nimport numpy as np\ndf = pd.read_csv('df2.csv')\ndf.head()")

# Dataframe
df = pd.read_csv('df2.csv')

st.write(df.head()[2:])  

st.header('Describing the data')
st.code('df.describe().T')
st.dataframe(df.describe().T) 


st.header('Null values in the data')
st.code('df.isnull().sum()')
st.write(df.isnull().sum())

st.markdown('As we can see, there is no null data.')

st.header('Columns in the data')
st.code('df.columns')
st.write(df.columns) 


st.title('Graphs')

st.header('make model with price')

# Add image
img = Image.open("p1.png")
st.image(img, caption="ML")
st.info('As we can see, some of the make models have a very low values, and this will not help us. So, I am going to drop them.')

st.markdown('#### First, I am going to make a variable with these values')
st.code('#Making a variable to see the classes that have less than 100, as it may affect the model\nlow = df.make_model.value_counts()[df.make_model.value_counts() <= 100].index\nlow\n')

st.markdown("#### Now, I am going to drop them with the for loop.")
st.code("#Dropping the low values of make_model\nfor i in low:\ndrop_index = df[df['make_model'] == i].index\ndf.drop(index = drop_index, inplace=True)\ndf.reset_index(drop=True, inplace=True)")

st.header('price')
st.code('#Seeing teh boxplot of price\nsns.boxplot(df["price"])\nplt.show()')
img2 = Image.open("p2.png")
st.image(img2, caption="ML")
st.markdown("As we can see, there are some outliers in price, I am going to drop them.")

st.markdown("##### Filtering the price")
st.code("#Dropping the prices that more than 40000\ndf = df[df['price'] < 40000]\ndf")


colors = ['purple']

st.header('Gears with price')
fig, ax = plt.subplots()
ax.bar(df['Gears'], df['price'], color=colors)
ax.set_xlabel('Gears')
ax.set_ylabel('price')
ax.set_title('Bar Plot')

# Display the plot in Streamlit
st.pyplot(fig)

c = ['navy']

st.header('age with price')
fig, ax = plt.subplots()
ax.bar(df['age'], df['price'], color=c)
ax.set_xlabel('age')
ax.set_ylabel('price')
ax.set_title('Bar Plot')

# Display the plot in Streamlit
st.pyplot(fig)
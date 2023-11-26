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

st.header('Columns in the data')
st.code('df.columns')
st.write(df.columns) 


st.title('Graphs')

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

c2 = ['orange']
st.header('make model with price')
fig, ax = plt.subplots()
ax.bar(df['make_model'], df['price'], color=c2)
ax.set_xlabel('make_model')
ax.set_ylabel('price')
ax.set_title('Bar Plot')

# Display the plot in Streamlit
st.pyplot(fig)



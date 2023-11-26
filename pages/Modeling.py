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

st.markdown("# Modeling page ❄️")
st.sidebar.markdown("# Modeling page ❄️")


# Dataframe
df = pd.read_csv('df2.csv')

# Split into X, y
st.markdown('### Splitting the data')
st.code("#Splitting the data to X and y \nX = df.drop(columns='price')\ny = df.price")



#Splitting the data to train and test
st.markdown('### Splitting X and y')
st.code("X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=t, random_state=s)")

#Objects values
st.markdown('### Object columns')
st.code("cat_ob = X_train.select_dtypes('object').columns\ncat_ob")

st.markdown('Now, we want to convert these object columns or values to numeric values to work with them in the modeling part')


#Converting Objects values
st.markdown('### Converting Object columns')
st.code("#First, importing OrdinalEncoder\nfrom sklearn.preprocessing import OrdinalEncoder\nord_enc = OrdinalEncoder()\ntrans = make_column_transformer((ord_enc, cat_ob), remainder='passthrough', verbose_feature_names_out=False).set_output(transform='pandas')")

#Eval function
st.markdown('### Creating Eval Function')
st.code('#Creating the eval function to see the metrics\n# Pre-defined functions\ndef eval(model, X_train, y_train, X_test, y_test):\ny_pred = model.predict(X_test)\ny_train_pred = model.predict(X_train)\nscores = {"train": {"R2" : r2_score(y_train, y_train_pred),\n"mae" : mean_absolute_error(y_train, y_train_pred),\n"mse" : mean_squared_error(y_train, y_train_pred),\n"rmse" : np.sqrt(mean_squared_error(y_train, y_train_pred))},\n"test": {"R2" : r2_score(y_test, y_pred),\n"mae" : mean_absolute_error(y_test, y_pred),\n"mse" : mean_squared_error(y_test, y_pred),\n"rmse" : np.sqrt(mean_squared_error(y_test, y_pred))}}\nreturn pd.DataFrame(scores)\n# Adjusted R2 Score\ndef adj_r2(y_test, y_pred, X):\nr2 = r2_score(y_test, y_pred)\nn = X.shape[0]   # number of observations\np = X.shape[1]   # number of independent variables\nadj_r2 = 1 - (1-r2)*(n-1)/(n-p-1)\nreturn adj_r2\n')

#RF Model
st.markdown('### RF Model')
st.code('#Importing RandomForestRegressor\nfrom sklearn.ensemble import RandomForestRegressor\n#Creating the model of rf with the best params\nrf_model = RandomForestRegressor(max_depth= 10, max_features = 4, n_estimators= 300,  random_state=s)\n#the operations of the pipeline\noperations = [("encoder", trans), ("RF_model", rf_model)]\n#Creating the pipeline\npipe_rf_m = Pipeline(steps=operations)\n#Fitting the model with the training data\npipe_rf_m.fit(X_train, y_train)')


#using eval function
st.markdown('### Seeing the evaluation of the model')
st.code("#seeing the result of the grid search model\neval(pipe_rf_m, X_train, y_train, X_test, y_test)")

#Visualizing
st.markdown('### Visualizing the result')
st.code("#Importing RadViz\nfrom yellowbrick.features import RadViz\n#Visualizing the result\nvisualizer = RadViz(size=(500, 1000))\nvisualizer = PredictionError(pipe_rf_m)\nvisualizer.fit(X_train, y_train) # Fit the training data to the visualizer\nvisualizer.score(X_test, y_test) # Evaluate the model on the test data\nvisualizer.show();")


# Add image
img = Image.open("m1.png")
st.image(img, caption="Visualizing")


#Final Modle
st.markdown('### Final Modle')
st.markdown('In the final step, we have to fit the model with the whole data X and y')
st.code('#Creating the model of rf with the best params\nrf = RandomForestRegressor(max_depth= 10, max_features = 4, n_estimators= 300,  random_state=s)\n#the operations of the pipeline\noperations = [("encoder", trans), ("RF_model", rf)]\n#Creating the pipeline\nfinal_rf = Pipeline(steps=operations)\n#Fitting the model with the whole data\nfinal_rf.fit(X, y)')

#Saving the model
st.markdown('### Saving the model')
st.code("pickle.dump(final_rf, open('final_rf', 'wb'))\nnew_model = pickle.load(open('final_rf', 'rb'))")


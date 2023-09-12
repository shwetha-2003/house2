#import package
import streamlit as st
import pandas as pd
import numpy as np
#from PIL import Image
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
#import time

#import the data
data = pd.read_csv("Data Clean.csv")
#image = Image.open("house.png")
st.title("House Price Prediction Application")
#st.image(image, use_column_width=True)

#cek data
st.write("This application was created to estimate the price range of houses that consumers will buy")
check_data = st.checkbox("Look at existing data")
if check_data:
    st.write(data.head(4312))
st.subheader("Let's start predicting the price of your dream home!")

#input angka
sqft_liv = st.slider("What size living room do you want (ft2) ?",int(data.sqft_living.min()),int(data.sqft_living.max()),int(data.sqft_living.mean()) )
bath     = st.slider("How many bathrooms?",int(data.bathrooms.min()),int(data.bathrooms.max()),int(data.bathrooms.mean()) )
bed      = st.slider("How many bedrooms?",int(data.bedrooms.min()),int(data.bedrooms.max()),int(data.bedrooms.mean()) )
floor    = st.slider("How many floors of the house do you want?",int(data.floors.min()),int(data.floors.max()),int(data.floors.mean()) )

#split data
X = data.drop('price', axis = 1)
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2, random_state=45)

#buat model
#import model
model=LinearRegression()
#fitting dan predict model
model.fit(X_train, y_train)
model.predict(X_test)
errors = np.sqrt(mean_squared_error(y_test,model.predict(X_test)))
predictions = model.predict([[sqft_liv,bath,bed,floor]])[0]

#cek prediksi harga rumah
if st.button("Run!"):
    st.header("Predict the price of your dream home = {} $".format(int(predictions)))
    st.subheader("Your home price range = {} $ - {} $".format(int(predictions-errors),int(predictions+errors)))
    

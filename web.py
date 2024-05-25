import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import matplotlib.pyplot as plt
import streamlit as st
import pandas_datareader as data
start='2010-01-01'
end='2019-12-31'

st.title('STOCK TREND AND PRICE PREDICTION OF SBI')


df=pd.read_csv('SBIN.NS.csv')

#Describing Data
st.subheader('Data From 2010-2024')
st.write(df.describe())
#visualization
st.subheader('closing price vs Time chart')
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('closing price vs Time chart with 100MA')
MA100=df.Close.rolling(100).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(MA100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('closing price vs Time chart with 200MA')
MA100=df.Close.rolling(100).mean()
MA200=df.Close.rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(MA100)
plt.plot(MA200)
plt.plot(df.Close)
st.pyplot(fig)

data_training=pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])
from sklearn.preprocessing import  MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
data_training_array= scaler.fit_transform(data_training)



#load my Model
model=load_model('keras_model.h5')

#testing part

past_100_days =data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
  x_test.append(input_data[i-100:i])
  y_test.append(input_data[i,0])


x_test,y_test = np.array(x_test),np.array(y_test)
y_predicted = model.predict(x_test)
scaler=scaler.scale_

scale_factor =1/scaler[0]
y_predicted=y_predicted*scale_factor
y_test = y_test * scale_factor



st.subheader('prediction vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test,"b" , label = "Original price")
plt.plot(y_predicted, 'r', label ="Predicted Price")

plt.legend()
st.pyplot(fig2)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSprop
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.layers import LSTM

df=pd.read_csv('temp1.csv',encoding='euc-kr')
df.fillna(method='ffill', inplace=True)
df.drop(['지점','지점명','일시','평균 이슬점온도(°C)',	'평균 증기압(hPa)','평균 현지기압(hPa)',
'평균 해면기압(hPa)','가조시간(hr)','평균 지면온도(°C)','최저 초상온도(°C)','평균 5cm 지중온도(°C)'], axis=1, inplace=True)
target_names = ['평균기온(°C)']
df_targets = df[target_names]
xs=df_targets.values[:-1]
num_data=len(xs)
train_split=0.8
num_train=int(train_split*num_data)

xs_train=xs[0:num_train]
xs_test=xs[num_train:]
ts=df_targets.values[1:]

xs=xs.transpose()
ts=ts.transpose()
xs_test=xs_test.transpose()
xs_train=xs_train.transpose()



'''
input_names = ['평균 이슬점온도(°C)',	'평균 증기압(hPa)','평균 현지기압(hPa)',
'평균 해면기압(hPa)','가조시간(hr)','평균 지면온도(°C)','최저 초상온도(°C)','평균 5cm 지중온도(°C)']
'''
'''
df_input= df[input_names]
x_data=df_input.values
y_data=df_targets.values
'''
'''
num_data=len(x_data)
train_split=0.8
num_train=int(train_split*num_data)
num_test=num_data-num_train

x_train=x_data[0:num_train]
x_test=x_data[num_train:]

y_train=y_data[0:num_train]
y_test=y_data[num_train:]

num_x_signals=x_data.shape[1]
num_y_signals=y_data.shape[1]
#print(num_x_signals)
#print(num_y_signals)
'''

'''
x_scaler = MinMaxScaler()
x_train_scaled = x_scaler.fit_transform(x_train)

x_test_scaled = x_scaler.transform(x_test)
y_scaler = MinMaxScaler()
y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled = y_scaler.transform(y_test)
'''
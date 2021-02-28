# DNN for DNA energy regression according to DNA's nucleoside sequence (Developed by Prof. Shin-Pon Ju 2020/Sep./20)
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import shutil
import json

from tensorflow import keras
from tensorflow.keras import layers
data = pd.read_csv("./groupenergy.csv") # data we use to train the ML for energy regression
data.to_csv("./test.csv",index=False) # check if we have read the same data as we think

data_num = data.shape[0]
#print ("data_num:{}".format(data_num))

indexes = np.random.permutation(data_num)

train_data = data.loc[indexes]

mean = train_data.mean()
std = train_data.std()
#print (train_data.head(1))
#print (mean.head(1))
#print (std.head(1))
#with open('model_mean.txt', 'w') as file:
#     file.write(mean) # use `json.load` to do the reverse
#
#with open('model_std.txt', 'w') as file:
#     file.write(std) # use `json.load` to do the reverse
#print (mean)
#print (type(mean))
mean.to_csv("./mean.csv",header=False)
std.to_csv("./std.csv",header=False)
#input()
#testdata = pd.read_csv("./mean.csv")
#df = pandas.read_csv('csvfile.txt', index_col=False, header=0);
#testmean = testdata.ix[0,:]
#print(type(testdata.iloc[:,1]))
#val =testdata.iloc[:,1].values
#print(type(val))
# = pd.Series(val, index=testdata.iloc[:,0])
#print(my_obj)
#print(type(my_obj))
#print(mean.index)
#print(mean.values)
#print(type(std))
#print(type(train_data))
#input()
train_data = (train_data - mean) / std
train_data['Energy'] = train_data['Energy']/4.0

#val_data = (train_data - mean) / std

x_train = np.array(train_data.drop('Energy', axis='columns'))
#print(type(x_train[0]))
#print(type(x_train[0]))
#print(x_train[0][0].shape)
y_train = np.array(train_data['Energy'])
#E_Mean = y_train.mean()
#E_std = y_train.std()
#y_train = (y_train - E_Mean)/E_std
#print(y_train.mean())
#print(type(y_train))
#print(y_train.std())
#input()
#x_val = np.array(val_data.drop('price', axis='columns'))
#y_val = np.array(val_data['price'])
input_shape = x_train.shape[1]

              
#model_dir = './Best/'
#shutil.rmtree(model_dir, ignore_errors=True)
#os.makedirs(model_dir)

model_2 = keras.Sequential(name='model-2')
model_2.add(layers.Dense(3600, kernel_initializer='normal',
                         #kernel_regularizer=keras.regularizers.l1_l2(0.000001), 
                         activation='relu', input_shape=(input_shape,)))
#model_2.add(layers.Dropout(0.25))
    
model_2.add(layers.Dense( 2400, 
                        # kernel_regularizer=keras.regularizers.l1_l2(0.000001), 
                         activation='relu'))    
model_2.add(layers.Dense( 1200, 
                         #kernel_regularizer=keras.regularizers.l1_l2(0.00001), 
                         activation='relu')) 
model_2.add(layers.Dense( 600, 
                         #kernel_regularizer=keras.regularizers.l1_l2(0.00001), 
                         activation='relu'))                                                     
#model_2.add(layers.Dense(16, 
#                         #kernel_regularizer=keras.regularizers.l1_l2(0.00001), 
#                         activation='relu'))                         
#model_2.add(layers.Dense(8, 
#                         #kernel_regularizer=keras.regularizers.l1_l2(0.00001), 
#                         activation='relu'))                         
#
#model_2.add(layers.Dropout(0.5))r
##model_2.add(layers.Dropout(0.5))                         
#model_2.add(layers.Dense(32, kernel_regularizer=keras.regularizers.l1_l2(0.05), activation='relu'))
#model_2.add(layers.Dropout(0.5))
#model_2.add(layers.Dense(30, kernel_regularizer=keras.regularizers.l1_l2(0.05), activation='relu'))
#model_2.add(layers.Dropout(0.5))
#model_2.add(layers.Dense(16, kernel_regularizer=keras.regularizers.l1_l2(0.05), activation='relu'))
#model_2.add(layers.Dropout(0.5))

 #model_2.add(layers.Dropout(0.5))
#model_2.add(layers.Dense(128, kernel_regularizer=keras.regularizers.l2(0.001), activation='elu'))
#model_2.add(layers.Dropout(0.5))
model_2.add(layers.Dense(1))
#keras.optimizers.Adam(0.001),tf.keras.optimizers.RMSprop(0.005)
model_2.compile(keras.optimizers.Adam(0.0001),
                loss=keras.losses.MeanSquaredError(),
                metrics=[keras.metrics.MeanAbsoluteError()])

log_dir = os.path.join('DNAsequence-logs', 'model-2')
model_cbk = keras.callbacks.TensorBoard(log_dir=log_dir)
model_mckp = keras.callbacks.ModelCheckpoint('Best-model.h5', 
                                             monitor='val_mean_absolute_error', 
                                          save_best_only=True, 
                                             mode='min')
#model_reduce = keras.callbacks.ReduceLROnPlateau(monitor='mean_absolute_error', factor=0.5, patience=10, verbose=0, mode='auto', epsilon=0.005, cooldown=0, min_lr=0)                                             
monitor = EarlyStopping(monitor='val_mean_absolute_error', min_delta=1e-3, patience=5, 
        verbose=1, mode='auto', restore_best_weights=True)
history = model_2.fit(x_train, y_train, 
            batch_size=20,
            epochs=100, 
            validation_split = 0.2, 
            callbacks=[model_cbk, model_mckp,monitor])
print(history.history.keys())  # 查看history儲存的資訊有哪些

with open('model_his.txt', 'w') as file:
     file.write(json.dumps(history.history)) # use `json.load` to do the reverse

#print (history.history)

#model_test = keras.models.load_model('./Best/Best-model-2.h5')
#x_test = np.array([x_train[0]])
#print(x_test)
#print(x_test[0].reshape(1,-1))
#print (x_test[0].shape)
#print (x_test[0].ndim)
#y_pred = model_test.predict(x_train[0].reshape(1,-1))

#print(y_train[0])
#print(y_pred)


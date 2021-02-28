import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import shutil
import json
from tensorflow import keras
from tensorflow.keras import layers
from scipy.stats import pearsonr

model_test = keras.models.load_model('./Best-model.h5')
data = pd.read_csv("./groupenergy.csv")
print (model_test.summary())
mean_data = pd.read_csv("./mean.csv",header=None)
mean_val =mean_data.iloc[:,1].values
mean = pd.Series(mean_val, index=mean_data.iloc[:,0])

std_data = pd.read_csv("./std.csv",header=None)
std_val =std_data.iloc[:,1].values
std = pd.Series(std_val, index=std_data.iloc[:,0])

data = (data - mean) / std

#print(mean)
#input()

x_train = np.array(data.drop('Energy', axis='columns'))

y_train = -(np.array(data['Energy'])*std["Energy"] + mean["Energy"])

y_pred = model_test.predict(x_train)
y_pred = -(y_pred*std["Energy"]*4.0 + mean["Energy"])

# calculate the Pearson's correlation between two variables
print (type(y_pred))
print (type(y_train))

y_predls= y_pred.flatten().tolist()
y_trainls= y_train.flatten().tolist()

with open("trainPred.txt","w") as f:
    f.write("Train Predict\n")
    for (train,predict) in zip(y_trainls,y_predls):
        f.write("{0} {1}\n".format(train,predict))

#print (y_predls)
#print (type(y_trainls))
# calculate Pearson's correlation
#print(pearsonr(y_predls, y_trainls))
#print('Pearsons correlation: %.3f' % corr)
#print(std["Energy"])
#print(mean["Energy"])
#print(type(y_train))
#print(type(y_pred))
#input()
x=y_train 
y=y_pred.flatten()
#print(min(x), max(x))
#print(min(y), max(y))

plt.figure(figsize=(8, 6), dpi=150)

plt.scatter(x,y,5)
#plt.title("Binding energy", {'fontsize':15})
plt.xlabel('Aptamer/EpCAM binding energy by MARTINI force field (kcal/mol)' , {'fontsize':12})
plt.ylabel('Aptamer/EpCAM binding energy predicted by DNN (kcal/mol)', {'fontsize':12})

#print (plt.xlim()) #plt.axis([-420,-240,-420,-240])
plt.plot(plt.xlim(),plt.xlim(),color='black') #
#plt.hlines(y=374.73, xmin=230, xmax=410,color='red')
#plt.vlines(x=370.27, ymin=270, ymax=380,color='red')

#plt.axis([230,410,270,380])
#plt.plot(plt.xlim(),plt.xlim(),color='black') #
#plt.hlines(y=374.73, xmin=240, xmax=400,color='red')
#plt.vlines(x=370.27, ymin=240, ymax=400,color='red')
plt.show()

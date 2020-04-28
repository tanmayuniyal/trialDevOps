#%%
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import sklearn
import sys
import os
import warnings
import matplotlib.pyplot as plt 

DeprecationWarning('ignore')
os.chdir('D:/machine_learning/digits')


#%%
df=pd.read_csv('train.csv')
df1=pd.read_csv('test.csv')
#%%
image=df.iloc[5:6,1:]
#%%
plt.imshow(image.values.reshape(28,28))


#%%
y=df['label']
x=df.iloc[:,1:].values
#%%
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=13)

#%%
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=18)
rfc.fit(x_train,y_train)
y_predict=rfc.predict(x_test)
score3=accuracy_score(y_test,y_predict)
print(score3)

#%%
predict1 = rfc.predict(df1)

#%%
import cv2
x=cv2.imread('testimage1.png',0)

#%%
img=np.asarray(x)

#%%
plt.imshow(img.reshape(784,784))

#%%
d=pd.DataFrame(img)


#%%
d.to_csv('pixel.csv')

#%%
df2=pd.read_csv('pixel.csv')

#%%
predict2 = rfc.predict(df2)


#%%

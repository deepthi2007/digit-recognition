import cv2 
import pandas as pd
import seaborn as sns 
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X,y = fetch_openml('mnist_784',version=1,return_X_y=True)
""" print(pd.Series(y).value_counts()) """
classes = ['0','1','2','3','4','5','6','7','8','9']
nclasses = len(classes)

samples_per_class = 5
figure = plt.figure(figsize=(nclasses*2,(1+samples_per_class*2)))
idx_cls = 0
for cls in classes :
    idxs = np.flatnonzero(y == cls)
    idxs = np.random.choice(idxs,samples_per_class,replace=False)
    print(idxs)
    i=0
    for idx in idxs :
        print(idx)
        plt_idx = i*nclasses+idx_cls+1
        p = plt.subplot(samples_per_class,nclasses,plt_idx)
        p = sns.heatmap(np.reshape(X[idx],(28,28)),cmap=plt.cm.gray,xticklabels=False,yticklabels=False,cbar=False)
        p = plt.axis("off")
        i+=1
    idx_cls+=1

""" print(len(X))
print(len(X[0]))
print(X[0])
print(y[0]) """

x_train,x_test,y_train,y_test = train_test_split(X,y,train_size = 7500,test_size=2500,random_state=0)

x_train_scaled = x_train/255
x_test_scaled = x_test/255

clf = LogisticRegression(solver="saga",multi_class='multinomial').fit(x_train_scaled,y_train)
predict = clf.predict(x_test_scaled)
acc = accuracy_score(predict,y_test)
print(acc)
print(predict)

import cv2 
import pandas as pd
import seaborn as sns 
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image
import PIL.ImageOps
import os,ssl,time

if (not os.environ.get('PYTHONHTTPSVERIFY','') and getattr(ssl,'_create_verified_context',None)):
    ssl._create_default_https_context = ssl._create_unverified_context
X,y = fetch_openml('mnist_784',version=1,return_X_y=True)
classes = ['0','1','2','3','4','5','6','7','8','9']
nclasses = len(classes)

x_train,x_test,y_train,y_test = train_test_split(X,y,train_size = 7500,test_size=2500,random_state=0)

x_train_scaled = x_train/255
x_test_scaled = x_test/255

clf = LogisticRegression(solver="saga",multi_class='multinomial').fit(x_train_scaled,y_train)
predict = clf.predict(x_test_scaled)
acc = accuracy_score(predict,y_test)

cap = cv2.VideoCapture(0)
while True :
    try:
        ret,frame = cap.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        height,width = gray.shape()
        upperleft = (int(width/2-56),int(height/2-56))
        bottomright = (int(width/2+56),int(height/2+56))
        cv2.rectangle(gray,upperleft,bottomright,(0,255,0),2)
        roi = gray[upperleft[1]:bottomright[1],upperleft[0]:bottomright[0]]
        impil = Image.fromarray(roi)
        imageBw = impil.convert('L')
        imageBw_resize = imageBw.resize((28,28),Image.ANTIALIAS)
        imageBw_resize_inverted = PIL.ImageOps.invert(imageBw_resize)
        pixelfilter = 20
        min_pixel = np.percentile(imageBw_resize_inverted,pixelfilter)
        imageBw_resize_inverted_scaled = np.clip(imageBw_resize_inverted-min_pixel,0,255)
        max_pixel = np.max(imageBw_resize_inverted)
        imageBw_resize_inverted_scaled = np.asarray(imageBw_resize_inverted_scaled)/max_pixel
        test_sample = np.array(imageBw_resize_inverted_scaled).reshape(1,784)

        predict = clf.predict(test_sample)
        print(predict)

        cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
    
    except Exception as e:
        pass

cap.release()
cv2.destroyAllWindows()

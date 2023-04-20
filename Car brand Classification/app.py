# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 22:44:43 2023

@author: hp
"""

from __future__ import division,print_function
import sys
import os
import glob
import re
import numpy as np
#keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input,decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
#flask utils
from flask import Flask,redirect,url_for,request,render_template
from werkzeug.utils import secure_filename

app=Flask(__name__)
MODEL_PATH='car_resnet50.h5'
model=load_model(MODEL_PATH)

#function to predict the car
def model_predict(img_path,model):
    img=image.load_img(img_path,target_size=(224,224))
    x=image.img_to_array(img)
    x=x/255
    x=np.expand_dims(x,axis=0)
    pred=model.predict(x)
    pred=np.argmax(pred,axis=1)
    if pred==0:
        preds="Car is Audi"
    elif preds==1:
        preds="The Car is Lamborghini"
    else:
        preds="The Car Is Mercedes"
    
    return pred

@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
#function to upload the img and predict the class
def upload():
    if request.method=='POST':
        f=request.files['file']
        #save the file
        basepath=os.path.dirname(__file__)
        file_path=os.path.join(
            basepath,'uploads',secure_filename(f.filename))
        f.save(file_path)
        #make prediction
        pred=model.predict(file_path,model)
        result=pred
        return result
    return None

if __name__=='__main__':
    app.run(debug=True)
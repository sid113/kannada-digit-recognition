from flask import Flask
from keras.models import load_model
from keras import backend as K
import jsonpickle
from flask import Flask,Response , request , flash , url_for,jsonify
import pandas as pd
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from PIL import Image
import matplotlib.pyplot as plt;
model = load_model('/home/zerome/kannada digit recognition/my_model_1.h5')
import tensorflow as tf
import keras
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import cv2
import csv

prediction =-1
app=Flask(__name__)
@app.route('/predict',methods=['GET','POST'])
def predict():
    global prediction
    app.logger.debug('Running classifier')
    filename=request.files['file'];
    
    print(prediction)
    image=load_image(filename);
    
    prediction=run_model(image)
    
    print(prediction)
    return  jsonify({"prediction": int(prediction)})
    
def load_image(filename):
      
    img = load_img(filename, target_size=(28, 28,1),grayscale=True)    
    img = img_to_array(img)    
    img = img/255
   
    img = img.reshape(1, 28, 28, 1)    
    
    return img
def run_model(image):
    K.clear_session()
    model = load_model('/home/zerome/kannada digit recognition/my_model_1.h5')
    
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    
    digit = model.predict_classes(image)
    # keras.clear_session()
    return digit;
    
    
if __name__ == '__main__':
    app.debug=True
    app.run(host='172.16.2.130', port=5000);

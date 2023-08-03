from __future__ import division,print_function
from flask import Flask, request, jsonify, render_template
from PIL import Image
import io
import os
import numpy as np
from keras.preprocessing import image
import tensorflow as tf
from keras.models import load_model,Sequential
from werkzeug.utils import secure_filename

global graph
graph=tf.compat.v1.disable_eager_execution()
app=Flask(__name__)


model=load_model("breastcancerpredictmodel.h5")

@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def upload():
    if request.method=='POST':
        f=request.files['image']
        basepath=os.path.dirname(__file__)
        file_path=os.path.join(
            basepath,'uploads',secure_filename(f.filename))
        f.save(file_path)
        img=image.load_img(file_path,target_size=(64,64))
        x=image.img_to_array(img)
        x=np.expand_dims(x,axis=0)      

        with graph.as_default():
            preds=model.predict_classes(x)
        if preds[0][0]==0:
            text = "The tumor is Benign.. Need not worry!"
        else:
            text = "The tumor is Malignant tumor.. Please Consult Doctor"
        text=text
        return jsonify({'prediction': text})
if __name__=='__main__':
    app.run(debug=True,threaded=False)
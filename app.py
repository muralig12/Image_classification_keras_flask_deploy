from __future__ import division, print_function
# coding=utf-8
import sys
import os
import numpy as np
import cv2

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/my_model.h5'

# Load your trained model
model = load_model(MODEL_PATH)
model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#print('Model loaded. Check http://127.0.0.1:5000/')

def model_predict(img_path, model):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (64, 64))
    img = img.reshape(1, 64, 64, 3)
    # print(np.argmax(loaded_model.predict(img)))
    preds = model.predict(img)
    return preds



@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make predictionpip
        preds = model_predict(file_path, model)
        if preds >= 0.5:
               result="its a dog"
               return result
        else:
               result="its a cat"
               return result

    return None


if __name__ == '__main__':
     app.run(port=5003, debug=True)

    # Serve the app with gevent
   # http_server = WSGIServer(('', 5000), app)
  #  http_server.serve_forever()

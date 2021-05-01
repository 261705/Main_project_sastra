# coding=utf-8
import tensorflow as tf
import numpy as np
import keras
import os
import time

# Keras
from keras.models import load_model, model_from_json
from keras.preprocessing import image
from PIL import Image
from keras.initializers import glorot_uniform

# Flask utils
from flask import Flask, url_for, render_template, request,send_from_directory,redirect
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)

# load json file before weights
loaded_json = open("D:\Academics\8th sem\Main Project\Webapp\models\crop.json", "r")
# read json architecture into variable
loaded_json_read = loaded_json.read()
# close file
loaded_json.close()
# retreive model from json
loaded_model = model_from_json(loaded_json_read)
# load weights
loaded_model.load_weights(r'D:\Academics\8th sem\Main Project\Webapp\cropdiseasedetection_weights.h5')
model1 = load_model(r'D:\Academics\8th sem\Main Project\Webapp\one-class.h5')
global graph
graph = tf.get_default_graph()



def leaf_predict(img_path):
    # load image with target size
    img = image.load_img(img_path, target_size=(256, 256))
    # convert to array
    img = image.img_to_array(img)
    # normalize the array
    img /= 255
    # expand dimensions for keras convention
    img = np.expand_dims(img, axis=0)

    with graph.as_default():
        opt = keras.optimizers.Adam(lr=0.001)
        loaded_model.compile(optimizer=opt, loss='mse')
        preds = model1.predict(img)
        dist = np.linalg.norm(img - preds)
        if dist <= 20:
            return "leaf"
        else:
            return "not leaf"


def model_predict(img_path):
    # load image with target size
    img = image.load_img(img_path, target_size=(256, 256))
    # convert to array
    img = image.img_to_array(img)
    # normalize the array
    img /= 255
    # expand dimensions for keras convention
    img = np.expand_dims(img, axis=0)

    with graph.as_default():
        opt = keras.optimizers.Adam(lr=0.001)
        loaded_model.compile(
            optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        preds = loaded_model.predict_classes(img)
        return int(preds)


@app.route('/', methods=['GET', 'POST'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['image']
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        img_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(img_path)
        leaf = leaf_predict(img_path)
        if leaf == "leaf":
            # Make prediction
            preds = model_predict(img_path)
            Classes = ["Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
                       "Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot", "Corn_(maize)___Common_rust",
                       "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot",
                       "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
                       "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy", "Tomato___Bacterial_spot",
                       "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
                       "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite",
                       "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
                       "Tomato___healthy"]
            Disease=Classes[preds]
            return render_template('output.html', result=Disease, filee=f.filename)
        else:
            return render_template('input.html', Error="ERROR: UPLOADED IMAGE IS NOT A LEAF (OR) MORE LEAVES IN ONE IMAGE")
        # return result
    return None

@app.route('/predict/<filename>')
def send_file(filename):
    return send_from_directory('uploads', filename)



if __name__ == '__main__':
    app.run()


import base64
import numpy as np
import io
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from flask import request
from flask import jsonify
from flask import Flask

app = Flask(__name__)

def get_model():
    global model
    model = load_model('image_generation_model.h5')
    print(" * Model loaded!")

def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

@app.route("/predict", methods=["POST"])
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    # print("B-"+encoded)
    # print("B-"+decoded)
    decoded = io.BytesIO(decoded)
    decoded.seek(0)
    image = Image.open(decoded)
    processed_image = preprocess_image(image, target_size=(224, 224))
    get_model()
    prediction = model.predict(processed_image).tolist()

    response = {
        'prediction': {
            'B10': prediction[0][0],
            'B20': prediction[0][1],
            'B50': prediction[0][2],
            'B100': prediction[0][3]
        }
    }
    print('prediction',prediction)
    return jsonify(response)

# export FLASK_APP=predict_app.py
# flask run --host=0.0.0.0
# check http://0.0.0.0:5000/static/bills_detect.html

# $Powersh3ll
# $fileName='/Users/romelldominguez/Pictures/download.png'
# $bytes=[IO.File]::ReadAllBytes($fileName)
# $base64Image=[Convert]::ToBase64String($bytes)
# $message=@{image=$base64Image}
# $jsonified=ConvertTo-Json $message
# $response=Invoke-RestMethod -Method Post -Uri "http://0.0.0.0:5000/predict" -Body $jsonified
# $response.prediction | format-list

# $curl
# fileName='/Users/romelldominguez/Pictures/download.png'
# base64Image=$(base64 $fileName)
# jsonified="{\"image\":\"${base64Image}\"}"
# echo $jsonified >> data.json
# curl -X POST --data @data.json http://0.0.0.0:5000/predict

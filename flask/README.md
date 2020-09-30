```sh
export FLASK_APP=predict_app.py
flask run --host=0.0.0.0
# check http://0.0.0.0:5000/static/bills_detect.html
# check http://0.0.0.0:81/predict-with-visuals.html
```

##  Powersh3ll
```sh
$fileName='/Users/romelldominguez/Pictures/download.png'
$bytes=[IO.File]::ReadAllBytes($fileName)
$base64Image=[Convert]::ToBase64String($bytes)
$message=@{image=$base64Image}
$jsonified=ConvertTo-Json $message
$response=Invoke-RestMethod -Method Post -Uri "http://0.0.0.0:5000/predict" -Body $jsonified
$response.prediction | format-list
```

## Transform TFkeras to TFjs
```sh
pip install tensorflowjs
npm install -g nodemon
```

```sh
python test.py
tensorflowjs_converter --input_format keras image_generation_model.h5 static/models/
cd local-server/;nodemon server.js
# ./static/models/
#     group1-shard1of1
#     group2-shard1of1
#     group3-shard1of1
#     model.json 
```

```python
from tensorflow import keras
import tensorflowjs as tfjs
vgg16 = keras.applications.vgg16.VGG16()
tfjs.converters.save_keras_model(vgg16,'static/models')
# change "Functional" to "Model" in model.js file
```

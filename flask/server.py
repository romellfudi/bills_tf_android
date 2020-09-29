from flask import request
from flask import jsonify
from flask import Flask

app = Flask(__name__)

@app.route('/hello',methods=['POST'])
def hello():
    message = request.get_json(force=True)
    name = message['name']
    response = {
        'greeting': 'Hello, ' + name + '!'
    }
    return jsonify(response)

@app.route('/')
def index():
    return 'Index Page'
    
# export FLASK_APP=server.py
# py -m flask run
# flask run --host=0.0.0.0


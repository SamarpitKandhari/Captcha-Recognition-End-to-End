import os
from flask_cors import CORS, cross_origin
from flask import Flask,request,render_template,jsonify

from src.pipeline.predict_pipeline import PredictPipeline
from src.utils import decodeImage

from src.logger import logging

app = Flask(__name__)
CORS(app)

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictPipeline(self.filename)

## Route for a home page

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('home.html')

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    image = request.json['image']
    logging.info("Image Loaded")
    decodeImage(image, clApp.filename)
    result = clApp.classifier.predict()
    return jsonify(result)



if __name__=="__main__":
    clApp = ClientApp()
    app.run(host="0.0.0.0") 
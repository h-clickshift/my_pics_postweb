from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.applications.resnet import ResNet50
from PIL import Image
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    # Get the uploaded image file
    img_file = request.files['image']
    filename = img_file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img_file.save(filepath)


    model = ResNet50(weights='imagenet')
    # Preprocess the image
    img = Image.open(filepath)
    img = img.resize((224, 224))
    x = preprocess_input(np.array(img))

    # Make a prediction using the model
    preds = model.predict(np.array([x]))

    # Decode the predictions and return the top 3 labels
    decoded_preds = decode_predictions(preds, top=3)[0]
    labels = []
    for label in decoded_preds:
        labels.append(label[1])

    return render_template('result.html', labels=labels)

if __name__ == '__main__':
    app.run(debug=True)

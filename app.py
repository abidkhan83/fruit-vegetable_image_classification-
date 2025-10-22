from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Load your model
model = load_model('image_class_model.keras')

# Class names
data_cat = ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum',
 'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant',
 'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce',
 'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple',
 'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn',
 'sweetpotato', 'tomato', 'turnip', 'watermelon']

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    filename = None

    if request.method == "POST":
        file = request.files["file"]
        if file:
            filename = file.filename
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Preprocess the image
            img_height, img_width = 180, 180
            img = Image.open(filepath).convert("RGB").resize((img_width, img_height))
            img_array = tf.keras.utils.img_to_array(img)
            img_batched = tf.expand_dims(img_array, 0)

            predictions = model.predict(img_batched)
            score = tf.nn.softmax(predictions[0])
            prediction = data_cat[np.argmax(score)]
            confidence = float(np.max(score)) * 100

 
    return render_template("index.html", prediction=prediction, confidence=confidence, filename=filename)


if __name__ == "__main__":
    app.run(debug=True)
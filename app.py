from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import joblib
from main import create_model
import base64
from PIL import Image
import io

app = Flask(__name__)
model = create_model()  # Load your model
class_names = joblib.load('class_names.joblib')

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        data = request.get_json()
        if "image" in data:
            # Decode base64 image data
            img_data = data["image"].split(",")[1]
            img = Image.open(io.BytesIO(base64.b64decode(img_data)))
            # Convert image to RGB format
            img = img.convert("RGB")
            # Preprocess image
            img = img.resize((224, 224))
            img = tf.keras.preprocessing.image.img_to_array(img)
            img = img / 255.0
            img = tf.expand_dims(img, axis=0)
            # Predict
            pred_prob = model.predict(img)
            pred_class = class_names[pred_prob.argmax()]
            return jsonify({'prediction': pred_class})
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)

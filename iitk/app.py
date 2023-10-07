
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

# Define wine quality labels in the desired order
wine_quality_labels = ["Very Poor (3)", "Poor (4)", "Below Average (5)", "Average (6)", "Good (7)", "Very Good (8)", "Excellent (9)"]

# Function to map integer predictions to labels
def map_to_quality_label(prediction):
    # Ensure the prediction is within the label range
    prediction = min(max(prediction, 3), 9)
    return wine_quality_labels[prediction - 3]

@flask_app.route("/")
def Home():
    return render_template("indexwine.html")

@flask_app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    # Round the prediction to the nearest integer
    rounded_prediction = round(prediction[0])
    # Map the rounded prediction to a quality label
    quality_label = map_to_quality_label(rounded_prediction)
    return render_template("indexwine.html", prediction_text=f"Quality of wine is {quality_label}")

if __name__ == "__main__":
    flask_app.run(debug=True)

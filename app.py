from transformers import pipeline
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load pre-trained model
sentiment_model = pipeline("sentiment-analysis")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.json.get("text")
    result = sentiment_model(text)
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

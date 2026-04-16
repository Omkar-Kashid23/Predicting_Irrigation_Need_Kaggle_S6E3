from flask import Flask, request, jsonify
from model_utils import predict

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    return {"status": "healthy"}

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    try:
        data = request.get_json(force=True)
        result = predict(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

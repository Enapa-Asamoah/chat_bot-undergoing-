from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from chatbot import get_response

app = Flask(__name__)
CORS(app)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if "message" in data:
        message = data["message"]
        response = get_response(message)
        return jsonify({"answer": response})
    else:
        return jsonify({"error": "No 'message' field provided in the request."}), 400

if __name__ == "__main__":
    app.run(debug=True)

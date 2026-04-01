from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("model/lightgbm.pkl")
features = joblib.load("model/features.pkl")

@app.route("/")
def home():
    return "Fraud Detection API Running 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        input_df = pd.DataFrame([data])

        # Fix columns
        for col in features:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[features]

        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        return jsonify({
            "fraud": int(pred),
            "probability": float(prob),
            "message": "Fraud 🚨" if pred == 1 else "Safe ✅"
        })

    except Exception as e:
        return jsonify({"error": str(e)})
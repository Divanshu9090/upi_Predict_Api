from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load models
xgb_model = joblib.load("./model/xgboost.pkl")
cat_model = joblib.load("./model/catboost.pkl")
lgb_model = joblib.load("./model/lightgbm.pkl")

# Load feature order
features = joblib.load("./model/features.pkl")


def prepare_input(data):
    input_df = pd.DataFrame([data])

    # Add missing columns
    for col in features:
        if col not in input_df.columns:
            input_df[col] = 0

    # Arrange order
    input_df = input_df[features]

    return input_df

def make_response(model_name, prediction, probability):
    return jsonify({
        "model": model_name,
        "fraud": int(prediction),
        "probability": float(probability),
        "message": "Fraud Detected 🚨" if prediction == 1 else "Legitimate Transaction ✅"
    })

# ROUTES
@app.route("/")
def home():
    return "Fraud Detection API's Running 🚀"

# XGBOOST
@app.route("/predictxt", methods=["POST"])
def predict_xgb():
    try:
        data = request.get_json()
        input_df = prepare_input(data)

        pred = xgb_model.predict(input_df)[0]
        prob = xgb_model.predict_proba(input_df)[0][1]

        return make_response("xgboost", pred, prob)

    except Exception as e:
        return jsonify({"error": str(e)})

# CATBOOST
@app.route("/predictct", methods=["POST"])
def predict_cat():
    try:
        data = request.get_json()
        input_df = prepare_input(data)

        pred = cat_model.predict(input_df)[0]
        prob = cat_model.predict_proba(input_df)[0][1]

        return make_response("catboost", pred, prob)

    except Exception as e:
        return jsonify({"error": str(e)})

# LIGHTGBM
@app.route("/predictlm", methods=["POST"])
def predict_lgb():
    try:
        data = request.get_json()
        input_df = prepare_input(data)

        pred = lgb_model.predict(input_df)[0]
        prob = lgb_model.predict_proba(input_df)[0][1]

        return make_response("lightgbm", pred, prob)

    except Exception as e:
        return jsonify({"error": str(e)})
    
#  Run the app
if __name__ == "__main__":
    app.run(debug=True)
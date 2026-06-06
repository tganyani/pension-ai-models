from flask import Flask, request, jsonify
import pandas as pd
import joblib
import json
from flask_cors import CORS
from xgboost import XGBClassifier



app = Flask(__name__)
CORS(app)
# Load model and metadata
model = joblib.load('compliance_model.pt')
fraud_model = XGBClassifier()
fraud_model.load_model("fraud_model.json")
fraud_preprocessor = joblib.load("fraud_preprocessor.pkl")

with open('model_metadata.json', 'r') as f:
    metadata = json.load(f)

feature_names = metadata['feature_names']


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        input_df = pd.DataFrame([data])

        # One-hot encode
        input_processed = pd.get_dummies(input_df)

        # Match training columns
        input_final = input_processed.reindex(
            columns=feature_names,
            fill_value=0
        )

        prediction_prob = model.predict_proba(input_final)[0][1]
        prediction_class = int(model.predict(input_final)[0])

        risk_level = "Low"

        if prediction_prob > 0.7:
            risk_level = "High"
        elif prediction_prob > 0.3:
            risk_level = "Medium"

        return jsonify({
            "prediction": prediction_class,
            "probability": round(float(prediction_prob), 4),
            "risk_level": risk_level,
            "status": "success"
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 400

# predict fraud
@app.route('/predict-fraud', methods=['POST'])
def predict_fraud():
    try:
        data = request.get_json()

        # Convert request to DataFrame
        input_df = pd.DataFrame([data])

        # Apply preprocessing
        input_processed = fraud_preprocessor.transform(input_df)

        # Predict
        prediction_prob = fraud_model.predict_proba(input_processed)[0][1]
        prediction_class = int(fraud_model.predict(input_processed)[0])

        if prediction_prob >= 0.7:
            risk_level = "High"
        elif prediction_prob >= 0.3:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        return jsonify({
            "prediction": prediction_class,
            "probability": round(float(prediction_prob), 4),
            "risk_level": risk_level,
            "fraud_detected": bool(prediction_class),
            "status": "success"
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 400
@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": True
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
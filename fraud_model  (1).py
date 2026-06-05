from flask import Flask, request, jsonify
import pandas as pd
import joblib
import json
import numpy as np

app = Flask(__name__)

# Load the model and metadata
model = joblib.load('compliance_model.pt')
with open('model_metadata.json', 'r') as f:
    metadata = json.load(f)
    feature_names = metadata['feature_names']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from POST request
        data = request.get_json()
        
        # Convert to DataFrame
        input_df = pd.DataFrame([data])
        
        # Apply the same one-hot encoding as during training
        # We need to ensure all training columns are present
        input_processed = pd.get_dummies(input_df)
        
        # Reindex to match training features, filling missing with 0
        input_final = input_processed.reindex(columns=feature_names, fill_value=0)
        
        # Make prediction
        prediction_prob = model.predict_proba(input_final)[0][1]
        prediction_class = int(model.predict(input_final)[0])
        
        # Determine risk level
        risk_level = "Low"
        if prediction_prob > 0.7:
            risk_level = "High"
        elif prediction_prob > 0.3:
            risk_level = "Medium"
            
        return jsonify({
            'prediction': prediction_class,
            'probability': float(round(prediction_prob, 4)),
            'risk_level': risk_level,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': True})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

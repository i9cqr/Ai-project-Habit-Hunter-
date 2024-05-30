import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template

# Create Flask app
app = Flask(__name__)

# Load the model and scaler
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Define routes
@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for making predictions."""
    try:
        # Get input data from request
        data = request.get_json(force=True)

        # Validate input data
        features = validate_input(data)

        # Scale the features
        scaled_features = scaler.transform(features)

        # Predict the class
        prediction = model.predict(scaled_features)
        output = int(prediction[0])  # Convert prediction to int

        return jsonify({'prediction': output})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Input validation function
def validate_input(data):
    """Validate and preprocess input data."""
    # Define required fields
    required_fields = ['Sex', 'age', 'height', 'weight','waistline', 'sight_left', 'sight_right',
                       'hear_left', 'hear_right', 'SBP', 'DBP', 'BLDS', 'tot_chole',
                       'HDL_chole', 'LDL_chole', 'triglyceride', 'hemoglobin',
                       'urine_protein', 'serum_creatinine', 'SGOT_AST', 'SGOT_ALT',
                       'gamma_GTP', 'DRK_YN']

    # Check if all required fields are present
    for field in required_fields:
        if field not in data:
            raise ValueError(f'Missing required field: {field}')

    # Extract features
    features = [data[field] for field in required_fields]
    features = np.array(features).reshape(1, -1)  # Convert to 2D array

    return features

if __name__ == '__main__':
    app.run(debug=True)

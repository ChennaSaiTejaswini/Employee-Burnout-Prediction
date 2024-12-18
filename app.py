from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model and scaler
with open('models/linear_regression_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('models/scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Get feature names from the scaler (this will give us the order)
feature_names = scaler.feature_names_in_

# Home route
@app.route('/')
def home():
    return render_template('index.html')  # Render the HTML form

# Route to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract data from the form
        input_data = {
            'Company Type': request.form.get('Company Type'),
            'WFH Setup Available': request.form.get('WFH Setup Available'),
            'Gender': request.form.get('Gender'),
            'Designation': float(request.form.get('Designation')),
            'Resource Allocation': float(request.form.get('Resource Allocation')),
            'Mental Fatigue Score': float(request.form.get('Mental Fatigue Score'))
        }

        # Convert the input data into a DataFrame
        input_df = pd.DataFrame([input_data])

        # One-hot encode categorical variables
        input_df = pd.get_dummies(input_df, columns=['Company Type', 'WFH Setup Available', 'Gender'], drop_first=True)

        # Ensure the columns in the input data match the feature names from training
        missing_cols = set(feature_names) - set(input_df.columns)
        for col in missing_cols:
            input_df[col] = 0  # Add missing columns with a value of 0

        # Reorder the columns to match the training data
        input_df = input_df[feature_names]

        # Scale the input data
        input_df = scaler.transform(input_df)

        # Make a prediction
        prediction = model.predict(input_df)
        predicted_burnout = round(prediction[0], 2)

        # Return a more user-friendly and formatted response
        response = {
            "status": "success",
            "message": "Burnout Rate Prediction",
            "data": {
                "predicted_burnout_rate": f"{predicted_burnout * 100}%",  # Displaying as percentage
                "confidence": "This is the predicted burnout rate based on the data provided."
            }
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': 'An error occurred during the prediction',
            'error': str(e)
        })


# Run the app
if __name__ == '__main__':
    app.run(debug=True)

# Import necessary libraries
import numpy as np
import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load dataset
data = pd.read_excel("employee_burnout_analysis-AI.xlsx")

# Data overview
print("Shape of the dataset:", data.shape)
print(data.head())
print(data.describe())

# Check for missing values
print("Missing values in each column:\n", data.isnull().sum())

# Drop rows with missing values in 'Burn Rate' and other columns
data = data.dropna(subset=['Burn Rate', 'Resource Allocation', 'Mental Fatigue Score'])

# Drop 'Employee ID' as it is not useful
data = data.drop('Employee ID', axis=1)

# Create a new feature 'Days' to represent seniority
data['Date of Joining'] = pd.to_datetime(data['Date of Joining'])
data['Days'] = (pd.to_datetime("2008-01-01") - data['Date of Joining']).dt.days

# Drop 'Date of Joining' and 'Days' if no strong correlation
data = data.drop(['Date of Joining', 'Days'], axis=1)

# One-hot encode categorical features
data = pd.get_dummies(data, columns=['Company Type', 'WFH Setup Available', 'Gender'], drop_first=True)

# Split the data into features and target
X = data.drop('Burn Rate', axis=1)
y = data['Burn Rate']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=1)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler
os.makedirs("models", exist_ok=True)
with open("models/scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

# Train a Linear Regression model
linear_regression_model = LinearRegression()
linear_regression_model.fit(X_train, y_train)

# Make predictions
y_pred = linear_regression_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nLinear Regression Model Performance Metrics:")
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("Mean Absolute Error:", mae)
print("R-squared Score:", r2)

# Save the model
with open("models/linear_regression_model.pkl", "wb") as model_file:
    pickle.dump(linear_regression_model, model_file)


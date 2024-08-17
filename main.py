import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv('insurance.csv')

# Encode categorical variables
df_encoded = pd.get_dummies(df, drop_first=True)

# Prepare the features and target variable
X = df_encoded.drop('charges', axis=1)
y = df_encoded['charges']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize the Random Forest regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions on the training and testing sets
y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)

# Evaluate the model
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

# Print the results
print(f"Training MSE: {mse_train:.2f}, R^2: {r2_train:.2f}")
print(f"Testing MSE: {mse_test:.2f}, R^2: {r2_test:.2f}")

# Plotting the results
plt.figure(figsize=(14, 7))

# Training data predictions
plt.subplot(1, 2, 1)
plt.scatter(y_train, y_train_pred, alpha=0.5, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--k')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Training Data: Actual vs. Predicted')

# Testing data predictions
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_test_pred, alpha=0.5, color='red')
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--k')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Testing Data: Actual vs. Predicted')

plt.show()

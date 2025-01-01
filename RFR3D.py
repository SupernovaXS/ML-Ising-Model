import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Step 1: Load Data
data = pd.read_csv('Iron 0-5 Data.csv', header=None, names=["Energy (MeV)", "Total Attenuation (cm^2/g)", "Attenuation Coefficient (cm^-1)"])
# print(data.head())

# Step 2: Convert to numeric, handle errors and drop NaNs
data['Energy (MeV)'] = pd.to_numeric(data['Energy (MeV)'], errors='coerce')
data['Attenuation Coefficient (cm^-1)'] = pd.to_numeric(data['Attenuation Coefficient (cm^-1)'], errors='coerce')
data = data.dropna()

# Step 3: Filter data range between 10^0 and 10^5 (for energy values)
mask = (data['Energy (MeV)'] >= 1) & (data['Energy (MeV)'] <= 100000)
data_range = data[mask]

# Extract X (Energy) and y (Attenuation Coefficient)
X = data_range['Energy (MeV)'].values.reshape(-1, 1)  # Ensure X is a 2D array for ML models
y = data_range['Attenuation Coefficient (cm^-1)'].values

# Step 4: Shuffle and split data ensuring that energy values are distributed across both sets
# We use a stratified split based on quantiles of energy values to ensure even distribution

# Create a new column for quantiles of energy
data_range['Quantile'] = pd.qcut(data_range['Energy (MeV)'], q=10, labels=False)

# Perform stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=data_range['Quantile'], random_state=42)

# Step 5: Train the Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Step 6: Predict for the testing set
y_test_pred = rf_model.predict(X_test)

# Generate a line for the predicted values over the full range of X (from training + testing set)
X_full = np.logspace(np.log10(X.min()), np.log10(X.max()), num=500).reshape(-1, 1)  # Using logspace for smoothness
y_full_pred = rf_model.predict(X_full)

'''

# Create the first plot (only blue dots and red x's for testing data)
plt.figure(figsize=(10, 6))

# Plot the testing set actual values as blue
plt.scatter(X_test.flatten(), y_test, color='blue', label='Actual Values', alpha=0.7)

# Plot the testing set predictions as red crosses
plt.scatter(X_test.flatten(), y_test_pred, color='red', label='Predicted Values', alpha=0.7, marker='x')

# Plot the predicted line in green
plt.plot(X_full.flatten(), y_full_pred, color='green', label='Predicted Line', linewidth=2)

# Set logarithmic scale for x-axis
plt.xscale('log')

# Add labels and title
plt.xlabel('Energy (MeV)', fontsize=12)
plt.ylabel('Attenuation Coefficient (cm^-1)', fontsize=12)
plt.title('Testing Set: Predicted vs Actual Attenuation Coefficients', fontsize=14)

# Add a legend
plt.legend(fontsize=12)

# Adjust the layout and display the plot
plt.tight_layout()

# Save the first plot to a file
plt.savefig("rfr3d_testing.png")

# Show the first plot
plt.show()

# Create the second plot (with black dots for training data)
plt.figure(figsize=(10, 6))

# Plot the training data points as gray
plt.scatter(X_train.flatten(), y_train, color='gray', label='Training Data', alpha=0.7)

# Plot the testing set actual values as blue
plt.scatter(X_test.flatten(), y_test, color='blue', label='Actual Values', alpha=0.7)

# Plot the testing set predictions as red crosses
plt.scatter(X_test.flatten(), y_test_pred, color='red', label='Predicted Values', alpha=0.7, marker='x')

# Plot the predicted line in green
plt.plot(X_full.flatten(), y_full_pred, color='green', label='Predicted Line', linewidth=2)

# Set logarithmic scale for x-axis
plt.xscale('log')

# Add labels and title
plt.xlabel('Energy (MeV)', fontsize=12)
plt.ylabel('Attenuation Coefficient (cm^-1)', fontsize=12)
plt.title('Model Performance: Training and Testing Data vs Predictions', fontsize=14)

# Add a legend
plt.legend(fontsize=12)

# Adjust the layout and display the plot
plt.tight_layout()

# Save the second plot to a file
plt.savefig("rfr3d_all.png")

# Show the second plot
plt.show()

'''

# Step 7: Evaluate the model
mse_test = mean_squared_error(y_test, y_test_pred)
# print(f"Test Mean Squared Error: {mse_test:.6f}")

# Print testing data along with predictions
test_results = pd.DataFrame({
    'Energy (MeV)': X_test.flatten(),
    'Actual Attenuation Coefficient (cm^-1)': y_test,
    'Predicted Attenuation Coefficient (cm^-1)': y_test_pred
})
# print("\nTesting Results (Actual vs Predicted):")
# print(test_results)

# Predict the attenuation coefficient for a given energy
def predict(given_energy, thickness_cm):
    given_energy_reshaped = np.array([[given_energy]])
    predicted_coeff = round(float(rf_model.predict(given_energy_reshaped)), 6)
    predicted_trans = round(float(np.exp(-predicted_coeff*thickness_cm)), 6)
    predicted_att = round(float(1 - predicted_trans), 6)
    return predicted_coeff, predicted_trans, predicted_att

# given_energy = 300
# thickness = 1

# dataset = predict(given_energy, thickness)
# coeff = dataset[0]
# trans = dataset[1]
# att = dataset[2]

# print(f"The predicted attenuation coefficient for energy {given_energy} MeV is: {coeff[0]:.6f}")
# print(f"The predicted transmission for energy {given_energy} MeV and a thickness of {thickness} is {trans[0]:.6f}%.")
# print(f"The predicted attenuation for energy {given_energy} MeV and a thickness of {thickness} is {att[0]:.6f}%.")


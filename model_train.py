import numpy as np
import pickle
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

# Create output directory for models and visualizations
os.makedirs('models', exist_ok=True)
os.makedirs('visualizations', exist_ok=True)

# Load the sequences data
sequences = np.load("sequences_test_Test001.npy")

#Converting the sequence to 2D as isolation forest requires 2D input
# Assuming the sequences are in the shape (n_samples, sequence_length, n_features)
# converting each seq in single vector
n_sequences, seq_length, n_features = sequences.shape
print(f"Original sequences shape: {sequences.shape}")

X = sequences.reshape(n_sequences, seq_length * n_features)
print(f"Reshaped data for Isolation Forest: {X.shape}")

# Standardize the features
print("Standardizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data for training and validating
X_train, X_val = train_test_split(X_scaled, test_size=0.2, random_state=42)
print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")

# Train model
contamination = 0.05  # Assuming 5% of data is anomalous
model = IsolationForest(
    n_estimators=100,
    max_samples='auto',
    contamination=contamination,
    random_state=42,
    n_jobs=-1  # Use all available cores
)

model.fit(X_train)

# Save the model and scaler
print("Saving model and scaler...")
with open('models/isolation_forest_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Predict on validation set (1: normal, -1: anomaly)
y_pred = model.predict(X_val)
# Convert to binary (0: anomaly, 1: normal)
y_pred_binary = np.where(y_pred == 1, 1, 0)

# Get anomaly scores
anomaly_scores = model.decision_function(X_val)
print(f"Anomaly score range: {np.min(anomaly_scores)} to {np.max(anomaly_scores)}")

# Plot anomaly scores
plt.figure(figsize=(12, 6))
plt.plot(anomaly_scores)
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
plt.title('Anomaly Scores on Validation Set')
plt.xlabel('Sample Index')
plt.ylabel('Anomaly Score (lower = more anomalous)')
plt.savefig('visualizations/anomaly_scores.png')

# Plot anomaly distribution
plt.figure(figsize=(12, 6))
plt.hist(anomaly_scores, bins=50)
plt.axvline(x=0, color='r', linestyle='-', alpha=0.3)
plt.title('Distribution of Anomaly Scores')
plt.xlabel('Anomaly Score')
plt.ylabel('Frequency')
plt.savefig('visualizations/anomaly_distribution.png')

print(f"Number of predicted anomalies: {np.sum(y_pred_binary == 0)} out of {len(y_pred_binary)}")
print(f"Percentage of anomalies: {np.sum(y_pred_binary == 0) / len(y_pred_binary) * 100:.2f}%")

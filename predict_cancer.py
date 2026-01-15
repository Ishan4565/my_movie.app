import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

# ==========================================
# PART 1: THE TRAINER (Building the Model)
# ==========================================
print("Step 1: Loading and Splitting Data...")
cancer = load_breast_cancer()
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the assembly line (Pipeline)
full_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=10)), # We start with 10, but GridSearch will optimize this
    ('knn', KNeighborsClassifier())
])

# Define the "Search Grid" for the robot to find the best settings
param_grid = {
    'pca__n_components': [5, 10, 15, 20],
    'knn__n_neighbors': [3, 5, 7]
}

print("Step 2: Optimizing with GridSearchCV (This may take a moment)...")
grid_search = GridSearchCV(full_pipeline, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Save the absolute best version of the model
best_model = grid_search.best_estimator_
joblib.dump(best_model, 'cancer_model.pkl')
print(f"Model Saved! Best CV Accuracy: {grid_search.best_score_:.2f}")

# ==========================================
# PART 2: THE PREDICTOR (Visualization & New Data)
# ==========================================
print("\nStep 3: Generating Visualization and Predictions...")

# Load the model back up
model = joblib.load('cancer_model.pkl')

# Extract components for the PCA plot (using only first 2 for 2D visual)
pca_step = model.named_steps['pca']
scaler_step = model.named_steps['scaler']

# Create pca_df for the plot
X_scaled = scaler_step.transform(X)
X_pca = pca_step.transform(X_scaled)
pca_df = pd.DataFrame(X_pca[:, :2], columns=['PC1', 'PC2'])

# Create the plot
plt.figure(figsize=(10, 6))
plt.scatter(pca_df['PC1'], pca_df['PC2'], c=y, cmap='RdBu', alpha=0.7, edgecolors='k')
plt.title(f"PCA Visualization (Using {grid_search.best_params_['pca__n_components']} Components)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label="0: Malignant, 1: Benign")
plt.show()

# Simulate one new patient
new_patient = np.random.rand(1, 30) 
prediction = model.predict(new_patient)
confidence = np.max(model.predict_proba(new_patient)) * 100

print(f"New Patient Diagnosis: {'Benign' if prediction[0] == 1 else 'Malignant'}")
print(f"Confidence: {confidence:.2f}%")
# A complete list of 30 features to match the StandardScaler's expectations
# Order: Mean (10), Error (10), and "Worst" (10) features
patient_measurements = [
    14.12, 19.29, 91.58, 616.1, 0.096, 0.104, 0.088, 0.048, 0.181, 0.062, # Means
    0.405, 1.216, 2.866, 40.33, 0.007, 0.025, 0.032, 0.011, 0.020, 0.004, # Errors
    16.26, 25.67, 107.2, 880.5, 0.132, 0.254, 0.272, 0.114, 0.290, 0.083  # "Worst"
]

# Reshape into a 2D array (1 row, 30 columns)
patient_data = np.array(patient_measurements).reshape(1, -1)

# Now, call predict
prediction = model.predict(patient_data)
print(f"Prediction: {prediction}")
# Print the names of all 30 features the model expects
for i, name in enumerate(cancer.feature_names):
    print(f"Feature {i+1}: {name}")
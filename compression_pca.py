import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Load the data
cancer = load_breast_cancer()
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)

# 2. CRITICAL STEP: Scaling
# PCA is a "distance-based" algorithm. If 'area' is 1000 and 'smoothness' is 0.1, 
# PCA will think 'area' is the only thing that matters.
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# 3. Apply PCA
# We want to reduce 30 features down to just 2 for visualization
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

# 4. Create a new "Squashed" DataFrame
pca_df = pd.DataFrame(data=pca_data, columns=['PC1', 'PC2'])
pca_df['Target'] = cancer.target # 0 = Malignant, 1 = Benign
plt.figure(figsize=(8,6))
plt.scatter(pca_df['PC1'], pca_df['PC2'], c=pca_df['Target'], cmap='plasma')
plt.xlabel('First Principal Component (PC1)')
plt.ylabel('Second Principal Component (PC2)')
plt.title('PCA: 30 Features reduced to 2 for Cancer Diagnosis')
plt.show()
print(pca.explained_variance_ratio_)
import numpy as np

# 1. Run PCA on all 30 features
pca_full = PCA().fit(scaled_data)

# 2. Plot the cumulative explained variance
plt.figure(figsize=(8,4))
plt.plot(np.cumsum(pca_full.explained_variance_ratio_), marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.axhline(y=0.95, color='r', linestyle='--') # 95% threshold
plt.title('Finding the Elbow Point')
plt.show()
# Create a DataFrame of the PCA components
components_df = pd.DataFrame(
    pca.components_, 
    columns=cancer.feature_names, 
    index=['PC1', 'PC2']
)

# Look at the 'Recipe' for PC1
pc1_recipe = components_df.iloc[0].sort_values(ascending=False)
print("Top features contributing to PC1:")
print(pc1_recipe.head(5))
# 1. Fit PCA on all original features
pca_all = PCA().fit(scaled_data)

# 2. Calculate the "Cumulative Explained Variance"
cumulative_variance = np.cumsum(pca_all.explained_variance_ratio_)

# 3. Find the first index where variance > 95%
optimal_components = np.argmax(cumulative_variance >= 0.95) + 1

print(f"To keep 95% of the information, you need {optimal_components} components.")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import load_breast_cancer

# 1. Load the raw data
cancer = load_breast_cancer()

# 2. Define X (The Measurements) and y (The Diagnosis)
X = cancer.data    # This is the 2D array of 30 features
y = cancer.target  # This is the 1D array of 0s and 1s

# 3. NOW you can split them
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 1. Split the data first
# Assuming 'X' is your 30 features and 'y' is your target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Scale the data (This defines the missing name!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Now PCA will work because X_train_scaled exists
pca = PCA(n_components=10)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

# 1. Load Data
cancer = load_breast_cancer()
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = cancer.target

# 2. Split (Keep the test set in a 'vault' for the very end)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Create the Pipeline (The Assembly Line)
# This defines the sequence so you never get a NameError again!
my_pipeline = Pipeline([
    ('scaler', StandardScaler()),         # Step 1: Scale
    ('pca', PCA(n_components=10)),        # Step 2: Compress to 10 dims
    ('knn', KNeighborsClassifier(n_neighbors=5)) # Step 3: Classify
])

# 4. Cross-Validation using the Pipeline
# This runs the WHOLE assembly line 5 times on different folds
scores = cross_val_score(my_pipeline, X_train, y_train, cv=5)

print(f"Cross-Validation Scores: {scores}")
print(f"Average Accuracy: {scores.mean():.2f}")

# 5. Final Test (Only do this once you are happy with the CV score!)
my_pipeline.fit(X_train, y_train)
print(f"Final Test Accuracy: {my_pipeline.score(X_test, y_test):.2f}")
from sklearn.model_selection import GridSearchCV

# 1. Define the parameter grid
# Format: 'stepname__parametername'
param_grid = {
    'pca__n_components': [5, 10, 15, 20],  # Testing different dimensions
    'knn__n_neighbors': [3, 5, 7, 9]       # Testing different neighbor counts
}

# 2. Create the GridSearch object
# It uses your pipeline as the base
grid_search = GridSearchCV(my_pipeline, param_grid, cv=5, scoring='accuracy')

# 3. Run the search
grid_search.fit(X_train, y_train)

# 4. See the results
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best CV Score: {grid_search.best_score_:.2f}")
import joblib

# 1. Grab the best model found by the robot
best_model = grid_search.best_estimator_

# 2. Save it to your desktop
joblib.dump(best_model, 'cancer_diagnosis_model.pkl')

print("Model saved and ready for the real world!")


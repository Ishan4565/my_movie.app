# ===============================
# 1️⃣ IMPORT LIBRARIES
# ===============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import joblib


# ===============================
# 2️⃣ LOAD DATA
# ===============================
df = pd.read_csv("cleaned_cafe_sales.csv")


print("Data loaded successfully")
print(df.head())


# ===============================
# 3️⃣ FEATURE ENGINEERING
# ===============================

# Average price per item
df["Item_Avg_Price"] = df.groupby("Item")["Price_Per_Unit"].transform("mean")

# Transaction day feature
df["Transaction_Date"] = pd.to_datetime(df["Transaction_Date"])
df["Transaction_Day"] = df["Transaction_Date"].dt.dayofweek


# ===============================
# 4️⃣ OUTLIER REMOVAL (IQR METHOD)
# ===============================

Q1 = df["Total_Spent"].quantile(0.25)
Q3 = df["Total_Spent"].quantile(0.75)
IQR = Q3 - Q1

df = df[
    (df["Total_Spent"] >= Q1 - 1.5 * IQR) &
    (df["Total_Spent"] <= Q3 + 1.5 * IQR)
]

print("Outliers removed")


# ===============================
# 5️⃣ ENCODING CATEGORICAL DATA
# ===============================

df_encoded = pd.get_dummies(
    df,
    columns=["Item", "Location", "Payment_Method"],
    drop_first=True
)

print("Encoding completed")


# ===============================
# 6️⃣ FEATURE SELECTION
# ===============================

features = [
    "Quantity",
    "Price_Per_Unit",
    "Item_Avg_Price",
    "Transaction_Day"
]

X = df_encoded[features]
y = df_encoded["Total_Spent"]


# ===============================
# 7️⃣ TRAIN-TEST SPLIT
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ===============================
# 8️⃣ TRAIN MODEL
# ===============================

model = LinearRegression()
model.fit(X_train, y_train)

print("Model trained successfully")


# ===============================
# 9️⃣ PREDICTION & EVALUATION
# ===============================

y_pred = model.predict(X_test)

print("\nModel Performance:")
print("MAE :", mean_absolute_error(y_test, y_pred))
print("MSE :", mean_squared_error(y_test, y_pred))
print("R²  :", r2_score(y_test, y_pred))


# ===============================
# 🔟 VISUALIZE RESULTS
# ===============================

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Total Spent")
plt.ylabel("Predicted Total Spent")
plt.title("Actual vs Predicted")
plt.show()


# ===============================
# 1️⃣1️⃣ SAVE MODEL & FEATURES
# ===============================

joblib.dump(model, "total_spent_model.pkl")
joblib.dump(features, "model_features.pkl")

print("Model and features saved successfully")
import os
print(os.listdir())


from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
# --- STEP 1: Load Data ---
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

# --- STEP 2: Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# --- STEP 3: Scale ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # This defines X_train_scaled

# --- STEP 4: PCA ---
pca = PCA(n_components=10)
X_train_pca = pca.fit_transform(X_train_scaled) # THIS defines X_train_pca

# --- STEP 5: Cross-Validation ---
# Now this line will work because X_train_pca was defined in Step 4
scores = cross_val_score(knn, X_train_pca, y_train, cv=5)

# Define your model
knn = KNeighborsClassifier(n_neighbors=5)

# Run 5-Fold Cross Validation
# We use X_train_pca (your 10 components) and y_train
scores = cross_val_score(knn, X_train_pca, y_train, cv=5)

print(f"Scores for each fold: {scores}")
print(f"Average CV Accuracy: {scores.mean():.2f}")
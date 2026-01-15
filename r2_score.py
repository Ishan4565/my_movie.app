import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# -----------------------------
# 1. Create dataset (real-life like)
# -----------------------------
data = {
    "Distance_City": [10, 9, 8, 8, 7, 6, 6, 5, 4, 3],
    "Price": [2000, 2300, 2700, 3000, 3300, 3600, 3900, 4300, 4700, 5200]
}

df = pd.DataFrame(data)

# -----------------------------
# 2. Features and targe
# -----------------------------
X = df[["Distance_City"]]
y = df["Price"]

# -----------------------------
# 3. Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 4. Train model
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# 5. Predictions & R²
# -----------------------------
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

print("R² Score:", r2)

# -----------------------------
# 6. VISUALIZATION (IMPORTANT)
# -----------------------------
plt.scatter(y_test, y_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()])
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(f"Room Price Prediction (R² = {r2:.2f})")
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LinearRegression

# -------------------------
# 1. Create meaningful data
# -------------------------
np.random.seed(42)
n = 500

room_size = np.random.normal(1000, 200, n)          # sqft
distance = np.random.normal(5, 2, n)                # km from city
bedrooms = np.random.randint(1, 5, n)
age = np.random.randint(0, 30, n)
parking = np.random.randint(0, 2, n)

# Price with strong signal
price = (
    room_size * 300 +
    bedrooms * 20000 -
    distance * 15000 -
    age * 1000 +
    parking * 25000 +
    np.random.normal(0, 20000, n)  # small noise
)

df = pd.DataFrame({
    "room_size": room_size,
    "distance": distance,
    "bedrooms": bedrooms,
    "age": age,
    "parking": parking,
    "price": price
})

X = df.drop("price", axis=1)
y = df["price"]

# -------------------------
# 2. Learning curve
# -------------------------
model = LinearRegression()

train_sizes, train_scores, val_scores = learning_curve(
    model,
    X,
    y,
    cv=5,
    scoring="r2",
    train_sizes=np.linspace(0.1, 1.0, 10)
)

train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)

# -------------------------
# 3. Plot
# -------------------------
plt.plot(train_sizes, train_mean, label="Training R²")
plt.plot(train_sizes, val_mean, label="Validation R²")
plt.xlabel("Training Size")
plt.ylabel("R² Score")
plt.title("Learning Curve (Good Model)")
plt.legend()
plt.show()
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

param_grid = {
    "max_depth": [2, 3, 4, 5, 6, 8, 10],
    "min_samples_split": [2, 5, 10]
}

model = DecisionTreeRegressor(random_state=42)

grid = GridSearchCV(
    model,
    param_grid,
    cv=5,
    scoring="r2"
)

grid.fit(X, y)

print("Best Parameters:", grid.best_params_)
print("Best CV R²:", grid.best_score_)


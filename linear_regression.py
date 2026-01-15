import pandas as pd
df=pd.read_csv("cleaned_cafe_sales.csv")
print(df.head())
X = df[["Quantity", "Price_Per_Unit"]]
y = df["Total_Spent"]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)
new_data = [[3, 120]]  # Quantity = 3, Price = 120
prediction = model.predict(new_data)

print("Predicted Total Spent:", prediction[0])
import joblib

joblib.dump(model, "sales_model.pkl")

import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.DataFrame({
    "Hours": [2, 4, 6, 8],
    "Marks": [45, 55, 65, 78]
})

X = df[["Hours"]]
y = df["Marks"]

model = LinearRegression()
model.fit(X, y)

new_hours = pd.DataFrame({"Hours": [5]})
prediction = model.predict(new_hours)

print("Predicted Marks:", prediction[0])

import pandas as pd
from sklearn.linear_model import LinearRegression
df = pd.DataFrame({
    "Experience": [1, 2, 3, 4, 5],
    "Salary": [20000, 25000, 30000, 38000, 45000]
})

X = df[["Experience"]]
y = df["Salary"]

model.fit(X, y)
print(model.predict([[6]]))

import pandas as pd

df = pd.read_csv("cleaned_cafe_sales.csv")
corr = df.corr(numeric_only=True)
print(corr["Total_Spent"].sort_values(ascending=False))
selected_features = ["Quantity", "Price_Per_Unit"]
X = df[selected_features]
y = df["Total_Spent"]
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(random_state=42)
model.fit(X, y)
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(random_state=42)
model.fit(X, y)
print(df.dtypes)
numeric_cols = ["Quantity", "Price_Per_Unit", "Total_Spent"]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
corr = df[numeric_cols].corr()
print(corr)
print(corr["Total_Spent"].sort_values(ascending=False))
import matplotlib.pyplot as plt

plt.matshow(corr)
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Correlation Matrix")
plt.show()
corr = df[["Quantity", "Price_Per_Unit", "Total_Spent"]].corr()
print(corr)
import matplotlib.pyplot as plt

plt.matshow(corr)
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns, rotation=45)
plt.yticks(range(len(corr)), corr.columns)
plt.show()

X = df[["Quantity", "Price_Per_Unit"]]


from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
import numpy as np

for depth in range(1, 15):
    model = DecisionTreeRegressor(max_depth=depth)
    scores = cross_val_score(model, X, y, cv=5, scoring="r2")
    print(f"Depth {depth} → Avg R²: {scores.mean():.3f}")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# -------------------------
# Create dataset
# -------------------------
np.random.seed(42)
n = 500

room_size = np.random.normal(1000, 200, n)
distance = np.random.normal(5, 2, n)
bedrooms = np.random.randint(1, 5, n)
age = np.random.randint(0, 30, n)
parking = np.random.randint(0, 2, n)

price = (
    room_size * 300 +
    bedrooms**2 * 15000 +     # NON-LINEAR part
    np.exp(-distance) * 50000 +
    parking * 25000 -
    age * 1200 +
    np.random.normal(0, 15000, n)
)

X = pd.DataFrame({
    "room_size": room_size,
    "distance": distance,
    "bedrooms": bedrooms,
    "age": age,
    "parking": parking
})

y = price

# -------------------------
# Models
# -------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

# -------------------------
# Plot learning curves
# -------------------------
plt.figure(figsize=(10, 6))

for name, model in models.items():
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5, scoring="r2",
        train_sizes=np.linspace(0.1, 1.0, 10)
    )
    plt.plot(train_sizes, val_scores.mean(axis=1), label=name)

plt.xlabel("Training Size")
plt.ylabel("Validation R²")
plt.title("Learning Curves: Linear vs Random Forest")
plt.legend()
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# -------------------------
# Create dataset
# -------------------------
np.random.seed(42)
n = 500

room_size = np.random.normal(1000, 200, n)
distance = np.random.normal(5, 2, n)
bedrooms = np.random.randint(1, 5, n)
age = np.random.randint(0, 30, n)

price = (
    room_size * 300 +
    bedrooms**2 * 15000 +
    np.exp(-distance) * 50000 -
    age * 1000 +
    np.random.normal(0, 15000, n)
)

X = pd.DataFrame({
    "room_size": room_size,
    "distance": distance,
    "bedrooms": bedrooms,
    "age": age
})

y = price

# -------------------------
# Train-test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# Gradient Boosting Model
# -------------------------
model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

# -------------------------
# Evaluation
# -------------------------
preds = model.predict(X_test)
print("R² Score:", r2_score(y_test, preds))
residuals = y_test - preds

plt.scatter(preds, residuals)
plt.axhline(0)
plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.show()
y_log = np.log(y)


import pandas as pd

df = pd.read_csv("cleaned_cafe_sales.csv")
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
predictions = model.predict(X_test)
print(predictions[:5])
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("MAE:", mae)
print("RMSE:", rmse)
import matplotlib.pyplot as plt

plt.scatter(y_test, predictions)
plt.xlabel("Actual Total Spent")
plt.ylabel("Predicted Total Spent")
plt.title("Actual vs Predicted")
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
poly2 = PolynomialFeatures(degree=2)
X_train_p2 = poly2.fit_transform(X_train)
X_test_p2 = poly2.transform(X_test)

model_good = LinearRegression()
model_good.fit(X_train_p2, y_train)



import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso

X = np.random.rand(100, 5)
y = X @ np.array([5, 4, 3, 2, 1]) + np.random.randn(100)

alphas = [0.01, 0.1, 1, 10, 100]

ridge_coefs = []
lasso_coefs = []

for a in alphas:
    ridge = Ridge(alpha=a)
    ridge.fit(X, y)
    ridge_coefs.append(ridge.coef_)

    lasso = Lasso(alpha=a, max_iter=5000)
    lasso.fit(X, y)
    lasso_coefs.append(lasso.coef_)

plt.figure()
for i in range(5):
    plt.plot(alphas, [c[i] for c in ridge_coefs])
plt.xscale("log")
plt.title("Ridge Coefficients")
plt.show()

plt.figure()
for i in range(5):
    plt.plot(alphas, [c[i] for c in lasso_coefs])
plt.xscale("log")
plt.title("Lasso Coefficients")
plt.show()


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample data
data = {
    "Hours": [1,2,3,4,5,6,7,8],
    "Pass":  [0,0,0,0,1,1,1,1]
}

df = pd.DataFrame(data)

X = df[["Hours"]]
y = df["Pass"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

model = LogisticRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# Actual results
y_true = [1, 0, 1, 1, 0, 0, 1, 0]

# Model predictions
y_pred = [1, 0, 1, 0, 0, 1, 1, 0]

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)

# Metrics
print("Precision:", precision_score(y_true, y_pred))
print("Recall:", recall_score(y_true, y_pred))
print("F1 Score:", f1_score(y_true, y_pred))


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Fake dataset
np.random.seed(42)
X = np.random.rand(200, 1)
y = (X[:, 0] > 0.5).astype(int)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Probabilities
y_probs = model.predict_proba(X_test)[:, 1]

# ROC values
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
auc = roc_auc_score(y_test, y_probs)

# Plot ROC
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")  # random line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()



from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Example data
X = [
    [300, 1],  # size, location (1 = premium)
    [150, 0],
    [400, 1],
    [200, 0],
    [350, 1],
    [120, 0]
]

y = [1, 0, 1, 0, 1, 0]  # 1 = expensive, 0 = cheap

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

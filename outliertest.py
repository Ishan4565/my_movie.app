import pandas as pd

df = pd.read_csv("cleaned_cafe_sales.csv")
Q1 = df["Total_Spent"].quantile(0.25)
Q3 = df["Total_Spent"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df_no_outliers = df[
    (df["Total_Spent"] >= lower_bound) &
    (df["Total_Spent"] <= upper_bound)
]
print("Original size:", df.shape)
print("After outlier removal:", df_no_outliers.shape)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = df_no_outliers[["Quantity", "Price_Per_Unit"]]
y = df_no_outliers["Total_Spent"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)





import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score
X, y = make_classification(
    n_samples=500,
    n_features=5,
    n_informative=3,
    n_redundant=1,
    random_state=42
)

df = pd.DataFrame(X, columns=[f"Feature_{i}" for i in range(5)])
df["Target"] = y
X_train, X_test, y_train, y_test = train_test_split(
    df.drop("Target", axis=1),
    df["Target"],
    test_size=0.3,
    random_state=42
)
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)

print("AUC Score:", auc_score)
plt.figure()
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random Model")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve")
plt.legend()
plt.show()
y_probs = model.predict_proba(X_test)[:, 1]
threshold = 0.3
y_pred_custom = (y_probs >= threshold).astype(int)
from sklearn.metrics import precision_score, recall_score

print("Precision:", precision_score(y_test, y_pred_custom))
print("Recall:", recall_score(y_test, y_pred_custom))


import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Model
model = LogisticRegression(max_iter=500)

# 5-fold cross-validation
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

print("Scores for each fold:", scores)
print("Average accuracy:", scores.mean())

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1. Create a "Miscorrelated" Dataset
np.random.seed(42)
data_size = 100

# Actual features (noisy and hard to learn)
age = np.random.randint(20, 80, data_size)
cholesterol = np.random.randint(150, 300, data_size)

# The target: 1 = Sick, 0 = Healthy
target = np.random.choice([0, 1], size=data_size)

# THE LEAKY FEATURE: A Blood Test ID that encodes the answer
# If target is 1, ID is 1000s. If target is 0, ID is 5000s.
blood_test_id = [1000 + np.random.randint(1, 10) if t == 1 else 5000 + np.random.randint(1, 10) for t in target]

df = pd.DataFrame({
    'Age': age,
    'Cholesterol': cholesterol,
    'Blood_Test_ID': blood_test_id,
    'Target': target
})

# 2. Train the Model
X = df.drop('Target', axis=1)
y = df['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# 3. Check Accuracy
print(f"Model Accuracy: {model.score(X_test, y_test)}") # Likely 1.0
# 4. Visualize the "Cheating"
importances = model.feature_importances_
features = X.columns

plt.bar(features, importances, color=['blue', 'blue', 'red'])
plt.title("Feature Importance: Spotting the Cheater")
plt.ylabel("Importance Score")
plt.show()

df = df.drop("Blood_Test_ID", axis=1)
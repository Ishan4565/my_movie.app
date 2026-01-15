import pandas as pd

data = {
    "message": [
        "Win money now",
        "Hello how are you",
        "Free gift available",
        "Let us meet tomorrow",
        "Claim your free prize",
        "Are you coming today"
    ],
    "label": ["spam", "ham", "spam", "ham", "spam", "ham"]
}

df = pd.DataFrame(data)
print(df)
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["message"])
y = df["label"]
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(X, y)
test_messages = [
    "Free money now",
    "Are you free today",
    "Win a prize"
]

test_vectors = vectorizer.transform(test_messages)
predictions = model.predict(test_vectors)

for msg, pred in zip(test_messages, predictions):
    print(f"Message: '{msg}' → {pred}")


with open("bayes.txt", "r") as f:
    messages = f.readlines()

messages = [msg.strip() for msg in messages]

vectors = vectorizer.transform(messages)
predictions = model.predict(vectors)

for msg, pred in zip(messages, predictions):
    print(f"{msg} → {pred}")


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# 1. Create a simple dataset
# -----------------------------
data = {
    "Hours_Studied": [1,2,3,4,5,6,7,8,9,10],
    "Sleep_Hours":  [5,6,5,6,7,7,8,8,9,9],
    "Passed":       [0,0,0,0,0,1,1,1,1,1]
}

df = pd.DataFrame(data)

X = df[["Hours_Studied", "Sleep_Hours"]]
y = df["Passed"]

# -----------------------------
# 2. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -----------------------------
# 3. Overfitting model (BAD)
# -----------------------------
overfit_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

overfit_model.fit(X_train, y_train)

train_acc = accuracy_score(y_train, overfit_model.predict(X_train))
test_acc = accuracy_score(y_test, overfit_model.predict(X_test))

print("OVERFITTED MODEL")
print("Train Accuracy:", train_acc)
print("Test Accuracy :", test_acc)

# -----------------------------
# 4. Controlled model (GOOD)
# -----------------------------
good_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=3,
    min_samples_split=4,
    random_state=42
)

good_model.fit(X_train, y_train)

train_acc2 = accuracy_score(y_train, good_model.predict(X_train))
test_acc2 = accuracy_score(y_test, good_model.predict(X_test))

print("\nCONTROLLED MODEL")
print("Train Accuracy:", train_acc2)
print("Test Accuracy :", test_acc2)

# -----------------------------
# 5. Cross-validation (REAL ML CHECK)
# -----------------------------
cv_scores = cross_val_score(good_model, X, y, cv=5)

print("\nCross-validation scores:", cv_scores)
print("Average CV Accuracy:", cv_scores.mean())


from sklearn.metrics import roc_curve, roc_auc_score

y_prob = model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

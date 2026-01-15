import pandas as pd

# Create a DataFrame directly
data = {
    "Name": ["Amit", "Sita", "Ram", "Gita"],
    "Age": [20, 22, 19, 21],
    "Marks": [85, 90, 78, 88]
}

df = pd.DataFrame(data)

# Display the DataFrame
print("Full DataFrame:")
print(df)

# Access columns
print("\nNames column:")
print(df["Name"])

# Filter rows (example: Age > 20)
print("\nStudents with Age > 20:")
print(df[df["Age"] > 20])

# Add a new column
df["Passed"] = df["Marks"] >= 80
print("\nUpdated DataFrame with Passed column:")
print(df)


import pandas as pd
import json

# Step 1: JSON data as a string
json_data = '''
[
    {"Name": "Amit", "Age": 20, "Marks": 85},
    {"Name": "Sita", "Age": 22, "Marks": 90},
    {"Name": "Ram", "Age": 19, "Marks": 78},
    {"Name": "Gita", "Age": 21, "Marks": 88}
]
'''

# Step 2: Convert JSON string → Python object (list of dicts)
data = json.loads(json_data)

# Step 3: Create a DataFrame from JSON data
df = pd.DataFrame(data)

# Step 4: Use pandas operations
print("Full DataFrame:")
print(df)

# Filter rows: Marks >= 85
print("\nStudents with Marks >= 85:")
print(df[df["Marks"] >= 85])

# Add a new column: Passed if Marks >= 80
df["Passed"] = df["Marks"] >= 80
print("\nUpdated DataFrame with Passed column:")
print(df)


import numpy as np
import pandas as pd
import json

# ------------------------------
# Step 1: JSON data (like dataset)
# ------------------------------
json_data = '''
[
    {"Hours_Studied": 2, "Previous_Score": 70, "Marks": 75},
    {"Hours_Studied": 3, "Previous_Score": 80, "Marks": 85},
    {"Hours_Studied": 1, "Previous_Score": 60, "Marks": 65},
    {"Hours_Studied": 4, "Previous_Score": 90, "Marks": 95},
    {"Hours_Studied": 5, "Previous_Score": 85, "Marks": 90}
]
'''

# Convert JSON string to Python object
data = json.loads(json_data)

# Create a pandas DataFrame
df = pd.DataFrame(data)

# ------------------------------
# Step 2: Inspect Data
# ------------------------------
print("Dataset:")
print(df)

# ------------------------------
# Step 3: Prepare features (X) and target (y)
# ------------------------------
X = df[["Hours_Studied", "Previous_Score"]].values  # features
y = df["Marks"].values                              # target

# ------------------------------
# Step 4: Train a simple linear regression model manually (using NumPy)
# ------------------------------
# Formula: y = X * w + b
# For simplicity, we'll just calculate weights using pseudo-inverse
X_b = np.c_[np.ones((X.shape[0], 1)), X]  # add bias term
w = np.linalg.pinv(X_b).dot(y)            # solve linear regression

# ------------------------------
# Step 5: Predict for a new student
# ------------------------------
new_student = np.array([1, 3, 75])  # 1 = bias, 3 hours studied, previous score 75
predicted_marks = new_student.dot(w)

print("\nPredicted marks for student studying 3 hours with previous score 75:", predicted_marks)


import pandas as pd

# 1. The JSON 'Input' (Machine & Human Readable Text)
raw_json = '{"name": "Alice", "score": 95, "city": "NY"}'

# 2. The Pandas 'Library' (Converting to a structured table)
df = pd.read_json(raw_json, typ='series')

# 3. Preprocessing (Cleaning the data for ML)
df['score'] = df['score'] / 100  # Scaling the number between 0 and 1
print(df)


import pandas as pd
from pandas import json_normalize
import json

nested_json = '''
[
  {
    "name": "Sita",
    "age": 22,
    "marks": {"math": 90, "science": 85},
    "hobbies": ["reading", "painting"]
  },
  {
    "name": "Ram",
    "age": 19,
    "marks": {"math": 70, "science": 75},
    "hobbies": ["football"]
  }
]
'''

data = json.loads(nested_json)

# Flatten the nested JSON
df = json_normalize(data, sep="_")

print(df)


import pandas as pd

# Create DataFrame
data = {
    "Name": ["Amit", "Sita", "Ram", "Gita"],
    "Age": [20, 22, 19, 21],
    "Marks": [85, 90, 78, 88]
}
df = pd.DataFrame(data)

# Analyze data
print(df.info())             # first rows
print(df.describe())         # summary stats
print(df[df["Marks"] > 85])  # students with Marks > 85
print(df["Age"].mean())      # average age


import pandas as pd

# Step 1: Create DataFrame
data = {
    "Calories_Burned": [250, 300, 220, 400, 350],
    "Heart_Rate": [80, 90, 75, 95, 85],
    "Steps": [5000, 7000, 4500, 8000, 7500]
}

df = pd.DataFrame(data)

# Step 2: Preview the data
print("Full DataFrame:")
print(df)

# Step 3: Summary statistics
print("\nSummary statistics:")
print(df.describe())

# Step 4: Filter rows (e.g., high calorie burn)
print("\nDays with Calories > 300:")
print(df[df["Calories_Burned"] > 300])

# Step 5: Average values
print("\nAverage Heart Rate:", df["Heart_Rate"].mean())
print("Average Steps:", df["Steps"].mean())

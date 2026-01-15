import os
print("Running from:", os.getcwd())

import pandas as pd

# Create a simple table
data = {
    "Name": ["Amit", "Sita", "Ram"],
    "Age": [20, 22, 19],
    "Marks": [85, 90, 78]
}

df = pd.DataFrame(data)

# Display the DataFrame
print(df)

# Basic operations
print("\nAverage Age:", df["Age"].mean())
print("Maximum Marks:", df["Marks"].max())

import pandas as pd

df = pd.read_csv("data.csv")

print(df.head())

import pandas as pd

# 1️⃣ Create a DataFrame (like a table) from a dictionary
data = {
    "Name": ["Amit", "Sita", "Ram", "Gita"],
    "Age": [20, 22, 19, 21],
    "Marks": [85, 90, 78, 88]
}

df = pd.DataFrame(data)

# 2️⃣ Display the whole table
print("Full DataFrame:")
print(df)

# 3️⃣ Access a column
print("\nNames column:")
print(df["Name"])

# 4️⃣ Access multiple columns
print("\nName and Marks columns:")
print(df[["Name", "Marks"]])

# 5️⃣ Filter rows (example: Age > 20)
print("\nStudents with Age > 20:")
print(df[df["Age"] > 20])

# 6️⃣ Basic statistics
print("\nAverage Marks:", df["Marks"].mean())
print("Maximum Age:", df["Age"].max())

# 7️⃣ Add a new column
df["Passed"] = df["Marks"] >= 80
print("\nUpdated DataFrame with Passed column:")
print(df)




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

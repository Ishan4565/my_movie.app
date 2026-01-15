import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("cleaned_cafe_sales.csv")  # change name if needed
numeric_df = df.select_dtypes(include="number")
print(numeric_df.columns)
corr = numeric_df.corr()
print(corr)
plt.figure()
plt.imshow(corr)
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()
plt.figure()
plt.scatter(df["Quantity"], df["Total_Spent"])
plt.xlabel("Quantity")
plt.ylabel("Total Spent")
plt.title("Quantity vs Total Spent")
plt.show()


import matplotlib.pyplot as plt

plt.hist(df["Total_Spent"], bins=20)
plt.xlabel("Total Spent")
plt.ylabel("Frequency")
plt.title("Distribution of Total Spent")

plt.show()
import matplotlib.pyplot as plt

payment_counts = df["Payment_Method"].value_counts()

plt.pie(
    payment_counts,
    labels=payment_counts.index,
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Payment Method Distribution")
plt.axis("equal")  # makes the pie a perfect circle
plt.show()

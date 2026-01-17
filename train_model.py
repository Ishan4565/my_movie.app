import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1. Load Data
df = pd.read_csv('dirty_cafe_sales.csv')
df.columns = df.columns.str.strip()

# 2. FORCE NUMERIC (This fixes the TypeError)
# 'coerce' turns words like "UNKNOWN" into NaN (empty), so we can drop them
df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
df['Price Per Unit'] = pd.to_numeric(df['Price Per Unit'], errors='coerce')

# Drop rows where Quantity or Price is missing/invalid
df = df.dropna(subset=['Quantity', 'Price Per Unit'])

# 3. Create Date Features
df['Transaction Date'] = pd.to_datetime(df['Transaction Date'], errors='coerce')
df['Month'] = df['Transaction Date'].dt.month.fillna(1)
df['DayOfWeek'] = df['Transaction Date'].dt.dayofweek.fillna(0)
df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)

# 4. Encoding
encoders = {}
for col in ['Item', 'Payment Method', 'Location']:
    if col in df.columns:
        le = LabelEncoder()
        # Fill missing text with 'Other' before encoding
        df[col] = df[col].astype(str).replace('nan', 'Other')
        df[f'{col}_Encoded'] = le.fit_transform(df[col])
        encoders[col] = le

# 5. Define X and y
# Rename for consistency with the Streamlit app
df.rename(columns={'Payment Method_Encoded': 'Payment_Encoded'}, inplace=True)

feature_cols = ['Item_Encoded', 'Quantity', 'Price Per Unit', 'Payment_Encoded', 
                'Location_Encoded', 'Month', 'DayOfWeek', 'IsWeekend']

X = df[feature_cols]
# Now the math will work because everything is a float/int!
y = df['Quantity'] * df['Price Per Unit']

# 6. Scaling & Training
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# 7. Save Assets
joblib.dump(model, 'cafe_sales_model.pkl')
joblib.dump(scaler, 'cafe_sales_scaler.pkl')
joblib.dump(encoders, 'cafe_sales_encoders.pkl')

print("\n" + "="*40)
print("✅ SCIENCE, B! The data is cleaned and the model is saved.")
print(f"Total rows trained: {len(df)}")
print("="*40)
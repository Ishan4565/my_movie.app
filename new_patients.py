import pandas as pd
import joblib
import numpy as np

# 1. Load your tuned best model
model = joblib.load('cancer_model.pkl')

# 2. Load the dummy data
df = pd.read_csv('cancer_patients.csv.txt')

# 3. Make predictions
# The pipeline automatically scales and applies PCA!
predictions = model.predict(df)
probs = model.predict_proba(df)

# 4. Display results
for i in range(len(df)):
    diag = "Malignant" if predictions[i] == 0 else "Benign"
    conf = np.max(probs[i]) * 100
    print(f"Patient {i+1}: {diag} ({conf:.2f}% Confidence)")
    # 1. Prepare the results table
results_df = pd.DataFrame({
    'Patient_Number': range(1, len(predictions) + 1),
    'Predicted_Status': ['Malignant' if p == 0 else 'Benign' for p in predictions],
    'Confidence_Score': [f"{max(pr)*100:.2f}%" for pr in probs]
})

# 2. Add the original measurements so the doctor can verify
final_report = pd.concat([results_df, df], axis=1)

# 3. Export to CSV (readable by Excel)
final_report.to_csv('final_medical_report.csv', index=False)

print("--- Report Successfully Generated ---")
print(results_df.to_string(index=False))
# 1. Create a DataFrame for the results
results_summary = pd.DataFrame({
    'Diagnosis': ['Malignant' if p == 0 else 'Benign' for p in predictions],
    'Confidence': [f"{np.max(prob)*100:.2f}%" for prob in probs]
})

# 2. Combine the results with the original patient measurements
# This way, the Excel sheet has the ID, the Result, AND the Data
final_excel_report = pd.concat([results_summary, df], axis=1)

# 3. SAVE THE FILE
# We use index=False so Excel doesn't add an extra column of numbers
final_excel_report.to_csv('Cancer_Diagnosis_Report.csv', index=False)

print("File saved as 'Cancer_Diagnosis_Report.csv' in your folder!")
from datetime import datetime

# 1. Get the current date and time
# This creates a string like '2023-10-27_14-30'
now = datetime.now().strftime("%Y-%m-%d_%H-%M")

# 2. Create a unique filename
filename = f"Cancer_Report_{now}.csv"

# 3. Save it
final_excel_report.to_csv(filename, index=False)

print(f"--- Success! Report saved as: {filename} ---")
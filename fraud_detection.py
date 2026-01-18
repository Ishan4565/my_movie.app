"""
CREDIT CARD FRAUD DETECTION MODEL
Goal: Detect fraudulent transactions using imbalanced data techniques
Features: Transaction amount, time, location, merchant type, etc.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report, roc_auc_score)
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("CREDIT CARD FRAUD DETECTION MODEL")
print("="*80)

# ============================================================================
# STEP 1: CREATE IMBALANCED DATASET (Simulating Real Fraud Data)
# ============================================================================
print("\n📊 STEP 1: Creating Imbalanced Fraud Dataset")
print("-"*80)

np.random.seed(42)

n_normal = 9900
n_fraud = 100

print(f"Creating dataset with severe imbalance:")
print(f"  Normal transactions: {n_normal} (99%)")
print(f"  Fraud transactions: {n_fraud} (1%)")

def generate_transactions(n_samples, is_fraud=False):
    data = []
    
    for _ in range(n_samples):
        if is_fraud:
            amount = np.random.uniform(500, 5000)
            hour = np.random.choice([0, 1, 2, 3, 23])
            location = np.random.choice(['Foreign', 'Online', 'ATM'], p=[0.5, 0.3, 0.2])
            merchant = np.random.choice(['Electronics', 'Jewelry', 'Online Store'], p=[0.4, 0.4, 0.2])
            card_present = np.random.choice([0, 1], p=[0.7, 0.3])
            num_transactions_today = np.random.randint(5, 20)
        else:
            amount = np.random.uniform(10, 500)
            hour = np.random.randint(6, 23)
            location = np.random.choice(['Local', 'Online', 'Store'], p=[0.5, 0.3, 0.2])
            merchant = np.random.choice(['Grocery', 'Gas Station', 'Restaurant', 'Retail'], p=[0.3, 0.3, 0.2, 0.2])
            card_present = np.random.choice([0, 1], p=[0.3, 0.7])
            num_transactions_today = np.random.randint(1, 5)
        
        data.append({
            'Transaction_Amount': round(amount, 2),
            'Hour_of_Day': hour,
            'Location_Type': location,
            'Merchant_Category': merchant,
            'Card_Present': card_present,
            'Num_Transactions_Today': num_transactions_today,
            'Is_Weekend': np.random.choice([0, 1], p=[0.7, 0.3]),
            'Is_Fraud': 1 if is_fraud else 0
        })
    
    return data

normal_data = generate_transactions(n_normal, is_fraud=False)
fraud_data = generate_transactions(n_fraud, is_fraud=True)

all_data = normal_data + fraud_data
df = pd.DataFrame(all_data)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n✓ Dataset created with {len(df)} transactions")
print(f"\nClass distribution:")
print(df['Is_Fraud'].value_counts())
print(f"\nImbalance ratio: {n_normal/n_fraud:.1f}:1 (Normal:Fraud)")

print(f"\nSample data:")
print(df.head(10))

print(f"\nFraud transactions preview:")
print(df[df['Is_Fraud'] == 1].head())

# ============================================================================
# STEP 2: FEATURE ENGINEERING
# ============================================================================
print("\n" + "="*80)
print("🔧 STEP 2: Feature Engineering")
print("-"*80)

df_processed = df.copy()

le_location = LabelEncoder()
le_merchant = LabelEncoder()

df_processed['Location_Encoded'] = le_location.fit_transform(df_processed['Location_Type'])
df_processed['Merchant_Encoded'] = le_merchant.fit_transform(df_processed['Merchant_Category'])

df_processed['Amount_Per_Transaction'] = df_processed['Transaction_Amount'] / (df_processed['Num_Transactions_Today'] + 1)
df_processed['Is_Night'] = (df_processed['Hour_of_Day'] >= 22) | (df_processed['Hour_of_Day'] <= 4)
df_processed['Is_Night'] = df_processed['Is_Night'].astype(int)

feature_columns = [
    'Transaction_Amount',
    'Hour_of_Day',
    'Location_Encoded',
    'Merchant_Encoded',
    'Card_Present',
    'Num_Transactions_Today',
    'Is_Weekend',
    'Amount_Per_Transaction',
    'Is_Night'
]

print(f"✓ Created {len(feature_columns)} features:")
for i, feat in enumerate(feature_columns, 1):
    print(f"  {i}. {feat}")

X = df_processed[feature_columns]
y = df_processed['Is_Fraud']

print(f"\n✓ Features shape: {X.shape}")
print(f"✓ Target shape: {y.shape}")

# ============================================================================
# STEP 3: TRAIN-TEST SPLIT (Before SMOTE!)
# ============================================================================
print("\n" + "="*80)
print("📦 STEP 3: Train-Test Split")
print("-"*80)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {len(X_train)} samples")
print(f"  Normal: {sum(y_train == 0)} ({sum(y_train == 0)/len(y_train)*100:.1f}%)")
print(f"  Fraud: {sum(y_train == 1)} ({sum(y_train == 1)/len(y_train)*100:.1f}%)")

print(f"\nTesting set: {len(X_test)} samples")
print(f"  Normal: {sum(y_test == 0)} ({sum(y_test == 0)/len(y_test)*100:.1f}%)")
print(f"  Fraud: {sum(y_test == 1)} ({sum(y_test == 1)/len(y_test)*100:.1f}%)")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n✓ Features scaled using StandardScaler")

# ============================================================================
# STEP 4: APPLY SMOTE
# ============================================================================
print("\n" + "="*80)
print("⚖️ STEP 4: Applying SMOTE (Synthetic Minority Over-sampling)")
print("-"*80)

print(f"\nBEFORE SMOTE:")
print(f"  Normal transactions: {sum(y_train == 0)}")
print(f"  Fraud transactions: {sum(y_train == 1)}")
print(f"  Imbalance ratio: {sum(y_train == 0)/sum(y_train == 1):.1f}:1")

smote = SMOTE(random_state=42, k_neighbors=3)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

print(f"\nAFTER SMOTE:")
print(f"  Normal transactions: {sum(y_train_balanced == 0)}")
print(f"  Fraud transactions: {sum(y_train_balanced == 1)}")
print(f"  Imbalance ratio: {sum(y_train_balanced == 0)/sum(y_train_balanced == 1):.1f}:1")

print(f"\n✓ Created {sum(y_train_balanced == 1) - sum(y_train == 1)} synthetic fraud examples!")

# ============================================================================
# STEP 5: TRAIN MODELS (With and Without SMOTE)
# ============================================================================
print("\n" + "="*80)
print("🤖 STEP 5: Training Models")
print("="*80)

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=15),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5)
}

results = {}

print("\n🔴 Training WITHOUT SMOTE (Imbalanced Data):")
print("-"*80)

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    if name == 'Logistic Regression':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    results[f"{name} (No SMOTE)"] = {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'predictions': y_pred
    }
    
    print(f"  Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

print("\n" + "="*80)
print("\n🟢 Training WITH SMOTE (Balanced Data):")
print("-"*80)

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    model_smote = type(model)(**model.get_params())
    
    if name == 'Logistic Regression':
        model_smote.fit(X_train_balanced, y_train_balanced)
        y_pred = model_smote.predict(X_test_scaled)
        y_pred_proba = model_smote.predict_proba(X_test_scaled)[:, 1]
    else:
        X_train_balanced_unscaled = scaler.inverse_transform(X_train_balanced)
        model_smote.fit(X_train_balanced_unscaled, y_train_balanced)
        y_pred = model_smote.predict(X_test)
        y_pred_proba = model_smote.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    results[f"{name} (SMOTE)"] = {
        'model': model_smote,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'predictions': y_pred
    }
    
    print(f"  Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

# ============================================================================
# STEP 6: MODEL COMPARISON
# ============================================================================
print("\n" + "="*80)
print("📊 STEP 6: Model Comparison")
print("="*80)

comparison_data = []
for name, metrics in results.items():
    comparison_data.append({
        'Model': name,
        'Accuracy': f"{metrics['accuracy']:.4f}",
        'Precision': f"{metrics['precision']:.4f}",
        'Recall': f"{metrics['recall']:.4f}",
        'F1-Score': f"{metrics['f1']:.4f}",
        'ROC-AUC': f"{metrics['roc_auc']:.4f}"
    })

comparison_df = pd.DataFrame(comparison_data)
print("\n" + comparison_df.to_string(index=False))

best_model_name = max(
    [k for k in results.keys() if 'SMOTE' in k], 
    key=lambda x: results[x]['recall']
)
best_model = results[best_model_name]['model']
best_metrics = results[best_model_name]

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   Recall (Fraud Detection Rate): {best_metrics['recall']:.2%}")
print(f"   Precision (Accuracy when predicting fraud): {best_metrics['precision']:.2%}")
print(f"   F1-Score: {best_metrics['f1']:.4f}")

# ============================================================================
# STEP 7: CONFUSION MATRIX
# ============================================================================
print("\n" + "="*80)
print("🎯 STEP 7: Confusion Matrix (Best Model)")
print("-"*80)

cm = confusion_matrix(y_test, best_metrics['predictions'])

print(f"\nConfusion Matrix:")
print(f"                  Predicted")
print(f"                Normal  Fraud")
print(f"Actual Normal    {cm[0][0]:5d}  {cm[0][1]:5d}")
print(f"       Fraud     {cm[1][0]:5d}  {cm[1][1]:5d}")

tn, fp, fn, tp = cm.ravel()

print(f"\nBreakdown:")
print(f"  True Negatives (Correct Normal): {tn}")
print(f"  False Positives (Normal flagged as Fraud): {fp}")
print(f"  False Negatives (Missed Fraud): {fn} ⚠️")
print(f"  True Positives (Caught Fraud): {tp} ✓")

fraud_caught_pct = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0
print(f"\n✓ Fraud Detection Rate: {fraud_caught_pct:.1f}% of frauds caught!")

# ============================================================================
# STEP 8: FEATURE IMPORTANCE
# ============================================================================
if 'Random Forest' in best_model_name or 'Gradient Boosting' in best_model_name:
    print("\n" + "="*80)
    print("🔍 STEP 8: Feature Importance")
    print("-"*80)
    
    importances = best_model.feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    print("\nMost Important Features for Fraud Detection:")
    for idx, row in feature_importance.iterrows():
        bar = '█' * int(row['Importance'] * 100)
        print(f"  {row['Feature']:<30} {row['Importance']:.4f} {bar}")

# ============================================================================
# STEP 9: SAVE MODEL
# ============================================================================
print("\n" + "="*80)
print("💾 STEP 9: Saving Model and Components")
print("-"*80)

joblib.dump(best_model, 'fraud_detection_model.pkl')
print("✓ Saved: fraud_detection_model.pkl")

joblib.dump(scaler, 'fraud_detection_scaler.pkl')
print("✓ Saved: fraud_detection_scaler.pkl")

encoders = {
    'location': le_location,
    'merchant': le_merchant
}
joblib.dump(encoders, 'fraud_detection_encoders.pkl')
print("✓ Saved: fraud_detection_encoders.pkl")

metadata = {
    'model_name': best_model_name,
    'feature_columns': feature_columns,
    'metrics': {
        'accuracy': best_metrics['accuracy'],
        'precision': best_metrics['precision'],
        'recall': best_metrics['recall'],
        'f1': best_metrics['f1'],
        'roc_auc': best_metrics['roc_auc']
    }
}
joblib.dump(metadata, 'fraud_detection_metadata.pkl')
print("✓ Saved: fraud_detection_metadata.pkl")

df.to_csv('fraud_transactions_dataset.csv', index=False)
print("✓ Saved: fraud_transactions_dataset.csv")

# ============================================================================
# STEP 10: TEST PREDICTION
# ============================================================================
print("\n" + "="*80)
print("🔮 STEP 10: Example Predictions")
print("="*80)

print("\n🟢 Example 1: Normal Transaction")
normal_example = {
    'Transaction_Amount': 45.50,
    'Hour_of_Day': 14,
    'Location_Encoded': le_location.transform(['Local'])[0],
    'Merchant_Encoded': le_merchant.transform(['Grocery'])[0],
    'Card_Present': 1,
    'Num_Transactions_Today': 2,
    'Is_Weekend': 0,
    'Amount_Per_Transaction': 45.50 / 3,
    'Is_Night': 0
}

normal_df = pd.DataFrame([normal_example])

if 'Logistic' in best_model_name:
    normal_scaled = scaler.transform(normal_df)
    normal_pred = best_model.predict(normal_scaled)[0]
    normal_proba = best_model.predict_proba(normal_scaled)[0]
else:
    normal_pred = best_model.predict(normal_df)[0]
    normal_proba = best_model.predict_proba(normal_df)[0]

print(f"Transaction: $45.50 at Grocery store, 2pm, Local, Card Present")
print(f"Prediction: {'🚨 FRAUD' if normal_pred == 1 else '✅ NORMAL'}")
print(f"Fraud Probability: {normal_proba[1]:.2%}")

print("\n🔴 Example 2: Suspicious Transaction")
fraud_example = {
    'Transaction_Amount': 2500.00,
    'Hour_of_Day': 2,
    'Location_Encoded': le_location.transform(['Foreign'])[0],
    'Merchant_Encoded': le_merchant.transform(['Electronics'])[0],
    'Card_Present': 0,
    'Num_Transactions_Today': 15,
    'Is_Weekend': 0,
    'Amount_Per_Transaction': 2500.00 / 16,
    'Is_Night': 1
}

fraud_df = pd.DataFrame([fraud_example])

if 'Logistic' in best_model_name:
    fraud_scaled = scaler.transform(fraud_df)
    fraud_pred = best_model.predict(fraud_scaled)[0]
    fraud_proba = best_model.predict_proba(fraud_scaled)[0]
else:
    fraud_pred = best_model.predict(fraud_df)[0]
    fraud_proba = best_model.predict_proba(fraud_df)[0]

print(f"Transaction: $2,500 at Electronics, 2am, Foreign, Card Not Present")
print(f"Prediction: {'🚨 FRAUD' if fraud_pred == 1 else '✅ NORMAL'}")
print(f"Fraud Probability: {fraud_proba[1]:.2%}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("✅ FRAUD DETECTION MODEL COMPLETE!")
print("="*80)

print(f"""
📊 Summary:
  • Dataset: {len(df)} transactions ({n_fraud} fraud, {n_normal} normal)
  • Imbalance: {n_normal/n_fraud:.1f}:1 ratio
  • SMOTE: Created {sum(y_train_balanced == 1) - sum(y_train == 1)} synthetic fraud examples
  • Best Model: {best_model_name}
  • Fraud Detection Rate: {best_metrics['recall']:.1%}
  • False Positive Rate: {fp/(fp+tn):.2%}
  
📁 Saved Files:
  • fraud_detection_model.pkl
  • fraud_detection_scaler.pkl
  • fraud_detection_encoders.pkl
  • fraud_detection_metadata.pkl
  • fraud_transactions_dataset.csv

🎯 Key Metrics:
  • Recall (Catch fraud): {best_metrics['recall']:.2%} ← Most Important!
  • Precision (Accuracy): {best_metrics['precision']:.2%}
  • F1-Score: {best_metrics['f1']:.4f}

💡 Why SMOTE Helped:
  Without SMOTE: Model ignores fraud (too rare)
  With SMOTE: Model learns fraud patterns (balanced data)
""")

print("="*80)
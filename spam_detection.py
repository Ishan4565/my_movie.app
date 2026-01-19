"""
EMAIL/SMS SPAM DETECTION MODEL
Goal: Classify messages as SPAM or HAM (not spam)
Algorithm: Naive Bayes (perfect for text classification!)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("SPAM DETECTION MODEL - NAIVE BAYES")
print("="*80)

# ============================================================================
# STEP 1: DOWNLOAD NLTK DATA
# ============================================================================
print("\n📥 STEP 1: Downloading NLTK Data")
print("-"*80)

try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    print("✓ NLTK data downloaded successfully")
except:
    print("⚠️  NLTK download failed (may already exist)")

# ============================================================================
# STEP 2: CREATE SPAM DATASET
# ============================================================================
print("\n" + "="*80)
print("📊 STEP 2: Creating Spam/Ham Dataset")
print("-"*80)

spam_messages = [
    "WINNER!! You have won a $1000 cash prize! Call now to claim!",
    "Congratulations! You've been selected for a FREE vacation to Bahamas!",
    "URGENT! Your account will be closed. Click here immediately!",
    "Get rich quick! Make $5000 per week working from home!",
    "Claim your FREE iPhone now! Limited offer! Click here!",
    "You have won the lottery! Send your bank details to claim!",
    "Hot singles in your area! Click to meet them now!",
    "Lose 30 pounds in 30 days! Buy our miracle pills now!",
    "CONGRATULATIONS!!! You are our lucky winner! Call 1-800-WINNER!",
    "Free money! Click here to get $1000 deposited to your account!",
    "Your package could not be delivered. Click to reschedule.",
    "FINAL NOTICE: Your payment is overdue. Pay now to avoid penalty!",
    "Get a loan approved in 5 minutes! Bad credit OK! Apply now!",
    "You qualify for a $250,000 loan! No credit check required!",
    "Exclusive offer just for you! Buy Viagra at 90% discount!",
    "Work from home and earn $10,000/month! No experience needed!",
    "BREAKING: Secret method to make money online revealed!",
    "Your email has won! Claim your prize of $500,000!",
    "Increase your income! Join our program and get rich fast!",
    "Limited time offer! Get 70% off on all products! Shop now!"
]

ham_messages = [
    "Hey, are we still meeting for coffee tomorrow afternoon?",
    "Can you send me the project report by end of day?",
    "Thanks for your help yesterday. Really appreciate it!",
    "Meeting has been rescheduled to 3 PM. See you then.",
    "Happy birthday! Hope you have a wonderful day!",
    "Could you please review the attached document?",
    "Dinner at 7? Let me know if that works for you.",
    "Great presentation today! Well done!",
    "The package will be delivered tomorrow between 2-5 PM.",
    "Reminder: Your appointment is scheduled for next Monday.",
    "Can we discuss the budget in tomorrow's meeting?",
    "Thanks for the information. I'll get back to you soon.",
    "Your order has been confirmed. Tracking number: ABC123",
    "Please find the attached invoice for last month.",
    "Looking forward to seeing you at the conference!",
    "Your subscription renewal is due next week.",
    "The weather looks great for the weekend trip!",
    "Can you pick up some groceries on your way home?",
    "Your flight is confirmed for departure at 6:30 AM.",
    "Thank you for your patience. Issue has been resolved."
]

# Create dataset
data = []
for msg in spam_messages:
    data.append({'message': msg, 'label': 'spam'})
for msg in ham_messages:
    data.append({'message': msg, 'label': 'ham'})

df = pd.DataFrame(data)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"✓ Dataset created with {len(df)} messages")
print(f"\nClass distribution:")
print(df['label'].value_counts())

print(f"\nSample spam messages:")
print(df[df['label'] == 'spam'].head(3)['message'].values)

print(f"\nSample ham messages:")
print(df[df['label'] == 'ham'].head(3)['message'].values)

# ============================================================================
# STEP 3: TEXT PREPROCESSING
# ============================================================================
print("\n" + "="*80)
print("🧹 STEP 3: Text Preprocessing")
print("-"*80)

def preprocess_text(text):
    """
    Clean and preprocess text for ML
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove stopwords (common words like 'the', 'is', 'and')
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    
    # Stemming (reduce words to root form: running → run)
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    
    return ' '.join(words)

print("Example preprocessing:")
original = "WINNER!! You've won $1000! Click HERE now!!!"
processed = preprocess_text(original)
print(f"\nOriginal:  {original}")
print(f"Processed: {processed}")

# Apply preprocessing to all messages
df['processed_message'] = df['message'].apply(preprocess_text)

print(f"\n✓ Preprocessing complete!")
print(f"\nBefore: {df['message'].iloc[0]}")
print(f"After:  {df['processed_message'].iloc[0]}")

# ============================================================================
# STEP 4: FEATURE EXTRACTION (TF-IDF)
# ============================================================================
print("\n" + "="*80)
print("🔧 STEP 4: Feature Extraction with TF-IDF")
print("-"*80)

# TF-IDF Vectorizer
tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
X = tfidf.fit_transform(df['processed_message'])
y = df['label'].map({'spam': 1, 'ham': 0})

print(f"✓ TF-IDF matrix shape: {X.shape}")
print(f"  {X.shape[0]} messages × {X.shape[1]} features")

print(f"\n✓ Top 10 most important words:")
feature_names = tfidf.get_feature_names_out()
print(feature_names[:10])

# ============================================================================
# STEP 5: TRAIN-TEST SPLIT
# ============================================================================
print("\n" + "="*80)
print("📦 STEP 5: Train-Test Split")
print("-"*80)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} messages")
print(f"  Spam: {sum(y_train == 1)}")
print(f"  Ham: {sum(y_train == 0)}")

print(f"\nTesting set: {X_test.shape[0]} messages")
print(f"  Spam: {sum(y_test == 1)}")
print(f"  Ham: {sum(y_test == 0)}")

# ============================================================================
# STEP 6: TRAIN MODELS (Multiple types of Naive Bayes!)
# ============================================================================
print("\n" + "="*80)
print("🤖 STEP 6: Training Models")
print("="*80)

models = {
    'Multinomial Naive Bayes': MultinomialNB(),
    'Bernoulli Naive Bayes': BernoulliNB(),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}

print("\nTraining and evaluating models...")
print("-"*80)

for name, model in models.items():
    print(f"\n🔄 Training {name}...")
    
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': y_pred
    }
    
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")

# ============================================================================
# STEP 7: MODEL COMPARISON
# ============================================================================
print("\n" + "="*80)
print("📊 STEP 7: Model Comparison")
print("="*80)

comparison_data = []
for name, metrics in results.items():
    comparison_data.append({
        'Model': name,
        'Accuracy': f"{metrics['accuracy']:.4f}",
        'Precision': f"{metrics['precision']:.4f}",
        'Recall': f"{metrics['recall']:.4f}",
        'F1-Score': f"{metrics['f1']:.4f}"
    })

comparison_df = pd.DataFrame(comparison_data)
print("\n" + comparison_df.to_string(index=False))

# Select best model
best_model_name = max(results, key=lambda x: results[x]['f1'])
best_model = results[best_model_name]['model']
best_metrics = results[best_model_name]

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   Accuracy: {best_metrics['accuracy']:.2%}")
print(f"   F1-Score: {best_metrics['f1']:.4f}")

# ============================================================================
# STEP 8: CONFUSION MATRIX
# ============================================================================
print("\n" + "="*80)
print("🎯 STEP 8: Confusion Matrix (Best Model)")
print("-"*80)

cm = confusion_matrix(y_test, best_metrics['predictions'])

print(f"\nConfusion Matrix:")
print(f"                  Predicted")
print(f"                Ham    Spam")
print(f"Actual Ham     {cm[0][0]:4d}   {cm[0][1]:4d}")
print(f"       Spam    {cm[1][0]:4d}   {cm[1][1]:4d}")

tn, fp, fn, tp = cm.ravel()

print(f"\nBreakdown:")
print(f"  True Negatives (Correct Ham): {tn}")
print(f"  False Positives (Ham marked as Spam): {fp} ⚠️")
print(f"  False Negatives (Missed Spam): {fn} ⚠️")
print(f"  True Positives (Caught Spam): {tp} ✓")

spam_caught_pct = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0
print(f"\n✓ Spam Detection Rate: {spam_caught_pct:.1f}%")

# ============================================================================
# STEP 9: SAVE MODEL
# ============================================================================
print("\n" + "="*80)
print("💾 STEP 9: Saving Model and Vectorizer")
print("-"*80)

joblib.dump(best_model, 'spam_detector_model.pkl')
print("✓ Saved: spam_detector_model.pkl")

joblib.dump(tfidf, 'spam_detector_vectorizer.pkl')
print("✓ Saved: spam_detector_vectorizer.pkl")

metadata = {
    'model_name': best_model_name,
    'metrics': {
        'accuracy': best_metrics['accuracy'],
        'precision': best_metrics['precision'],
        'recall': best_metrics['recall'],
        'f1': best_metrics['f1']
    }
}
joblib.dump(metadata, 'spam_detector_metadata.pkl')
print("✓ Saved: spam_detector_metadata.pkl")

df.to_csv('spam_dataset.csv', index=False)
print("✓ Saved: spam_dataset.csv")

# ============================================================================
# STEP 10: TEST WITH NEW MESSAGES
# ============================================================================
print("\n" + "="*80)
print("🔮 STEP 10: Testing with New Messages")
print("="*80)

test_messages = [
    "Congratulations! You won $1 million! Click here now!",
    "Hi, can we meet tomorrow at 3pm for the project discussion?",
    "URGENT: Your account has been compromised. Reset password immediately!",
    "Thanks for the dinner last night. Had a great time!",
    "Get 90% discount on all products! Limited time offer!"
]

print("\nPredicting new messages:")
print("-"*80)

for msg in test_messages:
    # Preprocess
    processed = preprocess_text(msg)
    
    # Vectorize
    vectorized = tfidf.transform([processed])
    
    # Predict
    prediction = best_model.predict(vectorized)[0]
    proba = best_model.predict_proba(vectorized)[0]
    
    label = "🚨 SPAM" if prediction == 1 else "✅ HAM"
    confidence = proba[prediction] * 100
    
    print(f"\nMessage: {msg[:60]}...")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.1f}%")

# ============================================================================
# STEP 11: INTERACTIVE PREDICTION
# ============================================================================
print("\n" + "="*80)
print("💬 STEP 11: Interactive Spam Detection")
print("="*80)

def predict_spam(message):
    """Predict if a message is spam"""
    processed = preprocess_text(message)
    vectorized = tfidf.transform([processed])
    prediction = best_model.predict(vectorized)[0]
    proba = best_model.predict_proba(vectorized)[0]
    
    return {
        'label': 'SPAM' if prediction == 1 else 'HAM',
        'confidence': proba[prediction] * 100,
        'spam_probability': proba[1] * 100,
        'ham_probability': proba[0] * 100
    }

# Test interactive prediction
print("\nTry your own message!")
user_input = input("\nEnter a message to check (or press Enter to skip): ").strip()

if user_input:
    result = predict_spam(user_input)
    
    print("\n" + "="*80)
    print("📊 PREDICTION RESULT")
    print("="*80)
    print(f"\nYour message: {user_input}")
    print(f"\nPrediction: {result['label']}")
    print(f"Confidence: {result['confidence']:.1f}%")
    print(f"\nDetailed probabilities:")
    print(f"  Spam probability: {result['spam_probability']:.1f}%")
    print(f"  Ham probability: {result['ham_probability']:.1f}%")
else:
    print("\nSkipped interactive mode.")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("✅ SPAM DETECTION MODEL COMPLETE!")
print("="*80)

print(f"""
📊 Summary:
  • Dataset: {len(df)} messages ({sum(df['label']=='spam')} spam, {sum(df['label']=='ham')} ham)
  • Best Model: {best_model_name}
  • Accuracy: {best_metrics['accuracy']:.1%}
  • Spam Detection Rate: {best_metrics['recall']:.1%}
  
📁 Saved Files:
  • spam_detector_model.pkl (trained model)
  • spam_detector_vectorizer.pkl (TF-IDF vectorizer)
  • spam_detector_metadata.pkl (model info)
  • spam_dataset.csv (training data)

🎯 What Makes Naive Bayes Perfect for Spam:
  • Fast predictions (real-time spam filtering)
  • Works great with text (word probabilities)
  • Simple and interpretable
  • Industry standard for spam detection

📧 Use Cases:
  • Email spam filtering
  • SMS spam detection
  • Comment spam detection
  • Social media spam filtering
  
💡 How It Works:
  1. User enters message
  2. Text is cleaned and preprocessed
  3. Converted to TF-IDF features
  4. Naive Bayes calculates probability
  5. Classified as SPAM or HAM
""")

print("="*80)
"""
SPAM DETECTION SYSTEM - STREAMLIT APP
Real-time spam/ham classification using Naive Bayes
"""

import streamlit as st
import pandas as pd
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

st.set_page_config(page_title="Spam Detector", page_icon="📧", layout="wide")

# Download NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        return True
    except:
        return False

download_nltk_data()

# Load model
@st.cache_resource
def load_spam_model():
    try:
        model = joblib.load('spam_detector_model.pkl')
        vectorizer = joblib.load('spam_detector_vectorizer.pkl')
        metadata = joblib.load('spam_detector_metadata.pkl')
        return model, vectorizer, metadata, True
    except FileNotFoundError:
        return None, None, None, False

model, vectorizer, metadata, model_loaded = load_spam_model()

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    
    return ' '.join(words)

# Prediction function
def predict_message(message):
    processed = preprocess_text(message)
    vectorized = vectorizer.transform([processed])
    prediction = model.predict(vectorized)[0]
    proba = model.predict_proba(vectorized)[0]
    
    return {
        'label': 'SPAM' if prediction == 1 else 'HAM',
        'is_spam': prediction == 1,
        'confidence': proba[prediction] * 100,
        'spam_prob': proba[1] * 100,
        'ham_prob': proba[0] * 100
    }

# ============================================================================
# HEADER
# ============================================================================

st.title("📧 Spam Detection System")
st.markdown("### Classify messages as Spam or Ham using Machine Learning")

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.header("📊 System Status")
    
    if model_loaded:
        st.success("✅ Model Active")
        
        st.markdown("---")
        st.header("🎯 Model Info")
        st.metric("Algorithm", metadata['model_name'])
        st.metric("Accuracy", f"{metadata['metrics']['accuracy']:.1%}")
        st.metric("Precision", f"{metadata['metrics']['precision']:.1%}")
        
        st.markdown("---")
        st.header("🚨 Classification")
        st.markdown("""
        - **SPAM:** Unsolicited/unwanted messages
          - Advertisements
          - Phishing attempts
          - Scams
          - Promotional offers
        
        - **HAM:** Legitimate messages
          - Personal communication
          - Work emails
          - Genuine notifications
        """)
        
        st.markdown("---")
        st.header("🧠 How It Works")
        st.markdown("""
        1. **Text Preprocessing**
           - Lowercase conversion
           - Remove special characters
           - Remove stopwords
           - Word stemming
        
        2. **Feature Extraction**
           - TF-IDF vectorization
           - Convert text to numbers
        
        3. **Classification**
           - Naive Bayes algorithm
           - Probability calculation
           - Spam/Ham prediction
        """)
    else:
        st.error("❌ Model Not Found")
        st.warning("Run training script first")

# ============================================================================
# MAIN APP
# ============================================================================

if not model_loaded:
    st.error("⚠️ Model files not found!")
    st.code("python spam_detection_training.py")
    st.stop()

tab1, tab2, tab3 = st.tabs(["🔍 Single Message", "📦 Batch Analysis", "📊 Examples"])

# ============================================================================
# TAB 1: SINGLE MESSAGE
# ============================================================================

with tab1:
    st.subheader("Check Single Message")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Enter Message")
        
        message_input = st.text_area(
            "Type or paste your message here:",
            height=200,
            placeholder="Example: Congratulations! You won $1000! Click here to claim your prize!"
        )
        
        analyze_button = st.button("🔍 Analyze Message", type="primary", use_container_width=True)
        
        # Quick test buttons
        st.markdown("#### 🎯 Quick Tests")
        col_a, col_b = st.columns(2)
        
        with col_a:
            if st.button("Test Spam Example", use_container_width=True):
                message_input = "WINNER!! You have won $1000 cash prize! Call now to claim your reward!"
                analyze_button = True
        
        with col_b:
            if st.button("Test Ham Example", use_container_width=True):
                message_input = "Hi, can we meet tomorrow at 3pm to discuss the project?"
                analyze_button = True
    
    with col2:
        st.markdown("#### Analysis Result")
        
        if analyze_button and message_input:
            try:
                result = predict_message(message_input)
                
                # Big result display
                if result['is_spam']:
                    st.error("🚨 SPAM DETECTED!")
                    st.markdown("### ⚠️ This message appears to be spam")
                else:
                    st.success("✅ LEGITIMATE MESSAGE")
                    st.markdown("### ✓ This message appears to be ham")
                
                st.markdown("---")
                
                # Confidence meter
                st.metric("Confidence", f"{result['confidence']:.1f}%")
                st.progress(result['confidence'] / 100)
                
                st.markdown("---")
                
                # Detailed probabilities
                st.markdown("#### 📊 Probability Breakdown")
                
                prob_col1, prob_col2 = st.columns(2)
                
                with prob_col1:
                    st.metric(
                        "Spam Probability",
                        f"{result['spam_prob']:.1f}%",
                        delta=None
                    )
                
                with prob_col2:
                    st.metric(
                        "Ham Probability",
                        f"{result['ham_prob']:.1f}%",
                        delta=None
                    )
                
                # Visual bars
                st.markdown("**Visual Comparison:**")
                spam_bar = "🟥" * int(result['spam_prob'] / 5)
                ham_bar = "🟩" * int(result['ham_prob'] / 5)
                
                st.markdown(f"Spam: {spam_bar} {result['spam_prob']:.1f}%")
                st.markdown(f"Ham:  {ham_bar} {result['ham_prob']:.1f}%")
                
                st.markdown("---")
                
                # Message preview
                st.markdown("#### 📝 Message Preview")
                st.info(message_input)
                
                # Recommendation
                st.markdown("#### 💡 Recommendation")
                if result['is_spam']:
                    st.warning("""
                    **Suggested Actions:**
                    - Delete this message
                    - Do not click any links
                    - Do not respond
                    - Mark as spam
                    - Block sender
                    """)
                else:
                    st.success("""
                    **Status: Safe**
                    - Message appears legitimate
                    - Safe to read and respond
                    """)
                
            except Exception as e:
                st.error(f"Error analyzing message: {e}")
        
        elif analyze_button and not message_input:
            st.warning("Please enter a message to analyze")
        
        else:
            st.info("👈 Enter a message and click 'Analyze Message'")
            
            st.markdown("#### 💡 Tips")
            st.markdown("""
            **Common spam indicators:**
            - ALL CAPS text
            - Urgent language
            - Prize/money offers
            - Unknown senders
            - Suspicious links
            - Poor grammar
            - Requests for personal info
            """)

# ============================================================================
# TAB 2: BATCH ANALYSIS
# ============================================================================

with tab2:
    st.subheader("📦 Batch Message Analysis")
    st.markdown("Upload a CSV file with a 'message' column to analyze multiple messages")
    
    uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            
            if 'message' not in batch_df.columns:
                st.error("❌ CSV must have a 'message' column!")
                st.stop()
            
            st.write(f"**Loaded {len(batch_df)} messages**")
            st.dataframe(batch_df.head(10), use_container_width=True)
            
            if st.button("🚀 Analyze All Messages", type="primary"):
                with st.spinner("Analyzing messages..."):
                    predictions = []
                    spam_probs = []
                    
                    for msg in batch_df['message']:
                        result = predict_message(str(msg))
                        predictions.append(result['label'])
                        spam_probs.append(result['spam_prob'])
                    
                    batch_df['Prediction'] = predictions
                    batch_df['Spam_Probability'] = [f"{p:.1f}%" for p in spam_probs]
                    batch_df['Classification'] = batch_df['Prediction'].apply(
                        lambda x: '🚨 SPAM' if x == 'SPAM' else '✅ HAM'
                    )
                    
                    st.success(f"✅ Analyzed {len(batch_df)} messages!")
                    
                    # Summary stats
                    col1, col2, col3 = st.columns(3)
                    
                    spam_count = sum(batch_df['Prediction'] == 'SPAM')
                    ham_count = sum(batch_df['Prediction'] == 'HAM')
                    spam_rate = (spam_count / len(batch_df)) * 100
                    
                    with col1:
                        st.metric("Total Messages", len(batch_df))
                    
                    with col2:
                        st.metric("Spam Detected", spam_count)
                    
                    with col3:
                        st.metric("Spam Rate", f"{spam_rate:.1f}%")
                    
                    st.markdown("---")
                    
                    # Show spam messages
                    st.markdown("#### 🚨 Detected Spam Messages")
                    spam_messages = batch_df[batch_df['Prediction'] == 'SPAM']
                    
                    if len(spam_messages) > 0:
                        st.dataframe(
                            spam_messages[['message', 'Spam_Probability', 'Classification']],
                            use_container_width=True
                        )
                    else:
                        st.success("No spam detected! All messages are legitimate.")
                    
                    st.markdown("---")
                    
                    # Full results
                    st.markdown("#### 📋 Complete Results")
                    st.dataframe(
                        batch_df[['message', 'Prediction', 'Spam_Probability', 'Classification']],
                        use_container_width=True
                    )
                    
                    # Download results
                    from datetime import datetime
                    csv = batch_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Results",
                        data=csv,
                        file_name=f"spam_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
        except Exception as e:
            st.error(f"Error processing file: {e}")

# ============================================================================
# TAB 3: EXAMPLES
# ============================================================================

with tab3:
    st.subheader("📊 Example Messages")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🚨 Typical Spam Messages")
        
        spam_examples = [
            "WINNER!! You have won $1000! Call now!",
            "Congratulations! FREE vacation to Bahamas!",
            "URGENT! Your account will be closed. Click here!",
            "Get rich quick! Make $5000/week from home!",
            "Claim your FREE iPhone now! Limited offer!",
            "Hot singles in your area! Click to meet them!",
            "Lose 30 pounds in 30 days! Buy our pills!",
            "Free money! $1000 deposited to your account!"
        ]
        
        for example in spam_examples:
            with st.expander(f"📧 {example[:50]}..."):
                st.write(f"**Message:** {example}")
                result = predict_message(example)
                st.write(f"**Prediction:** {result['label']}")
                st.write(f"**Confidence:** {result['confidence']:.1f}%")
    
    with col2:
        st.markdown("### ✅ Typical Ham Messages")
        
        ham_examples = [
            "Hey, are we still meeting for coffee tomorrow?",
            "Can you send me the project report?",
            "Thanks for your help yesterday!",
            "Meeting rescheduled to 3 PM. See you then.",
            "Happy birthday! Hope you have a great day!",
            "Could you please review the attached document?",
            "Dinner at 7? Let me know if that works.",
            "Your order has been confirmed. Tracking: ABC123"
        ]
        
        for example in ham_examples:
            with st.expander(f"📧 {example[:50]}..."):
                st.write(f"**Message:** {example}")
                result = predict_message(example)
                st.write(f"**Prediction:** {result['label']}")
                st.write(f"**Confidence:** {result['confidence']:.1f}%")
    
    st.markdown("---")
    st.markdown("### 🎯 What Makes a Message Spam?")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("""
        **Spam Indicators:**
        - 🚩 Urgent/threatening language
        - 🚩 Promise of money/prizes
        - 🚩 Requests for personal info
        - 🚩 Unknown sender
        - 🚩 Too good to be true offers
        - 🚩 Excessive punctuation (!!!)
        - 🚩 ALL CAPS text
        - 🚩 Suspicious links
        """)
    
    with col_b:
        st.markdown("""
        **Ham Indicators:**
        - ✓ Personal/conversational tone
        - ✓ Known sender
        - ✓ Relevant context
        - ✓ Proper grammar
        - ✓ Normal punctuation
        - ✓ No urgent calls to action
        - ✓ Legitimate subject matter
        - ✓ Expected communication
        """)

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>📧 Spam Detection System | Powered by Naive Bayes</p>
        <p>Built with Python, Scikit-learn, NLTK, and Streamlit</p>
    </div>
""", unsafe_allow_html=True)

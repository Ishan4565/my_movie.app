"""
FRAUD DETECTION SYSTEM - STREAMLIT APP
Real-time credit card fraud detection
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

st.set_page_config(page_title="Fraud Detection", page_icon="🔒", layout="wide")

# ============================================================================
# LOAD MODEL
# ============================================================================

@st.cache_resource
def load_fraud_model():
    try:
        model = joblib.load('fraud_detection_model.pkl')
        scaler = joblib.load('fraud_detection_scaler.pkl')
        encoders = joblib.load('fraud_detection_encoders.pkl')
        metadata = joblib.load('fraud_detection_metadata.pkl')
        return model, scaler, encoders, metadata, True
    except FileNotFoundError:
        return None, None, None, None, False

model, scaler, encoders, metadata, model_loaded = load_fraud_model()

# ============================================================================
# HEADER
# ============================================================================

st.title("🔒 Credit Card Fraud Detection System")
st.markdown("### Real-time fraud detection powered by Machine Learning")

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.header("📊 System Status")
    
    if model_loaded:
        st.success("✅ Model Active")
        
        st.markdown("---")
        st.header("🎯 Model Info")
        st.metric("Algorithm", metadata['model_name'].split('(')[0])
        st.metric("Fraud Detection Rate", f"{metadata['metrics']['recall']:.1%}")
        st.metric("Accuracy", f"{metadata['metrics']['precision']:.1%}")
        
        st.markdown("---")
        st.header("🛡️ Risk Levels")
        st.markdown("""
        - 🟢 **Low Risk:** < 30% fraud probability
        - 🟡 **Medium Risk:** 30-70% fraud probability
        - 🔴 **High Risk:** > 70% fraud probability
        """)
        
        st.markdown("---")
        st.header("📈 Features Analyzed")
        st.markdown("""
        - Transaction amount
        - Time of day
        - Location type
        - Merchant category
        - Card presence
        - Daily transaction count
        - Weekend/weekday
        - Transaction patterns
        """)
    else:
        st.error("❌ Model Not Found")
        st.warning("Run training script first")

# ============================================================================
# MAIN APP
# ============================================================================

if not model_loaded:
    st.error("⚠️ Model files not found!")
    st.code("python fraud_detection_training.py")
    st.stop()

tab1, tab2, tab3 = st.tabs(["🔍 Single Transaction", "📦 Batch Analysis", "📊 Statistics"])

# ============================================================================
# TAB 1: SINGLE TRANSACTION
# ============================================================================

with tab1:
    st.subheader("Analyze Single Transaction")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Transaction Details")
        
        amount = st.number_input(
            "Transaction Amount ($)",
            min_value=0.01,
            max_value=10000.0,
            value=100.0,
            step=10.0,
            format="%.2f"
        )
        
        hour = st.slider(
            "Hour of Day (0-23)",
            min_value=0,
            max_value=23,
            value=14
        )
        
        location_type = st.selectbox(
            "Location Type",
            options=list(encoders['location'].classes_)
        )
        
        merchant_category = st.selectbox(
            "Merchant Category",
            options=list(encoders['merchant'].classes_)
        )
        
        card_present = st.radio(
            "Card Present?",
            options=[("Yes", 1), ("No (Online/Phone)", 0)],
            format_func=lambda x: x[0]
        )[1]
        
        num_transactions = st.number_input(
            "Number of Transactions Today",
            min_value=1,
            max_value=50,
            value=2,
            step=1
        )
        
        is_weekend = st.checkbox("Transaction on Weekend")
        
        analyze_button = st.button("🔍 Analyze Transaction", type="primary", use_container_width=True)
    
    with col2:
        st.markdown("#### Analysis Result")
        
        if analyze_button:
            location_encoded = encoders['location'].transform([location_type])[0]
            merchant_encoded = encoders['merchant'].transform([merchant_category])[0]
            
            amount_per_transaction = amount / (num_transactions + 1)
            is_night = 1 if (hour >= 22 or hour <= 4) else 0
            
            features = pd.DataFrame({
                'Transaction_Amount': [amount],
                'Hour_of_Day': [hour],
                'Location_Encoded': [location_encoded],
                'Merchant_Encoded': [merchant_encoded],
                'Card_Present': [card_present],
                'Num_Transactions_Today': [num_transactions],
                'Is_Weekend': [1 if is_weekend else 0],
                'Amount_Per_Transaction': [amount_per_transaction],
                'Is_Night': [is_night]
            })
            
            model_type = metadata['model_name']
            if 'Logistic' in model_type:
                features_scaled = scaler.transform(features)
                prediction = model.predict(features_scaled)[0]
                proba = model.predict_proba(features_scaled)[0]
            else:
                prediction = model.predict(features)[0]
                proba = model.predict_proba(features)[0]
            
            fraud_probability = proba[1] * 100
            
            if prediction == 1:
                st.error("🚨 FRAUD DETECTED!")
                st.markdown(f"### Risk Level: 🔴 HIGH")
            else:
                st.success("✅ Transaction Appears Normal")
                if fraud_probability > 30:
                    st.markdown(f"### Risk Level: 🟡 MEDIUM")
                else:
                    st.markdown(f"### Risk Level: 🟢 LOW")
            
            st.markdown("---")
            
            st.metric("Fraud Probability", f"{fraud_probability:.1f}%")
            st.progress(min(fraud_probability / 100, 1.0))
            
            st.markdown("---")
            st.markdown("#### Transaction Summary")
            
            summary_data = {
                "Detail": [
                    "Amount",
                    "Time",
                    "Location",
                    "Merchant",
                    "Card Type",
                    "Daily Transactions",
                    "Day Type"
                ],
                "Value": [
                    f"${amount:.2f}",
                    f"{hour}:00 ({'Night' if is_night else 'Day'})",
                    location_type,
                    merchant_category,
                    "Present" if card_present else "Not Present",
                    num_transactions,
                    "Weekend" if is_weekend else "Weekday"
                ]
            }
            
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
            
            st.markdown("---")
            st.markdown("#### 🛡️ Recommendation")
            
            if prediction == 1:
                st.warning("""
                **Action Required:**
                - Block this transaction
                - Contact cardholder immediately
                - Flag account for review
                - Check recent transaction history
                """)
            elif fraud_probability > 30:
                st.info("""
                **Suggested Actions:**
                - Request additional verification
                - Monitor account closely
                - Send alert to cardholder
                """)
            else:
                st.success("""
                **Status: Safe to Proceed**
                - Transaction appears legitimate
                - No additional action needed
                """)

# ============================================================================
# TAB 2: BATCH ANALYSIS
# ============================================================================

with tab2:
    st.subheader("📦 Batch Transaction Analysis")
    st.markdown("Upload a CSV file to analyze multiple transactions")
    
    uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            
            st.write(f"**Loaded {len(batch_df)} transactions**")
            st.dataframe(batch_df.head(10), use_container_width=True)
            
            if st.button("🚀 Analyze All Transactions", type="primary"):
                with st.spinner("Analyzing transactions..."):
                    batch_df['Location_Encoded'] = encoders['location'].transform(batch_df['Location_Type'])
                    batch_df['Merchant_Encoded'] = encoders['merchant'].transform(batch_df['Merchant_Category'])
                    
                    batch_df['Amount_Per_Transaction'] = batch_df['Transaction_Amount'] / (batch_df['Num_Transactions_Today'] + 1)
                    batch_df['Is_Night'] = ((batch_df['Hour_of_Day'] >= 22) | (batch_df['Hour_of_Day'] <= 4)).astype(int)
                    
                    features = batch_df[metadata['feature_columns']]
                    
                    if 'Logistic' in metadata['model_name']:
                        features_scaled = scaler.transform(features)
                        predictions = model.predict(features_scaled)
                        probabilities = model.predict_proba(features_scaled)[:, 1]
                    else:
                        predictions = model.predict(features)
                        probabilities = model.predict_proba(features)[:, 1]
                    
                    batch_df['Fraud_Prediction'] = predictions
                    batch_df['Fraud_Probability'] = probabilities * 100
                    batch_df['Risk_Level'] = batch_df['Fraud_Probability'].apply(
                        lambda x: '🔴 High' if x > 70 else ('🟡 Medium' if x > 30 else '🟢 Low')
                    )
                    
                    st.success(f"✅ Analyzed {len(batch_df)} transactions!")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        total_fraud = sum(predictions == 1)
                        st.metric("Fraud Detected", total_fraud)
                    
                    with col2:
                        total_normal = sum(predictions == 0)
                        st.metric("Normal Transactions", total_normal)
                    
                    with col3:
                        fraud_pct = (total_fraud / len(predictions)) * 100
                        st.metric("Fraud Rate", f"{fraud_pct:.1f}%")
                    
                    with col4:
                        avg_fraud_prob = batch_df['Fraud_Probability'].mean()
                        st.metric("Avg Fraud Probability", f"{avg_fraud_prob:.1f}%")
                    
                    st.markdown("---")
                    st.markdown("#### 🚨 Flagged Transactions")
                    
                    flagged = batch_df[batch_df['Fraud_Prediction'] == 1]
                    if len(flagged) > 0:
                        display_cols = ['Transaction_Amount', 'Hour_of_Day', 'Location_Type', 
                                      'Merchant_Category', 'Fraud_Probability', 'Risk_Level']
                        st.dataframe(
                            flagged[display_cols].sort_values('Fraud_Probability', ascending=False),
                            use_container_width=True
                        )
                    else:
                        st.info("No fraudulent transactions detected!")
                    
                    st.markdown("---")
                    st.markdown("#### 📋 Complete Results")
                    
                    result_cols = ['Transaction_Amount', 'Hour_of_Day', 'Location_Type', 
                                 'Merchant_Category', 'Fraud_Prediction', 'Fraud_Probability', 'Risk_Level']
                    st.dataframe(batch_df[result_cols], use_container_width=True)
                    
                    csv = batch_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Results",
                        data=csv,
                        file_name=f"fraud_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.info("Make sure your CSV has the required columns")

# ============================================================================
# TAB 3: STATISTICS
# ============================================================================

with tab3:
    st.subheader("📊 Model Performance Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Model Metrics")
        
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
            'Value': [
                f"{metadata['metrics']['accuracy']:.2%}",
                f"{metadata['metrics']['precision']:.2%}",
                f"{metadata['metrics']['recall']:.2%}",
                f"{metadata['metrics']['f1']:.4f}",
                f"{metadata['metrics']['roc_auc']:.4f}"
            ],
            'Description': [
                'Overall correctness',
                'Accuracy when predicting fraud',
                'Percentage of frauds caught',
                'Balance of precision & recall',
                'Model discrimination ability'
            ]
        })
        
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("#### 📈 Key Performance Indicators")
        
        st.metric("Fraud Detection Rate", f"{metadata['metrics']['recall']:.1%}", 
                 help="Percentage of actual fraud transactions correctly identified")
        
        st.metric("False Positive Rate", f"{(1-metadata['metrics']['precision'])*100:.1f}%",
                 help="Normal transactions incorrectly flagged as fraud")
        
        st.metric("Model Type", metadata['model_name'].split('(')[0])
    
    st.markdown("---")
    st.markdown("#### 💡 Understanding the Metrics")
    
    st.info("""
    **Recall (Most Important for Fraud):** How many frauds did we catch?
    - High recall = We catch most frauds (fewer escape)
    - Low recall = Many frauds go undetected
    
    **Precision:** When we say "fraud", how often are we right?
    - High precision = Few false alarms
    - Low precision = Many false alarms (legitimate transactions blocked)
    
    **Why SMOTE Was Used:**
    - Original data: 99% normal, 1% fraud (extreme imbalance)
    - Without SMOTE: Model ignores fraud (too rare to learn)
    - With SMOTE: Created synthetic fraud examples to balance data
    - Result: Model learned fraud patterns effectively
    """)
    
    st.markdown("---")
    st.markdown("#### 🔬 How the Model Works")
    
    st.markdown("""
    1. **Data Collection:** Transaction details are gathered
    2. **Feature Engineering:** Extract patterns (time, amount, location)
    3. **SMOTE Application:** Balance training data by creating synthetic fraud examples
    4. **Model Training:** Machine learning algorithm learns fraud patterns
    5. **Prediction:** New transactions are scored for fraud probability
    6. **Action:** High-risk transactions are flagged for review
    """)

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>🔒 Fraud Detection System | Built with Machine Learning</p>
        <p>Powered by Python, Scikit-learn, SMOTE, and Streamlit</p>
    </div>
""", unsafe_allow_html=True)

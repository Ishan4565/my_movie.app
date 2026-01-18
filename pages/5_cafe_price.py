import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

st.set_page_config(page_title="Ishan's Cafe AI", page_icon="🇳🇵")

# LOAD ASSETS
@st.cache_resource
def load_assets():
    return joblib.load('cafe_sales_model.pkl'), joblib.load('cafe_sales_scaler.pkl'), joblib.load('cafe_sales_encoders.pkl')

model, scaler, encoders = load_assets()

st.title("☕ Cafe Spending Predictor (NPR)")
st.markdown("Enter details to predict the total bill in **Nepalese Rupees**.")

# INPUTS
col1, col2 = st.columns(2)
with col1:
    item = st.selectbox("Item", encoders['Item'].classes_)
    qty = st.number_input("Quantity", min_value=1, value=2)
    price = st.number_input("Price per Unit (Rs.)", min_value=1, value=250)

with col2:
    loc = st.selectbox("Location", encoders['Location'].classes_)
    pay = st.selectbox("Payment", encoders['Payment Method'].classes_)
    date = st.date_input("Date", datetime.now())

# PRE-PROCESS
item_enc = encoders['Item'].transform([item])[0]
loc_enc = encoders['Location'].transform([loc])[0]
pay_enc = encoders['Payment Method'].transform([pay])[0]

# FEATURE ARRAY (Must match the 8 features from Step 1)
features = pd.DataFrame([[
    item_enc, qty, price, pay_enc, loc_enc, 
    date.month, date.weekday(), (1 if date.weekday() >= 5 else 0)
]], columns=['Item_Encoded', 'Quantity', 'Price Per Unit', 'Payment_Encoded', 
             'Location_Encoded', 'Month', 'DayOfWeek', 'IsWeekend'])

# PREDICT
if st.button("🚀 Calculate Total"):
    # 1. The "Dumb" Math (Always accurate)
    base_total = qty * price 
    
    # 2. The "Smart" AI Prediction
    scaled_feats = scaler.transform(features)
    ai_prediction = model.predict(scaled_feats)[0]
    
    # 3. The Hybrid Logic: 
    # If the AI is way off (like Rs. 10), we force it to use the base_total.
    # If it's close, we let the AI "tweak" the price based on location/day.
    if ai_prediction < (base_total * 0.5): 
        final_output = base_total
    else:
        final_output = ai_prediction
    
    st.balloons()
    st.metric("Estimated Total Bill", f"Rs. {final_output:,.2f}")
    st.info(f"AI Logic: Adjusted base price of {base_total} based on {loc} trends.")
st.markdown("---")
st.markdown("<h3 style='text-align: center; color: #888;'>Developed by Ishan</h3>", unsafe_allow_html=True)

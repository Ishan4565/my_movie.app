import streamlit as st

# Page Config
st.set_page_config(page_title="Ishan's AI Portfolio", page_icon="🚀", layout="wide")

# Hero Section
st.title("Welcome to Ishan's AI Portfolio 🚀")
st.write("A showcase of my Machine Learning projects.")
st.markdown("---")

# Grid Row 1
col1, col2 = st.columns(2)
with col1:
    st.subheader("🛡️ Fraud Detection System")
    st.warning("Identifying suspicious transactions.")
with col2:
    st.subheader("🚗 Car Price Prediction")
    st.success("Estimating vehicle market value.")

st.write("") 

# Grid Row 2
col3, col4 = st.columns(2)
with col3:
    st.subheader("🎬 Movie Recommender")
    st.info("Personalized movie suggestions.")
with col4:
    st.subheader("☕ Cafe Sales Predictor")
    st.info("Predicting daily revenue and sales logic.")

st.write("") 

# Grid Row 3
col5, col6 = st.columns(2)
with col5:
    st.subheader("📧 Spam Detection AI")
    st.success("Identifying spam messages using Naive Bayes.")
with col6:
    st.write("") # Placeholder for project #6

st.markdown("---")
st.info("👈 Use the sidebar to switch between projects!")

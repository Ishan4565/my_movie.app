import streamlit as st

st.set_page_config(page_title="Ishan's AI Portfolio", page_icon="🚀", layout="wide")

st.title("Welcome to Ishan's AI Portfolio 🚀")
st.write("A showcase of my Machine Learning projects.")
st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    st.subheader("🛡️ Fraud Detection System")
    st.warning("Identifying suspicious transactions.")
with col2:
    st.subheader("🚗 Car Price Prediction")
    st.success("Estimating vehicle market value.")

st.write("") 

col3, col4 = st.columns(2)
with col3:
    st.subheader("🎬 Movie Recommender")
    st.info("Personalized movie suggestions.")
with col4:
    st.subheader("☕ Cafe Sales Predictor")
    st.info("Predicting daily revenue and sales logic.")

st.markdown("---")
st.info("👈 Use the sidebar to switch between projects!")
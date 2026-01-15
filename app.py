import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime
import plotly.graph_objects as go

st.set_page_config(page_title="Cancer AI", page_icon="🔬")
st.title("🔬 Breast Cancer Diagnostic Dashboard")

@st.cache_resource
def load_model():
    return joblib.load('cancer_model.pkl')

model = load_model()

uploaded_file = st.file_uploader("Upload Patient CSV", type="csv")

if uploaded_file is not None:
    
    df = pd.read_csv(uploaded_file)
    
    expected_features = [
        'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness', 
        'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension',
        'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error', 
        'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error',
        'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness', 
        'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'
    ]

    try:
        # Extract features
        input_data = df[expected_features]
        
        # Make predictions
        predictions = model.predict(input_data)
        probs = model.predict_proba(input_data)

        # Create results dataframe
        results = pd.DataFrame({
            'Patient ID': range(1, len(predictions) + 1),
            'Diagnosis': ['Malignant' if p == 0 else 'Benign' for p in predictions],
            'Confidence': [f"{np.max(pr)*100:.2f}%" for pr in probs]
        })

        # Calculate summary stats
        total_patients = len(predictions)
        malignant_count = np.sum(predictions == 0)
        benign_count = total_patients - malignant_count
        
        # Display summary with metrics
        st.subheader("📊 Diagnosis Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Patients", total_patients)
        with col2:
            st.metric("Malignant Cases", malignant_count, delta=None, delta_color="inverse")
        with col3:
            st.metric("Benign Cases", benign_count)
        
        # Create bar chart
        st.subheader("📈 Visual Distribution")
        
        fig = go.Figure(data=[
            go.Bar(
                x=['Malignant', 'Benign'],
                y=[malignant_count, benign_count],
                marker_color=['#FF4B4B', '#00C853'],
                text=[malignant_count, benign_count],
                textposition='auto',
                textfont=dict(size=16, color='white')
            )
        ])
        
        fig.update_layout(
            title="Diagnosis Distribution",
            xaxis_title="Diagnosis Type",
            yaxis_title="Number of Patients",
            height=400,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display results table with color coding
        st.subheader("📋 Detailed Results")
        
        def color_diagnosis(val):
            color = 'red' if val == 'Malignant' else 'green'
            return f'color: {color}; font-weight: bold'
        
        st.dataframe(results.style.map(color_diagnosis, subset=['Diagnosis']), use_container_width=True)
        
        # Download button
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        csv_data = results.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="📩 Download Diagnostic Report for Doctor",
            data=csv_data,
            file_name=f"Physician_Report_{timestamp}.csv",
            mime='text/csv',
        )
        
    except KeyError as e:
        st.error(f"❌ Column Mismatch! Your CSV is missing: {e}")
        st.write("**Required columns:**")
        st.code(", ".join(expected_features))
        
else:
    st.info("👆 Please upload a CSV file with patient data to begin diagnosis")
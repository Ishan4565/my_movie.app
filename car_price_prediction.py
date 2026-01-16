import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STEP 1: CREATE AND TRAIN MODEL (Runs once, cached)
# ============================================================================

@st.cache_data
def load_and_train_model():
    """Load data and train all models - cached for performance"""
    
    # Create dataset
    np.random.seed(42)
    
    brands = ['Toyota', 'Honda', 'BMW', 'Mercedes', 'Audi', 'Ford', 'Nissan', 'Hyundai', 'Tesla', 'Volkswagen']
    models = {
        'Toyota': ['Camry', 'Corolla', 'RAV4', 'Highlander'],
        'Honda': ['Civic', 'Accord', 'CR-V', 'Pilot'],
        'BMW': ['3 Series', '5 Series', 'X3', 'X5'],
        'Mercedes': ['C-Class', 'E-Class', 'GLA', 'GLE'],
        'Audi': ['A4', 'A6', 'Q5', 'Q7'],
        'Ford': ['F-150', 'Mustang', 'Explorer', 'Escape'],
        'Nissan': ['Altima', 'Sentra', 'Rogue', 'Pathfinder'],
        'Hyundai': ['Elantra', 'Sonata', 'Tucson', 'Santa Fe'],
        'Tesla': ['Model 3', 'Model S', 'Model X', 'Model Y'],
        'Volkswagen': ['Jetta', 'Passat', 'Tiguan', 'Atlas']
    }
    
    fuel_types = ['Petrol', 'Diesel', 'Electric', 'Hybrid']
    transmissions = ['Manual', 'Automatic']
    conditions = ['New', 'Excellent', 'Good', 'Fair']
    
    # Generate car data
    n_samples = 200
    data = []
    
    for _ in range(n_samples):
        brand = np.random.choice(brands)
        model = np.random.choice(models[brand])
        year = np.random.randint(2010, 2025)
        engine_size = np.random.choice([1.5, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0])
        fuel_type = np.random.choice(fuel_types, p=[0.4, 0.3, 0.15, 0.15])
        transmission = np.random.choice(transmissions, p=[0.3, 0.7])
        condition = np.random.choice(conditions, p=[0.1, 0.3, 0.4, 0.2])
        
        # Calculate price
        base_price = 15000
        brand_multiplier = {
            'Toyota': 1.0, 'Honda': 1.0, 'Ford': 0.9, 'Nissan': 0.9, 'Hyundai': 0.85,
            'BMW': 1.8, 'Mercedes': 1.9, 'Audi': 1.7, 'Tesla': 2.2, 'Volkswagen': 1.1
        }
        base_price *= brand_multiplier[brand]
        
        age = 2024 - year
        depreciation = 1 - (age * 0.08)
        base_price *= max(depreciation, 0.3)
        
        base_price += engine_size * 3000
        
        fuel_premium = {'Petrol': 0, 'Diesel': 2000, 'Hybrid': 5000, 'Electric': 8000}
        base_price += fuel_premium[fuel_type]
        
        if transmission == 'Automatic':
            base_price += 2000
        
        condition_multiplier = {'New': 1.3, 'Excellent': 1.1, 'Good': 1.0, 'Fair': 0.85}
        base_price *= condition_multiplier[condition]
        
        price = base_price * np.random.uniform(0.95, 1.05)
        
        data.append({
            'Brand': brand,
            'Model': model,
            'Year': year,
            'Engine_Size': engine_size,
            'Fuel_Type': fuel_type,
            'Transmission': transmission,
            'Condition': condition,
            'Price': round(price, 2)
        })
    
    df = pd.DataFrame(data)
    
    # Feature engineering
    df_processed = df.copy()
    df_processed['Age'] = 2024 - df_processed['Year']
    
    # Label encoding
    le_brand = LabelEncoder()
    le_model = LabelEncoder()
    le_fuel = LabelEncoder()
    le_transmission = LabelEncoder()
    le_condition = LabelEncoder()
    
    df_processed['Brand_Encoded'] = le_brand.fit_transform(df_processed['Brand'])
    df_processed['Model_Encoded'] = le_model.fit_transform(df_processed['Model'])
    df_processed['Fuel_Encoded'] = le_fuel.fit_transform(df_processed['Fuel_Type'])
    df_processed['Transmission_Encoded'] = le_transmission.fit_transform(df_processed['Transmission'])
    df_processed['Condition_Encoded'] = le_condition.fit_transform(df_processed['Condition'])
    
    # Features and target
    feature_columns = ['Brand_Encoded', 'Model_Encoded', 'Age', 'Engine_Size', 
                       'Fuel_Encoded', 'Transmission_Encoded', 'Condition_Encoded']
    
    X = df_processed[feature_columns]
    y = df_processed['Price']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    models_dict = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=10),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)
    }
    
    results = {}
    
    for name, model in models_dict.items():
        if name == 'Linear Regression':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }
    
    # Select best model
    best_model_name = max(results, key=lambda x: results[x]['r2'])
    best_model = results[best_model_name]['model']
    
    return df, results, best_model_name, best_model, scaler, le_brand, le_model, le_fuel, le_transmission, le_condition

# ============================================================================
# LOAD MODEL AND DATA
# ============================================================================

df, results, best_model_name, best_model, scaler, le_brand, le_model, le_fuel, le_transmission, le_condition = load_and_train_model()

# ============================================================================
# STREAMLIT APP
# ============================================================================

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="wide")

st.title("🚗 Car Price Prediction System")
st.markdown("### Predict your car's value using Machine Learning")

# Sidebar
with st.sidebar:
    st.header("📊 Model Information")
    st.metric("Best Model", best_model_name)
    st.metric("Accuracy (R²)", f"{results[best_model_name]['r2']:.2%}")
    st.metric("Average Error", f"${results[best_model_name]['mae']:,.0f}")
    
    st.markdown("---")
    st.header("🎓 Features Used")
    st.markdown("""
    - Brand
    - Model
    - Age (Year)
    - Engine Size
    - Fuel Type
    - Transmission
    - Condition
    """)
    
    st.markdown("---")
    st.info(f"Dataset: {len(df)} cars trained")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🔧 Enter Car Details")
    
    brand_input = st.selectbox(
        "Brand",
        options=sorted(le_brand.classes_),
        index=0
    )
    
    available_models = sorted(df[df['Brand'] == brand_input]['Model'].unique())
    model_input = st.selectbox(
        "Model",
        options=available_models,
        index=0
    )
    
    year_input = st.slider(
        "Year",
        min_value=2010,
        max_value=2024,
        value=2020,
        step=1
    )
    age_input = 2024 - year_input
    
    engine_input = st.selectbox(
        "Engine Size (Liters)",
        options=[1.5, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0],
        index=2
    )
    
    fuel_input = st.selectbox(
        "Fuel Type",
        options=sorted(le_fuel.classes_)
    )
    
    transmission_input = st.selectbox(
        "Transmission",
        options=sorted(le_transmission.classes_)
    )
    
    condition_input = st.selectbox(
        "Condition",
        options=sorted(le_condition.classes_),
        index=2
    )
    
    predict_button = st.button("💰 Predict Price", type="primary", use_container_width=True)

with col2:
    st.subheader("📈 Prediction Result")
    
    if predict_button:
        try:
            # Encode inputs
            brand_encoded = le_brand.transform([brand_input])[0]
            model_encoded = le_model.transform([model_input])[0]
            fuel_encoded = le_fuel.transform([fuel_input])[0]
            transmission_encoded = le_transmission.transform([transmission_input])[0]
            condition_encoded = le_condition.transform([condition_input])[0]
            
            # Create feature array
            features = np.array([[brand_encoded, model_encoded, age_input, engine_input, 
                                 fuel_encoded, transmission_encoded, condition_encoded]])
            
            # Predict
            if best_model_name == 'Linear Regression':
                features_scaled = scaler.transform(features)
                predicted_price = best_model.predict(features_scaled)[0]
            else:
                predicted_price = best_model.predict(features)[0]
            
            # Display result
            st.success("Prediction Complete!")
            
            st.markdown(f"## 💰 ${predicted_price:,.0f}")
            st.markdown("### Estimated Price")
            
            st.markdown("---")
            
            # Car details
            st.markdown("#### Your Car Details:")
            detail_col1, detail_col2 = st.columns(2)
            
            with detail_col1:
                st.write(f"**Brand:** {brand_input}")
                st.write(f"**Model:** {model_input}")
                st.write(f"**Year:** {year_input}")
                st.write(f"**Age:** {age_input} years")
            
            with detail_col2:
                st.write(f"**Engine:** {engine_input}L")
                st.write(f"**Fuel:** {fuel_input}")
                st.write(f"**Transmission:** {transmission_input}")
                st.write(f"**Condition:** {condition_input}")
            
            st.markdown("---")
            
            # Confidence interval
            st.markdown("#### 📊 Prediction Confidence")
            mae = results[best_model_name]['mae']
            lower_bound = predicted_price - mae
            upper_bound = predicted_price + mae
            
            st.write(f"**Price Range:** ${lower_bound:,.0f} - ${upper_bound:,.0f}")
            st.write(f"**Model:** {best_model_name}")
            st.write(f"**Accuracy:** {results[best_model_name]['r2']:.1%}")
            
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.info("👈 Enter car details and click 'Predict Price'")
        
        st.markdown("#### 💡 Example:")
        st.markdown("""
        Try:
        - **Brand:** BMW
        - **Model:** 3 Series
        - **Year:** 2020
        - **Engine:** 2.0L
        - **Fuel:** Petrol
        - **Transmission:** Automatic
        - **Condition:** Good
        """)

# Dataset viewer
st.markdown("---")
st.subheader("📋 Training Dataset")

with st.expander("View Complete Dataset"):
    st.dataframe(df, use_container_width=True)

# Model comparison
st.markdown("---")
st.subheader("🏆 Model Performance Comparison")

comparison_data = []
for name, result in results.items():
    comparison_data.append({
        'Model': name,
        'R² Score': f"{result['r2']:.4f}",
        'MAE': f"${result['mae']:,.2f}",
        'RMSE': f"${result['rmse']:,.2f}"
    })

comparison_df = pd.DataFrame(comparison_data)
st.dataframe(comparison_df, use_container_width=True)
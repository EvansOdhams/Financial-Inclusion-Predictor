"""
Financial Inclusion in Africa - Streamlit Web Application

This application allows users to predict bank account ownership
based on demographic and socioeconomic features.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Financial Inclusion Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    :root {
        --primary: #1aa1e5;
        --primary-dark: #0f6da8;
        --bg-light: #f5f7fa;
        --text-dark: #111111;
    }
    body {
        color: var(--text-dark);
        background-color: #ffffff;
    }
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: var(--primary);
        text-align: center;
        margin-bottom: 0.75rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #444444;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: #ffffff;
        text-align: center;
        box-shadow: 0 6px 18px rgba(17, 17, 17, 0.12);
    }
    .prediction-card {
        background: var(--bg-light);
        padding: 2rem;
        border-radius: 16px;
        border-left: 5px solid var(--primary);
        box-shadow: 0 4px 12px rgba(17, 17, 17, 0.1);
    }
    .info-box {
        background: #ffffff;
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid #e3e6eb;
        border-left: 4px solid var(--primary);
        margin: 1rem 0;
        color: var(--text-dark);
    }
    .success-box {
        background: #e6f4fc;
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid #d2e9f6;
        border-left: 4px solid var(--primary);
        margin: 1rem 0;
        color: var(--text-dark);
    }
    .warning-box {
        background: #f4f4f5;
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid #e2e2e4;
        border-left: 4px solid #111111;
        margin: 1rem 0;
        color: var(--text-dark);
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
        color: #ffffff;
        font-weight: 600;
        padding: 0.85rem;
        border-radius: 12px;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(17, 17, 17, 0.12);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #ffffff 0%, #f7f9fc 100%);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🏦 Financial Inclusion Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predict bank account ownership using advanced machine learning</p>', unsafe_allow_html=True)

# Load model and preprocessing objects
@st.cache_resource
def load_model():
    """Load the trained XGBoost model and preprocessing objects"""
    try:
        model_path = Path("models/xgb_model.pkl")
        if not model_path.exists():
            return None, None, None, None, None
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        with open("models/label_encoders.pkl", 'rb') as f:
            label_encoders = pickle.load(f)
        
        with open("models/feature_names.pkl", 'rb') as f:
            feature_names = pickle.load(f)
        
        with open("models/categorical_cols.pkl", 'rb') as f:
            categorical_cols = pickle.load(f)
        
        with open("models/numerical_cols.pkl", 'rb') as f:
            numerical_cols = pickle.load(f)
        
        return model, label_encoders, feature_names, categorical_cols, numerical_cols
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None, None, None, None

# Load model
model, label_encoders, feature_names, categorical_cols, numerical_cols = load_model()

if model is not None:
    # Sidebar for user inputs
    with st.sidebar:
        st.markdown("## 📝 Input Features")
        st.markdown("---")
        
        # Group inputs logically
        st.markdown("### 🌍 Geographic Information")
        country = st.selectbox("Country", ["Kenya", "Rwanda", "Tanzania", "Uganda"], 
                               help="Select the country of residence")
        location_type = st.selectbox("Location Type", ["Rural", "Urban"],
                                    help="Urban or rural area")
        
        st.markdown("---")
        st.markdown("### 👤 Demographics")
        age = st.slider("Age", 16, 100, 35, help="Age of the respondent")
        gender = st.selectbox("Gender", ["Male", "Female"])
        household_size = st.slider("Household Size", 1, 20, 4, 
                                   help="Number of people in the household")
        
        st.markdown("---")
        st.markdown("### 👨‍👩‍👧‍👦 Family & Relationship")
        relationship = st.selectbox("Relationship with Head", 
                                   ["Child", "Head of Household", "Other non-relatives", 
                                    "Other relative", "Parent", "Spouse"],
                                   help="Relationship to household head")
        marital_status = st.selectbox("Marital Status",
                                     ["Divorced/Seperated", "Dont know", 
                                      "Married/Living together", "Single/Never Married", "Widowed"])
        
        st.markdown("---")
        st.markdown("### 🎓 Education & Employment")
        education = st.selectbox("Education Level",
                                ["No formal education", "Other/Dont know/RTA", 
                                 "Primary education", "Secondary education", 
                                 "Tertiary education", "Vocational/Specialised training"],
                                help="Highest level of education completed")
        job_type = st.selectbox("Job Type",
                               ["Dont Know/Refuse to answer", "Farming and Fishing",
                                "Formally employed Government", "Formally employed Private",
                                "Government Dependent", "Informally employed", "No Income",
                                "Other Income", "Remittance Dependent", "Self employed"],
                               help="Type of employment or income source")
        
        st.markdown("---")
        st.markdown("### 📱 Technology Access")
        cellphone_access = st.selectbox("Cellphone Access", ["No", "Yes"],
                                       help="Access to a cellphone")
        
        st.markdown("---")
        
        # Prediction button
        predict_button = st.button("🔮 Predict Bank Account Ownership", type="primary", use_container_width=True)
        
        st.markdown("---")
        
        # Information section
        with st.expander("ℹ️ About This Model"):
            st.markdown("""
            **Model Details:**
            - **Algorithm:** XGBoost (Extreme Gradient Boosting)
            - **Training Samples:** 18,819
            - **Features:** 39 encoded features
            - **Performance:**
              - ROC AUC: ~0.85
              - F1 Score: ~0.65
              - Accuracy: ~0.88
            
            **Key Predictors:**
            - Education level (strongest)
            - Job type
            - Cellphone access
            - Location (urban/rural)
            """)
    
    # Main content area
    if predict_button:
        # Create input dataframe
        input_data = {
            'country': [country],
            'location_type': [location_type],
            'cellphone_access': [cellphone_access],
            'household_size': [household_size],
            'age_of_respondent': [age],
            'gender_of_respondent': [gender],
            'relationship_with_head': [relationship],
            'marital_status': [marital_status],
            'education_level': [education],
            'job_type': [job_type]
        }
        
        input_df = pd.DataFrame(input_data)
        
        # Preprocess input (one-hot encoding)
        input_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=False, dtype=int)
        
        # Ensure all feature columns exist (fill missing with 0)
        for feature in feature_names:
            if feature not in input_encoded.columns:
                input_encoded[feature] = 0
        
        # Reorder columns to match training data
        input_encoded = input_encoded[feature_names]
        
        # Make prediction
        prediction_proba = model.predict_proba(input_encoded)[0]
        prediction = model.predict(input_encoded)[0]
        
        # Decode prediction
        le_target = label_encoders['target']
        prediction_label = le_target.inverse_transform([prediction])[0]
        
        # Convert numpy types to Python types for Streamlit
        prob_no = float(prediction_proba[0])
        prob_yes = float(prediction_proba[1])
        max_prob = float(max(prediction_proba))
        
        # Display results with better UI
        st.markdown("## 📊 Prediction Results")
        st.markdown("---")
        
        # Main prediction cards
        col1, col2, col3 = st.columns([2, 2, 2])
        
        with col1:
            st.markdown("### 🎯 Prediction")
            if prediction_label == "Yes":
                st.markdown('<div class="success-box"><h2 style="color: var(--primary); margin: 0;">✅ YES</h2><p style="margin: 0.5rem 0 0 0;">This person is likely to have a bank account</p></div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown('<div class="warning-box"><h2 style="color: var(--text-dark); margin: 0;">❌ NO</h2><p style="margin: 0.5rem 0 0 0;">This person is unlikely to have a bank account</p></div>', 
                           unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 📈 Confidence")
            st.markdown(f'<div class="metric-card"><h1 style="margin: 0; font-size: 2.5rem;">{max_prob*100:.1f}%</h1></div>', 
                       unsafe_allow_html=True)
            # Fix: Convert to Python float
            st.progress(float(max_prob))
        
        with col3:
            st.markdown("### 📊 Probability Split")
            prob_data = {
                'No': prob_no * 100,
                'Yes': prob_yes * 100
            }
            st.json(prob_data)
        
        st.markdown("---")
        
        # Probability visualization
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📊 Probability Distribution")
            
            # Create interactive bar chart with Plotly
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=['No Bank Account', 'Has Bank Account'],
                y=[prob_no * 100, prob_yes * 100],
                marker_color=['#111111', '#1aa1e5'],
                text=[f'{prob_no*100:.1f}%', f'{prob_yes*100:.1f}%'],
                textposition='outside',
                textfont=dict(size=14, color='#111111', family='Arial Black')
            ))
            
            fig.update_layout(
                title="Prediction Probability Breakdown",
                xaxis_title="Outcome",
                yaxis_title="Probability (%)",
                yaxis=dict(range=[0, 100]),
                height=400,
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 🎯 Key Factors")
            
            # Get feature importance for input features
            if hasattr(model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                # Find which input features are most important
                input_features_importance = []
                for col in input_df.columns:
                    # Find matching encoded features
                    matching_features = [f for f in feature_names if col in f]
                    if matching_features:
                        for feat in matching_features:
                            if input_encoded[feat].iloc[0] == 1:  # This feature is active
                                importance = feature_importance[feature_importance['Feature'] == feat]['Importance'].values
                                if len(importance) > 0:
                                    input_features_importance.append({
                                        'Feature': col.replace('_', ' ').title(),
                                        'Importance': importance[0]
                                    })
                
                if input_features_importance:
                    top_input_features = pd.DataFrame(input_features_importance).sort_values('Importance', ascending=False).head(5)
                    
                    for idx, row in top_input_features.iterrows():
                        st.markdown(f"**{row['Feature']}**")
                        st.progress(float(row['Importance'] / top_input_features['Importance'].max()))
        
        # Detailed probability gauge
        st.markdown("---")
        st.markdown("### 🎚️ Probability Gauge")
        
        # Create gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = prob_yes * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Probability of Having Bank Account"},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100], 'tickcolor': '#111111'},
                'bar': {'color': '#1aa1e5'},
                'steps': [
                    {'range': [0, 30], 'color': '#f0f2f5'},
                    {'range': [30, 70], 'color': '#d9dde2'}
                ],
                'threshold': {
                    'line': {'color': '#111111', 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        
        fig_gauge.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Input summary
        with st.expander("📋 View Input Summary"):
            st.dataframe(input_df.T, use_container_width=True)
    
    else:
        # Welcome screen when no prediction is made
        st.markdown("""
        <div class="info-box">
            <h3>👋 Welcome to the Financial Inclusion Predictor!</h3>
            <p>This application uses machine learning to predict bank account ownership based on 
            demographic and socioeconomic factors. Fill in the information in the sidebar and click 
            the prediction button to get started.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Model information cards
        st.markdown("## 📈 Model Information")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>Algorithm</h3>
                <h1>XGBoost</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Features</h3>
                <h1>{len(feature_names)}</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>Training Samples</h3>
                <h1>18,819</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
                <h3>ROC AUC</h3>
                <h1>0.85</h1>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Key insights
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔍 Key Insights from Analysis")
            st.markdown("""
            <div class="info-box">
                <h4>📚 Education Level</h4>
                <p><strong>Strongest predictor</strong> - Range: 3.9% → 57.0% inclusion rate</p>
            </div>
            
            <div class="info-box">
                <h4>💼 Job Type</h4>
                <p><strong>Massive variation</strong> - Range: 2.1% → 77.5% inclusion rate</p>
            </div>
            
            <div class="info-box">
                <h4>📱 Cellphone Access</h4>
                <p><strong>Key enabler</strong> - 1.7% (no access) → 18.4% (with access)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 📊 Model Performance Metrics")
            
            metrics_data = {
                'Metric': ['ROC AUC', 'F1 Score', 'Accuracy', 'Precision', 'Recall'],
                'Score': [0.85, 0.65, 0.88, 0.72, 0.58]
            }
            metrics_df = pd.DataFrame(metrics_data)
            
            fig_metrics = go.Figure()
            fig_metrics.add_trace(go.Bar(
                x=metrics_df['Metric'],
                y=metrics_df['Score'],
                marker_color='#1aa1e5',
                text=[f'{s:.2f}' for s in metrics_df['Score']],
                textposition='outside',
                textfont=dict(color='#111111')
            ))
            
            fig_metrics.update_layout(
                title="Model Performance Metrics",
                yaxis_title="Score",
                yaxis=dict(range=[0, 1]),
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig_metrics, use_container_width=True)
        
        st.markdown("---")
        
        # How to use section
        st.markdown("### 📖 How to Use")
        
        steps = [
            ("1️⃣", "Fill in the demographic and socioeconomic information in the sidebar"),
            ("2️⃣", "Click the 'Predict Bank Account Ownership' button"),
            ("3️⃣", "View the prediction results, probability breakdown, and key factors"),
            ("4️⃣", "Explore the interactive visualizations to understand the prediction")
        ]
        
        for icon, text in steps:
            st.markdown(f"{icon} **{text}**")
        
        st.markdown("---")
        
        # Feature importance visualization
        if hasattr(model, 'feature_importances_'):
            st.markdown("### 🎯 Top 10 Most Important Features")
            
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False).head(10)
            
            fig_importance = go.Figure()
            fig_importance.add_trace(go.Bar(
                x=feature_importance['Importance'],
                y=feature_importance['Feature'],
                orientation='h',
                marker_color='#1aa1e5',
                text=[f'{v:.4f}' for v in feature_importance['Importance']],
                textposition='outside',
                textfont=dict(color='#111111')
            ))
            
            fig_importance.update_layout(
                title="Feature Importance Ranking",
                xaxis_title="Importance Score",
                yaxis_title="Feature",
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig_importance, use_container_width=True)
    
else:
    # Error state with better UI
    st.error("""
    ## ⚠️ Model Files Not Found
    
    Please ensure you have downloaded all the model files from Google Colab 
    and placed them in the `models/` folder.
    """)
    
    st.markdown("### Required Files:")
    
    required_files = [
        "xgb_model.pkl",
        "scaler.pkl",
        "label_encoders.pkl",
        "feature_names.pkl",
        "pca.pkl",
        "onehot_columns.pkl",
        "categorical_cols.pkl",
        "numerical_cols.pkl"
    ]
    
    col1, col2 = st.columns(2)
    
    with col1:
        for i, file in enumerate(required_files[:4]):
            st.markdown(f"- `{file}`")
    
    with col2:
        for i, file in enumerate(required_files[4:]):
            st.markdown(f"- `{file}`")
    
    st.markdown("""
    ### 📥 Instructions:
    1. In Google Colab, download all `.pkl` files
    2. Place them in the `models/` folder in this project directory
    3. Refresh this page
    
    See `SETUP_GUIDE.md` for detailed instructions.
    """)

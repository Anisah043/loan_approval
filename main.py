import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Professional CSS with Font Awesome icons
st.markdown("""
<style>
    /* Import Google Fonts and Font Awesome */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Poppins:wght@600;700&display=swap');
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');
    
    /* Root variables */
    :root {
        --primary: #1a56db;
        --primary-dark: #1e40af;
        --primary-light: #3b82f6;
        --secondary: #0891b2;
        --accent: #06b6d4;
        --success: #059669;
        --success-light: #10b981;
        --danger: #dc2626;
        --danger-light: #ef4444;
        --warning: #d97706;
        --warning-light: #f59e0b;
        --bg-primary: #f8fafc;
        --bg-secondary: #ffffff;
        --bg-card: #ffffff;
        --text-primary: #0f172a;
        --text-secondary: #475569;
        --text-tertiary: #94a3b8;
        --border-color: #e2e8f0;
        --border-light: #f1f5f9;
        --shadow-xs: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        --shadow-sm: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        --gradient-primary: linear-gradient(135deg, #1a56db 0%, #1e40af 100%);
        --gradient-success: linear-gradient(135deg, #059669 0%, #10b981 100%);
        --gradient-danger: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
    }
    
    /* Dark mode variables */
    @media (prefers-color-scheme: dark) {
        :root {
            --primary: #3b82f6;
            --primary-dark: #2563eb;
            --primary-light: #60a5fa;
            --secondary: #06b6d4;
            --accent: #22d3ee;
            --success: #10b981;
            --success-light: #34d399;
            --danger: #ef4444;
            --danger-light: #f87171;
            --warning: #f59e0b;
            --warning-light: #fbbf24;
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-card: #1e293b;
            --text-primary: #f1f5f9;
            --text-secondary: #cbd5e1;
            --text-tertiary: #94a3b8;
            --border-color: #334155;
            --border-light: #475569;
            --gradient-primary: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
            --gradient-success: linear-gradient(135deg, #10b981 0%, #34d399 100%);
            --gradient-danger: linear-gradient(135deg, #ef4444 0%, #f87171 100%);
        }
    }
    
    /* Global styles */
    .stApp {
        background: var(--bg-primary);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container */
    .main > div {
        padding: 2rem 1rem;
        max-width: 1400px;
        margin: 0 auto;
    }
    
    /* Professional Header */
    .professional-header {
        background: var(--gradient-primary);
        border-radius: 20px;
        padding: 3rem 2.5rem;
        margin-bottom: 2.5rem;
        box-shadow: var(--shadow-xl);
        position: relative;
        overflow: hidden;
    }
    
    .professional-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.05'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
        opacity: 0.4;
    }
    
    .header-content {
        position: relative;
        z-index: 1;
    }
    
    .header-icon {
        width: 64px;
        height: 64px;
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 1rem;
        border: 2px solid rgba(255, 255, 255, 0.3);
    }
    
    .header-icon i {
        font-size: 2rem;
        color: white;
    }
    
    .header-title {
        font-family: 'Poppins', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        color: white;
        margin: 0 0 0.5rem 0;
        letter-spacing: -0.025em;
    }
    
    .header-subtitle {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.125rem;
        margin: 0;
        font-weight: 400;
    }
    
    /* Section containers */
    .section-container {
        background: var(--bg-card);
        border-radius: 16px;
        padding: 2rem;
        border: 1px solid var(--border-color);
        box-shadow: var(--shadow-sm);
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .section-container:hover {
        box-shadow: var(--shadow-md);
        border-color: var(--primary-light);
    }
    
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid var(--border-light);
    }
    
    .section-icon {
        width: 40px;
        height: 40px;
        background: var(--gradient-primary);
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: var(--shadow-sm);
    }
    
    .section-icon i {
        color: white;
        font-size: 1.25rem;
    }
    
    .section-title {
        font-family: 'Poppins', sans-serif;
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 0;
    }
    
    /* Professional form inputs */
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        border-radius: 10px !important;
        border: 2px solid var(--border-color) !important;
        background: var(--bg-primary) !important;
        color: var(--text-primary) !important;
        padding: 0.75rem 1rem !important;
        font-size: 0.9375rem !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
        box-shadow: var(--shadow-xs) !important;
    }
    
    .stNumberInput > div > div > input:hover,
    .stSelectbox > div > div > select:hover {
        border-color: var(--primary-light) !important;
    }
    
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 4px rgba(26, 86, 219, 0.1) !important;
        outline: none !important;
    }
    
    /* Labels */
    label {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
        font-size: 0.875rem !important;
        margin-bottom: 0.5rem !important;
        text-transform: uppercase;
        letter-spacing: 0.025em;
    }
    
    /* Professional Button */
    .stButton > button {
        background: var(--gradient-primary) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 1rem 2rem !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        box-shadow: var(--shadow-lg) !important;
        transition: all 0.3s ease !important;
        width: 100%;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        font-family: 'Poppins', sans-serif;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-xl) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Professional metric cards */
    .pro-metric-card {
        background: var(--bg-card);
        border-radius: 16px;
        padding: 2rem;
        border: 1px solid var(--border-color);
        box-shadow: var(--shadow-md);
        transition: all 0.3s ease;
        height: 100%;
        position: relative;
        overflow: hidden;
    }
    
    .pro-metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: var(--gradient-primary);
    }
    
    .pro-metric-card:hover {
        transform: translateY(-4px);
        box-shadow: var(--shadow-xl);
    }
    
    .metric-icon-wrapper {
        width: 56px;
        height: 56px;
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 1rem;
        box-shadow: var(--shadow-sm);
    }
    
    .metric-icon-wrapper.decision {
        background: linear-gradient(135deg, #1a56db20, #1a56db10);
    }
    
    .metric-icon-wrapper.success {
        background: linear-gradient(135deg, #05966920, #05966910);
    }
    
    .metric-icon-wrapper.danger {
        background: linear-gradient(135deg, #dc262620, #dc262610);
    }
    
    .metric-icon-wrapper i {
        font-size: 1.5rem;
    }
    
    .metric-icon-wrapper.decision i {
        color: var(--primary);
    }
    
    .metric-icon-wrapper.success i {
        color: var(--success);
    }
    
    .metric-icon-wrapper.danger i {
        color: var(--danger);
    }
    
    .metric-label-pro {
        font-size: 0.8125rem;
        color: var(--text-secondary);
        font-weight: 600;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .metric-value-pro {
        font-family: 'Poppins', sans-serif;
        font-size: 2.25rem;
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1;
        margin-bottom: 0.75rem;
    }
    
    .metric-description {
        font-size: 0.875rem;
        color: var(--text-secondary);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .metric-description i {
        font-size: 1rem;
    }
    
    .metric-description.positive {
        color: var(--success);
    }
    
    .metric-description.negative {
        color: var(--danger);
    }
    
    /* Professional progress bar */
    .progress-container {
        margin-top: 1rem;
    }
    
    .progress-bar-pro {
        width: 100%;
        height: 12px;
        background: var(--border-light);
        border-radius: 9999px;
        overflow: hidden;
        position: relative;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.06);
    }
    
    .progress-fill-pro {
        height: 100%;
        border-radius: 9999px;
        transition: width 0.6s ease;
        position: relative;
        overflow: hidden;
    }
    
    .progress-fill-pro::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(90deg, 
            rgba(255,255,255,0) 0%, 
            rgba(255,255,255,0.3) 50%, 
            rgba(255,255,255,0) 100%);
        animation: shimmer 2s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    .progress-fill-pro.success-bar {
        background: var(--gradient-success);
    }
    
    .progress-fill-pro.danger-bar {
        background: var(--gradient-danger);
    }
    
    /* Result container */
    .result-container-pro {
        background: var(--bg-card);
        border-radius: 20px;
        padding: 2.5rem;
        border: 1px solid var(--border-color);
        box-shadow: var(--shadow-lg);
        margin: 2rem 0;
    }
    
    .result-header-pro {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 2rem;
    }
    
    .result-icon-large {
        width: 64px;
        height: 64px;
        border-radius: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: var(--gradient-primary);
        box-shadow: var(--shadow-md);
    }
    
    .result-icon-large i {
        font-size: 2rem;
        color: white;
    }
    
    .result-title {
        font-family: 'Poppins', sans-serif;
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0;
    }
    
    /* Professional status badge */
    .status-badge-pro {
        display: inline-flex;
        align-items: center;
        gap: 0.75rem;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        font-weight: 700;
        font-size: 1.125rem;
        margin-bottom: 2rem;
        box-shadow: var(--shadow-md);
        font-family: 'Poppins', sans-serif;
    }
    
    .status-badge-pro i {
        font-size: 1.5rem;
    }
    
    .status-badge-pro.approved {
        background: var(--gradient-success);
        color: white;
    }
    
    .status-badge-pro.rejected {
        background: var(--gradient-danger);
        color: white;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: var(--bg-secondary) !important;
        border-right: 1px solid var(--border-color) !important;
        padding: 2rem 1rem;
    }
    
    /* Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes scaleIn {
        from {
            opacity: 0;
            transform: scale(0.95);
        }
        to {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    .section-container, .pro-metric-card, .result-container-pro {
        animation: fadeInUp 0.4s ease;
    }
    
    .professional-header {
        animation: scaleIn 0.5s ease;
    }
    
    /* Column spacing */
    [data-testid="column"] {
        padding: 0 0.5rem;
    }
    .result-container {
        animation: fadeIn 0.3s ease;
    }
    
    .result-container {
        background: var(--bg-secondary);
        border-radius: 16px;
        padding: 2rem;
        border: 1px solid var(--border-color);
        box-shadow: var(--shadow);
        margin-top: 2rem;
    }
    .result-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 1.5rem;
    }
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.875rem;
        margin-bottom: 1rem;
    }
    .status-badge.approved {
        background: rgba(16, 185, 129, 0.1);
        color: var(--success);
        border: 1px solid var(--success);
    }
    
    .status-badge.rejected {
        background: rgba(239, 68, 68, 0.1);
        color: var(--danger);
        border: 1px solid var(--danger);
    }
            
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .professional-header {
            padding: 2rem 1.5rem;
        }
        
        .header-title {
            font-size: 1.75rem;
        }
        
        .section-container {
            padding: 1.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    model = joblib.load("loan_status_predictor.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    return model, preprocessor

def predict_loan_status(applicant_data, model, preprocessor):
    processed_data = preprocessor.transform(applicant_data)
    prediction = model.predict(processed_data)
    probabilities = model.predict_proba(processed_data)
    
    return {
        'status': prediction[0],
        'probability_approved': probabilities[0][1],
        'probability_rejected': probabilities[0][0]
    }

# Professional Header
st.markdown("""
<div class="professional-header">
    <div class="header-content">
        <div class="header-icon">
            <i class="fas fa-university"></i>
        </div>
        <h1 class="header-title">Loan Approval Predictor</h1>
        <p class="header-subtitle">Advanced AI-powered credit assessment and loan approval prediction system</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Load models
try:
    model, preprocessor = load_models()
except:
    st.error("⚠️ Error loading the model. Please check if the model files exist.")
    st.stop()

# Input sections in 3 columns
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="section-container">
        <div class="section-header">
            <div class="section-icon">
                <i class="fas fa-user-tie"></i>
            </div>
            <div class="section-title">Applicant Profile</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    person_age = st.number_input("Age", min_value=18, max_value=100, value=30, key="age")
    person_income = st.number_input("Annual Income ($)", min_value=0, value=50000, step=1000, key="income")
    person_education = st.selectbox(
        "Education Level",
        ['High School', 'Associate', 'Bachelor', 'Master', 'Doctorate'],
        key="education"
    )

with col2:
    st.markdown("""
    <div class="section-container">
        <div class="section-header">
            <div class="section-icon">
                <i class="fas fa-home"></i>
            </div>
            <div class="section-title">Housing & Credit</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    person_home_ownership = st.selectbox(
        "Home Ownership",
        ['RENT', 'OWN', 'MORTGAGE', 'OTHER'],
        key="ownership"
    )
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700, key="credit")
    cb_person_cred_hist_length = st.number_input(
        "Credit History (years)",
        min_value=0.0,
        max_value=50.0,
        value=5.0,
        step=0.5,
        key="history"
    )
    previous_loan_defaults = st.selectbox(
        "Previous Defaults",
        ['No', 'Yes'],
        key="defaults"
    )

with col3:
    st.markdown("""
    <div class="section-container">
        <div class="section-header">
            <div class="section-icon">
                <i class="fas fa-file-invoice-dollar"></i>
            </div>
            <div class="section-title">Loan Information</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    loan_intent = st.selectbox(
        "Loan Purpose",
        ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'],
        key="intent"
    )
    loan_amnt = st.number_input("Loan Amount ($)", min_value=0, value=10000, step=500, key="amount")
    loan_int_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0, value=8.5, step=0.1, key="rate")
    loan_percent_income = st.number_input(
        "Loan % of Income",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.01,
        key="percent"
    )

# Professional Predict button
st.markdown("<br>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_clicked = st.button("Predict", type="primary", use_container_width=True)

if predict_clicked:
    
    new_applicant = pd.DataFrame({
        'person_age': [person_age],
        'person_income': [person_income],
        'person_home_ownership': [person_home_ownership],
        'loan_intent': [loan_intent],
        'loan_amnt': [loan_amnt],   
        'loan_int_rate': [loan_int_rate],
        'loan_percent_income': [loan_percent_income],
        'credit_score': [credit_score],
        'person_education': [person_education],
        'previous_loan_defaults_on_file': [previous_loan_defaults],
        'cb_person_cred_hist_length': [cb_person_cred_hist_length]
    })

    result = predict_loan_status(new_applicant, model, preprocessor)
    
    # Result display
    status_text = "Approved" if result['status'] == 1 else "Rejected"
    status_class = "approved" if result['status'] == 1 else "rejected"
    status_icon_class = "fa-circle-check" if result['status'] == 1 else "fa-circle-xmark"
    
    status_icon = "✓" if result['status'] == 1 else "✗"
    
    st.markdown(f"""
    <div class="result-container">
        <div class="result-header">📊 Prediction Results</div>
        <span class="status-badge {status_class}">
            <span>{status_icon}</span>
            Loan {status_text}
        </span>
    </div>
    <br>
    """, unsafe_allow_html=True)
    
    # Professional Metrics in cards
    col1, col2, col3 = st.columns(3)
    
    confidence_level = "High" if max(result['probability_approved'], result['probability_rejected']) > 0.75 else "Moderate"
    confidence_icon = "fa-bolt" if confidence_level == "High" else "fa-gauge-high"
    
    with col1:
        st.markdown(f"""
        <div class="pro-metric-card">
            <div class="metric-icon-wrapper decision">
                <i class="fas fa-gavel"></i>
            </div>
            <div class="metric-label-pro">Final Decision</div>
            <div class="metric-value-pro">{status_text}</div>
            <div class="metric-description {'positive' if result['status'] == 1 else 'negative'}">
                <i class="fas {confidence_icon}"></i>
                {confidence_level} Confidence
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        approval_pct = result['probability_approved'] * 100
        st.markdown(f"""
        <div class="pro-metric-card">
            <div class="metric-icon-wrapper success">
                <i class="fas fa-thumbs-up"></i>
            </div>
            <div class="metric-label-pro">Approval Probability</div>
            <div class="metric-value-pro">{approval_pct:.1f}%</div>
            <div class="progress-container">
                <div class="progress-bar-pro">
                    <div class="progress-fill-pro success-bar" style="width: {approval_pct}%"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        rejection_pct = result['probability_rejected'] * 100
        st.markdown(f"""
        <div class="pro-metric-card">
            <div class="metric-icon-wrapper danger">
                <i class="fas fa-thumbs-down"></i>
            </div>
            <div class="metric-label-pro">Rejection Probability</div>
            <div class="metric-value-pro">{rejection_pct:.1f}%</div>
            <div class="progress-container">
                <div class="progress-bar-pro">
                    <div class="progress-fill-pro danger-bar" style="width: {rejection_pct}%"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
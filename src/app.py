# $env:PYTHONPATH="."
# streamlit run src/app.py
"""
🩺 Clinical Heart Disease AI - Streamlit Application
World-class medical AI interface for cardiovascular risk assessment.

Author: Ridwan Oladipo, MD | AI Specialist
"""

# ── Imports ────────────────────────────────────────────────────────────────────
from datetime import datetime

import streamlit as st
from markdown import markdown

from src.app_helpers import (
    load_custom_css, check_api_health, call_api_endpoint,
    create_risk_gauge, create_shap_waterfall, create_population_comparison,
    get_sample_patients, validate_patient_data, API_BASE_URL
)

# ================ 🛠 SIDEBAR TOGGLE ================
if 'sidebar_state' not in st.session_state:
    st.session_state.sidebar_state = 'expanded'

st.set_page_config(
    page_title="Clinical Heart Disease AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state=st.session_state.sidebar_state
)

if st.button("🩺", help="Toggle sidebar"):
    st.session_state.sidebar_state = (
        'collapsed' if st.session_state.sidebar_state == 'expanded' else 'expanded'
    )
    st.rerun()

st.markdown(
    '<div style="font-size:0.75rem; color:#6b7280; margin-top:-10px;">Menu</div>',
    unsafe_allow_html=True
)

# ================ 🔧 PAGE CONFIGURATION ================
load_custom_css()

# ================ 🏥 MAIN APPLICATION ================
def main():
    """Main Streamlit application."""

    # ---------- 📋 HEADER ----------
    st.markdown("""
    <div class="medical-header">
        <h1>🩺 Clinical Heart Disease AI</h1>
        <p>Advanced cardiovascular risk assessment with AI-powered explainability</p>
        <p><strong>By Ridwan Oladipo, MD | AI Specialist</strong></p>
    </div>
    """, unsafe_allow_html=True)

    # ---------- 🔧 API HEALTH CHECK ----------
    health_status = check_api_health()

    if not health_status:
        st.error("🚨 **API Connection Failed** - Please ensure the FastAPI service is running on localhost:8000")
        st.code("uvicorn src.api:app --reload", language="bash")
        st.stop()

    if not health_status.get('model_loaded', False):
        st.error("🚨 **Model Not Loaded** - Please check API logs")
        st.stop()

    st.success("✅ **System Online** - Model loaded and ready for predictions")

    # ---------- 🩺 PATIENT INPUT PANEL ----------
    st.markdown("## 🩺 Patient Input Panel")

    with st.container():
        # Sidebar for sample data
        with st.sidebar:
            st.markdown("### 🎯 Quick Demo")
            sample_patients = get_sample_patients()
            selected_sample = st.selectbox(
                "Load Sample Patient:",
                ["Custom Input"] + list(sample_patients.keys())
            )

            if st.button("🔄 Load Sample Data"):
                if selected_sample != "Custom Input":
                    for key, value in sample_patients[selected_sample].items():
                        st.session_state[key] = value
                    st.rerun()

            st.markdown("---")
            st.markdown("### ℹ️ About This Tool")
            st.markdown("""
            This AI system predicts coronary artery disease risk using:
            - **XGBoost** machine learning model
            - **SHAP** explainable AI 
            - **Clinical guidelines** integration
            - **Population comparisons**
            """)

            st.markdown("### 📊 Model Performance")
            if health_status:
                st.metric("ROC-AUC", "0.91")
                st.metric("Sensitivity", "97%")
                st.metric("Specificity", "71%")

        # Main input form
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 👤 Demographics & Vitals")

            age = st.slider("Age (years)", 18, 100,
                            st.session_state.get('age', 54),
                            help="Patient's age in years")

            sex = st.selectbox("Sex",
                               options=[0, 1],
                               format_func=lambda x: "Female" if x == 0 else "Male",
                               index=st.session_state.get('sex', 1))

            trestbps = st.slider("Resting Blood Pressure (mmHg)", 80, 300,
                                 st.session_state.get('trestbps', 132),
                                 help="Resting blood pressure measurement")

            chol = st.slider("Cholesterol (mg/dl)", 100, 600,
                             st.session_state.get('chol', 246),
                             help="Serum cholesterol level")

            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl",
                               options=[0, 1],
                               format_func=lambda x: "No" if x == 0 else "Yes",
                               index=st.session_state.get('fbs', 0))

            thalach = st.slider("Maximum Heart Rate", 60, 220,
                                st.session_state.get('thalach', 150),
                                help="Maximum heart rate achieved during exercise")

        with col2:
            st.markdown("### ❤️ Cardiac Assessment")

            cp = st.selectbox("Chest Pain Type",
                              options=[0, 1, 2, 3],
                              format_func=lambda x: ["Typical Angina", "Atypical Angina",
                                                     "Non-anginal Pain", "Asymptomatic"][x],
                              index=st.session_state.get('cp', 0))

            restecg = st.selectbox("Resting ECG",
                                   options=[0, 1, 2],
                                   format_func=lambda x: ["Normal", "ST-T Abnormality",
                                                          "LV Hypertrophy"][x],
                                   index=st.session_state.get('restecg', 0))

            exang = st.selectbox("Exercise Induced Angina",
                                 options=[0, 1],
                                 format_func=lambda x: "No" if x == 0 else "Yes",
                                 index=st.session_state.get('exang', 0))

            oldpeak = st.slider("ST Depression", 0.0, 10.0,
                                st.session_state.get('oldpeak', 1.0),
                                step=0.1,
                                help="ST depression induced by exercise")

            slope = st.selectbox("ST Slope",
                                 options=[0, 1, 2],
                                 format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x],
                                 index=st.session_state.get('slope', 1))

            ca = st.selectbox("Major Vessels (Angiography)",
                              options=[0, 1, 2, 3],
                              format_func=lambda x: f"{x} vessels",
                              index=st.session_state.get('ca', 0),
                              help="Number of major coronary vessels with significant stenosis")

            thal = st.selectbox("Thalassemia",
                                options=[1, 2, 3],
                                format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect"][x - 1],
                                index=st.session_state.get('thal', 2) - 1)

    patient_data = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }
    for k, v in patient_data.items():
        st.session_state[k] = v

    # ---------- 🔮 PREDICTION BUTTON ----------
    st.markdown("---")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🔮 **Analyze Cardiovascular Risk**",
                                   use_container_width=True, type="primary")

    if predict_button:
        # ---------- 📊 INSTANT RISK PANEL ----------
        with st.spinner("🧠 Analyzing patient data with AI..."):
            prediction_data, pred_error = call_api_endpoint("predict", patient_data)
            shap_data, shap_error = call_api_endpoint("shap", patient_data)
            positions_data, pos_error = call_api_endpoint("positions", patient_data)
            metrics_data, metrics_error = call_api_endpoint("metrics")

        if pred_error:
            st.error(f"❌ {pred_error}")
            return

        st.markdown("## 📊 Instant Risk Assessment")
        col1a, col1b = st.columns([1, 1])

        with col1a:
            st.markdown('<div class="risk-gauge-container">', unsafe_allow_html=True)
            risk_fig = create_risk_gauge(prediction_data['probability'], prediction_data['risk_class'])
            st.plotly_chart(risk_fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col1b:
            risk_class = prediction_data['risk_class']
            probability = prediction_data['probability']
            if "Low" in risk_class:
                box = "risk-low"
            elif "Moderate" in risk_class:
                box = "risk-moderate"
            else:
                box = "risk-high"
            st.markdown(f'<div class="{box}">{"🟩" if box=="risk-low" else "🟧" if box=="risk-moderate" else "🟥"} '
                        f'{risk_class}<br>{probability:.1%} Risk</div>', unsafe_allow_html=True)

            confidence_score = abs(probability - 0.5) * 2
            if confidence_score >= 0.6:
                confidence_level = "High"
            elif confidence_score >= 0.3:
                confidence_level = "Moderate"
            else:
                confidence_level = "Low"

            st.metric("Model Confidence", f"{confidence_level}",
                      help="How certain the model is about this prediction")

            if metrics_data:
                col_a, col_b = st.columns(2)
                col_a.metric("Sensitivity", f"{metrics_data['sensitivity']:.1%}")
                col_b.metric("Specificity", f"{metrics_data['specificity']:.1%}")

        # ---------- ⚕️ CLINICAL SUPPORT ----------
        st.markdown("## ⚕️ Clinical Decision Support")
        summary_html = markdown(prediction_data['clinical_summary'])
        st.markdown(f"""
        <div class="clinical-summary">
            <h4>🩺 Clinical Interpretation & Recommendations</h4>
            {summary_html}
        </div>
        """, unsafe_allow_html=True)

        if positions_data and not pos_error:
            st.markdown("### 📋 Clinical Guidelines Assessment")
            guidelines = positions_data['guideline_categories']

            gcol1, gcol2 = st.columns(2)

        with gcol1:
            st.markdown(
                '<div class="medical-panel"><div class="medical-panel-header">🩸 Cardiovascular Risk Factors</div>',
                unsafe_allow_html=True)
            st.markdown(f"**Blood Pressure:** {guidelines.get('blood_pressure', 'N/A')}",
                        help="AHA/ACC Guidelines: Normal <120, Elevated 120-129, Stage 1: 130-139, Stage 2: ≥140 mmHg")
            st.markdown(f"**Cholesterol:** {guidelines.get('cholesterol', 'N/A')}",
                        help="NCEP ATP III Guidelines: Desirable <200, Borderline 200-239, High ≥240 mg/dl")
            st.markdown('</div>', unsafe_allow_html=True)

        with gcol2:
            st.markdown('<div class="medical-panel"><div class="medical-panel-header">🏃 Functional Assessment</div>',
                        unsafe_allow_html=True)
            st.markdown(f"**Age Category:** {guidelines.get('age_risk', 'N/A')}",
                        help="Cardiovascular risk increases with age, especially >65 years")
            st.markdown(f"**Exercise Capacity:** {guidelines.get('exercise_capacity', 'N/A')}",
                        help="Based on % of age-predicted max HR (220-age). Poor capacity indicates cardiac dysfunction")
            st.markdown('</div>', unsafe_allow_html=True)


        # ---------- 🧠 SHAP DASHBOARD ----------
        if shap_data and not shap_error:
            st.markdown("## 🧠 Explainable AI Dashboard")
            s_col1, s_col2 = st.columns([2, 1])

            with s_col1:
                shap_fig = create_shap_waterfall(shap_data)
                st.plotly_chart(shap_fig, use_container_width=True)

            with s_col2:
                st.markdown("### 🎯 Top Risk Drivers")
                for feature in shap_data['top_features'][:5]:
                    impact_class = "feature-increase" if feature['shap_value'] > 0 else "feature-decrease"
                    st.markdown(f"""
                    <div class="feature-card {impact_class}">
                        <strong>{feature['feature'].replace('_', ' ').title()}</strong><br>
                        <small>{feature['clinical_explanation']}</small><br>
                        <span style="font-size: 0.8em;">{feature['impact']}: {abs(feature['shap_value']):.3f}</span>
                    </div>
                    """, unsafe_allow_html=True)

            if positions_data and not pos_error:
                st.markdown("### 📈 Population Comparison")
                pcol1, pcol2 = st.columns([1, 1])

                with pcol1:
                    pop_fig = create_population_comparison(positions_data)
                    st.plotly_chart(pop_fig, use_container_width=True)

                with pcol2:
                    st.markdown("#### 📊 Percentile Rankings")
                    feature_positions = positions_data['feature_positions']
                    for feature, data in feature_positions.items():
                        if feature in ['age', 'trestbps', 'chol', 'thalach']:
                            st.write(f"**{feature.replace('_', ' ').title()}:** {data['comparison']}")

        # ---------- 💡 INSIGHTS ----------
        st.markdown("---")
        with st.expander("💡 **Model Insights & Clinical Context**", expanded=False):
            st.markdown("""
            <div class="insight-box">
                <h5>⚠️ Important Clinical Context</h5>
                <p>This model reflects patterns from the Cleveland Heart Disease dataset, which showed some atypical findings compared to textbook cardiology:</p>
                <ul>
                    <li><strong>Typical Angina Paradox:</strong> Patients with classic chest pain symptoms were often found to have <em>lower</em> disease rates, likely due to rapid diagnostic workup and exclusion.</li>
                    <li><strong>Exercise Response Patterns:</strong> Some patients with confirmed CAD showed better exercise tolerance, reflecting the complex referral patterns in this cohort.</li>
                    <li><strong>Thalassemia Findings:</strong> Reversible defects appeared more frequently in the no-disease group, highlighting dataset-specific diagnostic workflows.</li>
                </ul>
                <p><strong>Clinical Recommendation:</strong> Always interpret AI predictions within the full clinical context, patient history, and current guidelines. This tool is designed to support, not replace, clinical judgment.</p>
            </div>
            """, unsafe_allow_html=True)

            if metrics_data:
                st.markdown("#### 📊 Detailed Model Performance")
                colA, colB, colC, colD = st.columns(4)

                colA.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{metrics_data['roc_auc']:.3f}</div>
                    <div class="metric-label">ROC-AUC</div>
                </div>
                """, unsafe_allow_html=True)

                colB.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{metrics_data['sensitivity']:.1%}</div>
                    <div class="metric-label">Sensitivity</div>
                </div>
                """, unsafe_allow_html=True)

                colC.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{metrics_data['specificity']:.1%}</div>
                    <div class="metric-label">Specificity</div>
                </div>
                """, unsafe_allow_html=True)

                colD.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{metrics_data['ppv']:.1%}</div>
                    <div class="metric-label">PPV</div>
                </div>
                """, unsafe_allow_html=True)

                st.write(
                    f"• **High Sensitivity ({metrics_data['sensitivity']:.1%}):** Excellent at detecting disease - minimizes missed diagnoses")
                st.write(
                    f"• **Moderate Specificity ({metrics_data['specificity']:.1%}):** May flag some healthy patients for further evaluation")
                st.write(
                    f"• **Positive Predictive Value ({metrics_data['ppv']:.1%}):** When model predicts disease, it's correct {metrics_data['ppv']:.1%} of the time")
                st.write(
                    f"• **Negative Predictive Value ({metrics_data['npv']:.1%}):** When model predicts no disease, it's correct {metrics_data['npv']:.1%} of the time")

    # ---------- 💼 FOOTER ----------
    st.markdown("---")

    st.warning(
        "⚠️ This AI tool is for **decision support only** and is not a substitute for professional clinical judgment. "
        "All medical decisions should be made in consultation with qualified healthcare providers. "
        "The predictions are based on statistical patterns and should always be interpreted within the full clinical context."
    )

    fcol1, fcol2, fcol3 = st.columns(3)

    with fcol1:
        st.subheader("🔗 Links")
        st.markdown("[GitHub Repository](https://github.com/dr-ridwanoladipo/cardio-ai-predictor)")
        st.markdown(
            "[Kaggle Notebook](https://www.kaggle.com/code/ridwanoladipoai/medical-ai-for-heart-disease-eda-engineering)")
        st.markdown("[API Documentation](https://cardio.mednexai.com/docs)")

    with fcol2:
        st.subheader("📊 Data Source")
        st.write("Cleveland Heart Disease Dataset")
        st.write("UCI Machine Learning Repository")

    with fcol3:
        st.subheader("🏥 Model Info")
        st.write("XGBoost with SHAP Explainability")
        st.write("Deployed on AWS ECS Fargate")

    st.caption(
        "© 2025 Ridwan Oladipo, MD | Medical AI Specialist  \n"
        "**cardio.mednexai.com** | Advanced Healthcare AI Solutions"
    )

# ================ 🚀 ENTRY POINT ================
if __name__ == "__main__":
    main()



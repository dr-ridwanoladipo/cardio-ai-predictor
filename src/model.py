"""
🩺 Clinical Heart Disease AI - Model Module
Reusable prediction and explanation functions for both FastAPI and Streamlit.

Author: Ridwan Oladipo, MD | AI Specialist
"""

# ── Imports ────────────────────────────────────────────────────────────────────
import json
import logging
import warnings
from typing import Dict, List, Tuple, Any

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#                            Core Predictor Class
# ═══════════════════════════════════════════════════════════════════════════════
class HeartDiseasePredictor:
    """Clinical Heart Disease Prediction and Explanation System."""

    def __init__(self):
        """Create empty placeholders; populate later via load_artifacts()."""
        self.model = None
        self.scaler = None
        self.explainer = None
        self.feature_names = None
        self.numerical_features = None
        self.metrics = None
        self.cohort_data = None
        self.feature_explanations = self._get_clinical_explanations()

    # ───────────────────────────────────────────────────────────────────────────
    # Artifact loading
    # ───────────────────────────────────────────────────────────────────────────
    def load_artifacts(self) -> bool:
        """Load model, scaler, explainer, feature lists, metrics, cohort data."""
        try:
            logger.info("Loading model artifacts...")

            # Core model components
            self.model = joblib.load("artifacts/xgb_model.pkl")
            self.scaler = joblib.load("artifacts/scaler.pkl")
            self.explainer = joblib.load("artifacts/shap_explainer.pkl")

            # Feature metadata
            with open("artifacts/feature_names.json") as f:
                self.feature_names = json.load(f)
            with open("artifacts/numerical_features.json") as f:
                self.numerical_features = json.load(f)

            # Metrics
            with open("artifacts/metrics.json") as f:
                self.metrics = json.load(f)

            # Cohort data for population comparisons
            self.cohort_data = pd.read_csv("data/heartdisease_processed.csv")

            logger.info("All artifacts loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load artifacts: {str(e)}")
            return False

    # ───────────────────────────────────────────────────────────────────────────
    # Static clinical explanations
    # ───────────────────────────────────────────────────────────────────────────
    def _get_clinical_explanations(self) -> Dict[str, str]:
        """Return clinical explanations for each feature."""
        return {
            'age': "Patient's age in years - cardiovascular risk increases with age",
            'sex': "Gender (0=Female, 1=Male) - males have higher CAD risk before age 65",
            'cp': "Chest pain type - typical angina suggests coronary artery disease",
            'trestbps': "Resting blood pressure (mmHg) - hypertension damages arteries",
            'chol': "Serum cholesterol (mg/dl) - high levels contribute to atherosclerosis",
            'fbs': "Fasting blood sugar >120 mg/dl - diabetes accelerates CAD",
            'restecg': "Resting ECG findings - abnormalities may indicate heart disease",
            'thalach': "Maximum heart rate achieved - lower values may indicate disease",
            'exang': "Exercise-induced angina - chest pain during stress suggests CAD",
            'oldpeak': "ST depression during exercise - measure of cardiac ischemia",
            'slope': "ST segment slope - downsloping suggests significant CAD",
            'ca': "Number of major coronary vessels - more blockages = higher risk",
            'thal': "Thalassemia stress test - reversible defects indicate active ischemia",
            'age_group': "Age risk category (0=Young, 1=Middle-aged, 2=Older, 3=Elderly)",
            'cp_severity': "Chest pain severity score (1=Asymptomatic, 4=Typical Angina)",
            'bp_category': "Blood pressure category per AHA guidelines",
            'chol_risk': "Cholesterol risk level per NCEP guidelines",
            'hr_achievement': "Heart rate achievement ratio (% of age-predicted maximum)",
            'age_chol_interaction': "Combined age and cholesterol risk factor",
            'cp_exang_interaction': "Combined chest pain and exercise angina risk"
        }

    # ───────────────────────────────────────────────────────────────────────────
    # Derived feature engineering
    # ───────────────────────────────────────────────────────────────────────────
    def compute_auto_fields(self, patient_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Compute derived clinical fields from basic patient inputs."""
        enhanced_patient = patient_dict.copy()

        # Age group
        age = patient_dict['age']
        if age < 40:
            enhanced_patient['age_group'] = 0  # Young
        elif age < 55:
            enhanced_patient['age_group'] = 1  # Middle-aged
        elif age < 65:
            enhanced_patient['age_group'] = 2  # Older
        else:
            enhanced_patient['age_group'] = 3  # Elderly

        # Chest pain severity
        cp_severity_map = {0: 4, 1: 3, 2: 2, 3: 1}
        enhanced_patient['cp_severity'] = cp_severity_map[patient_dict['cp']]

        # Blood pressure category
        bp = patient_dict['trestbps']
        if bp < 120:
            enhanced_patient['bp_category'] = 0
        elif bp < 130:
            enhanced_patient['bp_category'] = 1
        elif bp < 140:
            enhanced_patient['bp_category'] = 2
        else:
            enhanced_patient['bp_category'] = 3

        # Cholesterol risk category
        chol = patient_dict['chol']
        if chol < 200:
            enhanced_patient['chol_risk'] = 0
        elif chol < 240:
            enhanced_patient['chol_risk'] = 1
        else:
            enhanced_patient['chol_risk'] = 2

        # Heart rate achievement
        enhanced_patient['hr_achievement'] = patient_dict['thalach'] / (220 - age)

        # Interaction terms
        enhanced_patient['age_chol_interaction'] = age * chol / 1000
        enhanced_patient['cp_exang_interaction'] = patient_dict['cp'] * patient_dict['exang']

        return enhanced_patient

    # ───────────────────────────────────────────────────────────────────────────
    # Prediction & clinical summary
    # ───────────────────────────────────────────────────────────────────────────
    def predict_proba(self, input_df: pd.DataFrame) -> Tuple[float, str, str]:
        """Predict probability and return risk classification + summary."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_artifacts() first.")

        # Scale numerical features
        input_scaled = input_df.copy()
        input_scaled[self.numerical_features] = self.scaler.transform(input_df[self.numerical_features])

        probability = self.model.predict_proba(input_scaled)[0, 1]

        # Classify risk level
        if probability < 0.3:
            risk_class = "Low Risk"
        elif probability < 0.7:
            risk_class = "Moderate Risk"
        else:
            risk_class = "High Risk"

        # Clinical summary
        clinical_summary = self._generate_clinical_summary(probability, input_df.iloc[0])

        return probability, risk_class, clinical_summary

    def _generate_clinical_summary(self, probability: float, patient_data: pd.Series) -> str:
        """Generate a clinical summary based on prediction and key features."""
        risk_pct = probability * 100
        high_risk_factors = []

        if patient_data['ca'] >= 2:
            high_risk_factors.append("multi-vessel coronary disease")
        if patient_data['thal'] == 3:
            high_risk_factors.append("reversible perfusion defect")
        if patient_data['exang'] == 1:
            high_risk_factors.append("exercise-induced angina")
        if patient_data['slope'] == 2:
            high_risk_factors.append("downsloping ST segment")
        if patient_data['cp'] == 0:
            high_risk_factors.append("typical angina")

        if probability >= 0.7:
            summary = (
                f"🟥 **High Risk** ({risk_pct:.1f}%): likely driven by "
                f"{', '.join(high_risk_factors[:2]) if high_risk_factors else 'multiple adverse factors'}.\n\n"
                "🚨 **Immediate actions:**\n\n"
                "- Arrange **urgent cardiology referral** for angiography or coronary CTA.\n\n"
                "- Initiate **high-intensity statin therapy** (≥50% LDL reduction).\n\n"
                "- Consider **antiplatelet agents** per guideline indications.\n\n"
                "- Adopt a **Mediterranean or DASH diet** (vegetables, nuts, fish; sodium <1,500 mg/day).\n\n"
                "- Target **≥150 min/week of moderate aerobic exercise**.\n\n"
                "- Ensure smoking cessation, weight management, stress reduction."
            )
        elif probability >= 0.3:
            summary = (
                f"🟧 **Moderate Risk** ({risk_pct:.1f}%): further evaluation with "
                "**stress echocardiography or coronary CTA** is recommended.\n\n"
                "📋 **Lifestyle prescriptions:**\n\n"
                "- Mediterranean or DASH diet, sodium <1,500 mg/day.\n\n"
                "- **≥150 min/week of moderate aerobic activity**.\n\n"
                "- Start or optimize **moderate-intensity statin therapy** if LDL elevated.\n\n"
                "- Maintain BP <130/80 mmHg and monitor glucose regularly."
            )
        else:
            summary = (
                f"🟩 **Low Risk** ({risk_pct:.1f}%): continue a **heart-healthy lifestyle**.\n\n"
                "- Mediterranean/DASH diet.\n\n"
                "- Regular exercise (≥150 min/week).\n\n"
                "- BP <130/80 mmHg.\n\n"
                "- Strict avoidance of smoking.\n\n"
                "- Annual labs (lipids, BP, glucose) to sustain excellent cardiovascular health."
            )

        return summary

    # ───────────────────────────────────────────────────────────────────────────
    # SHAP explanations
    # ───────────────────────────────────────────────────────────────────────────
    def get_shap_values(self, input_df: pd.DataFrame) -> np.ndarray:
        """Calculate SHAP values for explanation."""
        if self.explainer is None:
            raise ValueError("SHAP explainer not loaded. Call load_artifacts() first.")

        input_scaled = input_df.copy()
        input_scaled[self.numerical_features] = self.scaler.transform(input_df[self.numerical_features])
        shap_values = self.explainer.shap_values(input_scaled)

        return shap_values

    def get_top_features(self, shap_values: np.ndarray, top_n: int = 5) -> List[Dict[str, Any]]:
        """Return top contributing features with explanations."""
        feature_contributions = []
        for i, feature_name in enumerate(self.feature_names):
            contribution = shap_values[0, i]
            feature_contributions.append({
                'feature': feature_name,
                'shap_value': float(contribution),
                'abs_contribution': abs(contribution),
                'impact': 'Increases Risk' if contribution > 0 else 'Decreases Risk',
                'clinical_explanation': self.feature_explanations.get(feature_name, "Clinical factor"),
                'magnitude': 'High' if abs(contribution) > 0.1 else 'Moderate' if abs(contribution) > 0.05 else 'Low'
            })
        feature_contributions.sort(key=lambda x: x['abs_contribution'], reverse=True)
        return feature_contributions[:top_n]

    # ───────────────────────────────────────────────────────────────────────────
    # Cohort comparisons
    # ───────────────────────────────────────────────────────────────────────────
    def get_feature_positions(self, input_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Get patient's feature values relative to the cohort."""
        patient_positions = {}
        patient_data = input_df.iloc[0]
        comparison_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'hr_achievement']

        for feature in comparison_features:
            if feature in patient_data and feature in self.cohort_data.columns:
                patient_value = patient_data[feature]
                cohort_values = self.cohort_data[feature].dropna()
                percentile = (cohort_values <= patient_value).mean() * 100

                if percentile >= 80:
                    comparison = f"Higher than {percentile:.0f}% of similar patients"
                elif percentile >= 60:
                    comparison = f"Above average ({percentile:.0f}th percentile)"
                elif percentile >= 40:
                    comparison = f"Average range ({percentile:.0f}th percentile)"
                elif percentile >= 20:
                    comparison = f"Below average ({percentile:.0f}th percentile)"
                else:
                    comparison = f"Lower than {100 - percentile:.0f}% of similar patients"

                patient_positions[feature] = {
                    'value': float(patient_value),
                    'percentile': float(percentile),
                    'comparison': comparison,
                    'cohort_mean': float(cohort_values.mean()),
                    'cohort_std': float(cohort_values.std())
                }

        return patient_positions

    # ───────────────────────────────────────────────────────────────────────────
    # Misc utilities
    # ───────────────────────────────────────────────────────────────────────────
    def get_metrics(self) -> Dict[str, Any]:
        """Return model performance metrics."""
        if self.metrics is None:
            raise ValueError("Metrics not loaded. Call load_artifacts() first.")
        return self.metrics

    def get_guideline_categories(self, patient_data: Dict[str, Any]) -> Dict[str, str]:
        """Return clinical guideline interpretations for key values."""
        categories = {}

        # Blood Pressure (AHA Guidelines)
        bp = patient_data.get('trestbps', 0)
        if bp < 120:
            categories['blood_pressure'] = "Normal (<120 mmHg)"
        elif bp < 130:
            categories['blood_pressure'] = "Elevated (120-129 mmHg)"
        elif bp < 140:
            categories['blood_pressure'] = "Stage 1 Hypertension (130-139 mmHg)"
        else:
            categories['blood_pressure'] = "Stage 2 Hypertension (≥140 mmHg)"

        # Cholesterol (NCEP Guidelines)
        chol = patient_data.get('chol', 0)
        if chol < 200:
            categories['cholesterol'] = "Desirable (<200 mg/dl)"
        elif chol < 240:
            categories['cholesterol'] = "Borderline High (200-239 mg/dl)"
        else:
            categories['cholesterol'] = "High (≥240 mg/dl)"

        # Age Risk Category
        age = patient_data.get('age', 0)
        if age < 40:
            categories['age_risk'] = "Young Adult (<40 years)"
        elif age < 55:
            categories['age_risk'] = "Middle-aged (40-54 years)"
        elif age < 65:
            categories['age_risk'] = "Older Adult (55-64 years)"
        else:
            categories['age_risk'] = "Elderly (≥65 years)"

        # Heart Rate Achievement
        hr_achievement = patient_data.get('hr_achievement', 0)
        if hr_achievement >= 0.85:
            categories['exercise_capacity'] = "Excellent (≥85% predicted max HR)"
        elif hr_achievement >= 0.75:
            categories['exercise_capacity'] = "Good (75-84% predicted max HR)"
        elif hr_achievement >= 0.65:
            categories['exercise_capacity'] = "Fair (65-74% predicted max HR)"
        else:
            categories['exercise_capacity'] = "Poor (<65% predicted max HR)"

        return categories


# ═══════════════════════════════════════════════════════════════════════════════
#                        Global instance + Convenience wrappers
# ═══════════════════════════════════════════════════════════════════════════════
predictor = HeartDiseasePredictor()


def initialize_model() -> bool:
    """Initialize the global predictor instance."""
    return predictor.load_artifacts()


def predict_heart_disease(patient_data: Dict[str, Any]) -> Dict[str, Any]:
    """End-to-end helper: derive features → predict → explain → compare."""
    enhanced_patient = predictor.compute_auto_fields(patient_data)
    input_df = pd.DataFrame([enhanced_patient])[predictor.feature_names]

    probability, risk_class, clinical_summary = predictor.predict_proba(input_df)
    shap_values = predictor.get_shap_values(input_df)
    top_features = predictor.get_top_features(shap_values)
    feature_positions = predictor.get_feature_positions(input_df)
    guideline_categories = predictor.get_guideline_categories(enhanced_patient)
    model_metrics = predictor.get_metrics()

    return {
        'prediction': {
            'probability': float(probability),
            'risk_class': risk_class,
            'clinical_summary': clinical_summary
        },
        'explanations': {
            'shap_values': shap_values.tolist(),
            'top_features': top_features
        },
        'comparisons': {
            'feature_positions': feature_positions,
            'guideline_categories': guideline_categories
        },
        'model_info': {
            'metrics': model_metrics,
            'feature_names': predictor.feature_names
        }
    }


# ── Back-compat helper functions (signatures preserved) ────────────────────────
def load_artifacts():
    """Load model artifacts (backwards compatibility)."""
    return predictor.load_artifacts()


def compute_auto_fields(patient_dict):
    """Compute auto fields (backwards compatibility)."""
    return predictor.compute_auto_fields(patient_dict)


def predict_proba(input_df):
    """Predict probability (backwards compatibility)."""
    return predictor.predict_proba(input_df)


def get_shap_values(input_df):
    """Get SHAP values (backwards compatibility)."""
    return predictor.get_shap_values(input_df)


def get_top_features(shap_values):
    """Get top features (backwards compatibility)."""
    return predictor.get_top_features(shap_values)


def get_feature_positions(input_df, cohort_df=None):
    """Get feature positions (backwards compatibility)."""
    # cohort_df kept for legacy calls; predictor already owns cohort_data.
    return predictor.get_feature_positions(input_df)


def get_metrics():
    """Get model metrics (backwards compatibility)."""
    return predictor.get_metrics()

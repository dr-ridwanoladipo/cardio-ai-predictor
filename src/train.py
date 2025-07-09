"""
🩺 Clinical Heart Disease AI - Training Pipeline
Complete training script that replicates notebook analysis and saves all artifacts.

Author: Ridwan Oladipo, MD | AI Specialist
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import optuna
import joblib
import json
import logging
import os
import warnings
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score
)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('outputs/train.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def setup_directories():
    """Create necessary directories if they don't exist."""
    directories = ['artifacts', 'outputs', 'data']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    logger.info("Directories verified/created")


def load_and_clean_data():
    """Load raw data and apply clinical corrections."""
    logger.info("Loading raw dataset...")
    df = pd.read_csv('data/heartdisease.csv')

    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Missing values: {df.isnull().sum().sum()}")

    # Clinical data corrections (from notebook)
    logger.info("Applying clinical data corrections...")

    # Fix thal (thalassemia) - replace 0 with mode
    num_thal_errors = df[df['thal'] == 0].shape[0]
    if num_thal_errors > 0:
        thal_mode = df['thal'][df['thal'] != 0].mode()[0]
        df['thal'] = df['thal'].replace(0, thal_mode)
        logger.info(f"Corrected {num_thal_errors} 'thal' rows by replacing 0 with mode: {thal_mode}")

    # Fix ca (coronary arteries) - replace 4 with mode
    num_ca_errors = df[df['ca'] == 4].shape[0]
    if num_ca_errors > 0:
        ca_mode = df['ca'][df['ca'] != 4].mode()[0]
        df['ca'] = df['ca'].replace(4, ca_mode)
        logger.info(f"Corrected {num_ca_errors} 'ca' rows by replacing 4 with mode: {ca_mode}")

    # Save cleaned data
    df.to_csv('data/heartdisease_cleaned.csv', index=False)
    logger.info("Cleaned dataset saved")

    return df


def create_clinical_features(df):
    """Apply clinical feature engineering from notebook."""
    logger.info("Creating clinical features...")

    df_processed = df.copy()

    # Age groups (cardiovascular risk stratification)
    def create_age_groups(age):
        if age < 40:
            return 0  # Young
        elif age < 55:
            return 1  # Middle-aged
        elif age < 65:
            return 2  # Older
        else:
            return 3  # Elderly

    df_processed['age_group'] = df_processed['age'].apply(create_age_groups)

    # Chest Pain Severity Score (clinical relevance)
    def chest_pain_severity(cp):
        severity_map = {0: 4, 1: 3, 2: 2, 3: 1}  # Typical Angina=4, Asymptomatic=1
        return severity_map[cp]

    df_processed['cp_severity'] = df_processed['cp'].apply(chest_pain_severity)

    # Blood Pressure Categories (AHA Guidelines)
    def bp_category(bp):
        if bp < 120:
            return 0  # Normal
        elif bp < 130:
            return 1  # Elevated
        elif bp < 140:
            return 2  # Stage 1 Hypertension
        else:
            return 3  # Stage 2 Hypertension

    df_processed['bp_category'] = df_processed['trestbps'].apply(bp_category)

    # Cholesterol Risk Categories
    def chol_risk(chol):
        if chol < 200:
            return 0  # Desirable
        elif chol < 240:
            return 1  # Borderline High
        else:
            return 2  # High

    df_processed['chol_risk'] = df_processed['chol'].apply(chol_risk)

    # Maximum Heart Rate Achievement (% of age-predicted max)
    df_processed['hr_achievement'] = df_processed['thalach'] / (220 - df_processed['age'])

    # Create interaction features (clinically meaningful)
    df_processed['age_chol_interaction'] = df_processed['age'] * df_processed['chol'] / 1000
    df_processed['cp_exang_interaction'] = df_processed['cp'] * df_processed['exang']

    # Add target label for consistency
    df_processed['target_label'] = df_processed['target'].map({0: 'No Disease', 1: 'Disease'})

    logger.info("Clinical features created:")
    logger.info("- age_group, cp_severity, bp_category, chol_risk")
    logger.info("- hr_achievement, age_chol_interaction, cp_exang_interaction")

    # Save processed data
    df_processed.to_csv('data/heartdisease_processed.csv', index=False)
    logger.info("Processed dataset saved")

    return df_processed


def prepare_features_and_split(df):
    """Prepare features and create train/test split."""
    logger.info("Preparing features and splitting data...")

    # Separate features and target
    X = df.drop(['target', 'target_label'], axis=1)
    y = df['target']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale numerical features
    numerical_features = [
        'age', 'trestbps', 'chol', 'thalach', 'oldpeak',
        'hr_achievement', 'age_chol_interaction'
    ]

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    X_test_scaled[numerical_features] = scaler.transform(X_test[numerical_features])

    logger.info(f"Training set: {X_train_scaled.shape}")
    logger.info(f"Test set: {X_test_scaled.shape}")
    logger.info(
        f"Target distribution - No Disease: {(y_train == 0).sum()} ({(y_train == 0).sum() / len(y_train) * 100:.1f}%)")
    logger.info(
        f"Target distribution - Disease: {(y_train == 1).sum()} ({(y_train == 1).sum() / len(y_train) * 100:.1f}%)")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, numerical_features


def handle_class_imbalance(X_train, y_train):
    """Apply SMOTE if needed for class imbalance."""
    logger.info("Checking class balance...")

    minority_class_pct = min((y_train == 0).sum(), (y_train == 1).sum()) / len(y_train)

    if minority_class_pct < 0.4:
        logger.info(f"Applying SMOTE (minority class: {minority_class_pct:.2%})")
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        logger.info(f"After SMOTE: {X_train_balanced.shape[0]} samples")
        logger.info(f"Class 0: {(y_train_balanced == 0).sum()}, Class 1: {(y_train_balanced == 1).sum()}")
    else:
        logger.info(f"Classes well balanced ({minority_class_pct:.2%} minority), no SMOTE needed")
        X_train_balanced = X_train
        y_train_balanced = y_train

    return X_train_balanced, y_train_balanced


def optimize_xgboost(X_train, y_train):
    """Optimize XGBoost hyperparameters using Optuna."""
    logger.info("Starting Optuna hyperparameter optimization for XGBoost...")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def objective_xgb(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'eval_metric': 'logloss'
        }
        model = XGBClassifier(**params, random_state=42)
        score = cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring='roc_auc').mean()
        return score

    study_xgb = optuna.create_study(direction='maximize')
    study_xgb.optimize(objective_xgb, n_trials=50)
    best_params_xgb = study_xgb.best_params

    logger.info(f"Best XGBoost parameters: {best_params_xgb}")
    logger.info(f"Best ROC-AUC score: {study_xgb.best_value:.4f}")

    return best_params_xgb


def train_models(X_train, y_train, best_params_xgb):
    """Train individual models and ensemble."""
    logger.info("Training models...")

    # Initialize models
    xgb_model = XGBClassifier(**best_params_xgb, random_state=42, eval_metric='logloss')
    rf_model = RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_split=5, min_samples_leaf=2,
        random_state=42, class_weight='balanced'
    )
    lr_model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')

    # Create ensemble
    ensemble_model = VotingClassifier(
        estimators=[
            ('rf', rf_model),
            ('xgb', xgb_model),
            ('lr', lr_model)
        ],
        voting='soft'
    )

    models = {
        'Random Forest': rf_model,
        'XGBoost': xgb_model,
        'Logistic Regression': lr_model,
        'Ensemble': ensemble_model
    }

    # Cross-validation
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}

    logger.info("Cross-validation results:")
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring='roc_auc')
        cv_results[name] = scores
        logger.info(f"{name}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

    # Train final models
    logger.info("Training final models...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        logger.info(f"{name} training completed")

    return models, cv_results


def evaluate_models(models, X_test, y_test):
    """Evaluate all models and generate metrics."""
    logger.info("Evaluating models on test set...")

    predictions = {}
    probabilities = {}

    for name, model in models.items():
        predictions[name] = model.predict(X_test)
        probabilities[name] = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    results_df = pd.DataFrame()
    for name in models.keys():
        y_pred = predictions[name]
        y_proba = probabilities[name]

        results_df.loc[name, 'ROC-AUC'] = roc_auc_score(y_test, y_proba)
        results_df.loc[name, 'Accuracy'] = (y_pred == y_test).mean()
        results_df.loc[name, 'Precision'] = precision_score(y_test, y_pred)
        results_df.loc[name, 'Recall'] = recall_score(y_test, y_pred)
        results_df.loc[name, 'F1-Score'] = f1_score(y_test, y_pred)

    results_df = results_df.round(4)
    logger.info("\nPERFORMANCE COMPARISON:")
    logger.info(f"\n{results_df}")

    # Detailed metrics for XGBoost
    xgb_pred = predictions['XGBoost']
    cm = confusion_matrix(y_test, xgb_pred)
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)

    detailed_metrics = {
        'model': 'XGBoost',
        'roc_auc': float(results_df.loc['XGBoost', 'ROC-AUC']),
        'accuracy': float(results_df.loc['XGBoost', 'Accuracy']),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'ppv': float(ppv),
        'npv': float(npv),
        'precision': float(results_df.loc['XGBoost', 'Precision']),
        'recall': float(results_df.loc['XGBoost', 'Recall']),
        'f1_score': float(results_df.loc['XGBoost', 'F1-Score']),
        'confusion_matrix': {
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
        }
    }

    logger.info("\nXGBoost Clinical Metrics:")
    logger.info(f"Sensitivity: {sensitivity:.3f} ({sensitivity * 100:.1f}%)")
    logger.info(f"Specificity: {specificity:.3f} ({specificity * 100:.1f}%)")
    logger.info(f"PPV: {ppv:.3f} ({ppv * 100:.1f}%)")
    logger.info(f"NPV: {npv:.3f} ({npv * 100:.1f}%)")

    return results_df, detailed_metrics, predictions, probabilities


def create_evaluation_plots(models, X_test, y_test, predictions, probabilities):
    """Create and save evaluation plots."""
    logger.info("Creating evaluation plots...")

    plt.style.use('seaborn-v0_8')

    # Confusion Matrix and ROC Curve
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Confusion Matrix for XGBoost
    cm = confusion_matrix(y_test, predictions['XGBoost'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'],
                ax=axes[0])
    axes[0].set_title('Confusion Matrix - XGBoost Model')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')

    # Add performance annotations
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)

    axes[0].text(0.5, -0.15, f'Sensitivity: {sensitivity:.3f} | Specificity: {specificity:.3f}',
                 transform=axes[0].transAxes, ha='center', fontsize=10)
    axes[0].text(0.5, -0.20, f'PPV: {ppv:.3f} | NPV: {npv:.3f}',
                 transform=axes[0].transAxes, ha='center', fontsize=10)

    # ROC Curve Comparison
    for name in ['Random Forest', 'XGBoost', 'Logistic Regression', 'Ensemble']:
        y_proba = probabilities[name]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_score = roc_auc_score(y_test, y_proba)

        if name == 'XGBoost':
            axes[1].plot(fpr, tpr, linewidth=3, label=f'{name} (AUC = {auc_score:.3f})')
        else:
            axes[1].plot(fpr, tpr, alpha=0.7, label=f'{name} (AUC = {auc_score:.3f})')

    axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC Curves - Model Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/model_evaluation.png', dpi=300, bbox_inches='tight')
    plt.close()

    logger.info("Evaluation plots saved to outputs/model_evaluation.png")


def create_shap_analysis(model, X_train, X_test):
    """Create SHAP explainer and analysis."""
    logger.info("Creating SHAP analysis...")

    # Create explainer
    explainer = shap.TreeExplainer(model)

    # Calculate SHAP values for test set
    shap_values = explainer.shap_values(X_test)

    logger.info(f"SHAP values calculated for {X_test.shape[0]} test samples")

    # Global feature importance
    feature_importance = pd.DataFrame({
        'Feature': X_test.columns,
        'Mean_SHAP_Importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('Mean_SHAP_Importance', ascending=False)

    logger.info("\nTop 10 features by SHAP importance:")
    logger.info(f"\n{feature_importance.head(10)}")

    # Create SHAP plots
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, feature_names=X_test.columns, show=False)
    plt.title('SHAP Summary Plot - Feature Importance for Heart Disease Prediction',
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('outputs/shap_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, feature_names=X_test.columns,
                      plot_type="bar", show=False)
    plt.title('SHAP Feature Importance - Heart Disease Prediction',
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/shap_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

    logger.info("SHAP plots saved to outputs/")

    return explainer, shap_values, feature_importance


def save_artifacts(model, scaler, explainer, feature_names, numerical_features,
                   feature_importance, detailed_metrics, shap_values):
    """Save all artifacts for production use."""
    logger.info("Saving production artifacts...")

    # Core model artifacts
    joblib.dump(model, "artifacts/xgb_model.pkl")
    joblib.dump(scaler, "artifacts/scaler.pkl")
    joblib.dump(explainer, "artifacts/shap_explainer.pkl")

    # Feature metadata
    with open("artifacts/feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    with open("artifacts/numerical_features.json", "w") as f:
        json.dump(numerical_features, f, indent=2)

    # Model metrics
    with open("artifacts/metrics.json", "w") as f:
        json.dump(detailed_metrics, f, indent=2)

    # Feature importance
    feature_importance.to_csv("artifacts/feature_importance.csv", index=False)

    # SHAP values for population analysis
    np.save("artifacts/global_shap_values.npy", shap_values)

    logger.info("All artifacts saved to artifacts/ directory:")
    logger.info("- xgb_model.pkl, scaler.pkl, shap_explainer.pkl")
    logger.info("- feature_names.json, numerical_features.json")
    logger.info("- metrics.json, feature_importance.csv")
    logger.info("- global_shap_values.npy")

    # Validation test
    loaded_model = joblib.load("artifacts/xgb_model.pkl")
    logger.info("Artifact validation passed")


def main():
    """Main training pipeline."""
    logger.info("Clinical Heart Disease AI Training Pipeline")
    logger.info(f"Training started at: {datetime.now()}")

    try:
        # Setup
        setup_directories()

        # Data loading and preprocessing
        df = load_and_clean_data()
        df_processed = create_clinical_features(df)

        # Feature preparation
        X_train, X_test, y_train, y_test, scaler, numerical_features = prepare_features_and_split(df_processed)

        # Handle class imbalance
        X_train_balanced, y_train_balanced = handle_class_imbalance(X_train, y_train)

        # Hyperparameter optimization
        best_params_xgb = optimize_xgboost(X_train_balanced, y_train_balanced)

        # Model training
        models, cv_results = train_models(X_train_balanced, y_train_balanced, best_params_xgb)

        # Model evaluation
        results_df, detailed_metrics, predictions, probabilities = evaluate_models(models, X_test, y_test)

        # Create plots
        create_evaluation_plots(models, X_test, y_test, predictions, probabilities)

        # SHAP analysis
        xgb_model = models['XGBoost']
        explainer, shap_values, feature_importance = create_shap_analysis(xgb_model, X_train, X_test)

        # Save artifacts
        save_artifacts(
            xgb_model, scaler, explainer, X_train.columns.tolist(),
            numerical_features, feature_importance, detailed_metrics, shap_values
        )

        logger.info("Training pipeline completed successfully!")
        logger.info(f"Final XGBoost ROC-AUC: {detailed_metrics['roc_auc']:.4f}")
        logger.info(f"Sensitivity: {detailed_metrics['sensitivity']:.3f}")
        logger.info(f"Specificity: {detailed_metrics['specificity']:.3f}")

    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
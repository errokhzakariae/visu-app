import json
import math
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve
)

# =========================================================
# CONFIGURATION DE LA PAGE
# =========================================================
st.set_page_config(
    page_title="Loan Approval App",
    page_icon="📊",
    layout="wide"
)

# =========================================================
# CHEMINS DE BASE
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# =========================================================
# FONCTIONS UTILES
# =========================================================
def is_approved(value):
    if pd.isna(value):
        return False
    value = str(value).strip().lower()
    return value in ["y", "yes", "1", "true", "approved"]


def find_csv_file():
    csv_files_data = list(DATA_DIR.glob("*.csv")) if DATA_DIR.exists() else []
    if csv_files_data:
        return csv_files_data[0]

    csv_files_root = list(BASE_DIR.glob("*.csv"))
    if csv_files_root:
        return csv_files_root[0]

    return None


# =========================================================
# CHARGEMENT DU DATASET
# =========================================================
@st.cache_data
def load_data():
    csv_path = find_csv_file()
    if csv_path is None:
        raise FileNotFoundError(
            "Aucun fichier CSV trouvé. Mets ton dataset dans le dossier 'data' "
            "ou dans le même dossier que app.py."
        )
    df = pd.read_csv(csv_path)
    return df, csv_path


# =========================================================
# CHARGEMENT DU MODÈLE
# =========================================================
@st.cache_resource
def load_model(model_name):
    model_paths = {
        "Logistic Regression": MODELS_DIR / "logistic_regression.pkl",
        "Random Forest": MODELS_DIR / "random_forest.pkl",
    }

    model_path = model_paths[model_name]

    debug_info = {
        "dossier_courant": os.getcwd(),
        "base_dir": str(BASE_DIR),
        "models_dir": str(MODELS_DIR),
        "model_path_absolu": str(model_path.resolve()),
        "model_existe": model_path.exists(),
    }

    if not model_path.exists():
        raise FileNotFoundError(
            f"Le fichier modèle est introuvable : {model_path.resolve()}"
        )

    model = joblib.load(model_path)
    return model, debug_info


# =========================================================
# CHARGEMENT DU SCALER
# =========================================================
@st.cache_resource
def load_scaler():
    scaler_path = MODELS_DIR / "scaler.pkl"

    debug_info = {
        "scaler_path_absolu": str(scaler_path.resolve()),
        "scaler_existe": scaler_path.exists(),
    }

    if not scaler_path.exists():
        return None, debug_info

    scaler = joblib.load(scaler_path)
    return scaler, debug_info


# =========================================================
# CHARGEMENT DES MÉTADONNÉES
# =========================================================
@st.cache_data
def load_metadata():
    metadata_path = MODELS_DIR / "metadata.json"
    if not metadata_path.exists():
        return None

    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================================================
# PRÉPARATION DES DONNÉES POUR LA PRÉDICTION
# =========================================================
def prepare_input_for_model(
    gender,
    married,
    dependents,
    education,
    self_employed,
    applicant_income,
    coapplicant_income,
    loan_amount,
    loan_amount_term,
    credit_history,
    property_area,
    metadata=None
):
    # Encodages conformes aux features finales du modèle
    dependents_num = {"0": 0, "1": 1, "2": 2, "3+": 3}[dependents]
    education_num = 1 if education == "Graduate" else 0
    gender_male = 1 if gender == "Male" else 0
    married_yes = 1 if married == "Yes" else 0
    self_employed_yes = 1 if self_employed == "Yes" else 0

    area_semiurban = 1 if property_area == "Semiurban" else 0
    area_urban = 1 if property_area == "Urban" else 0

    # Features engineerées attendues par le TP / le modèle
    total_income = applicant_income + coapplicant_income
    loan_amount_to_income = loan_amount / total_income if total_income > 0 else 0
    emi = loan_amount / loan_amount_term if loan_amount_term > 0 else 0
    emi_to_income = emi / total_income if total_income > 0 else 0
    log_loan_amount = math.log1p(loan_amount)
    log_total_income = math.log1p(total_income)
    has_coapplicant = 1 if coapplicant_income > 0 else 0

    data = {
        "Dependents": dependents_num,
        "Education": education_num,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_amount_term,
        "Credit_History": credit_history,
        "TotalIncome": total_income,
        "LoanAmountToIncome": loan_amount_to_income,
        "EMI": emi,
        "EMIToIncome": emi_to_income,
        "Log_LoanAmount": log_loan_amount,
        "Log_TotalIncome": log_total_income,
        "Has_Coapplicant": has_coapplicant,
        "Area_Semiurban": area_semiurban,
        "Area_Urban": area_urban,
        "Gender_Male": gender_male,
        "Married_Yes": married_yes,
        "SelfEmployed_Yes": self_employed_yes,
    }

    input_df = pd.DataFrame([data])

    # Réindexation stricte sur les colonnes du metadata
    if metadata is not None and "feature_names" in metadata:
        expected_cols = metadata["feature_names"]
        input_df = input_df.reindex(columns=expected_cols, fill_value=0)

    return input_df


# =========================================================
# EXPLICATION LOCALE
# =========================================================
def get_local_feature_importance(model, input_df):
    if hasattr(model, "coef_"):
        contributions = model.coef_[0] * input_df.iloc[0].values
        importance_df = pd.DataFrame({
            "feature": input_df.columns,
            "importance": contributions
        })
    elif hasattr(model, "feature_importances_"):
        contributions = model.feature_importances_ * input_df.iloc[0].values
        importance_df = pd.DataFrame({
            "feature": input_df.columns,
            "importance": contributions
        })
    else:
        return None

    importance_df["abs_importance"] = importance_df["importance"].abs()
    importance_df = (
        importance_df.sort_values("abs_importance", ascending=False)
        .head(5)
        .sort_values("importance", ascending=True)
    )
    return importance_df


# =========================================================
# IMPORTANCE GLOBALE
# =========================================================
def get_global_feature_importance(model, metadata=None):
    feature_names = metadata["feature_names"] if metadata and "feature_names" in metadata else None

    if hasattr(model, "feature_importances_") and feature_names is not None:
        return pd.DataFrame({
            "feature": feature_names,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False).head(10)

    if hasattr(model, "coef_") and feature_names is not None:
        return pd.DataFrame({
            "feature": feature_names,
            "importance": np.abs(model.coef_[0])
        }).sort_values("importance", ascending=False).head(10)

    return None


# =========================================================
# CALCUL DES MÉTRIQUES DE PERFORMANCE
# =========================================================
def compute_model_metrics(model, scaler, df, metadata=None):
    if "Loan_Status" in df.columns:
        target_name = "Loan_Status"
    elif "Approved" in df.columns:
        target_name = "Approved"
    else:
        return None

    if metadata is None or "feature_names" not in metadata:
        return None

    y_true = df[target_name].apply(lambda x: 1 if is_approved(x) else 0)

    # On reconstruit un X minimal compatible si le dataset contient déjà
    # les features finales. Sinon, on s'appuie sur metadata seulement pour l'UI.
    expected_cols = metadata["feature_names"]

    if all(col in df.columns for col in expected_cols):
        X = df[expected_cols].copy()
    else:
        return None

    X_eval = X.copy()
    if scaler is not None:
        X_eval = scaler.transform(X_eval)

    y_pred = model.predict(X_eval)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_eval)[:, 1]
        metrics["auc"] = roc_auc_score(y_true, y_proba)
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        metrics["fpr"] = fpr
        metrics["tpr"] = tpr
        metrics["y_proba"] = y_proba

    return metrics


# =========================================================
# CHARGEMENT DES DONNÉES
# =========================================================
try:
    df, csv_path = load_data()
except Exception as e:
    st.error("Erreur lors du chargement du dataset.")
    st.exception(e)
    st.stop()

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.title("🏦 Loan Approval App")
    st.markdown("Application de scoring de prêt")

    selected_model_name = st.selectbox(
        "Choisir un modèle",
        ["Logistic Regression", "Random Forest"]
    )

    st.markdown("---")
    st.subheader("Filtres")

    if "ApplicantIncome" not in df.columns:
        st.error("La colonne 'ApplicantIncome' est introuvable dans le dataset.")
        st.write("Colonnes disponibles :", list(df.columns))
        st.stop()

    if "Education" not in df.columns:
        st.error("La colonne 'Education' est introuvable dans le dataset.")
        st.write("Colonnes disponibles :", list(df.columns))
        st.stop()

    income_min = int(df["ApplicantIncome"].min())
    income_max = int(df["ApplicantIncome"].max())

    selected_income_range = st.slider(
        "Revenu du demandeur",
        min_value=income_min,
        max_value=income_max,
        value=(income_min, income_max)
    )

    education_options = ["Tous"] + sorted(df["Education"].dropna().astype(str).unique().tolist())
    selected_education = st.selectbox(
        "Niveau d'éducation",
        education_options
    )

# =========================================================
# CHARGEMENT DU MODÈLE ET DU SCALER
# =========================================================
model = None
model_debug_info = {}
scaler = None
scaler_debug_info = {}
metadata = load_metadata()

try:
    model, model_debug_info = load_model(selected_model_name)
    model_loading_error = None
except Exception as e:
    model_loading_error = e

try:
    scaler, scaler_debug_info = load_scaler()
    scaler_loading_error = None
except Exception as e:
    scaler_loading_error = e

# =========================================================
# APPLICATION DES FILTRES
# =========================================================
filtered_df = df[
    (df["ApplicantIncome"] >= selected_income_range[0]) &
    (df["ApplicantIncome"] <= selected_income_range[1])
]

if selected_education != "Tous":
    filtered_df = filtered_df[filtered_df["Education"].astype(str) == selected_education]

# =========================================================
# IDENTIFICATION DE LA COLONNE CIBLE
# =========================================================
target_col = None
if "Loan_Status" in filtered_df.columns:
    target_col = "Loan_Status"
elif "Approved" in filtered_df.columns:
    target_col = "Approved"

# =========================================================
# EN-TÊTE PRINCIPAL
# =========================================================
st.title("📊 Application de prédiction de prêt")
st.write(f"Modèle actuellement sélectionné : **{selected_model_name}**")
st.caption(f"Dataset chargé : {csv_path.name}")

# =========================================================
# ONGLETS
# =========================================================
tab1, tab2, tab3 = st.tabs([
    "📊 Exploration des données",
    "🤖 Prédiction",
    "📈 Performance du modèle"
])

# =========================================================
# ONGLET 1 : EXPLORATION DES DONNÉES
# =========================================================
with tab1:
    st.subheader("📊 Exploration des données")

    if filtered_df.empty:
        st.warning("Aucune donnée ne correspond aux filtres sélectionnés.")
    else:
        missing_columns = []
        for col in ["ApplicantIncome", "LoanAmount", "Education"]:
            if col not in filtered_df.columns:
                missing_columns.append(col)

        if missing_columns:
            st.error(f"Colonnes manquantes dans le dataset : {missing_columns}")
            st.write("Colonnes disponibles :", list(filtered_df.columns))
        else:
            total_demandes = len(filtered_df)
            montant_moyen = filtered_df["LoanAmount"].mean()
            revenu_moyen = filtered_df["ApplicantIncome"].mean()

            if target_col is not None:
                approval_rate = filtered_df[target_col].apply(is_approved).mean() * 100
            else:
                approval_rate = 0.0

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Nombre total de demandes", f"{total_demandes}")
            col2.metric("Taux d'approbation global", f"{approval_rate:.1f}%")
            col3.metric("Montant moyen des prêts", f"{montant_moyen:.2f}")
            col4.metric("Revenu moyen", f"{revenu_moyen:.2f}")

            st.markdown("---")

            st.subheader("Distributions")
            dist_col1, dist_col2 = st.columns(2)

            with dist_col1:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.hist(filtered_df["ApplicantIncome"].dropna(), bins=30)
                ax.set_title("Histogramme des revenus")
                ax.set_xlabel("ApplicantIncome")
                ax.set_ylabel("Fréquence")
                st.pyplot(fig)

            with dist_col2:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.boxplot(filtered_df["LoanAmount"].dropna(), vert=True)
                ax.set_title("Box plot des montants de prêt")
                ax.set_ylabel("LoanAmount")
                st.pyplot(fig)

            st.markdown("---")

            st.subheader("Analyses")
            ana_col1, ana_col2 = st.columns(2)

            with ana_col1:
                if target_col is not None:
                    approval_by_education = (
                        filtered_df.groupby("Education")[target_col]
                        .apply(lambda x: x.apply(is_approved).mean() * 100)
                        .sort_values()
                    )

                    fig, ax = plt.subplots(figsize=(8, 4))
                    approval_by_education.plot(kind="bar", ax=ax)
                    ax.set_title("Taux d'approbation par éducation")
                    ax.set_xlabel("Education")
                    ax.set_ylabel("Taux d'approbation (%)")
                    plt.xticks(rotation=0)
                    st.pyplot(fig)
                else:
                    st.info("Colonne cible non trouvée pour calculer le taux d'approbation.")

            with ana_col2:
                if target_col is not None:
                    approved_count = filtered_df[target_col].apply(is_approved).sum()
                    rejected_count = len(filtered_df) - approved_count

                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.pie(
                        [approved_count, rejected_count],
                        labels=["Approved", "Rejected"],
                        autopct="%1.1f%%",
                        startangle=90
                    )
                    ax.set_title("Répartition Approved / Rejected")
                    st.pyplot(fig)
                else:
                    st.info("Colonne cible non trouvée pour afficher le pie chart.")

            st.markdown("---")

            st.subheader("Corrélations")
            numeric_df = filtered_df.select_dtypes(include=["number"])

            if not numeric_df.empty and numeric_df.shape[1] >= 2:
                corr = numeric_df.corr()
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
                ax.set_title("Heatmap de corrélation")
                st.pyplot(fig)
            else:
                st.info("Pas assez de colonnes numériques pour calculer une corrélation.")

            st.markdown("---")
            st.subheader("Aperçu des données filtrées")
            st.dataframe(filtered_df, use_container_width=True)

# =========================================================
# ONGLET 2 : PRÉDICTION
# =========================================================
with tab2:
    st.subheader("🤖 Prédiction")

    if model_loading_error is not None:
        st.error("Le modèle n'a pas pu être chargé.")
        st.exception(model_loading_error)
    else:
        st.success(f"Le modèle '{selected_model_name}' est bien chargé.")

        if scaler_loading_error is None and scaler is not None:
            st.success("Scaler chargé avec succès.")
        else:
            st.info("Aucun scaler chargé ou scaler non nécessaire.")

        st.markdown("### Formulaire de prédiction")

        with st.form("prediction_form"):
            col1, col2 = st.columns(2)

            with col1:
                applicant_income = st.number_input("Revenu du demandeur", min_value=0.0, value=5000.0, step=100.0)
                coapplicant_income = st.number_input("Revenu du co-demandeur", min_value=0.0, value=0.0, step=100.0)
                loan_amount = st.number_input("Montant du prêt", min_value=1.0, value=120.0, step=1.0)
                loan_amount_term = st.number_input("Durée du prêt (mois)", min_value=1.0, value=360.0, step=1.0)

            with col2:
                credit_history = st.selectbox("Historique de crédit", [1.0, 0.0])
                education = st.selectbox("Niveau d'éducation", ["Graduate", "Not Graduate"])
                married = st.selectbox("Marié", ["Yes", "No"])
                dependents = st.selectbox("Nombre de personnes à charge", ["0", "1", "2", "3+"])
                self_employed = st.selectbox("Travailleur indépendant", ["Yes", "No"])
                property_area = st.selectbox("Zone du bien", ["Rural", "Semiurban", "Urban"])
                gender = st.selectbox("Genre", ["Male", "Female"])

            submitted = st.form_submit_button("Prédire")

        if submitted:
            warnings = []

            if applicant_income <= 0:
                warnings.append("Le revenu du demandeur doit être supérieur à 0.")
            if loan_amount <= 0:
                warnings.append("Le montant du prêt doit être supérieur à 0.")
            if loan_amount_term <= 0:
                warnings.append("La durée du prêt doit être supérieure à 0.")

            if warnings:
                for w in warnings:
                    st.warning(w)
            else:
                try:
                    input_df = prepare_input_for_model(
                        gender=gender,
                        married=married,
                        dependents=dependents,
                        education=education,
                        self_employed=self_employed,
                        applicant_income=applicant_income,
                        coapplicant_income=coapplicant_income,
                        loan_amount=loan_amount,
                        loan_amount_term=loan_amount_term,
                        credit_history=credit_history,
                        property_area=property_area,
                        metadata=metadata
                    )

                    st.markdown("### Données envoyées au modèle")
                    st.dataframe(input_df, use_container_width=True)

                    X_pred = input_df.copy()

                    progress = st.progress(0)
                    progress.progress(25)

                    if scaler is not None:
                        X_pred = scaler.transform(X_pred)

                    progress.progress(60)
                    prediction = model.predict(X_pred)[0]
                    progress.progress(85)

                    approved_values = ["Y", "Yes", "1", 1, True, "Approved"]

                    if prediction in approved_values:
                        st.success("✅ Prêt approuvé")
                    else:
                        st.error("❌ Prêt rejeté")

                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(X_pred)[0]

                        if len(proba) == 2:
                            rejection_proba = proba[0] * 100
                            approval_proba = proba[1] * 100

                            st.markdown("### Probabilités")
                            prob_col1, prob_col2 = st.columns(2)
                            prob_col1.metric("Probabilité de rejet", f"{rejection_proba:.2f}%")
                            prob_col2.metric("Probabilité d'approbation", f"{approval_proba:.2f}%")

                    progress.progress(100)

                    st.markdown("### Explication de la décision")
                    local_importance = get_local_feature_importance(model, input_df)

                    if local_importance is not None:
                        fig, ax = plt.subplots(figsize=(8, 4))
                        ax.barh(local_importance["feature"], local_importance["importance"])
                        ax.set_title("Top 5 des features influentes")
                        ax.set_xlabel("Influence")
                        st.pyplot(fig)
                    else:
                        st.info("Impossible de calculer l'explication locale pour ce modèle.")

                except Exception as e:
                    st.error("Erreur pendant la prédiction.")
                    st.exception(e)

                    st.markdown("### Vérification utile")
                    if metadata is not None and "feature_names" in metadata:
                        st.write("Colonnes attendues :", metadata["feature_names"])
                    st.write("Colonnes envoyées :", list(input_df.columns))

# =========================================================
# ONGLET 3 : PERFORMANCE DU MODÈLE
# =========================================================
with tab3:
    st.subheader("📈 Performance du modèle")

    if model_loading_error is not None:
        st.warning("Impossible d'afficher la performance tant que le modèle n'est pas chargé.")
    else:
        st.write(f"Le modèle sélectionné est : **{selected_model_name}**")

        if metadata is not None and "models" in metadata:
            key = "logistic_regression" if selected_model_name == "Logistic Regression" else "random_forest"
            if key in metadata["models"]:
                model_metrics = metadata["models"][key]
            else:
                model_metrics = None
        else:
            model_metrics = compute_model_metrics(model, scaler, df, metadata=metadata)

        if model_metrics is None:
            st.info("Impossible de calculer les métriques avec les données disponibles.")
        else:
            c1, c2, c3, c4, c5 = st.columns(5)

            c1.metric("Accuracy", f"{model_metrics['accuracy']:.3f}")
            c2.metric("Precision", f"{model_metrics['precision']:.3f}")
            c3.metric("Recall", f"{model_metrics['recall']:.3f}")
            c4.metric("F1", f"{model_metrics.get('f1', model_metrics.get('f1_score', 0)):.3f}")

            auc_value = model_metrics["auc"] if "auc" in model_metrics else None
            c5.metric("AUC", f"{auc_value:.3f}" if auc_value is not None else "N/A")

            st.markdown("---")

            # Si on n'a pas les données détaillées de prédiction, on garde l'UI informative
            computed_metrics = compute_model_metrics(model, scaler, df, metadata=metadata)

            st.subheader("Matrice de confusion")
            if computed_metrics is not None:
                cm = np.array(computed_metrics["confusion_matrix"])
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                ax.set_xlabel("Prédit")
                ax.set_ylabel("Réel")
                ax.set_title("Confusion Matrix")
                st.pyplot(fig)
            else:
                st.info("Matrice de confusion indisponible avec le dataset actuel.")

            st.markdown("---")

            st.subheader("Courbe ROC")
            if computed_metrics is not None and "fpr" in computed_metrics and "tpr" in computed_metrics:
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.plot(computed_metrics["fpr"], computed_metrics["tpr"], label="ROC")
                ax.plot([0, 1], [0, 1], linestyle="--")
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title("ROC Curve")
                ax.legend()
                st.pyplot(fig)
            else:
                st.info("Courbe ROC indisponible pour ce modèle.")

            st.markdown("---")

            st.subheader("Feature importance globale")
            global_importance = get_global_feature_importance(model, metadata=metadata)

            if global_importance is not None:
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.barh(
                    global_importance["feature"][::-1],
                    global_importance["importance"][::-1]
                )
                ax.set_title("Top 10 des features les plus importantes")
                ax.set_xlabel("Importance")
                st.pyplot(fig)
            else:
                st.info("Importance globale indisponible pour ce modèle.")

            st.markdown("---")

            st.subheader("Distribution des probabilités")
            if computed_metrics is not None and "y_proba" in computed_metrics:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.hist(computed_metrics["y_proba"], bins=20)
                ax.set_title("Distribution des probabilités d'approbation")
                ax.set_xlabel("Probabilité")
                ax.set_ylabel("Fréquence")
                st.pyplot(fig)
            else:
                st.info("Distribution des probabilités indisponible.")
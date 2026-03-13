
# 🏦 Loan Approval Prediction App

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![License](https://img.shields.io/badge/License-Educational-green)

Application web interactive développée avec **Streamlit** permettant :

- d’explorer un dataset de demandes de prêt  
- de prédire l’approbation d’un prêt avec des modèles de machine learning  
- d’analyser les performances des modèles  

L'application transforme un **modèle de machine learning en outil interactif accessible aux utilisateurs non techniques**.

---

# 🎯 Objectif du projet

Ce projet a pour objectif de passer **d’un modèle ML développé dans un notebook à une application web utilisable**.

L’application permet :

- l’exploration des données
- l’analyse statistique
- la prédiction de l’approbation d’un prêt
- l’interprétation des décisions du modèle

---

# 🚀 Fonctionnalités

## 📊 Exploration des données

L'application propose plusieurs outils d’analyse :

- métriques globales du dataset
- histogramme des revenus
- boxplot des montants de prêt
- analyse du taux d’approbation par niveau d’éducation
- graphique de répartition des prêts approuvés / rejetés
- heatmap de corrélation
- affichage du dataset filtré

---

## 🤖 Prédiction

L'utilisateur peut saisir les caractéristiques d’un demandeur :

- revenu du demandeur
- revenu du co-demandeur
- montant du prêt
- durée du prêt
- historique de crédit
- niveau d’éducation
- statut marital
- nombre de dépendants
- statut professionnel
- zone géographique

Le modèle retourne :

- décision **Approved / Rejected**
- **probabilité d’approbation**
- **variables les plus influentes dans la décision**

---

## 📈 Performance du modèle

L'application affiche plusieurs métriques de machine learning :

- Accuracy
- Precision
- Recall
- F1-score
- AUC

Elle affiche également :

- matrice de confusion
- courbe ROC
- importance globale des variables
- distribution des probabilités de prédiction

---

# 🛠️ Technologies utilisées

- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Joblib

---

# 📁 Structure du projet

```
loan-approval-app/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   └── loan_data_clean.csv
│
├── models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── scaler.pkl
│   └── metadata.json
```

---

# ⚙️ Installation

### 1️⃣ Cloner le projet

```bash
git clone https://github.com/ton-username/loan-approval-app.git
cd loan-approval-app
```

### 2️⃣ Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3️⃣ Lancer l'application

```bash
streamlit run app.py
```

L'application sera accessible sur :

```
http://localhost:8501
```

---

# 🐳 Utilisation avec Docker

### Construire l’image

```bash
docker build -t loan-approval-app .
```

### Lancer le conteneur

```bash
docker run -p 8501:8501 loan-approval-app
```

Puis ouvrir :

```
http://localhost:8501
```

---

# 🧠 Modèles utilisés

## Logistic Regression

- modèle linéaire simple
- facile à interpréter
- rapide à entraîner

## Random Forest

- modèle d'ensemble
- capture les relations non linéaires
- généralement plus robuste

---

# 🔎 Explicabilité du modèle

L'application fournit :

### Importance globale

Les variables les plus importantes pour le modèle.

### Importance locale

Les variables qui influencent **une prédiction spécifique**.

---

# 📊 Dataset

Le dataset contient des informations sur les demandes de prêt :

- revenus
- historique de crédit
- statut professionnel
- caractéristiques du demandeur

Objectif : **prédire si un prêt sera approuvé ou rejeté**.

---

# 📌 Améliorations possibles

- visualisations interactives avec **Plotly**
- ajout de nouveaux modèles ML
- déploiement cloud
- monitoring du modèle
- authentification utilisateur
- API REST pour les prédictions

---

# 👨‍💻 Auteur

Projet réalisé dans le cadre du module :

**Application web pour visualiser les données – M2 Data / IA**

---

# 📜 Licence

Projet éducatif.

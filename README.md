# 💳 Credit Default Risk Analyzer

A machine learning project for predicting **credit card default risk** using the **UCI Credit Card Default Dataset**. This project implements and compares multiple machine learning classification algorithms and provides an interactive **Streamlit web application** for predicting default risk on new customer data.

---

## 📋 Problem Statement

Financial institutions need to identify customers who are likely to default on their credit card payments. Early prediction helps reduce financial losses and improve lending decisions.

The objective of this project is to build and compare multiple machine learning models capable of predicting whether a customer will default on their credit card payment in the following month.

This is a **Binary Classification** problem.

- **0 → No Default**
- **1 → Default**

---

# 📊 Dataset

**Source**

UCI Machine Learning Repository

Dataset:
Default of Credit Card Clients Dataset

### Dataset Statistics

| Property | Value |
|-----------|------:|
| Samples | 30,000 |
| Original Features | 23 |
| Engineered Features | 4 |
| Total Features Used | 27 |
| Missing Values | None |
| Target Variable | default_flag |

---

# 📈 Class Distribution

| Class | Percentage |
|-------|-----------:|
| No Default | 77.88% |
| Default | 22.12% |

The dataset is moderately imbalanced, therefore evaluation uses multiple metrics beyond accuracy.

---

# 🚀 Live Demo

## Streamlit Application

[https://credit-default-analyzer.streamlit.app](https://sagarrayala--credit-default-analyzer-app-llsdit.streamlit.app/)

---

## GitHub Repository

https://github.com/sagarrayala/-credit_default_analyzer

---

# 🤖 Machine Learning Models

The following supervised learning algorithms were implemented.

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Random Forest Classifier

Each model was trained and evaluated using exactly the same train/test split.

---

# 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | MCC | AUC |
|------|---------:|----------:|--------:|---------:|----:|----:|
| Logistic Regression | 0.6922 | 0.7712 | 0.6922 | 0.7154 | 0.3121 | 0.7435 |
| Decision Tree | 0.7120 | 0.7732 | 0.7120 | 0.7315 | 0.3248 | 0.7427 |
| K-Nearest Neighbors | **0.8123** | **0.7929** | **0.8123** | **0.7923** | 0.3773 | 0.7516 |
| Naive Bayes | 0.6418 | 0.7771 | 0.6418 | 0.6723 | 0.3024 | 0.7555 |
| Random Forest | 0.7687 | 0.7923 | 0.7687 | 0.7778 | **0.3922** | **0.7874** |

---
# 📝 Model Analysis

## Logistic Regression

- Strong baseline model.
- Accuracy: **69.22%**
- AUC: **0.7435**
- Easy to interpret and provides consistent performance.

---

## Decision Tree

- Accuracy improved to **71.20%**.
- Simple and interpretable model.
- Controlled tree depth reduces overfitting.
- AUC: **0.7427**.

---

## K-Nearest Neighbors (KNN)

- Highest Accuracy (**81.23%**).
- Highest Precision (**79.29%**).
- Highest Recall (**81.23%**).
- Highest F1 Score (**79.23%**).
- Performs well after feature scaling.
- Selected as the best-performing model based on accuracy.

---

## Gaussian Naive Bayes

- Fastest model.
- Lowest accuracy (**64.18%**).
- Gaussian assumption does not perfectly fit the dataset.
- Moderate ROC-AUC (**0.7555**).

---

## Random Forest

- Highest ROC-AUC (**0.7874**).
- Highest MCC (**0.3922**).
- Excellent balance between precision and recall.
- Robust ensemble model with strong generalization performance.

---

# 🏆 Best Performing Models

## Highest Accuracy

🥇 **K-Nearest Neighbors (KNN)**

Accuracy: **81.23%**

---

## Highest ROC-AUC

🥇 **Random Forest**

ROC-AUC: **0.7874**

---

## Overall Winner

Since **accuracy is considered the primary evaluation metric for this project**, **K-Nearest Neighbors (KNN)** is selected as the overall best-performing model.

KNN achieved:

- ✅ Highest Accuracy (81.23%)
- ✅ Highest Precision (79.29%)
- ✅ Highest Recall (81.23%)
- ✅ Highest F1 Score (79.23%)

Although **Random Forest** achieved the highest ROC-AUC (0.7874) and Matthews Correlation Coefficient (0.3922), KNN demonstrated the best predictive accuracy on the test dataset. Therefore, KNN is selected as the final model for this implementation.

---

# 📂 Project Structure

```text
credit_default_analyzer/
│
├── src/
│   ├── __init__.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_estimator.py
│   │   ├── logistic_risk_model.py
│   │   ├── decision_tree_risk_model.py
│   │   ├── knn_risk_model.py
│   │   ├── naive_bayes_risk_model.py
│   │   └── random_forest_risk_model.py
│   │
│   └── utils/
│       ├── __init__.py
│       └── data_processor.py
│
├── models/
│   ├── logistic_risk_v1.joblib
│   ├── decision_tree_v1.joblib
│   ├── knn_v1.joblib
│   ├── naive_bayes_v1.joblib
│   ├── random_forest_v1.joblib
│   ├── scaler.joblib
│   └── feature_columns.joblib
│
├── data/
│   └── credit_card_default.xls
│
├── app.py
├── train_and_evaluate_all_models.ipynb
├── test_data.csv
├── requirements.txt
├── README.md
└── .gitignore
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/sagarrayala/-credit_default_analyzer.git

cd credit-default-analyzer
```

---

## Create Virtual Environment

### Linux / macOS

```bash
python -m venv .venv

source .venv/bin/activate
```

### Windows

```bash
python -m venv .venv

.venv\Scripts\activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🏃 Train Models

Launch Jupyter Notebook.

```bash
jupyter notebook
```

Open:

```
train_and_evaluate_all_models.ipynb
```

Run all cells.

The notebook will:

- Load dataset
- Perform preprocessing
- Engineer features
- Train all models
- Evaluate all models
- Save trained models
- Generate comparison metrics

---

# ▶️ Run Streamlit App

```bash
streamlit run app.py
```

The application will open in your browser.

---

# 🖥️ Streamlit Features

The deployed application supports:

- ✅ Upload CSV file
- ✅ Choose any trained model
- ✅ Predict defaults
- ✅ Display Accuracy
- ✅ Display Precision
- ✅ Display Recall
- ✅ Display F1 Score
- ✅ Display MCC
- ✅ Display ROC AUC
- ✅ Confusion Matrix
- ✅ Classification Report

---

# 📦 Requirements

```text
streamlit
scikit-learn
numpy
pandas
matplotlib
seaborn
joblib
scipy
openpyxl
xlrd
```

---

# 📸 Virtual Lab Screenshot

Add your BITS Virtual Lab screenshot below.

```
virtual_lab_screenshot.png
```

Example:

```markdown

```

---

# 🎯 Future Improvements

Possible enhancements include:

- XGBoost
- LightGBM
- CatBoost
- Hyperparameter tuning using GridSearchCV
- SMOTE for handling class imbalance
- Explainable AI using SHAP values
- Docker deployment
- CI/CD using GitHub Actions
- REST API using FastAPI

---

# 👨‍💻 Author

**Sagar Rayala**

Senior Software Developer

M.Tech (AIML)

BITS Pilani

GitHub:
https://github.com/sagarrayala/

---

# 📜 License

This project has been developed solely for educational purposes as part of the **Machine Learning Assignment 2** submitted to **BITS Pilani Work Integrated Learning Programme**.

---

# 🙏 Acknowledgements

- UCI Machine Learning Repository
- Scikit-learn
- Streamlit
- Pandas
- NumPy
- Matplotlib
- BITS Pilani WILP

---

# ✅ Submission Checklist

| Item | Status |
|------|:------:|
| GitHub Repository | ✅ |
| Five ML Models | ✅ |
| Feature Engineering | ✅ |
| Evaluation Metrics | ✅ |
| Model Comparison | ✅ |
| Performance Analysis | ✅ |
| Best Model Identified | ✅ |
| Streamlit Application | ✅ |
| Live Deployment | ✅ |
| README Documentation | ✅ |
| requirements.txt | ✅ |
| test_data.csv | ✅ |
| Trained Models | ✅ |
| Virtual Lab Screenshot | ✅ |

---

## ⭐ If you found this project useful, consider giving the repository a star.

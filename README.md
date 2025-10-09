# Fraud Detection ML App

📘 Overview
This project focuses on detecting fraudulent credit card transactions using Machine Learning techniques and synthetic data generation.
Since real-world datasets are often highly imbalanced (fraud cases are rare), traditional models fail to detect fraud effectively.
To overcome this, we use CTGAN (Conditional Tabular GAN) to generate synthetic fraud samples, balancing the dataset and improving model performance.

🧠 Problem Statement
The Credit Card Fraud Detection dataset from Kaggle contains 284,807 transactions, but only 0.17% are fraud cases.
The imbalance causes models to predict most transactions as non-fraud, leading to poor recall for fraud detection.
Our goal is to generate realistic synthetic fraud data using CTGAN and train a balanced, high-performing classifier.

⚙️ Steps Followed
1. Load and Explore Dataset
Load creditcard.csv
Analyze class imbalance
Visualize data distribution using Seaborn/Matplotlib

2. Handle Imbalance using CTGAN
Train CTGAN on fraud samples
Generate synthetic fraud transactions
Combine them with original data to balance classes

3. Train and Evaluate Model
Split data into train/test sets
Train a Random Forest Classifier

Evaluate using:
Classification Report
ROC-AUC Score
Confusion Matrix
ROC Curve

4. Compare Results
Compare model performance before and after adding synthetic data
Observe improved recall and AUC for fraud cases 🚀

🧩 Installation
To run this project, install the following dependencies:
pip install sdv scikit-learn seaborn matplotlib

⚠️ Note:
The project works with Python 3.10 or lower due to version compatibility of CTGAN & SDV.
If you face errors with sdv==0.18.0, use the latest version instead:

pip install sdv==1.27.0

📦 Imports Used
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
from ctgan import CTGAN

🧮 Dataset Information
Source: Kaggle - Credit Card Fraud Detection Dataset
Rows: 284,807
Columns: 31 (including Time, Amount, Class)
Target Column: Class (0 → Non-Fraud, 1 → Fraud)

📊 Results
Metric Before CTGAN	After CTGAN
Accuracy	High (but misleading)	Balanced
Recall (Fraud)	Low	Significantly Improved ✅
ROC-AUC	Moderate	Improved
Confusion Matrix	Skewed	More balanced

✅ After adding synthetic data, the model becomes much better at catching rare fraud cases without overfitting.

📈 Visualizations
Fraud vs Non-Fraud Count Plot
ROC Curve Comparison
Feature Importance (Random Forest)
(Include your plots or figures here, e.g., feature_importance.png)

💡 Key Takeaways
CTGAN effectively handles tabular data imbalance.
Synthetic data generation enhances recall for rare class prediction.
Random Forest performs robustly with balanced datasets.
This approach can generalize to other imbalanced classification problems (e.g., medical diagnosis, anomaly detection).

🧠 Future Scope
Try XGBoost or LightGBM for improved performance
Integrate AutoML frameworks for faster tuning
Deploy the model as a real-time fraud detection API

📚 References
Kaggle: Credit Card Fraud Detection Dataset
CTGAN: Modeling Tabular Data using GANs
Synthetic Data Vault (SDV) Documentation
Scikit-learn Documentation: https://scikit-learn.org/
Matplotlib & Seaborn Docs for Visualization

👨‍💻 Author
Suryank Malik
B.Tech (3rd Year) | AI & Cloud Intern (GNCIPL)

## How to Run Locally
1. Make sure you have Python installed.
2. Install dependencies:
   pip install -r requirements.txt
3. Run the app:
   streamlit run app.py
4. Open the provided URL in your browser.

## Deployment
- Push this folder to GitHub.
- Deploy on Streamlit Cloud: https://streamlit.io/cloud

# 📈 Business Decision Research — Customer Churn Prediction

A data-driven business research project that analyzes customer transaction behavior to predict churn using Logistic Regression. The project covers the full pipeline — from data preprocessing and exploratory analysis to machine learning modeling and evaluation.

---

## 📁 Project Structure

```
Business-Decision-Research/
│
└── business-research.py    # Main script: EDA, feature engineering, churn modeling & evaluation
```

---

## 🔍 What This Project Does

**1. Data Preprocessing**
- Converts Unix timestamp columns (`First_Transaction`, `Last_Transaction`) to datetime format
- Classifies customers as churned or active based on last transaction date (`is_churn`)
- Removes unnecessary columns and engineers new time-based features (`Year_First_Transaction`, `Year_Last_Transaction`, `Year_Diff`)

**2. Exploratory Data Analysis (EDA)**
- Customer acquisition trend over year (bar chart)
- Total transactions per year (bar chart)
- Average transaction amount per product over time (point plot)
- Churn proportion by product (pie chart)
- Customer distribution by transaction frequency group
- Customer distribution by average transaction amount group

**3. Feature Engineering**
- Categorizes `Count_Transaction` into 5 groups (1, 2–3, 4–6, 7–10, >10)
- Categorizes `Average_Transaction_Amount` into 8 spending tiers
- Computes `Year_Diff` as the span between first and last transaction year

**4. Machine Learning — Churn Prediction**
- Features: `Average_Transaction_Amount`, `Count_Transaction`, `Year_Diff`
- Target: `is_churn` (binary)
- Model: **Logistic Regression**
- Train/test split: 75% / 25%
- Evaluation: Confusion Matrix, Accuracy, Precision, Recall

---

## 🛠️ Tools & Libraries

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat&logo=python&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=flat&logo=python&logoColor=white)

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
```

---

## 🚀 How to Run

1. Clone this repository
   ```bash
   git clone https://github.com/wirsem/Business-Decision-Research.git
   ```

2. Install required dependencies
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

3. Run the script
   ```bash
   python business-research.py
   ```

---

## 📊 Model Evaluation Output

The script outputs the following metrics after training:

| Metric | Description |
|---|---|
| **Confusion Matrix** | Visualized as a heatmap using Seaborn |
| **Accuracy** | Overall correct prediction rate |
| **Precision** | Proportion of true churn predictions that are correct |
| **Recall** | Proportion of actual churned customers correctly identified |

---

## 👤 Author

**Wira Tarumta Timothy Sembiring**
- 🔗 [LinkedIn](https://linkedin.com/in/wira-tarumta-timothy-sembiring)
- 🐙 [GitHub](https://github.com/wirsem)

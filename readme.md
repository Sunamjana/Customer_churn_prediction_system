# 📞 Telecom Customer Churn Prediction

A end-to-end machine learning project that predicts customer churn for a telecom company — from raw data analysis to a deployed Streamlit application.

---

## 🧩 Business Problem

A telecom company is losing customers and needs to know **who will churn, why, and what to do about it**.

Acquiring a new customer costs 5x more than retaining an existing one. Even a 1% reduction in churn has significant revenue impact. This project builds a data-driven solution that enables the retention team to act before a customer leaves.

---

## 📁 Project Structure

```
telecom-churn/
│
├── data/
│   └── telecom_churn.csv          # Raw dataset
│
├── telecom_churn.ipynb            # Main notebook (EDA + Modeling)
├── app.py                         # Streamlit deployment app
├── churn_pipeline.pkl             # Saved model pipeline (scaler + RF)
├── requirements.txt               # Dependencies
└── README.md
```

---

## 📊 Dataset

- **Source:** Telecom customer dataset
- **Records:** ~3,333 customers
- **Target:** `Churn` (1 = churned, 0 = retained)
- **Class imbalance:** ~14.5% churn rate — handled using SMOTE

**Features used:**

| Feature | Description |
|---|---|
| AccountWeeks | How long the customer has been with the company |
| ContractRenewal | Whether the customer renewed their contract (0/1) |
| DataPlan | Whether the customer has a data plan (0/1) |
| DataUsage | Monthly data usage in GB |
| CustServCalls | Number of customer service calls made |
| DayMins | Total day minutes used |
| DayCalls | Total day calls made |
| MonthlyCharge | Monthly bill amount |
| OverageFee | Extra charges beyond plan |
| RoamMins | Roaming minutes used |

---

## 🔍 Key EDA Findings

**1. Contract renewal is the strongest single predictor**
Customers without a contract churn at 40%+. Contracted customers stay below 15% churn — regardless of any other factor.

**2. Service failure drives churn, not price**
The highest-risk segment (high charge + high service calls + no contract) churns at **44% — 3x the overall rate**. These are 2-year tenure, active users averaging 204 daily minutes. The company is losing its best customers due to unresolved service complaints, not pricing.

**3. Established customers who escalate calls are the real early warning**
Long-tenure customers making 3+ service calls churn at 26.4%, double the 11.4% baseline. New customers with high calls churn at only 15.4% — they are likely asking onboarding questions, not complaining.

**4. Data usage alone does not drive churn**
When controlling for contract renewal, data usage adds almost no predictive signal. Contract renewal dominates.

---

## 🧪 Statistical Validation

| Test | Variables | Result |
|---|---|---|
| Chi-Square | Contract Renewal vs Churn | p < 0.001 — highly significant |
| T-Test | CustServCalls vs Churn | p < 0.001 — significant |
| T-Test | MonthlyCharge vs Churn | p < 0.05 — moderately significant |
| Chi-Square | Combined (Contract + Service Calls) vs Churn | p ≈ 9.73e-70 — extremely significant |

---

## 🤖 Models Trained

| Model | Precision | Recall | F1-Score | AUC-ROC |
|---|---|---|---|---|
| Logistic Regression | — | — | — | — |
| Decision Tree | — | — | — | — |
| Random Forest | — | — | — | — |
| Gradient Boosting | — | — | — | — |
| XGBoost | — | — | — | — |
| ANN (Keras) | — | — | — | — |

> Fill in your actual numbers from the model comparison table output.

**Why Random Forest was selected as the final model:**
- Best balance of recall and precision after threshold tuning
- XGBoost had similar AUC but lower recall on the minority class
- ANN had high recall but too many false positives — wasteful for the retention team

---

## ⚙️ Modeling Pipeline

```
Raw Data
   │
   ▼
Train-Test Split (80/20)
   │
   ▼
SMOTE (handle class imbalance on train set only)
   │
   ▼
StandardScaler + RandomForestClassifier (Pipeline)
   │
   ▼
GridSearchCV (5-fold CV, scoring=recall)
   │
   ▼
Threshold Tuning (0.4 selected over default 0.5)
   │
   ▼
churn_pipeline.pkl
```

**Why recall as the scoring metric?**
It is cheaper to offer a retention discount to a customer who was not going to leave than to miss a customer who was. Minimizing false negatives (missed churners) is the business priority.

**Why threshold = 0.4 instead of 0.5?**
Lowering the threshold increases recall — the model catches more churners. The trade-off is more false positives, which in this context means offering retention discounts to customers who would have stayed anyway. That cost is acceptable.

---

## 🚀 Streamlit App

The app allows the retention team to score any customer in real time before a service call.

**Features:**
- Input customer profile via form
- Outputs churn probability + risk level (Low / Medium / High)
- Explains which risk factors are present for that customer
- Recommends a specific retention action based on the risk profile

**Run locally:**

```bash
pip install -r requirements.txt
streamlit run app.py
```

The pipeline handles scaling internally — no manual preprocessing needed.

---

## 💼 Business Recommendations

Based on the model findings, three specific actions are recommended:

**1. Contract renewal campaigns**
Target medium and high-risk customers 60–90 days before contract expiry with a discount or loyalty reward. Contract renewal is the single most effective retention lever.

**2. Proactive service escalation**
Flag any customer making 3+ service calls in 30 days for outreach by the retention team. Do not wait for them to cancel. Long-tenure customers who escalate calls are the highest churn risk segment.

**3. High-risk segment — dedicated retention agent**
For customers flagged as: high monthly charge + no contract + 3+ service calls — assign a dedicated retention agent, not automated outreach. This segment churns at 44% and represents the company's most experienced, highest-usage customers.



---

## 🛠️ Tech Stack

- **Python** — pandas, numpy, matplotlib, seaborn
- **Scikit-learn** — modeling, pipeline, GridSearchCV, SMOTE
- **XGBoost** — gradient boosting classifier
- **TensorFlow / Keras** — ANN implementation
- **Scipy** — statistical hypothesis testing
- **Joblib** — model serialization
- **Streamlit** — deployment and UI

---

## ▶️ How to Run the Notebook

```bash
git clone https://github.com/your-username/telecom-churn
cd telecom-churn
pip install -r requirements.txt
jupyter notebook telecom_churn.ipynb
```

---

## 📌 Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
xgboost
tensorflow
scipy
joblib
streamlit
```
# Customer Churn Prediction — Telecom
### Business Data Mining · IDS 472

> **Goal:** Predict which telecom customers are at risk of churning and identify the key drivers behind their decision to leave.

---

## Project Overview

Customer churn is one of the most costly problems a subscription-based business faces — losing a customer means losing their recurring revenue AND spending significantly more to acquire a replacement. This project uses the **Telco Customer Churn dataset (7,043 customers)** to build and compare four machine learning models that predict whether a customer will leave, and surfaces the business factors that matter most.

A key challenge from the start: only **26.5% of customers in the dataset actually churned**. A lazy model could hit ~74% accuracy just by predicting "No Churn" for everyone. We built around this by optimizing for **F1 score and ROC-AUC** rather than raw accuracy.

---

## Dataset

- **Source:** Telco Customer Churn dataset
- **Size:** 7,043 customers, 20+ features
- **Target variable:** Churn (Yes/No)
- **Features include:** Demographics, contract type, payment method, tenure, services (internet, phone, streaming, tech support, online security)
- **Class imbalance:** 26.5% churned vs. 73.5% did not churn

---

## Data Preprocessing

- Simplified redundant categories ("No internet service" / "No phone service" → "No") to reduce noise
- Dropped `customerID` (non-predictive unique identifier)
- Fixed `TotalCharges` being stored as text — converted to numeric and filled missing values with the median ($1,397.48) to preserve all 7,043 rows
- Used **Variance Inflation Factor (VIF)** to detect and remove multicollinearity:
  - `MonthlyCharges` VIF: **4,872** — removed
  - `TotalCharges` VIF: **15** — removed
  - All remaining features brought under VIF of 10
- Applied **one-hot encoding** to categorical variables
- **80/20 train-test split**
- Standardized `tenure` for Logistic Regression (mean = 0, std = 1)
- Separate full dataset (including MonthlyCharges and TotalCharges) used for tree-based models, which are robust to multicollinearity

---

## Models & Methodology

Four models were built and compared, representing a clear progression in complexity:

| Model | Why We Used It |

| **Logistic Regression** | Baseline model; interpretable, handles mixed feature types, outputs probabilities for ROC-AUC |

| **Decision Tree** | Visualizable splits; built-in feature importance |

| **Random Forest** | Ensemble upgrade; handles class imbalance better than a single tree |

| **Gradient Boosting** | Sequential tree building; each tree corrects the previous one's mistakes; strongest on imbalanced classification |

**Hyperparameter tuning:** Used `GridSearchCV` with `StratifiedKFold` to preserve class imbalance across every fold. F1 score was the optimization metric throughout.

---

## Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.759 | 0.507 | 0.833 | 0.630 | **0.856** |
| Decision Tree | 0.744 | 0.542 | 0.757 | 0.632 | 0.795 |
| Random Forest | 0.757 | 0.562 | 0.735 | 0.637 | 0.839 |
| Gradient Boosting | 0.784 | 0.677 | 0.488 | 0.567 | 0.833 |

**Best model: Logistic Regression** — highest ROC-AUC (0.856), indicating the relationship between customer features and churn is largely linear. While Gradient Boosting had the highest accuracy, accuracy is a misleading metric here given the class imbalance.

---

## Key Findings & Business Recommendations

**1. Contract type is the #1 churn driver**
Month-to-month customers churn at dramatically higher rates than those on 1 or 2-year contracts. Offering discounts, bundles, or promotional incentives to move customers to longer contracts could meaningfully reduce churn.

**2. Fiber Optic customers churn more — despite it being a premium service**
Fiber Optic had the highest positive coefficient in the Logistic Regression and consistently showed elevated churn across all charge levels. This suggests a value perception problem. Targeted retention outreach and satisfaction surveys for Fiber Optic users are worth exploring.

**3. New customers are the highest risk**
Tenure was strongly linked to churn — newer customers are significantly more likely to leave. A strong onboarding experience, early check-ins, or a welcome loyalty reward could make a significant difference in long-term retention.

**4. Online Security and Tech Support reduce churn**
Customers with these services were less likely to leave, suggesting that feeling protected and supported drives loyalty. Bundling these into starter packages or offering them at reduced rates to new customers could improve early retention.

---

## Limitations

- Dataset is a static snapshot — customer behavior changes over time and models would need regular retraining
- Class imbalance (26.5% churners) was addressed but not fully solved — lower precision scores mean some non-churners get flagged, leading to unnecessary retention spending
- Dropping MonthlyCharges and TotalCharges for Logistic Regression may have removed some useful signal
- Dataset lacks potentially important features like customer service call history, competitor pricing, or recent service outages

---

## Tools & Methods

- **Language:** Python (Jupyter Notebook)
- **Libraries:** pandas, scikit-learn, matplotlib, seaborn
- **Methods:** Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, GridSearchCV, StratifiedKFold, VIF analysis, one-hot encoding, ROC-AUC evaluation

---

## Team

Aleksandra S. · Dan N. · Nabila Baklia · Tanvi G.

# Module 3 – Fraud Detection

## Problem Statement (from the brief)

NovaFin processes thousands of transactions every day.
Management wants to identify suspicious transactions while minimizing unnecessary investigation.

**Dataset:** `resources/data/nova_transactions.csv`

**Tasks**
1. **Exploratory analysis** – fraud rate; fraud by transaction type, amount, location and customer behaviour.
2. **Supervised fraud model** – target `Fraud_Flag`; build models and compare performance.
3. **Unsupervised detection** – Isolation Forest, to flag unusual transactions without using the fraud label. Compare supervised vs anomaly detection.
4. **Cost-based threshold** – missed fraud costs ₹10,000, a false-positive investigation costs ₹500. Find the threshold that minimizes expected operational cost.

## Data at a Glance

- 30,000 transactions, 12 columns, no missing values
- Fraud rate: **2.28%** (~684 fraud cases) – highly imbalanced
- Period: starts 2025-01-01, one transaction roughly every 29 minutes

| Column | Meaning |
|---|---|
| Transaction_ID | Unique ID (not a feature) |
| Customer_ID | Customer who made it (not a feature directly) |
| Timestamp | When it happened (can derive hour, weekday) |
| Amount | Transaction value in ₹ |
| Transaction_Type | Payment / Transfer / Withdrawal |
| Merchant_Category | Retail, Travel, Investment, Utilities, Online, Healthcare, Other |
| Location | One of 7 Indian cities |
| Device_ID | Device used (ID number, not a quantity) |
| Account_Age_Months | How long the account has existed |
| Historical_Avg_Transaction | Customer's usual transaction size |
| Transaction_Frequency | How often the customer transacts |
| Fraud_Flag | 1 = fraud, 0 = genuine (**target**) |

## Key Concepts

- **Imbalanced data:** with 2.28% fraud, a model that says "never fraud" is 97.7% accurate but useless. Use precision, recall, F1 and PR-AUC instead of accuracy.
- **Precision:** of the transactions we flag, how many are really fraud (low precision = wasted investigations).
- **Recall:** of all real fraud, how much we catch (low recall = missed fraud).
- **False negative (missed fraud):** costs ₹10,000. **False positive (false alarm):** costs ₹500. A miss is 20× worse, so the best threshold is usually well below 0.5.
- **Isolation Forest:** an unsupervised model that scores how "unusual" a transaction is, without seeing the fraud label.

## Approach

1. EDA – fraud rates by segment, amount vs customer's usual amount
2. Feature engineering – e.g. `Amount / Historical_Avg_Transaction`, hour of day
3. Supervised models – logistic regression, random forest, gradient boosting / XGBoost
4. Isolation Forest – compare with supervised models
5. Cost-based threshold – total cost = FN × ₹10,000 + FP × ₹500

## Results

### Step 1 – EDA (notebook: `m3_fraud_detection.ipynb`, charts: `results/eda_*.png`)

| Factor | Signal | Finding |
|---|---|---|
| Amount vs customer's usual | **Strong** | Median fraud is ~5× the customer's usual amount (genuine ~1.2×); transactions above 3× usual are 22% of volume but hold ~73% of fraud |
| Amount | **Strong** | Median fraud ₹26,438 vs ₹6,411 genuine; largest 20% of transactions have a 7.6% fraud rate |
| Transaction type | Moderate | Transfers 2.97% vs Payments 2.09%, Withdrawals 2.15% |
| Account age | Weak | Newest accounts (0–6 months) ~2.8% |
| Merchant, location, frequency, hour | Weak | All close to the 2.28% average |

ID columns (`Transaction_ID`, `Customer_ID`, `Device_ID`) are excluded from modelling; each Device_ID is shared by ~20 customers.

**Business message:** fraud is mainly about unusually large transactions compared with the customer's normal behaviour, especially transfers.

### Step 2 – Supervised model (charts: `results/model_*.png`, tables: `results/m3_*.csv`)

Features: log amount, log(amount ÷ customer's usual), account age, frequency, hour, transaction type, merchant, location. Stratified 75/25 split, 5-fold cross-validation.

| Model (test set) | ROC-AUC | PR-AUC | Brier | Fraud caught in top 5% |
|---|---|---|---|---|
| Logistic Regression | 0.794 | 0.121 | 0.179 | 32.2% |
| Random Forest | 0.777 | 0.131 | 0.086 | 33.3% |
| **Gradient Boosting (chosen)** | 0.784 | 0.129 | **0.021** | **35.1%** |
| Baseline: amount ÷ usual | 0.778 | 0.106 | – | 27.5% |

- Random-guess PR-AUC is 0.023, so the models are ~5× better than random. Models differ by less than the cross-validation spread.
- A time-based check (train before Apr 2026, test Apr–Aug 2026) gives similar scores, so the patterns are stable.
- Main drivers: transaction amount and how unusual it is for the customer, then transfers.
- **Gradient Boosting** chosen for its well-calibrated probabilities (needed for Step 4); Logistic Regression kept as the explainable challenger.
- Limitation: the available features separate fraud only moderately; richer data would matter more than model choice.

### Step 3 – Isolation Forest (charts: `results/iforest_*.png`)

Trained on the same numeric features without using `Fraud_Flag`; `contamination=0.05`.

| Test set | ROC-AUC | PR-AUC | Fraud caught in top 5% |
|---|---|---|---|
| Gradient Boosting (supervised) | 0.784 | 0.129 | 35.1% |
| Isolation Forest (unsupervised) | 0.676 | 0.054 | 17.0% |
| Baseline: amount ÷ usual | 0.778 | 0.106 | 27.5% |
| Random | 0.500 | 0.023 | 5.0% |

- Isolation Forest beats random but is about half as effective as the supervised model.
- Reason: it flags anything unusual in either direction. 55% of its anomalies are **below** the customer's usual amount (median ₹2,419, against ₹25,766 for fraud).
- Of 171 test frauds: 21 caught by both, 39 only by GB, 8 only by IF, 103 by neither. Blending gives no reliable gain.
- **Recommendation:** supervised model for alerts; Isolation Forest for monitoring new fraud patterns, data drift, and new products without labels.

### Step 4 – Cost-based threshold (charts: `results/cost_*.png`, tables: `results/m3_policy_comparison.csv`, `results/m3_cost_sensitivity.csv`)

Total cost = missed frauds × ₹10,000 + false alarms × ₹500. Threshold chosen on out-of-fold training predictions, checked on the test set.

- Theory: investigate when p > 500 / (500 + 10,000) = **0.0476**. The data-driven threshold is **0.0475**, which confirms the probabilities are well calibrated.

| Policy (test set, 7,500 tx, 171 frauds) | Investigated | Fraud caught | Total cost | Saving |
|---|---|---|---|---|
| Do nothing | 0 | 0 | ₹17.1 lakh | – |
| Investigate everything | 7,500 | 171 | ₹36.6 lakh | −114% |
| Default threshold 0.5 | 0 | 0 | ₹17.1 lakh | 0% |
| Rule: > 3× usual amount | 1,666 | 123 | ₹12.5 lakh | 27% |
| **Cost-optimal threshold 0.0475** | **554** | **84** | **₹11.1 lakh** | **35%** |

- About 1 in 7 investigations finds fraud; ~49% of fraud is caught.
- Sensitivity: the threshold follows the cost ratio; savings range from 1% to 56% across the cost scenarios.
- Illustrative scale (assumed 5,000 tx/day): ~₹14.7 crore saved per year, needing ~370 investigations/day.

## Conclusions

1. Fraud is driven by transactions that are unusually large for the customer, and is more common in transfers.
2. Use the supervised Gradient Boosting model, with Logistic Regression as the challenger; Isolation Forest only for monitoring.
3. **Investigate transactions with a fraud probability above ~4.8%**, not the default 50%. This cuts fraud-related cost by ~35%.
4. Risks: moderate signal in the data (about half of fraud is missed), synthetic data, assumed costs, investigation capacity, model drift. Re-validate on real data and retrain regularly.
5. Link to Module 11: effective fraud controls make growth in digital banking safer.

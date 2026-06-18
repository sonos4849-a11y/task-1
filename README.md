# task-1
## Project Overview: Telco Customer Churn Prediction

This project aimed to build a predictive model to identify customers likely to churn from a telecommunications company. The process involved comprehensive data analysis, feature engineering, model training, evaluation, and the development of a framework for an automated prediction pipeline. The ultimate goal is to enable proactive retention strategies by identifying high-risk customers.

### 1. Data Loading and Initial Exploration

*   **Data Source:** The project utilized the `WA_Fn-UseC_-Telco-Customer-Churn.csv` dataset, which contains various customer attributes and their churn status.
*   **Initial Checks:**
    *   Loaded the dataset into a pandas DataFrame.
    *   Inspected the first few rows (`df.head()`) to understand the data structure and content.
    *   Checked data types and non-null counts (`df.info()`) to identify potential issues.
    *   Identified and dropped the `customerID` column as it's a unique identifier and not a predictive feature.
    *   Converted the `TotalCharges` column from object to numeric, coercing errors to `NaN` and then dropping rows with missing `TotalCharges` values.
    *   Confirmed no other missing values remained after `TotalCharges` handling using `df.isnull().sum()` and `missingno.matrix()`.
    *   Generated summary statistics (`df.describe()`) for numerical and categorical features.

### 2. Exploratory Data Analysis (EDA)

EDA was performed to understand data distributions, relationships between variables, and identify potential patterns:

*   **Churn Distribution:** Visualized the overall distribution of churned vs. non-churned customers using a count plot, revealing an imbalanced dataset.
*   **Churn Rate by Contract Type:** Analyzed how churn rates vary across different contract types (`Month-to-month`, `One year`, `Two year`), highlighting that month-to-month contracts have significantly higher churn.
*   **Monthly Charges Distribution:** Plotted the distribution of `MonthlyCharges` to understand typical billing amounts.
*   **Outlier Analysis:** Used box plots for `tenure`, `MonthlyCharges`, and `TotalCharges` to identify and understand the presence of outliers, noting some higher values in charges and tenure.

### 3. Feature Engineering

Several new features were created and existing ones transformed to enhance the predictive power of the models:

*   **Standardizing Service Columns:** Values like 'No phone service' and 'No internet service' were harmonized to 'No' across relevant service columns for consistency.
*   **Binary Feature Encoding:** All binary categorical features (e.g., `gender`, `Partner`, `PhoneService`, `PaperlessBilling`, and various service-related flags) were converted into numerical representations (0s and 1s). The `Churn` target variable was also encoded as 0 for 'No' and 1 for 'Yes'.
*   **`tenure` Binning:** The continuous `tenure` feature was binned into five categorical groups (e.g., '0-12 Mths', '12-24 Mths') to capture non-linear relationships and potentially aid interpretation.
*   **Interaction Features:** New features were engineered by multiplying `MonthlyCharges` with binary indicators for each `Contract` type (`MonthlyCharges_MonthToMonth`, `MonthlyCharges_OneYear`, `MonthlyCharges_TwoYear`). This captures how monthly charges impact churn differently based on the contract.
*   **`TotalServices` Feature:** A `TotalServices` feature was created by summing up the number of services a customer subscribes to, serving as a proxy for customer engagement or usage frequency.
*   **One-Hot Encoding:** The remaining nominal categorical features (`Contract`, `InternetService`, `PaymentMethod`, and the new `tenure_group`) were one-hot encoded to convert them into a format suitable for machine learning algorithms, avoiding ordinal assumptions.

### 4. Data Preparation for Modeling

Before training, the data was prepared as follows:

*   **Feature-Target Split:** The dataset was split into features (`X`) and the target variable (`y`, which is `Churn`).
*   **Feature Scaling:** Numerical features in `X` were scaled using `StandardScaler` to have a mean of 0 and a standard deviation of 1. This is crucial for algorithms sensitive to feature magnitudes.
*   **Train-Test Split:** The data was divided into training (80%) and testing (20%) sets (`X_train`, `X_test`, `y_train`, `y_test`) using stratification to maintain the original churn ratio in both sets.

### 5. Model Building and Evaluation

Three different classification models were trained and rigorously evaluated using common metrics:

*   **Logistic Regression (Baseline Model):**
    *   **Accuracy:** 0.7903
    *   **Precision (Churn):** 0.6278
    *   **Recall (Churn):** 0.5187
    *   **F1-Score (Churn):** 0.5681
    *   Provided a strong baseline, demonstrating good overall performance with interpretable coefficients.

*   **Random Forest Classifier:**
    *   **Accuracy:** 0.7740
    *   **Precision (Churn):** 0.5915
    *   **Recall (Churn):** 0.4840
    *   **F1-Score (Churn):** 0.5324
    *   **ROC AUC:** 0.8187
    *   An ensemble method that generally offers robustness and good predictive power. It showed a strong ROC AUC.

*   **XGBoost Classifier:**
    *   **Accuracy:** 0.7683
    *   **Precision (Churn):** 0.5706
    *   **Recall (Churn):** 0.5187
    *   **F1-Score (Churn):** 0.5434
    *   **ROC AUC:** 0.8113
    *   A highly efficient and powerful gradient boosting algorithm. Its performance was competitive, particularly matching Logistic Regression's recall.

**Comparative Summary of Model Performance:**

| Metric        | Logistic Regression | Random Forest | XGBoost |
| :------------ | :------------------ | :------------ | :------ |
| **Accuracy**  | 0.7903              | 0.7740        | 0.7683  |
| **Precision** | 0.6278              | 0.5915        | 0.5706  |
| **Recall**    | 0.5187              | 0.4840        | 0.5187  |
| **F1-Score**  | 0.5681              | 0.5324        | 0.5434  |
| **ROC AUC**   | N/A                 | 0.8187        | 0.8113  |

Logistic Regression currently stands out with the highest accuracy and precision. Random Forest achieved the best ROC AUC, indicating its ability to distinguish between classes across various thresholds. XGBoost offers comparable recall.

### 6. Churn Risk Categorization

To make the model's output more actionable, churn probabilities from the XGBoost model were used to assign risk categories to customers:

*   **Thresholds:** Probabilities were categorized into:
    *   **Low Risk:** `P(Churn) < 0.3`
    *   **Medium Risk:** `0.3 <= P(Churn) < 0.7`
    *   **High Risk:** `P(Churn) >= 0.7`
*   **Visualization:** A count plot was generated to show the distribution of customers across these risk categories in the test set, providing an immediate understanding of the customer base's churn risk profile.

### 7. Automated Weekly Prediction Pipeline

To operationalize the churn prediction, a framework for an automated weekly pipeline was developed:

*   **Model and Scaler Persistence:** The trained `XGBoost model` and the `StandardScaler` were saved using `joblib` (`xgboost_churn_model.joblib` and `scaler.joblib`, respectively) to ensure they can be loaded and reused consistently without retraining.
*   **`preprocess_new_data` Function:** A comprehensive function was created to encapsulate all the data cleaning and feature engineering steps. This ensures that any new raw data fed into the pipeline is transformed identically to the training data.
*   **`make_predictions_and_assign_risk` Function:** This function integrates the loading of the saved model and scaler, applies the `preprocess_new_data` function to new raw customer data, generates churn probabilities, and then assigns the 'Low', 'Medium', or 'High' risk categories based on predefined thresholds.
*   **Simulation:** The pipeline's functionality was demonstrated by simulating new weekly data (sampling from the original dataset) and processing it through the `make_predictions_and_assign_risk` function, producing a DataFrame with predicted churn probabilities and risk categories.

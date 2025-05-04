# Breast Cancer Classification using Logistic Regression

This repository contains the Breast Cancer Classification using Logistic Regression, which involves building and evaluating a binary classification model using Logistic Regression to predict breast cancer diagnosis based on diagnostic features.

## Objective
To implement a logistic regression classifier to distinguish between Malignant (M) and Benign (B) breast tumors using the Breast Cancer Wisconsin (Diagnostic) dataset. The task includes data preprocessing, model training, evaluation using various classification metrics, and understanding the concepts of the sigmoid function and threshold tuning.

## Dataset
- **Source:** Breast Cancer Wisconsin (Diagnostic) Data Set
- **File:** `Data Set.csv`
- **Description:** The dataset contains features computed from digitized images of fine needle aspirates (FNA) of breast mass samples. These features describe characteristics of the cell nuclei present in the images.
- **Target Variable:** `diagnosis` (Categorical: 'M' = Malignant, 'B' = Benign). This was encoded as `1` for Malignant and `0` for Benign during preprocessing.

## Files in this Repository
- `BreastCancer_Logistic_Regression.ipynb`: Jupyter Notebook containing all the Python code for data loading, preprocessing, model training, evaluation, and visualization.
-  `BreastCancer_LR.py`: Contains the code. 
- `README.md`: This explanatory file.
- `.gitignore` (Optional): Specifies files intentionally untracked by Git (e.g., virtual environment files).

## Tools and Libraries Used
- Python 3.x
- Pandas: For data loading and manipulation.
- NumPy: For numerical operations.
- Scikit-learn:
    - `train_test_split`: For splitting data.
    - `StandardScaler`: For feature scaling.
    - `LogisticRegression`: The classification model.
    - `confusion_matrix`, `classification_report`, `accuracy_score`, `precision_score`, `recall_score`, `f1_score`: For model evaluation.
    - `roc_curve`, `auc`, `RocCurveDisplay`: For ROC curve analysis.
- Matplotlib & Seaborn: For data visualization (Confusion Matrix, ROC Curve, Sigmoid function plot).
- Jupyter Notebook: For interactive code development and documentation.

## Methodology / Steps Taken
1.  **Load Data:** Imported the dataset from the provided CSV file using Pandas.
2.  **Inspect & Clean:** Examined the data structure (`info()`, `head()`, `describe()`), dropped the irrelevant `id` column, handled potential empty columns from CSV formatting, and checked for missing values.
3.  **Encode Target:** Converted the categorical `diagnosis` column ('M', 'B') into numerical format (`1` for Malignant, `0` for Benign).
4.  **Feature/Target Split:** Separated the dataset into features (X) and the target variable (y).
5.  **Train/Test Split:** Divided the data into training (70%) and testing (30%) sets using `train_test_split`, ensuring stratification (`stratify=y`) to maintain class proportions.
6.  **Feature Scaling:** Standardized the numerical features using `StandardScaler` (fitted *only* on the training data, then transformed both training and testing data).
7.  **Model Training:** Initialized and trained a `LogisticRegression` model using the scaled training data (`X_train_scaled`, `y_train`).
8.  **Prediction:** Generated class predictions (`predict`) and probability predictions (`predict_proba`) for the scaled test set (`X_test_scaled`).
9.  **Evaluation:** Assessed the model's performance on the test set using:
    *   Accuracy Score.
    *   Confusion Matrix (visualized).
    *   Classification Report (Precision, Recall, F1-Score per class).
    *   ROC Curve and AUC Score.
10. **Sigmoid Function:** Explained and visualized the sigmoid function, which forms the core of logistic regression's probability estimation.
11. **Threshold Tuning:** Demonstrated how changing the classification threshold (default 0.5) affects predictions and impacts metrics like precision and recall, showing the trade-off involved.

## Key Results

- **Target Encoding:** Malignant ('M') = 1, Benign ('B') = 0.
- **Model Performance (Default Threshold = 0.5):**
    - Accuracy: `0.9708`
    - ROC AUC Score: `0.9975`
- **Performance for Malignant Class (Class 1):**
    - Precision: `0.9836` (Of those predicted Malignant, this proportion actually were).
    - Recall: `0.9375` (Of all actual Malignant cases, this proportion were correctly identified).
    - F1-Score: `0.9600`
- **Confusion Matrix Insights:** The model correctly identified `60` Malignant cases and `106` Benign cases. There were `4` False Negatives (Malignant cases missed) and `1` False Positives (Benign cases misclassified as Malignant).
- **Threshold Tuning:**
    - Lowering the threshold to `0.3` increased Recall for Malignant to `0.9688` but decreased Precision to `0.9841`.
    - Increasing the threshold to `0.7` increased Precision for Malignant to `1.0000` but decreased Recall to `0.9219`.
    - This highlights the importance of choosing a threshold based on whether minimizing False Negatives (prioritize Recall) or False Positives (prioritize Precision) is more critical.

## How to Run
1.  Clone this repository to your local machine:
2.  Navigate to the cloned directory:
3.  Ensure you have Python and the required libraries installed. You can typically install them using pip:
4.  Launch Jupyter Notebook:
5.  Open the `BreastCancer_Logistic_Regression.ipynb` file in Jupyter and run the cells sequentially.

## Questions and Answers

### 1. How does logistic regression differ from linear regression?

*   **Output:** Linear Regression predicts a continuous value. Logistic Regression predicts the probability of a binary outcome (between 0 and 1), which is then often converted to a class label.
*   **Function:** Linear Regression uses an identity function (output = linear combination). Logistic Regression passes the linear combination through a *sigmoid (logistic) function* to constrain the output to (0, 1).
*   **Goal:** Linear Regression aims to model the mean of a continuous dependent variable. Logistic Regression aims to model the probability of a specific class occurring in a binary classification problem.
*   **Assumptions:** Linear Regression assumes normally distributed errors with constant variance. Logistic Regression assumes errors follow a binomial distribution.

### 2. What is the sigmoid function?

*   It's a mathematical function with an "S" shaped curve.
*   **Formula:** `σ(z) = 1 / (1 + e^(-z))` where `z` is the linear combination output (`b₀ + b₁x₁ + ...`).
*   **Purpose:** It maps any real-valued input `z` to an output value between 0 and 1. In logistic regression, this output is interpreted as the probability of the positive class (P(Y=1)).

### 3. What is precision vs recall?

*   **Precision:** Measures the accuracy of positive predictions. Out of all instances predicted as positive, what proportion were actually positive?
    `Precision = TP / (TP + FP)`. High precision means few false positives.
*   **Recall (Sensitivity / True Positive Rate):** Measures how well the model finds all the actual positive instances. Out of all actual positive instances, what proportion did the model correctly identify?
    `Recall = TP / (TP + FN)`. High recall means few false negatives.
*   **Trade-off:** Often, increasing precision decreases recall, and vice-versa. The choice depends on whether false positives or false negatives are more costly for the specific application.

### 4. What is the ROC-AUC curve?

*   **ROC (Receiver Operating Characteristic) Curve:** A plot of the True Positive Rate (Recall) against the False Positive Rate (`FPR = FP / (FP + TN)`) at various classification threshold settings.
*   **AUC (Area Under the Curve):** The area under the ROC curve. It provides a single aggregate measure of a classifier's performance across all possible thresholds.
    *   AUC = 1: Perfect classifier.
    *   AUC = 0.5: Performance equivalent to random guessing.
    *   AUC < 0.5: Performance worse than random guessing.
*   **Significance:** AUC measures the model's ability to correctly rank positive instances higher than negative instances. It's useful for comparing models and is less sensitive to class imbalance than accuracy.

### 5. What is the confusion matrix?

*   A table used to evaluate the performance of a classification algorithm. It compares the actual target values with those predicted by the model.
*   **Structure (for binary classification):**
    *   **True Positives (TP):** Correctly predicted positive instances.
    *   **True Negatives (TN):** Correctly predicted negative instances.
    *   **False Positives (FP):** Incorrectly predicted positive instances (Type I Error).
    *   **False Negatives (FN):** Incorrectly predicted negative instances (Type II Error).
*   It provides the raw counts needed to calculate metrics like accuracy, precision, recall, specificity, etc.

### 6. What happens if classes are imbalanced?

Class imbalance occurs when one class is much more frequent than the other(s).

*   **Problems:**
    *   Models (especially those optimizing for accuracy) tend to become biased towards the majority class, predicting it more often.
    *   The model might achieve high accuracy simply by predicting the majority class always, while performing poorly on the minority class (which is often the class of interest).
    *   Standard accuracy becomes a misleading metric.
*   **Solutions:**
    *   **Resampling:** Undersample the majority class or oversample the minority class (e.g., SMOTE).
    *   **Class Weights:** Assign higher misclassification costs to the minority class during model training (e.g., `class_weight='balanced'` in `LogisticRegression`).
    *   **Use Appropriate Metrics:** Focus on metrics like Precision, Recall, F1-Score, AUC, or the Precision-Recall curve instead of just accuracy.
    *   **Algorithmic Approaches:** Use algorithms inherently better suited for imbalance (e.g., some tree-based methods).

### 7. How do you choose the threshold?

*   The default threshold for converting probability to a class label is usually 0.5.
*   The optimal threshold depends on the **business objective** and the **relative cost of False Positives vs. False Negatives**:
    *   If **minimizing False Negatives** is critical (e.g., disease detection - don't want to miss a sick patient), **lower the threshold** (increases Recall, might decrease Precision).
    *   If **minimizing False Positives** is critical (e.g., spam filtering - don't want to block legitimate emails), **raise the threshold** (increases Precision, might decrease Recall).
*   **Tools for choosing:**
    *   **ROC Curve:** Can help find a threshold that balances TPR and FPR.
    *   **Precision-Recall Curve:** Particularly useful for imbalanced datasets; helps find a threshold that balances precision and recall.
    *   Domain knowledge and cost analysis.

### 8. Can logistic regression be used for multi-class problems?

*   **Yes.** Standard logistic regression is binary, but it can be extended for multi-class classification (where there are more than two possible outcome classes) using techniques like:
    *   **One-vs-Rest (OvR) / One-vs-All (OvA):** Trains a separate binary logistic regression classifier for each class, pitting that class against all other classes combined. The class whose classifier outputs the highest probability is chosen. This is the default strategy in `sklearn.linear_model.LogisticRegression`.
    *   **Multinomial Logistic Regression (Softmax Regression):** Generalizes logistic regression directly to handle multiple classes by using the Softmax function instead of the sigmoid function to calculate probabilities for all classes simultaneously. You can enable this in `sklearn` by setting `multi_class='multinomial'`.

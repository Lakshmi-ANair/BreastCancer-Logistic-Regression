import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn imports
# No need for load_breast_cancer now
from sklearn.model_selection import train_test_split # To split data
from sklearn.preprocessing import StandardScaler # To standardize features
from sklearn.linear_model import LogisticRegression # The classification model
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score # Evaluation metrics
from sklearn.metrics import precision_score, recall_score, f1_score # More metrics
from sklearn.metrics import roc_curve, auc, RocCurveDisplay # For ROC-AUC curve

# Configure settings
%matplotlib inline
plt.style.use('seaborn-v0_8-darkgrid')
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x) # Adjust precision if needed
import warnings
warnings.filterwarnings('ignore')

print("Libraries imported successfully.")

# 1. Load data from CSV
file_path = 'Breast_Cancer_Wisconsin_Diagnostic_Data_Set.csv' # Make sure filename matches
df = pd.read_csv(file_path)

print(f"Original DataFrame shape: {df.shape}")
print("Original columns:", df.columns)

# 2. Initial Inspection & Cleanup
print("\nFirst 5 Rows (Original):")
display(df.head())

print("\nDataset Information:")
df.info()

# --- Data Cleaning ---
# Drop the 'id' column as it's not a predictor
if 'id' in df.columns:
    df = df.drop('id', axis=1)
    print("\nDropped 'id' column.")

# Handle potential empty trailing column if it exists (check df.info() or df.columns)
# Common names are 'Unnamed: 32', etc.
# Let's check for columns starting with 'Unnamed'
unnamed_cols = [col for col in df.columns if col.startswith('Unnamed:')]
if unnamed_cols:
    df = df.drop(unnamed_cols, axis=1)
    print(f"Dropped unnamed columns: {unnamed_cols}")

print("\nChecking for Missing Values after initial cleanup:")
print(df.isnull().sum())
# Optional: Handle missing values if any are found (e.g., df.dropna() or imputation)
# This dataset typically doesn't have missing values after dropping the empty column.
print(f"Total missing values: {df.isnull().sum().sum()}")

# 3. Encode the Target Variable ('diagnosis')
# Convert 'M' (Malignant) and 'B' (Benign) to numerical format (e.g., 1 and 0)
# Let's choose M=1 (Malignant - often the class of interest) and B=0 (Benign)
if 'diagnosis' in df.columns:
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    print("\nEncoded 'diagnosis' column: Mapped 'M' to 1, 'B' to 0.")
    print("Target Variable Distribution (after encoding):")
    print(df['diagnosis'].value_counts())
else:
    print("\nError: 'diagnosis' column not found!")

# Define target names based on our mapping
target_names = ['Benign (0)', 'Malignant (1)']

# 4. Separate Features (X) and Target (y)
X = df.drop('diagnosis', axis=1) # Features are all remaining columns
y = df['diagnosis']             # Encoded target variable

print("\n--- Final Shapes ---")
print("Shape of Features (X):", X.shape)
print("Shape of Target (y):", y.shape)
print("\nFeatures columns:", X.columns.tolist())
print("\nSample of Features (X):")
display(X.head(3))
print("\nSample of Target (y):")
print(y.head(3).to_string()) # .to_string() for better display of Series head

# Split data into training and testing sets
# Stratify=y ensures class proportions are maintained in train/test splits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print("--- Data Split Shapes ---")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

print("\nTraining set target distribution:")
print(y_train.value_counts(normalize=True))

print("\nTest set target distribution:")
print(y_test.value_counts(normalize=True))

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler ONLY on the training data features
scaler.fit(X_train)

# Transform both the training and testing data features
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Optional: Convert scaled arrays back to DataFrames for inspection
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

print("--- Feature Scaling Completed ---")
print("Scaled X_train_scaled description (Mean ~0, Stddev ~1):")
display(X_train_scaled_df.describe())

# Initialize the Logistic Regression model
model = LogisticRegression(random_state=42) # Default uses L2 regularization

# Train (fit) the model using the SCALED training data
model.fit(X_train_scaled, y_train)

print("--- Model Training ---")
print("Logistic Regression model trained successfully.")

# Display coefficients (optional, but insightful)
print("\nModel coefficients (weights for each feature):")
# Create a DataFrame for coefficients for better readability
# model.coef_[0] gets the coefficients for the positive class (Malignant=1)
coef_df = pd.DataFrame(model.coef_[0], index=X.columns, columns=['Coefficient'])
# Display coefficients sorted by magnitude (absolute value) to see most influential features
coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()
print(coef_df.sort_values('Abs_Coefficient', ascending=False).drop('Abs_Coefficient', axis=1))

print("\nModel intercept:", model.intercept_[0])

# Predict class labels (0 or 1) on the SCALED test data (using default 0.5 threshold)
y_pred = model.predict(X_test_scaled)

# Predict probabilities for each class [Prob(Class 0), Prob(Class 1)]
y_pred_proba = model.predict_proba(X_test_scaled)

# Extract probabilities for the positive class (Malignant = 1)
y_pred_proba_positive = y_pred_proba[:, 1]

print("--- Predictions ---")
print("Sample Predicted Labels (y_pred):", y_pred[:10])
print("Sample Actual Labels      (y_test):", y_test.values[:10])
print("\nSample Predicted Probabilities [Prob(Benign=0), Prob(Malignant=1)]:")
print(y_pred_proba[:5])

print("--- Model Evaluation (Default Threshold 0.5) ---")

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Plotting the confusion matrix
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, # Use defined names ['Benign (0)', 'Malignant (1)']
            yticklabels=target_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (0=Benign, 1=Malignant)')
plt.show()

tn, fp, fn, tp = cm.ravel()
print(f"True Negatives (TN - Correctly predicted Benign): {tn}")
print(f"False Positives (FP - Predicted Malignant, was Benign): {fp}")
print(f"False Negatives (FN - Predicted Benign, was Malignant): {fn}")
print(f"True Positives (TP - Correctly predicted Malignant): {tp}")

# --- Classification Report ---
report = classification_report(y_test, y_pred, target_names=target_names)
print("\nClassification Report:")
print(report)

# --- Individual Metrics ---
accuracy = accuracy_score(y_test, y_pred)
# Calculate precision, recall, F1 for the POSITIVE class (Malignant = 1)
# Explicitly set pos_label=1
precision_malignant = precision_score(y_test, y_pred, pos_label=1)
recall_malignant = recall_score(y_test, y_pred, pos_label=1)
f1_malignant = f1_score(y_test, y_pred, pos_label=1)

print("\n--- Key Metrics Summary ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (for Malignant=1): {precision_malignant:.4f}")
print(f"Recall (Sensitivity for Malignant=1): {recall_malignant:.4f}")
print(f"F1-Score (for Malignant=1): {f1_malignant:.4f}")

# Calculate FPR, TPR for the ROC curve
# Use the probabilities of the POSITIVE class (Malignant = 1)
fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred_proba_positive, pos_label=1)

# Calculate AUC
roc_auc = auc(fpr, tpr)

print("\n--- ROC Curve and AUC ---")
print(f"Area Under the Curve (AUC): {roc_auc:.4f}")

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guessing')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity/Recall)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# --- Sigmoid Function Explanation ---
# (Code is the same as previous example)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z_values = np.linspace(-10, 10, 200)
probabilities = sigmoid(z_values)

plt.figure(figsize=(8, 5))
plt.plot(z_values, probabilities, label='Sigmoid Function')
plt.axhline(y=0.5, color='red', linestyle='--', label='Default Threshold = 0.5')
plt.axhline(y=0.0, color='grey', linestyle=':', lw=1)
plt.axhline(y=1.0, color='grey', linestyle=':', lw=1)
plt.xlabel('z (Linear Combination Input)')
plt.ylabel('Probability P(Y=1)')
plt.title('Sigmoid (Logistic) Function')
plt.legend()
plt.grid(True)
plt.ylim(-0.05, 1.05) # Adjust y-limits for clarity
plt.show()

print("The sigmoid function maps the linear model output 'z' to a probability between 0 and 1.")
print("This probability is used to classify the instance, typically using a 0.5 threshold.")

print("\n--- Threshold Tuning ---")
print("Default threshold (0.5) results (from Cell 7):")
print(f"  Precision (Malignant): {precision_malignant:.4f}")
print(f"  Recall (Malignant):    {recall_malignant:.4f}")
print(f"  F1-Score (Malignant):  {f1_malignant:.4f}")

# --- Try a LOWER threshold (e.g., 0.3) to increase RECALL for Malignant ---
# This means we classify as Malignant even if probability is only >= 0.3
threshold_low = 0.3
y_pred_low_threshold = (y_pred_proba_positive >= threshold_low).astype(int)

print(f"\n--- Evaluation with Threshold = {threshold_low} ---")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_low_threshold))
print("\nClassification Report:\n", classification_report(y_test, y_pred_low_threshold, target_names=target_names))
print(f"  Precision (Malignant): {precision_score(y_test, y_pred_low_threshold, pos_label=1):.4f}")
print(f"  Recall (Malignant):    {recall_score(y_test, y_pred_low_threshold, pos_label=1):.4f}")
print(f"  F1-Score (Malignant):  {f1_score(y_test, y_pred_low_threshold, pos_label=1):.4f}")
print("--> Note: Recall likely increased, Precision likely decreased.")

# --- Try a HIGHER threshold (e.g., 0.7) to increase PRECISION for Malignant ---
# This means we only classify as Malignant if probability is >= 0.7
threshold_high = 0.7
y_pred_high_threshold = (y_pred_proba_positive >= threshold_high).astype(int)

print(f"\n--- Evaluation with Threshold = {threshold_high} ---")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_high_threshold))
print("\nClassification Report:\n", classification_report(y_test, y_pred_high_threshold, target_names=target_names))
print(f"  Precision (Malignant): {precision_score(y_test, y_pred_high_threshold, pos_label=1):.4f}")
print(f"  Recall (Malignant):    {recall_score(y_test, y_pred_high_threshold, pos_label=1):.4f}")
print(f"  F1-Score (Malignant):  {f1_score(y_test, y_pred_high_threshold, pos_label=1):.4f}")
print("--> Note: Precision likely increased, Recall likely decreased.")

print("\nConclusion: The best threshold depends on the application.")
print("For cancer detection, minimizing False Negatives (maximizing Recall for Malignant) is often prioritized,")
print("which might suggest using a threshold lower than 0.5, accepting lower precision.")

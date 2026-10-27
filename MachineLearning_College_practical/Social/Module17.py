import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# --- 1. Load Data & Preprocess ---
df = pd.read_csv('C:/Users/91970/OneDrive/Desktop/Parthml/Social/social_network_ads.csv')
df = df.drop_duplicates()

# Define features and target
X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Standardise features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --- Objective 1: Train an RBF Kernel SVM using available customer features ---
svm_rbf = SVC(kernel='rbf', random_state=42)
svm_rbf.fit(X_train_scaled, y_train)

# Predict on test set
y_pred = svm_rbf.predict(X_test_scaled)


# --- Objective 2: Identify false positives and false negatives from test-set predictions ---
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"Confusion Matrix Counts -> TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")

# Create a DataFrame containing the misclassified customers from the test set
X_test_df = pd.DataFrame(X_test_scaled, columns=['Age', 'EstimatedSalary'], index=X_test.index)
X_test_df['Actual_Purchased'] = y_test.values
X_test_df['Predicted_Purchased'] = y_pred

# Filter out rows where actual and predicted do not match
misclassified_df = X_test_df[X_test_df['Actual_Purchased'] != X_test_df['Predicted_Purchased']]

print(f"\nTotal Misclassified Customers in Test Set: {len(misclassified_df)}")
print("\n--- Sample of Misclassified Customers ---")
print(misclassified_df.head(10))
print("-----------------------------------------------")


# --- Objective 3: Calculate precision and recall, and discuss implications ---
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("\n--- Performance Metrics ---")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print("-----------------------------------------------")

# Discussion of Practical Implications
print("\n--- Practical Implications of Errors ---")
print("1. False Positives (FP): Model predicts the customer WILL buy, but they do NOT.")
print("   -> Implication: Wasted marketing budget (e.g., spending money sending target ads/promotions to uninterested users).")
print("\n2. False Negatives (FN): Model predicts the customer will NOT buy, but they DO.")
print("   -> Implication: Lost sales opportunity (e.g., missing out on a potential buyer because no ad was targeted at them).")
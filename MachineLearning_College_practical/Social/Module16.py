import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.inspection import DecisionBoundaryDisplay

# --- 1. Load Data & Preprocess ---
df = pd.read_csv('C:/Users/91970/OneDrive/Desktop/Parthml/Social/social_network_ads.csv')
df = df.drop_duplicates()

# Select features (Age and EstimatedSalary for 2D visualization) and target
X = df[['Age', 'EstimatedSalary']]
Y = df['Purchased']

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# --- Objective 1: Standardise Age and Estimated Salary ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --- Objective 2: Train Linear and RBF Kernel SVM Classifiers ---
linear_svm = SVC(kernel='linear', random_state=42)
linear_svm.fit(X_train_scaled, Y_train)

rbf_svm = SVC(kernel='rbf', random_state=42)
rbf_svm.fit(X_train_scaled, Y_train)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# --- 1. Linear SVM Decision Boundary ---
DecisionBoundaryDisplay.from_estimator(
    linear_svm,                # Trained model
    X_train_scaled,            # Training data features
    response_method="predict", # Use prediction labels for regions
    cmap=plt.cm.coolwarm,      # Color scheme
    alpha=0.6,                 # Background transparency
    ax=axes[0]                 # Selects the first subplot
)
axes[0].scatter(
    X_train_scaled[:, 0], X_train_scaled[:, 1], 
    c=Y_train, cmap=plt.cm.coolwarm, edgecolors='k'
)
axes[0].set_title('Linear SVM')
axes[0].set_xlabel('Standardized Age')
axes[0].set_ylabel('Standardized Estimated Salary')

# --- 2. RBF SVM Decision Boundary ---
DecisionBoundaryDisplay.from_estimator(
    rbf_svm,                   # Trained model
    X_train_scaled,            # Training data features
    response_method="predict", # Use prediction labels for regions
    cmap=plt.cm.coolwarm,      # Color scheme
    alpha=0.6,                 # Background transparency
    ax=axes[1]                 # Selects the second subplot
)
axes[1].scatter(
    X_train_scaled[:, 0], X_train_scaled[:, 1], 
    c=Y_train, cmap=plt.cm.coolwarm, edgecolors='k'
)
axes[1].set_title('RBF SVM')
axes[1].set_xlabel('Standardized Age')
axes[1].set_ylabel('Standardized Estimated Salary')

plt.tight_layout()
plt.show()

# --- Objective 3: Compare Test-Set Performance ---
y_pred_linear = linear_svm.predict(X_test_scaled)
y_pred_rbf = rbf_svm.predict(X_test_scaled)

comparison_q16 = pd.DataFrame({
    'SVM Model Kernel': ['Linear Kernel', 'RBF Kernel'],
    'Test Accuracy': [
        accuracy_score(Y_test, y_pred_linear), 
        accuracy_score(Y_test, y_pred_rbf)
    ]
})

print("--- Question 16: Linear vs RBF SVM Performance Comparison ---")
print(comparison_q16)
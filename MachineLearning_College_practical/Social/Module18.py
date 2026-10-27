import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split , cross_val_score , GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error , r2_score , accuracy_score , precision_score , recall_score , f1_score , confusion_matrix
from sklearn.preprocessing import LabelEncoder , StandardScaler

from sklearn.svm import SVC

df = pd.read_csv('C:/Users/91970/OneDrive/Desktop/Parthml/Social/social_network_ads.csv')

print(df.head())
print("-----------------------------------------------")
print(df.info())
print("-----------------------------------------------")
print(df.describe())
print("-----------------------------------------------")
print(df.isnull().sum())
print("-----------------------------------------------")
print(df.shape)
print("-----------------------------------------------")
df=df.drop_duplicates()
print("-----------------------------------------------")

X = df.drop('Purchased' , axis=1)
Y = df['Purchased']

print(X.shape , Y.shape)

X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size=0.2 , random_state=42)
print(X_train.shape , X_test.shape , Y_train.shape)

# Objective 1: Standardise features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

Log_model = LogisticRegression(random_state=42)
Log_model.fit(X_train_scaled , Y_train)
Log_pred = Log_model.predict(X_test_scaled)

svm_rbf = SVC(kernel='rbf', random_state=42)
svm_rbf.fit(X_train_scaled, Y_train)
y_pred_svm = svm_rbf.predict(X_test_scaled)

# --- Objective 3: Compare metrics, confusion matrices, and evaluate models ---
# Metrics for Logistic Regression
cm_lr = confusion_matrix(Y_test, Log_pred)
acc_lr = accuracy_score(Y_test, Log_pred)
prec_lr = precision_score(Y_test, Log_pred)
rec_lr = recall_score(Y_test, Log_pred)
f1_lr = f1_score(Y_test, Log_pred)


# Metrics for RBF SVM
cm_svm = confusion_matrix(Y_test, y_pred_svm)
acc_svm = accuracy_score(Y_test, y_pred_svm)
prec_svm = precision_score(Y_test, y_pred_svm)
rec_svm = recall_score(Y_test, y_pred_svm)
f1_svm = f1_score(Y_test, y_pred_svm)

# Build comparison DataFrame
comparison_q18 = pd.DataFrame({
    'Model': ['Logistic Regression', 'RBF Kernel SVM'],
    'Accuracy': [acc_lr, acc_svm],
    'Precision': [prec_lr, prec_svm],
    'Recall': [rec_lr, rec_svm],
    'F1-Score': [f1_lr, f1_svm]
})

print(comparison_q18)
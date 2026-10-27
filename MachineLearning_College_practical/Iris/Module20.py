# --- 1. Imports ---
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, accuracy_score

# --- Data Preparation (Iris Dataset) ---
iris = pd.read_csv('C:/Users/91970/OneDrive/Desktop/Parthml/Iris/Iris.csv')

# Drop 'Id' column if present in the CSV file
if 'Id' in iris.columns:
    iris = iris.drop(columns=['Id'])

print(iris.head())
print("------------------------------------")
print(iris.isnull().sum())
print("------------------------------------")
print(iris['Species'].value_counts())
print("-----------------------------------")

X = iris.drop('Species', axis=1)
Y = iris['Species']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# --- 1. Gini Impurity Criterion ---
model = DecisionTreeClassifier(criterion='gini', random_state=42, max_depth=3)
model.fit(X_train, Y_train)
y_pred_gini = model.predict(X_test)

# --- 2. Entropy (Information Gain) Criterion ---
dt_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
dt_entropy.fit(X_train, Y_train)
y_pred_entropy = dt_entropy.predict(X_test)

# --- Objective 2: Compare confusion matrices and classification accuracy ---
cm_gini = confusion_matrix(Y_test, y_pred_gini)
acc_gini = accuracy_score(Y_test, y_pred_gini)

cm_entropy = confusion_matrix(Y_test, y_pred_entropy)
acc_entropy = accuracy_score(Y_test, y_pred_entropy)

# Build comparison DataFrame
comparison_q20 = pd.DataFrame({
    'Criterion': ['Gini Impurity', 'Entropy (Information Gain)'],
    'Accuracy': [acc_gini, acc_entropy]
})

print("--- Question 20: Impurity Criteria Performance Comparison ---")
print(comparison_q20)

print("\n--- Gini Confusion Matrix ---")
print(cm_gini)

print("\n--- Entropy Confusion Matrix ---")
print(cm_entropy)

# --- Objective 3: Visualise the trees and discuss differences in splitting decisions ---
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Get class names automatically from the trained model
class_names = model.classes_.tolist()

# Plot Gini Tree (FIXED: X.columns and class_names defined)
plot_tree(
    model, feature_names=X.columns.tolist(), class_names=class_names,
    filled=True, rounded=True, ax=axes[0]
)
axes[0].set_title("Decision Tree with Gini Impurity")

# Plot Entropy Tree
plot_tree(
    dt_entropy, feature_names=X.columns.tolist(), class_names=class_names,
    filled=True, rounded=True, ax=axes[1]
)
axes[1].set_title("Decision Tree with Entropy (Information Gain)")

plt.tight_layout()
plt.show()
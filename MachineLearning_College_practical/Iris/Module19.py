import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

depths = [2,3,4,5]
train_accuracies = []
test_accuracies = []

for depth in depths:
    model = DecisionTreeClassifier(max_depth=depth , random_state=42)
    model.fit(X_train , y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train , train_pred)
    test_acc = accuracy_score(y_test , test_pred)

    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

# Create a comparison DataFrame
comparison_q19 = pd.DataFrame({
    'Max Depth': depths,
    'Training Accuracy': train_accuracies,
    'Testing Accuracy': test_accuracies
})

print("--- Question 19: Decision Tree Depth Performance ---")
print(comparison_q19)


# --- Objective 3: Plot the results and identify the balance ---
plt.figure(figsize=(8, 5))
plt.plot(depths, train_accuracies, marker='o', label='Training Accuracy', linewidth=2, color='blue')
plt.plot(depths, test_accuracies, marker='s', label='Testing Accuracy', linewidth=2, color='orange')
plt.title('Effect of Decision Tree Depth on Accuracy (Overfitting Analysis)')
plt.xlabel('Maximum Depth')
plt.ylabel('Accuracy')
plt.grid(True, linestyle='--', alpha=0.9)
plt.show()
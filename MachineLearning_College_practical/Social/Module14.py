import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split , cross_val_score , GridSearchCV
from sklearn.linear_model import LinearRegression
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


X = df.drop('Purchased' , axis=1)
Y = df['Purchased']


X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size=.2 , random_state=42)

standard = StandardScaler()
X_train = standard.fit_transform(X_train)
X_test = standard.transform(X_test)

default_svm = SVC(kernel='rbf' , random_state=42)
default_svm.fit(X_train , Y_train)

default_preds = default_svm.predict(X_test)
default_acc = accuracy_score(Y_test, default_preds)
default_f1 = f1_score(Y_test, default_preds)
default_cv = cross_val_score(default_svm, X_train, Y_train , cv=5, scoring='accuracy').mean()

param_grid = {
    'C':[.1 , 1 , 10 ,100],
    'gamma':['scale' , 'auto' , 0.01 , .1 , 1]
}

grid_search = GridSearchCV(
    estimator=SVC(kernel='rbf' , random_state=42),
    param_grid=param_grid,
    cv = 5,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train , Y_train)

best_model = grid_search.best_estimator_

best_preds = best_model.predict(X_test)
best_acc = accuracy_score(Y_test, best_preds)
best_f1 = f1_score(Y_test, best_preds)
best_cv = grid_search.best_score_

comparision_df = pd.DataFrame({
    'Model Configuration': ['Default RBF SVM', 'Tuned RBF SVM (GridSearchCV)'],
    'Cross-Validation Accuracy': [default_cv, best_cv],
    'Test Accuracy': [default_acc, best_acc],
    'Test F1-Score': [default_f1, best_f1]}
)


print(comparision_df)
print(grid_search.best_params_)
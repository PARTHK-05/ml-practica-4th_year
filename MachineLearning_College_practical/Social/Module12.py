import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
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

model_linear = SVC(kernel="linear" , random_state=42)
model_linear.fit(X_train , Y_train)
y_pred_linear = model_linear.predict(X_test)

accuracy_linear = accuracy_score(Y_test , y_pred_linear)
precision_linear = precision_score(Y_test , y_pred_linear)
recall_linear = recall_score(Y_test , y_pred_linear)
f1_linear = f1_score(Y_test , y_pred_linear)
cm_linear = confusion_matrix(Y_test , y_pred_linear)

model_rbf = SVC(kernel="rbf" ,  random_state=42)
model_rbf.fit(X_train , Y_train)
y_pred_rbf = model_rbf.predict(X_test)

accuracy_rbf = accuracy_score(Y_test , y_pred_rbf)
precision_rbf = precision_score(Y_test , y_pred_rbf)
recall_rbf = recall_score(Y_test , y_pred_rbf)
f1_rbf = f1_score(Y_test , y_pred_rbf)
cm_rbf = confusion_matrix(Y_test , y_pred_rbf)

results = pd.DataFrame({
    'kernal':['Linear' , 'RBF'],
    'accuracy':[accuracy_linear , accuracy_rbf],
    'precision':[precision_linear , precision_rbf],
    'recall':[recall_linear , recall_rbf],
    'f1_score':[f1_linear , f1_rbf],    
})

print(results)

print(cm_linear , cm_rbf)
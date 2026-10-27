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


X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size=.25 , random_state=42)

model_rbf = SVC(kernel="rbf" ,  random_state=42)
model_rbf.fit(X_train , Y_train)
y_pred_rbf = model_rbf.predict(X_test)

accuracy_rbf = accuracy_score(Y_test , y_pred_rbf)
precision_rbf = precision_score(Y_test , y_pred_rbf)
recall_rbf = recall_score(Y_test , y_pred_rbf)
f1_rbf = f1_score(Y_test , y_pred_rbf)
cm_rbf = confusion_matrix(Y_test , y_pred_rbf)

standard = StandardScaler()
X_train = standard.fit_transform(X_train)
X_test = standard.transform(X_test)

model_rbf_std = SVC(kernel="rbf" ,  random_state=42)
model_rbf_std.fit(X_train , Y_train)
y_pred_rbf_std = model_rbf_std.predict(X_test)

accuracy_rbf_std = accuracy_score(Y_test , y_pred_rbf_std)
precision_rbf_std = precision_score(Y_test , y_pred_rbf_std)
recall_rbf_std = recall_score(Y_test , y_pred_rbf_std)
f1_rbf_std = f1_score(Y_test , y_pred_rbf_std)
cm_rbf_std = confusion_matrix(Y_test , y_pred_rbf_std)


results = pd.DataFrame({
    'kernal':['RBF' , 'Standard RBF'],
    'accuracy':[accuracy_rbf , accuracy_rbf_std],
    'precision':[precision_rbf , precision_rbf_std],
    'recall':[recall_rbf , recall_rbf_std],
    'f1_score':[f1_rbf , f1_rbf_std],    
})

print(results)

print(cm_rbf)

print(cm_rbf_std)
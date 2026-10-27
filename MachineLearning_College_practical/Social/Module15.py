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
print("-----------------------------------------------")

X = df.drop('Purchased' , axis=1)
Y = df['Purchased']

print(X.shape , Y.shape)

X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size=0.2 , random_state=42)
print(X_train.shape , X_test.shape , Y_train.shape)



features = {
    'Age only': ['Age'],
    'EstimatedSalary only':['EstimatedSalary'],

    'Both':['Age' , 'EstimatedSalary']
}

results = []

for name,feature in features.items():
    X_tr = X_train[feature]
    X_te = X_test[feature]
    model = SVC(kernel='rbf' , random_state=42)

    standard = StandardScaler()
    X_train_std = standard.fit_transform(X_tr)
    X_test_std = standard.transform(X_te)

    model.fit(X_train_std , Y_train)
    y_pred_rbf = model.predict(X_test_std)

    accuracy_rbf = accuracy_score(Y_test , y_pred_rbf)
    precision_rbf = precision_score(Y_test , y_pred_rbf)
    recall_rbf = recall_score(Y_test , y_pred_rbf)
    f1_rbf = f1_score(Y_test , y_pred_rbf)

    results.append({
        'Features':name,
        'accuracy':accuracy_rbf,
        'precision':precision_rbf,
        'recall':recall_rbf,
        'f1_score':f1_rbf
    })

comparision_df = pd.DataFrame(results)

print(comparision_df)



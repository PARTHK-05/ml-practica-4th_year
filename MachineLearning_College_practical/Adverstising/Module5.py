import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error , r2_score

df = pd.read_csv('C:/Users/91970/OneDrive/Desktop/Parthml/Adverstising/advertising.csv')

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

X = df.drop('Sales' , axis=1)
Y = df['Sales']

print(X.shape , Y.shape)

X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size=0.2 , random_state=42)
print(X_train.shape , X_test.shape , Y_train.shape)

size_of_training_set = len(X)

training_size = {
    "50% of training data" :int(0.50*size_of_training_set),
    "70% of training data" :int(0.70*size_of_training_set),
    "80% of training data" :int(0.80*size_of_training_set)
}

results = []

for percentage,size in training_size.items():
    X_tr = X_train.iloc[:size]
    Y_tr = Y_train.iloc[:size]
    model = LinearRegression()
    model.fit(X_tr , Y_tr)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(Y_test , y_pred))
    r2 = r2_score(Y_test , y_pred)

    results.append({
        'Percentage of training data':percentage,
        'rmse':rmse,
        'r2_score':r2
    })

comparision_df = pd.DataFrame(results)

comparision_df = comparision_df.sort_values(by='r2_score' , ascending=False)

print(comparision_df)



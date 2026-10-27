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



features = {
    'TV and Radio':['TV' , 'Radio'],
    'All festures':['TV' , 'Radio' ,"Newspaper"]
}

def calculate_adjusted_r2(r2 , n , p):
    return 1-(((1-r2)*(n-1))/(n-p-1))

results = []

for name,feature in features.items():
    X_tr = X_train[feature]
    X_te = X_test[feature]
    model = LinearRegression()
    model.fit(X_tr , Y_train)
    y_pred = model.predict(X_te)

    rmse = np.sqrt(mean_squared_error(Y_test , y_pred))
    r2 = r2_score(Y_test , y_pred)
    n = X_te.shape[0]
    p = X_te.shape[1]
    r2_adj = calculate_adjusted_r2(r2, n, p)

    results.append({
        'Features':name,
        'rmse':rmse,
        'r2_score':r2,
        'r2_adjusted':r2_adj
    })

comparision_df = pd.DataFrame(results)
comparision_df = comparision_df.sort_values(by='r2_score' , ascending=False)

print(comparision_df)



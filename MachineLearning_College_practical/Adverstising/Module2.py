import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error , r2_score
from sklearn.preprocessing import PolynomialFeatures

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


X = df[['TV','Radio']]
Y = df['Sales']

print(X.shape , Y.shape)

X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size=0.2 , random_state=42)
print(X_train.shape , X_test.shape , Y_train.shape)


model = LinearRegression()
model.fit(X_train , Y_train)
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(Y_test , y_pred))
r2 = r2_score(Y_test , y_pred)

model2 = PolynomialFeatures(degree=2 , include_bias=True)
X_poly_train = model2.fit_transform(X_train)
X_poly_test = model2.transform(X_test)

model2 = LinearRegression()
model2.fit(X_poly_train , Y_train)
y_pred_poly = model2.predict(X_poly_test)

rmse_poly = np.sqrt(mean_squared_error(Y_test , y_pred_poly))
r2_poly = r2_score(Y_test , y_pred_poly)



result = pd.DataFrame({
    'Model': ['Standard Linear Regression (TV & Radio)', 'Polynomial Regression (Degree 2 - TV & Radio)'],
    'RMSE': [rmse, rmse_poly],
    'R2 Score': [r2, r2_poly]
})


print(result)
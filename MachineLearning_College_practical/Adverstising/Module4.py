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


X = df[['TV','Radio' ,'Newspaper']]
Y = df['Sales']

print(X.shape , Y.shape)

X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size=0.2 , random_state=42)
print(X_train.shape , X_test.shape , Y_train.shape)


model = LinearRegression()
model.fit(X_train , Y_train)
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(Y_test , y_pred))
r2 = r2_score(Y_test , y_pred)

X_test_second = X_test.copy()
X_test_second["Actual_score"] = Y_test
X_test_second["Predicted_score"] = y_pred 
X_test_second["Residual"] = Y_test - y_pred
X_test_second["Absolute Residual"] = abs( Y_test - y_pred)
X_test_second = X_test_second.sort_values(by='Absolute Residual' , ascending=False)
print(X_test_second.head())

# 1. Create the drawing canvas
plt.figure(figsize=(8, 5))

# 2. Plot actual data points vs model predictions
plt.scatter(Y_test, y_pred, color='blue', alpha=0.7)

# 3. Draw the red ideal diagonal line (y = x)
max_val = max(Y_test.max(), y_pred.max())
plt.plot([0, max_val], [0, max_val], color='red', linestyle='--')

# 4. Add labels and title
plt.title('Actual vs Predicted Sales')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')

# 5. Display the grid and show graph
plt.grid(True)
plt.show()

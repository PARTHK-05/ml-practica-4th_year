import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error , r2_score
from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('C:/Users/91970/OneDrive/Desktop/Parthml/Used_cars/used_cars.csv')

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
print(df['Transmission'].value_counts())
print("-----------------------------------------------")
print(df['Owner'].value_counts())
print("-----------------------------------------------")

# Initialize the encoder
encoder = LabelEncoder()

# Apply Label Encoding column by column safely
for col in ['Transmission', 'Owner']:
    df[col] = encoder.fit_transform(df[col])

# Now df is still a DataFrame, so .head() will work perfectly!
print(df.head())
print("-----------------------------------------------")

print(df['Transmission'].value_counts())
print("-----------------------------------------------")
print(df['Owner'].value_counts())
print("-----------------------------------------------")

models={
    'Linear Regresiion':LinearRegression(),
    'Decision tree':DecisionTreeRegressor(random_state=42),
    'Random Forest':RandomForestRegressor(random_state=42),
}
X = df.drop('Selling_Price' , axis=1)
Y=df['Selling_Price']

print(X.shape , Y.shape)

X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size=0.2 , random_state=42)
print(X_train.shape , X_test.shape , Y_train.shape)

results = []

for name , model in models.items():
    model.fit(X_train , Y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(Y_test , y_pred))

    r2 = r2_score(Y_test , y_pred)

    results.append({
        'model for training':name,
        'rmse':rmse,
        'r2_score':r2
    })

comparitive_prediction = pd.DataFrame(results)
print(comparitive_prediction)

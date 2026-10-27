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

df['Car_age'] = 2026-df['Year']

X = df.drop('Selling_Price' , axis=1)
Y = df['Selling_Price']

print(X.shape , Y.shape)

X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size=0.2 , random_state=42)
print(X_train.shape , X_test.shape , Y_train.shape)

features = {
    'Numerical Features' : ['Year' , 'Kms_Driven'],
    'All_Features':['Year','Kms_Driven','Transmission' , 'Owner']
}

results = []

for name,feature in features.items():
    X_tr = X_train[feature]
    X_te = X_test[feature]
    model = LinearRegression()
    model.fit(X_tr , Y_train)
    y_pred = model.predict(X_te)

    rmse = np.sqrt(mean_squared_error(Y_test , y_pred))
    r2 = r2_score(Y_test , y_pred)

    results.append({
        'Features':name,
        'rmse':rmse,
        'r2_score':r2
    })

comparision_df = pd.DataFrame(results)

comparision_df = comparision_df.sort_values(by='r2_score' , ascending=False)

print(comparision_df)



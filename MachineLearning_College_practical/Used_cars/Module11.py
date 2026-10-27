import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestRegressor

# --- 1. Load Dataset ---
df = pd.read_csv('C:/Users/91970/OneDrive/Desktop/Parthml/Used_cars/used_cars.csv')

print(df.head())
print("-----------------------------------------------")
print(df.info())
print("-----------------------------------------------")
print(df.describe())
print("-----------------------------------------------")
print(df.isnull().sum())
print("-----------------------------------------------")
print("Initial Shape:", df.shape)
print("-----------------------------------------------")

# --- 2. Remove Duplicates ---
df = df.drop_duplicates()
print("Shape after dropping duplicates:", df.shape)
print("-----------------------------------------------")

# Inspect categorical distributions
print(df['Transmission'].value_counts())
print("-----------------------------------------------")
print(df['Owner'].value_counts())
print("-----------------------------------------------")

# --- 3. Encode Categorical Features ---
encoder = LabelEncoder()
for col in ['Transmission', 'Owner']:
    df[col] = encoder.fit_transform(df[col])

print("Data after Label Encoding:")
print(df.head())
print("-----------------------------------------------")

# --- 4. Define Features and Target ---
X = df.drop('Selling_Price', axis=1)
Y = df['Selling_Price']

print("X shape:", X.shape, "| Y shape:", Y.shape)
print("-----------------------------------------------")

# --- 5. Train-Test Split ---
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print("Train shape:", X_train.shape, "| Test shape:", X_test.shape)
print("-----------------------------------------------")

# --- 6. Baseline Linear Regression Model ---
model = RandomForestRegressor(n_estimators=100 , random_state=42)
model.fit(X_train, Y_train)

y_pred = model.predict(X_test)

rmse_base = np.sqrt(mean_squared_error(Y_test, y_pred))
r2_base = r2_score(Y_test, y_pred)


features_importance_df = pd.DataFrame({
    'features': X.columns,
    'importance':model.feature_importances_
}) 

print(features_importance_df)

features_importance_df = features_importance_df.sort_values(by='importance' , ascending=False)

top_3_fetures = features_importance_df['features'].iloc[:3].tolist()
print(top_3_fetures)

X_reduced = df[top_3_fetures]
Y_reduced = df['Selling_Price']

X_train_red , X_test_red , y_train_red , y_test_red = train_test_split(X_reduced,Y_reduced,test_size=0.2,random_state=42)

model_reduced = RandomForestRegressor(n_estimators=100 , random_state=42)
model_reduced.fit(X_train_red, y_train_red)

y_pred_reduced = model_reduced.predict(X_test_red)

rmse_base_reduced = np.sqrt(mean_squared_error(Y_test, y_pred_reduced))
r2_base_reduced = r2_score(Y_test, y_pred_reduced)

result = pd.DataFrame({
    'Model':['Not Reduced' , 'Reduced'],
    'rsme':[rmse_base , rmse_base_reduced],
    'r2_score':[r2_base , r2_base_reduced],
})

print(result)

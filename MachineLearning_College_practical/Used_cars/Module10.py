import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

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
model = LinearRegression()
model.fit(X_train, Y_train)

y_pred = model.predict(X_test)

rmse_base = np.sqrt(mean_squared_error(Y_test, y_pred))
r2_base = r2_score(Y_test, y_pred)

print(f"Baseline Linear Regression -> RMSE: {rmse_base:.4f} | R2 Score: {r2_base:.4f}")
print("-----------------------------------------------")

# --- 7. Outlier Removal using Training Set IQR Method ---
train_data = X_train.copy()
train_data['Selling_Price'] = Y_train

# Calculate Q1, Q3, and IQR strictly on training target
Q1 = train_data['Selling_Price'].quantile(0.25)
Q3 = train_data['Selling_Price'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter out extreme selling-price observations from training data
train_cleaned = train_data[
    (train_data['Selling_Price'] >= lower_bound) &
    (train_data['Selling_Price'] <= upper_bound)
]

X_train_clean = train_cleaned.drop(columns=['Selling_Price'])
Y_train_clean = train_cleaned['Selling_Price']

print(f"Original training rows: {len(X_train)} | Cleaned training rows: {len(X_train_clean)}")
print("-----------------------------------------------")

# --- 8. Train Cleaned Linear Regression Model ---
model_clean = LinearRegression()
model_clean.fit(X_train_clean, Y_train_clean)

y_pred_clean = model_clean.predict(X_test)

rmse_clean = np.sqrt(mean_squared_error(Y_test, y_pred_clean))
r2_clean = r2_score(Y_test, y_pred_clean)

print(f"Cleaned Linear Regression  -> RMSE: {rmse_clean:.4f} | R2 Score: {r2_clean:.4f}")
print("-----------------------------------------------")
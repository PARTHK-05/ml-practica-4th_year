import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split , cross_val_score , GridSearchCV
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
df = df.drop_duplicates()
print(df['Transmission'].value_counts())
print("-----------------------------------------------")
print(df['Owner'].value_counts())
print("-----------------------------------------------")

# Initialize the encoder
encoder = LabelEncoder()

# Apply Label Encoding column by column safely
for col in ['Transmission', 'Owner']:
    df[col] = encoder.fit_transform(df[col])

print(df.head())
print("-----------------------------------------------")

print(df['Transmission'].value_counts())
print("-----------------------------------------------")
print(df['Owner'].value_counts())
print("-----------------------------------------------")

X = df.drop('Selling_Price' , axis=1)
Y = df['Selling_Price']

print(X.shape , Y.shape)

X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size=0.2 , random_state=42)
print(X_train.shape , X_test.shape , Y_train.shape)

# --- 1. Baseline Model ---
baseline_dt = DecisionTreeRegressor(random_state=42)
baseline_dt.fit(X_train , Y_train)

y_pred = baseline_dt.predict(X_test)
base_cv_rmse = np.mean(np.sqrt(-cross_val_score(baseline_dt, X_train, Y_train, cv=5, scoring='neg_mean_squared_error')))
base_r2 = r2_score(Y_test , y_pred)


# --- 2. GridSearchCV Tuning ---
params_grid = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

grid_search = GridSearchCV(
    estimator=DecisionTreeRegressor(random_state=42), # Fixed: changed to DecisionTreeRegressor
    cv=5,
    param_grid=params_grid,
    scoring='neg_mean_squared_error',             # Fixed: corrected score name string
    n_jobs=-1
)

# FIXED: You must fit the grid search before calling best_estimator_!
grid_search.fit(X_train, Y_train)

# Extract best model
best_dt = grid_search.best_estimator_

# --- 3. Tuned Model Evaluation ---
tuned_pred = best_dt.predict(X_test)
tuned_r2 = r2_score(Y_test, tuned_pred)

# Fixed: use cross_val_score for the tuned model so it matches how you evaluated the baseline
tuned_cv_rmse = np.mean(np.sqrt(-cross_val_score(best_dt, X_train, Y_train, cv=5, scoring='neg_mean_squared_error')))

# Print comparison summary table
comparison_q8 = pd.DataFrame({
    'Model': ['Baseline Decision Tree', 'Tuned Decision Tree'],
    'CV RMSE': [base_cv_rmse, tuned_cv_rmse],
    'Test R2 Score': [base_r2, tuned_r2]
})

print("--- Question 8: Baseline vs Tuned Model Comparison ---")
print(comparison_q8)
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv('gdpWorld.csv')

# Choose the six factors that affect GDP
features = ['population', 'area', 'population density', 'coastline area', 'net migration', 'infant mortality']

# Split the data into independent variables (X) and dependent variable (y)
X = df[features]
y = df['GDP']

# Linear Model
reg = LinearRegression().fit(X, y)
y_pred_linear = reg.predict(X)
linear_mse = mean_squared_error(y, y_pred_linear)

# Non-Linear Model (KNN Regression)
knn = KNeighborsRegressor().fit(X, y)
y_pred_nonlinear = knn.predict(X)
nonlinear_mse = mean_squared_error(y, y_pred_nonlinear)

# Compare the two models
if linear_mse < nonlinear_mse:
    print("Linear Regression is more accurate with MSE:", linear_mse)
else:
    print("KNN Regression is more accurate with MSE:", nonlinear_mse)

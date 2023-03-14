import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the data into a pandas DataFrame
data = pd.read_csv("gdpWorld.csv")

# Plot a heatmap to visualize the correlation between the variables
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True)
plt.show()

# Split the data into features (X) and target (y)
X = data[['population', 'capital_stock', 'labor_force', 'technology', 'natural_resources', 'trade']]
y = data['GDP']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the linear regression model
reg = LinearRegression().fit(X_train, y_train)

# Make predictions on the test data
y_pred_reg = reg.predict(X_test)

# Evaluate the linear regression model's accuracy
print("Linear Regression R2 Score:", reg.score(X_test, y_test))
print("Linear Regression MSE:", mean_squared_error(y_test, y_pred_reg))

# Train the non-linear SVM model
svm = SVR(kernel='rbf').fit(X_train, y_train)

# Make predictions on the test data
y_pred_svm = svm.predict(X_test)

# Evaluate the non-linear SVM model's accuracy
print("SVM R2 Score:", svm.score(X_test, y_test))
print("SVM MSE:", mean_squared_error(y_test, y_pred_svm))

# Compare the two models
if mean_squared_error(y_test, y_pred_reg) < mean_squared_error(y_test, y_pred_svm):
    print("Linear Regression gives a more accurate prediction")
else:
    print("Non-linear SVM gives a more accurate prediction")
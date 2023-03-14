import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the CSV dataset into a pandas DataFrame
data = pd.read_csv('weatherAUS.csv')

# Drop irrelevant columns
data = data.drop(['Date', 'Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm'], axis=1)

# Handle missing values
data = data.dropna()

# Encode categorical variables using one-hot encoding
ct = ColumnTransformer([('encoder', OneHotEncoder(), ['RainToday'])], remainder='passthrough')
X = ct.fit_transform(data.drop('RainTomorrow', axis=1))
y = data['RainTomorrow'].map({'No': 0, 'Yes': 1})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
clf = LogisticRegression(random_state=42)
clf.fit(X_train, y_train)

# Evaluate the performance of the model on the testing set
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

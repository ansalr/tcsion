import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Read the CSV file into a Pandas DataFrame
df = pd.read_csv('gdpWorld.csv')
df.fillna(df.mean(), inplace=True, axis=0)
df = pd.get_dummies(df, columns=['region'], prefix='region')

# Handle missing values
df.fillna(df.mean(), inplace=True)

# Convert categorical variables to numerical
df = pd.get_dummies(df, columns=['region'])

# Scale the variables
scaler = StandardScaler()
df[['population', 'area', 'population_density', 'coastline_area', 'net_migration',
    'infant_mortality', 'literacy', 'phones_per_1000', 'arable', 'crops', 'climate',
    'birth_rate', 'death_rate', 'agriculture', 'industry', 'service']] = scaler.fit_transform(df[['population', 'area', 'population_density', 'coastline_area', 'net_migration',
    'infant_mortality', 'literacy', 'phones_per_1000', 'arable', 'crops', 'climate',
    'birth_rate', 'death_rate', 'agriculture', 'industry', 'service']])

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
df = pd.read_csv('gdpWorld.csv')

# Handling missing values
df.fillna(df.mean(), inplace=True) # fill missing values with mean

# Converting categorical variables to numerical
df = pd.get_dummies(df, columns=['region']) # one hot encoding of categorical columns

# Scaling the variables
scaler = MinMaxScaler()
df[df.columns] = scaler.fit_transform(df[df.columns])

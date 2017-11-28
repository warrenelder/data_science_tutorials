# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset
dataSet = pd.read_csv('Data.csv')
x = dataSet.iloc[:, :-1].values
y = dataSet.iloc[:, 3].values

# Missing values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

# Encoding Categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x[:, 0] = labelencoder_x.fit_transform(x[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
x = onehotencoder.fit_transform(x).toarray()
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Splitting data into test and train
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Feature Scaling : 'Standardisation', 'Normalisation'
# Can do for categorical variables depending on wether the problem is a regression of categorisation
from sklearn.preprocessing import StandardScaler
sk_x = StandardScaler()
x_train = sk_x.fit_transform(x_train)
x_test = sk_x.transform(x_test)

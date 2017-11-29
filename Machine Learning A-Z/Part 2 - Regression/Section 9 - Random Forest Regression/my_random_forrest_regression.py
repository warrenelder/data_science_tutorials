#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 17:49:01 2017

@author: warren
"""

# Random Forest Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting the Random Forrest Regression Model to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=10,
                                  criterion="mse",
                                  max_depth=None,
                                  min_samples_split=2,
                                  min_samples_leaf=1,
                                  min_weight_fraction_leaf=0.,
                                  max_features="auto",
                                  max_leaf_nodes=None,
                                  min_impurity_decrease=0.,
                                  min_impurity_split=None,
                                  bootstrap=True,
                                  oob_score=False,
                                  n_jobs=1,
                                  random_state=0,
                                  verbose=0,
                                  warm_start=False)
regressor.fit(X, y)

# Predicting a new result
y_pred = regressor.predict(6.5)


# Visualising the Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forrest Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
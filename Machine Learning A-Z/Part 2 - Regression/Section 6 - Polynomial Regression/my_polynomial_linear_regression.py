#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 13:28:02 2017

@author: warren
"""

# Tasks: Determine if a person who was at level 6.5 in previous company should 
# of receiveda salary of $160K. 

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# As the number of variable are small then we shall not do a split of test and train

# Fit a linear regression model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X=X, y=y)

# Fit a polynomial regression model
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

# Visulise
plt.scatter(X, y, color = "red")
plt.title("Salary and Job title")
plt.ylabel("Salary")
plt.xlabel("Level")
# Visulalise linear regression results
plt.plot(X, lin_reg.predict(X), color="blue")
# Visulaise polynomial regression results
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)), color="green")
plt.show()

# Predict
x_employee = 6.5
# Linear Regression
y_lin = lin_reg.predict(x_employee)
# Polynomial Linear Regression
y_poly = lin_reg2.predict(poly_reg.fit_transform(x_employee))

















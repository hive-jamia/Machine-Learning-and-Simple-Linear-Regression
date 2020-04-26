# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 04:51:13 2020

@author: tarun
"""

import numpy as np    # for linear algebra
import pandas as pd   # dataframes
import matplotlib.pyplot as plt   # visual

df = pd.read_csv('Salary_Data.csv')   # dataframe

X = df.iloc[:, :-1].values   # experience
y = df.iloc[:, 1].values     # salary

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 1/3, random_state = 0)

# Simple Linear Regressor
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)


# for training set
plt.scatter(X_train, y_train, color='green')
plt.plot(X_train, model.predict(X_train), color='red')
plt.title('exp and salary (train set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

# for test set
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_train, model.predict(X_train), color='red')
plt.title('exp and salary (test set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

y_pred = model.predict(X_test)

pred = np.asarray([[6.5]])
print("Est. Salary with 6.5 yrs of experince"+ str(model.predict(pred)))
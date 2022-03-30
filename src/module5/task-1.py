import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

X = pd.read_csv("data/table.csv", delimiter=',')
n = len(X)
x = X["X"].to_list()
y = X["Y"].to_list()

x_mean = np.mean(x)
print(f"Выборочное среднее X:", x_mean)
y_mean = np.mean(y)
print(f"Выборочное среднее Y:", y_mean)

x = np.array([[1, var] for var in x])

#    x-> [nx2] y -> [1xn]
reg = LinearRegression().fit(x, y)

print(f"Параметры регрессии b_0:", reg.intercept_)
print(f"Параметры регрессии b_1:", reg.coef_)

#r^2
r2 = reg.score(x, y)
print(f"R^2:", r2)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor #For Decision tree

#For datas
dataset = pd.read_csv("Position_Salaries.csv")
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

#Create a new object and use it
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(x, y)

#Predict new value
z = regressor.predict([[6.5]])

x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, regressor.predict(x_grid), color = 'blue')
plt.title('True or false old salary')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
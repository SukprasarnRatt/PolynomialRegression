import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Insurance.csv')
# Sorting the dataset by age
sorted_dataset = dataset.sort_values(by=['ages']) 
X = sorted_dataset.iloc[:, 0:1].values
Y = sorted_dataset.iloc[:, -1].values

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, Y)

# Visualising the Linear Regression results
plt.scatter(X, Y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Insurance Charges (Linear Regression)')
plt.xlabel('Age')
plt.ylabel('Charges')
plt.show()

# Creating Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
print(X_poly)

# integrate the X_poly matrix of features into the linear regression model
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, Y)

# Visualising the Polynomial Regression results
plt.scatter(X, Y, color = 'red')
plt.plot(X, lin_reg_2.predict(X_poly), color = 'blue')
plt.title('Insurance Charges (Polynomial Regression)')
plt.xlabel('Age')
plt.ylabel('Charges')
plt.show()

#Import the Libraries
import numpy as np
import pandas as pd
import seaborn as sns 

#Import the Dataset
from sklearn.datasets import load_boston
boston_dataset = load_boston()

Dataset = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
Dataset['MEDV'] = boston_dataset.target

#There are 13 independent variables amd 1 dependent variable. So we need to see what is needed most importantly using the Correlation Matrix.
correlation_matrix = Dataset.corr().round(2)

# annot = True to print the values inside the square
sns.heatmap(data=correlation_matrix, annot=True)

X = Dataset.iloc[:,[5,12]].values
y = Dataset.iloc[:,-1].values

#So here introduce the polynomial features of degree 3.
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=3)
X = poly_reg.fit_transform(X)

#Feature scaling the independent variable
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

#Splitting the data into train and test
from sklearn.model_selection import train_test_split
X_train ,X_test, y_train ,y_test = train_test_split(X, y, test_size=0.25, random_state = 5, shuffle =True)

#Importing the linear regression models(using SGD)
from sklearn.linear_model import SGDRegressor, LinearRegression
regressor = SGDRegressor(max_iter=1000, tol=1e-3, alpha =0.001, random_state = 0, learning_rate = 'constant' , eta0 = 0.001)
regressor.fit(X_train, y_train)

#Predicting the output for our SGD Linear Model with the test set
y_pred = regressor.predict(X_test)

#Now lets calculate the Coefficient of Determination and the RMSE for our training set
from sklearn.metrics import r2_score , mean_squared_error

rmse_train = (np.sqrt(mean_squared_error(y_train, regressor.predict(X_train) )))
r_squared_train = r2_score(y_train , regressor.predict(X_train))
print("R squared for the training set")
print("---------------------------------")
print(r_squared_train)
print("---------------------------------")
print("RMSEfor the training set")
print("---------------------------------")
print(rmse_train)
print()
#Now lets calculate the Coefficient of Determination and the RMSE for our training set
rmse_test = (np.sqrt(mean_squared_error(y_test, regressor.predict(X_test) )))
r_squared_test = r2_score(y_test , regressor.predict(X_test))
print("R squared for the testing set")
print("---------------------------------")
print(r_squared_test)
print("---------------------------------")
print("RMSEfor the testing set")
print("---------------------------------")
print(rmse_test)
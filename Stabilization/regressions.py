import numpy as np
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
import sklearn.preprocessing as pp
from sklearn.model_selection import train_test_split

np.random.seed(2)
X = np.loadtxt("X_stab.txt", delimiter=";")
Y = np.loadtxt("Y_stab.txt", delimiter=";")

X[:,0:8] = X[:,0:8]/20000
X[:,8:10] = (X[:,8:10]+10000)/20000
X[:,10:11] = (X[:,10:11]+30000)/60000

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

LR = lm.LinearRegression()
LR.fit(x_train, y_train)
y_pred = LR.predict(x_test)

loss1 = mean_squared_error(y_test, y_pred)
print(loss1)

x_train_hp, x_test_1, y_train_hp, y_test_1 = train_test_split(x_train, y_train, test_size=0.5, random_state=42)

alphas = np.arange(0.0001, 1, 0.0001)

Lasso = lm.Lasso()
grid = GridSearchCV(estimator=Lasso, param_grid=dict(alpha=alphas))
grid.fit(x_train_hp, y_train_hp)
print(grid)
print(grid.best_score_)
print(grid.best_estimator_.alpha)

lasso_alpha = grid.best_estimator_.alpha
Lasso = lm.Lasso(alpha = lasso_alpha)
Lasso.fit(x_train, y_train)
y_pred = Lasso.predict(x_test)
print('MSE (Lasso)', mean_squared_error(y_test, y_pred))
print(Lasso.coef_)
Bonus_2 = Lasso.coef_

x_test1 = x_test.copy()
x_test1 = x_test[:,[0,4,6,7,8,9,10]]
x_train1 = x_train[:,[0,4,6,7,8,9,10]]

KNN = KNeighborsRegressor()
KNN.fit(x_train1, y_train)
y_pred = KNN.predict(x_test1)
mean_squared_error(y_test, y_pred)

X_validate, x_test_1, y_validate, y_test_1 = train_test_split(x_train1, y_train, test_size=0.5, random_state=42)

from sklearn.model_selection import GridSearchCV
neighbors = np.arange(1, 30, 1)
ps = np.arange(1, 10, 1)

KNN = KNeighborsRegressor()
grid = GridSearchCV(estimator=KNN, param_grid=[dict(n_neighbors = neighbors),
                                               dict(p = ps)])
grid.fit(X_validate, y_validate)
print(grid)
print(grid.best_score_)
print(grid.best_estimator_.n_neighbors)
print(grid.best_estimator_.p) 

n_n = grid.best_estimator_.n_neighbors
p_p = grid.best_estimator_.p
KNN_new = KNeighborsRegressor(n_neighbors = n_n, p= p_p)
KNN_new.fit(x_train1, y_train)
y_pred = KNN_new.predict(x_test1)
mean_squared_error(y_test, y_pred)

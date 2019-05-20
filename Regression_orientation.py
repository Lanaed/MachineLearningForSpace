import numpy as np
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
import sklearn.preprocessing as pp
from sklearn.model_selection import train_test_split
np.random.seed(2)
data = np.loadtxt("dataset_clear_;.txt", delimiter=";")
X = data[:,0:6]
Y = data[:,6]

X[:,0:4] = X[:,0:4]/300000
X[:,4:6] = X[:,4:6]/1000000
X[:,6:7] = (X[:,6:7]+30000)/60000

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

LR = lm.LinearRegression()
LR.fit(x_train, y_train)
y_pred = LR.predict(x_test)

mean_squared_error(y_test, y_pred)

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

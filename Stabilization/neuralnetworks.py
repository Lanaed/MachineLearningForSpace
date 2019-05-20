import numpy as np
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split

np.random.seed(2)
X = np.loadtxt("X_stab.txt", delimiter=";")
Y = np.loadtxt("Y_stab.txt", delimiter=";")

X[:,0:8] = X[:,0:8]/20000
X[:,8:10] = (X[:,8:10]+10000)/20000
X[:,10:11] = (X[:,10:11]+30000)/60000

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
x_test1 = x_test.copy()
x_test1 = x_test[:,[0,4,6,7,8,9,10]]
x_train1 = x_train[:,[0,4,6,7,8,9,10]]

from keras.models import Sequential 
from keras.layers import Dense

model_c = Sequential() 
model_c.add(Dense(12, input_dim=7, activation='relu')) 
model_c.add(Dense(15, activation='relu')) 
model_c.add(Dense(8, activation='relu')) 
model_c.add(Dense(10, activation='relu')) 
model_c.add(Dense(1, activation='sigmoid'))

def train_b (model_c, x_train1, y_train, x_test1, y_test):
    losses = [] 
    for b in range (80): 
        model_c.compile(loss="mean_squared_error", optimizer="adam", metrics=['mse']) 
        model_c.fit(x_train1, y_train, epochs = 100, batch_size=50+b*50) 
        scores = model_c.evaluate(x_test1, y_test, verbose=0)     
        losses.append(scores[1])
     return loses

#обработка losses
def train_e (model_c, x_train1, y_train, x_test1, y_test):
    losses = [] 
    N = range(50, 500, 30)
    for b in N: 
        model_c.compile(loss="mean_squared_error", optimizer="adam", metrics=['mse']) 
        model_c.fit(x_train1, y_train, epochs = b, batch_size=500) 
        scores = model_c.evaluate(x_test1, y_test, verbose=0)
        print(b)
        losses.append(scores[1])
    return losses
    
 
plt.plot(range(50, 4050, 50), losses)
plt.xlabel("batch size")
plt.ylabel("Mean Squared Error")
plt.title("Модель с удаленными признаками")
plt.show()


model_c = Sequential() 
model_c.add(Dense(14, input_dim=7, activation='relu')) 
model_c.add(Dense(28, input_shape=(14,), activation='relu')) 
model_c.add(Dense(14, input_shape=(28,), activation='relu'))
model_c.add(Dense(7, input_shape=(14,), activation='relu')) 
model_c.add(Dense(1, input_shape=(7,),activation='sigmoid'))

model_c.compile(loss="mean_squared_error", optimizer="adam", metrics=['mse']) 
model_c.fit(x_train1, y_train, epochs = 100, batch_size=500) 
scores = model_c.evaluate(x_test1, y_test, verbose=0)
model_json = model_c.to_json()
# Записываем модель в файл
json_file = open("model1.json", "w")
json_file.write(model_json)
json_file.close()
model_c.save_weights("model1.h5")
#обработка losses
model_c = Sequential() 
model_c.add(Dense(7, input_dim=7, activation='relu')) 
model_c.add(Dense(14, input_shape=(7,), activation='relu')) 
model_c.add(Dense(14, input_shape=(14,), activation='relu'))
model_c.add(Dense(7, input_shape=(14,), activation='relu')) 
model_c.add(Dense(1, input_shape=(7,),activation='sigmoid'))
losses1 = train_e (model_c, x_train1, y_train, x_test1, y_test)
losses2 = train_e (model_c, x_train1, y_train, x_test1, y_test)
#обработка losses
model_c = Sequential() 
model_c.add(Dense(7, input_dim=7, activation='relu')) 
model_c.add(Dense(14, input_shape=(7,), activation='relu')) 
model_c.add(Dense(14, input_shape=(14,), activation='relu'))
model_c.add(Dense(7, input_shape=(14,), activation='relu')) 
model_c.add(Dense(1, input_shape=(7,),activation='sigmoid'))
losses1 = train_e (model_c, x_train1, y_train, x_test1, y_test)
losses2 = train_e (model_c, x_train1, y_train, x_test1, y_test)
#обработка losses
model_c = Sequential() 
model_c.add(Dense(7, input_dim=7, activation='relu')) 
model_c.add(Dense(14, input_shape=(7,), activation='relu')) 
model_c.add(Dense(1, input_shape=(7,),activation='sigmoid'))
losses1 = train_e (model_c, x_train1, y_train, x_test1, y_test)
losses2 = train_e (model_c, x_train1, y_train, x_test1, y_test)
#обработка losses

model_c = Sequential() 
model_c.add(Dense(7, input_dim=7, activation='relu')) 
model_c.add(Dense(7, input_shape=(7,), activation='relu')) 
model_c.add(Dense(1, input_shape=(7,),activation='sigmoid'))
losses1 = train_e (model_c, x_train1, y_train, x_test1, y_test)
losses2 = train_e (model_c, x_train1, y_train, x_test1, y_test)
#обработка losses

model_c = Sequential() 
model_c.add(Dense(7, input_dim=7, activation='relu')) 
model_c.add(Dense(7, input_shape=(7,), activation='relu')) 
model_c.add(Dense(7, input_shape=(7,), activation='relu'))
model_c.add(Dense(7, input_shape=(7,), activation='relu')) 
model_c.add(Dense(1, input_shape=(7,),activation='sigmoid'))
losses1 = train_e (model_c, x_train1, y_train, x_test1, y_test)
losses2 = train_e (model_c, x_train1, y_train, x_test1, y_test)
#обработка losses
model_c.compile(loss="mean_squared_error", optimizer="adam", metrics=['mse']) 
model_c.fit(x_train1, y_train, epochs = 2000, batch_size=100) 
scores = model_c.evaluate(x_test1, y_test, verbose=0)
model_json = model_c.to_json()
# Записываем модель в файл
json_file = open("model2.json", "w")
json_file.write(model_json)
json_file.close()
model_c.save_weights("model2.h5")


model_c = Sequential() 
model_c.add(Dense(10, input_dim=7, activation='relu')) 
model_c.add(Dense(10, activation='relu')) 
model_c.add(Dense(10, activation='relu'))
model_c.add(Dense(10, activation='relu')) 
model_c.add(Dense(1,activation='sigmoid'))
losses1 = train_e (model_c, x_train1, y_train, x_test1, y_test)
losses2 = train_e (model_c, x_train1, y_train, x_test1, y_test)
#обработка losses

model_c = Sequential() 
model_c.add(Dense(10, input_dim=7, activation='relu')) 
model_c.add(Dense(20, activation='relu')) 
model_c.add(Dense(30, activation='relu'))
model_c.add(Dense(20, activation='relu')) 
model_c.add(Dense(10, activation='relu'))
model_c.add(Dense(10, activation='relu')) 
model_c.add(Dense(1,activation='sigmoid'))
losses1 = train_e (model_c, x_train1, y_train, x_test1, y_test)
losses2 = train_e (model_c, x_train1, y_train, x_test1, y_test)
#обработка losses

model_c = Sequential() 
model_c.add(Dense(10, input_dim=7, activation='relu')) 
model_c.add(Dense(10, activation='relu')) 
model_c.add(Dense(1,activation='sigmoid'))
losses1 = train_e (model_c, x_train1, y_train, x_test1, y_test)
losses2 = train_e (model_c, x_train1, y_train, x_test1, y_test)
#обработка losses

model_c = Sequential() 
model_c.add(Dense(7, input_dim=7, activation='relu')) 
model_c.add(Dense(49, activation='relu')) 
model_c.add(Dense(7, activation='relu'))
model_c.add(Dense(1,activation='sigmoid'))
losses1 = train_e (model_c, x_train1, y_train, x_test1, y_test)
losses2 = train_e (model_c, x_train1, y_train, x_test1, y_test)
#обработка losses

model_c = Sequential() 
model_c.add(Dense(7, input_dim=7, activation='relu')) 
model_c.add(Dense(14, activation='relu')) 
model_c.add(Dense(21, activation='relu')) 
model_c.add(Dense(28, activation='relu')) 
model_c.add(Dense(21, activation='relu'))
model_c.add(Dense(14, activation='relu'))
model_c.add(Dense(7, activation='relu'))
model_c.add(Dense(1,activation='sigmoid'))
losses1 = train_e (model_c, x_train1, y_train, x_test1, y_test)
losses2 = train_e (model_c, x_train1, y_train, x_test1, y_test)
#обработка losses

model_c = Sequential() 
model_c.add(Dense(7, input_dim=7, activation='relu')) 
model_c.add(Dense(14, activation='relu')) 
model_c.add(Dense(14, activation='relu'))
model_c.add(Dense(7, activation='relu'))
model_c.add(Dense(1,activation='sigmoid'))
losses1 = train_e (model_c, x_train1, y_train, x_test1, y_test)
losses2 = train_e (model_c, x_train1, y_train, x_test1, y_test)
#обработка losses

model_c = Sequential() 
model_c.add(Dense(7, input_dim=7, activation='relu')) 
model_c.add(Dense(14, activation='relu')) 
model_c.add(Dense(7, activation='relu'))
model_c.add(Dense(1,activation='sigmoid'))
losses1 = train_e (model_c, x_train1, y_train, x_test1, y_test)
losses2 = train_e (model_c, x_train1, y_train, x_test1, y_test)
#обработка losses

model_c = Sequential() 
model_c.add(Dense(7, input_dim=7, activation='relu')) 
model_c.add(Dense(14, activation='relu')) 
model_c.add(Dense(21, activation='relu')) 
model_c.add(Dense(28, activation='relu'))
model_c.add(Dense(35, activation='relu'))
model_c.add(Dense(28, activation='relu'))
model_c.add(Dense(21, activation='relu'))
model_c.add(Dense(14, activation='relu'))
model_c.add(Dense(7, activation='relu'))
model_c.add(Dense(1,activation='sigmoid'))
losses1 = train_e (model_c, x_train1, y_train, x_test1, y_test)
losses2 = train_e (model_c, x_train1, y_train, x_test1, y_test)
#обработка losses

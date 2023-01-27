from keras.datasets import mnist
from keras.datasets import cifar10

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

from matplotlib import pyplot as plt

import numpy as np

# # mnist dataset
# # Loading the Data
# (x_train,y_train), (x_test,y_test) = mnist.load_data()
# # Shaping the Data
# x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]**2)
# x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]**2)

# x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
# x_train, x_test = x_train/255., x_test/255.


# model = SVC()

# model.fit(x_train,y_train)

# pred = model.predict(x_test)

# a_1 = accuracy_score(y_test, pred)

# print(a_1)
#Cifar-10 dataset


#Loading Data
(x_train,y_train), (x_test,y_test) = cifar10.load_data()
n_images,n_rows,n_colimns,n_channels = x_train.shape

x_train = x_train.reshape((x_train.shape[0], -1))
x_test = x_test.reshape((x_test.shape[0], -1))
n_training = 5000
n_testing = 1000

x_train = x_train[0:n_training,:]
x_test = x_test[0:n_testing,:]
y_train = y_train[0:n_training]
y_test = y_test[0:n_testing]
#Shaping Data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Hyperparameters Arrays

accuracy_results=[]
kernels = ["rbf", "poly", "linear","sigmoid"]
c_list = [0.01,0.1,0.5,0.8,1]

for k in kernels:
    temp = []
    for temp_c in c_list:       
        model2 = SVC(kernel = k, C = temp_c)
        model2.fit(x_train,y_train.ravel())
        pred_2 = model2.predict(x_test)
        a_2 = accuracy_score(y_test, pred_2)
        temp.append(a_2)
        print("Accucacy Score with kernel:", k ,"and C:", temp_c , "----is:", a_2)
    accuracy_results.append(temp)

plt.plot(c_list, accuracy_results[0],'o')
plt.title="rbf"
plt.xlabel('Hyperparameter "C"') 
plt.ylabel('Accuracy Score') 
plt.show()
plt.plot(c_list, accuracy_results[1],'o')
plt.title="poly"
plt.xlabel('Hyperparameter "C"') 
plt.ylabel('Accuracy Score') 
plt.show()
plt.plot(c_list, accuracy_results[2],'o')
plt.title="linear"
plt.xlabel('Hyperparameter "C"') 
plt.ylabel('Accuracy Score') 
plt.show()
plt.plot(c_list, accuracy_results[3],'o')
plt.title="sigmoid"
plt.xlabel('Hyperparameter "C"') 
plt.ylabel('Accuracy Score') 
plt.show()
from keras.datasets import cifar10
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.metrics import accuracy_score
import time

clf1 = KNeighborsClassifier(1)
clf2 = KNeighborsClassifier(3) 
CentroidClassifier = NearestCentroid()



(x_train,y_train), (x_test,y_test) = cifar10.load_data()
n_images,n_rows,n_colimns,n_channels = x_train.shape

x_train = x_train.reshape((x_train.shape[0], -1))
x_test = x_test.reshape((x_test.shape[0], -1))

start_time1 = time.time()
clf1.fit(x_train,y_train)
pred1 = clf1.predict(x_test)
print("Execution Time for kNN-1 Classifier --- %s seconds ---" % (time.time() - start_time1))
print("Accuracy for kNN1 --- %s ---" % accuracy_score(y_test, pred1))

start_time2 = time.time()
clf2.fit(x_train,y_train)
pred2 = clf2.predict(x_test)
print("Execution for kNN-3 Classifier --- %s seconds ---" % (time.time() - start_time2))
print("Accuracy for kNN1 --- %s ---" % accuracy_score(y_test, pred2))

start_time3 = time.time()
CentroidClassifier.fit(x_train,y_train)
pred3 = CentroidClassifier.predict(x_test)
print("Execution Time for Centroid Classifier --- %s seconds ---" % (time.time() - start_time3))
print("Accuracy for kNN1 --- %s ---" % accuracy_score(y_test, pred3))
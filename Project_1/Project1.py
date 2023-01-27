from tensorflow import keras 
from keras import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import pandas as pd
from keras import models, layers
from matplotlib import pyplot as plt
import numpy as np
import sys
#sys.stdout = open(r"C:\Users\Koukas\Desktop\Project 1 logs\log4.txt", "w")
num_features = 28*28
num_classes = 10

(x_train,y_train), (x_test,y_test) = datasets.mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]**2)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]**2)

x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
x_train, x_test = x_train/255., x_test/255.
model = keras.Sequential([layers.Flatten(input_shape=(num_features,)),
                          layers.Dense(128, activation='relu'),
                          layers.Dense(128, activation='relu'),
                          layers.Dense(num_classes)])

model.compile(optimizer='sgd',
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
metrics=['accuracy'])

model.summary()

history = model.fit(x_train,y_train, 
validation_data = (x_test, y_test),
epochs = 50)

plt.figure(1)
plt.plot(history.history['accuracy'], label = 'train')
plt.plot(history.history['val_accuracy'], label = 'test')
plt.legend()
plt.title('Performance on training and validation sets')
plt.show()
#sys.stdout.close()
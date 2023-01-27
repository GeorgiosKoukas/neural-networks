from tensorflow import keras
from keras.datasets import mnist
from matplotlib import pyplot as plt
import numpy as np
from keras import backend as K
from keras.layers import Layer
from keras import models, layers
from keras.initializers import RandomUniform, Initializer, Constant
import numpy as np
from keras.initializers import Initializer
from sklearn.cluster import KMeans
import pandas as pd
from keras.datasets import cifar10

class InitCentersKMeans(Initializer):
    """ Initializer for initialization of centers of RBF network
        by clustering the given data set.
    # Arguments
        X: matrix, dataset
    """

    def __init__(self, X, max_iter=100):
        self.X = X
        self.max_iter = max_iter

    def __call__(self, shape, dtype=None):
        assert shape[1] == self.X.shape[1]

        n_centers = shape[0]
        km = KMeans(n_clusters=n_centers, max_iter=self.max_iter, verbose=0)
        km.fit(self.X)
        return km.cluster_centers_

class InitCentersRandom(Initializer):
    """ Initializer for initialization of centers of RBF network
        as random samples from the given data set.
    # Arguments
        X: matrix, dataset to choose the centers from (random rows
          are taken as centers)
    """

    def __init__(self, X):
        self.X = X

    def __call__(self, shape, dtype=None):
        assert shape[1] == self.X.shape[1]
        idx = np.random.randint(self.X.shape[0], size=shape[0])
  	    
        if type(self.X) == np.ndarray:
            return self.X[idx, :]
        elif type(self.X) == pd.core.frame.DataFrame:
            return self.X.iloc[idx, :]


	    
	# type checking to access elements of data correctly
  	    
    	    
  	    
    		

class RBFLayer(Layer):
    """ Layer of Gaussian RBF units.
    # Example
    ```python
        model = Sequential()
        model.add(RBFLayer(10,
                           initializer=InitCentersRandom(X),
                           betas=1.0,
                           input_shape=(1,)))
        model.add(Dense(1))
    ```
    # Arguments
        output_dim: number of hidden units (i.e. number of outputs of the
                    layer)
        initializer: instance of initiliazer to initialize centers
        betas: float, initial value for betas
    """

    def __init__(self, output_dim, initializer=None, betas=1.0, **kwargs):
        self.output_dim = output_dim
        self.init_betas = betas
        if not initializer:
            self.initializer = RandomUniform(0.0, 1.0)
        else:
            self.initializer = initializer
        super(RBFLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        self.centers = self.add_weight(name='centers',
                                       shape=(self.output_dim, input_shape[1]),
                                       initializer=self.initializer,
                                       trainable=True)
        self.betas = self.add_weight(name='betas',
                                     shape=(self.output_dim,),
                                     initializer=Constant(
                                         value=self.init_betas),
                                     # initializer='ones',
                                     trainable=True)

        super(RBFLayer, self).build(input_shape)

    def call(self, x):

        C = K.expand_dims(self.centers)
        H = K.transpose(C-K.transpose(x))
        return K.exp(-self.betas * K.sum(H**2, axis=1))

        # C = self.centers[np.newaxis, :, :]
        # X = x[:, np.newaxis, :]

        # diffnorm = K.sum((C-X)**2, axis=-1)
        # ret = K.exp( - self.betas * diffnorm)
        # return ret

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        # have to define get_config to be able to use model_from_json
        config = {
            'output_dim': self.output_dim
        }
        base_config = super(RBFLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



# mnist dataset
# Loading the Data
(x_train,y_train), (x_test,y_test) = mnist.load_data()
# Shaping the Data
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]**2)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]**2)

x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
x_train, x_test = x_train/255., x_test/255.
num_inputs = 28*28
num_outputs = 10

rbflayer = RBFLayer(50,
                      initializer=InitCentersRandom(x_train),
                      betas=0.02,
                      input_shape=(x_test.shape[1],))

model = keras.Sequential()
model.add(rbflayer)
model.add(layers.Dense(10, activation="sigmoid"))


model.compile(optimizer='Adamax',
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
metrics=['accuracy'])


history = model.fit(x_train,y_train, 
    validation_data = (x_test, y_test),
    epochs = 20)

plt.figure(1)
plt.plot(history.history['accuracy'], label = 'train')
plt.plot(history.history['val_accuracy'], label = 'test')
plt.legend()
plt.title('Performance on training and validation sets')
plt.show()




sample_idx = int(np.random.random() * len(y_test)) # Choose a random sample index

x_sample = x_test[sample_idx]
x_sample = np.expand_dims(x_sample, axis=0)#Changing he shape to vector from(784,) to (1,784)
prediction = model.predict(x_sample) #Returns a vector of probabilities for each class


predicted_number = np.argmax(prediction)#Get the next number by adding 1 to the predicted number
next_number = (predicted_number + 1) % 10
print("Predicted",predicted_number)

x_sample = x_sample.reshape(28,28)
plt.imshow(x_sample, cmap='gray_r')
plt.show()



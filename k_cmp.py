import keras
from keras.datasets import mnist
from keras.layers import Input, Dense, Flatten, ReLU, Softmax
from keras.models import Model, Sequential
import numpy as np

# import keras.backend as k
# k.categorical_crossentropy()

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    n_classes = 10
    x_train = x_train[0:100] / 255
    y_train = np.eye(n_classes)[y_train[0:100]]
    print(x_train.shape, y_train[0], y_train[0].shape)

    input = Input(shape=(28, 28))
    f1 = Flatten()(input)
    d1 = Dense(units=32)(f1)
    r1 = ReLU()(d1)
    d2 = Dense(units=10)(r1)
    r2 = ReLU()(d2)
    sm = Softmax()(r2)

    model = Model(input=input, output=sm)
    model.compile(optimizer='SGD', loss='categorical_crossentropy')
    model.fit(x=x_train, y=y_train, epochs=100, batch_size=1)


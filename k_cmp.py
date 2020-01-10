# compare between my code and keras

import keras
from keras.datasets import mnist
from keras.layers import Input, Dense, Flatten, ReLU, Softmax, Convolution2D, AveragePooling2D
from keras.models import Model, Sequential
import numpy as np
from matplotlib import pyplot as plt
# from Layers import Input, Output, Dense, Relu, Flatten, Convolution2D, Softmax, AveragePooling2D
from Model import Model
import Loss
from util import Image_generator, int_to_one_hot

# import keras.backend as k
# k.categorical_crossentropy()

if __name__ == '__main__':
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # n_classes = 10
    # x_train = x_train[0:100] / 255
    # y_train = np.eye(n_classes)[y_train[0:100]]
    # print(x_train.shape, y_train[0], y_train[0].shape)
    #
    # input = Input(shape=(28, 28))
    # f1 = Flatten()(input)
    # d1 = Dense(units=32)(f1)
    # r1 = ReLU()(d1)
    # d2 = Dense(units=10)(r1)
    # r2 = ReLU()(d2)
    # sm = Softmax()(r2)
    #
    # model = Model(input=input, output=sm)
    # model.compile(optimizer='SGD', loss='categorical_crossentropy')
    # model.fit(x=x_train, y=y_train, epochs=100, batch_size=1)

    # date preprocess
    g = Image_generator()
    x_train = np.zeros(shape=(48, 80, 80, 3))
    y_train = np.zeros(shape=(48, 3, 1))
    n_classes = 3

    # , is_brighter=False, is_darker=False, is_flip_X=False, is_flip_Y=False
    temp_img = np.array(plt.imread('test_pictures/train_picture/1.jpg'), dtype=np.float64)
    temp_label = int_to_one_hot(x=0, n_classes=n_classes)
    x_train[0:16], y_train[0:16] = g.one_input_flow_batch(input=temp_img, label=temp_label, batch_size=16)

    temp_img = np.array(plt.imread('test_pictures/train_picture/2.jpg'), dtype=np.float64)
    temp_label = int_to_one_hot(x=1, n_classes=n_classes)
    x_train[16:32], y_train[16:32] = g.one_input_flow_batch(input=temp_img, label=temp_label, batch_size=16)

    temp_img = np.array(plt.imread('test_pictures/train_picture/3.jpg'), dtype=np.float64)
    temp_label = int_to_one_hot(x=2, n_classes=n_classes)
    x_train[32:48], y_train[32:48] = g.one_input_flow_batch(input=temp_img, label=temp_label, batch_size=16)

    # x_train = np.zeros(shape=(3, 80, 80, 3))
    # y_train = np.zeros(shape=(3, 3, 1))
    # y_train[0] = int_to_one_hot(0, 3)
    # y_train[1] = int_to_one_hot(1, 3)
    # y_train[2] = int_to_one_hot(2, 3)
    # x_train[0] = np.array(plt.imread('test_pictures/train_picture/1.jpg'), dtype=np.float64)
    # x_train[1] = np.array(plt.imread('test_pictures/train_picture/2.jpg'), dtype=np.float64)
    # x_train[2] = np.array(plt.imread('test_pictures/train_picture/3.jpg'), dtype=np.float64)

    x_train /= 255
    y_train = np.squeeze(y_train, axis=-1)

    # for x in x_train:
    #     plt.imshow(x)
    #     plt.show()
    # print(y_train)
    # print(x_train)

    print(x_train.shape, y_train.shape)

    input = Input((80, 80, 3))
    av = AveragePooling2D()(input)
    avv = AveragePooling2D()(av)
    conv1 = Convolution2D(filters=8, kernel_size=(3, 3), use_bias=False)(avv)
    r1 = ReLU()(conv1)
    av1 = AveragePooling2D()(r1)
    f = Flatten()(av1)
    d1 = Dense(units=32)(f)
    d2 = Dense(units=3)(d1)
    sm = Softmax()(d2)
    model = keras.models.Model(input, sm)
    sgd = keras.optimizers.SGD(lr=0.01)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    model.fit(x=x_train, y=y_train, epochs=5, shuffle=True, batch_size=1)

    # test
    cnt = 0
    oo = model.predict(x_train)
    for i in range(48):
        # o = model.predict(x_train[i])
        o = oo[i]
        t = y_train[i]

        # convert (n, 1) to shape (n, )
        # o = o[..., 0]


        pred = np.argmax(o)
        true = np.argmax(t)
        print('output=', o)
        print('true=', t)
        print('predict=', pred, 'true=', true)
        if pred == true:
            cnt += 1
        print('\n')
    print('accruacy is ', cnt / 48)


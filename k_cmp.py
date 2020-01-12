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
from train import data_prepare

'''
this .py is for compare my frame with keras，for debug
这个文件是为了比较框架和标准框架keras之间的区别，方便改bug
'''

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

    x_train, y_train = data_prepare()
    y_train = np.squeeze(y_train, axis=-1)

    input = Input((80, 80, 3))
    # av = AveragePooling2D()(input)
    conv1 = Convolution2D(filters=8, kernel_size=(3, 3), use_bias=False)(input)
    r1 = ReLU()(conv1)
    av1 = AveragePooling2D()(r1)
    f = Flatten()(av1)
    d1 = Dense(units=32)(f)
    d2 = Dense(units=3)(d1)
    sm = Softmax()(d2)
    model = keras.models.Model(input, sm)
    sgd = keras.optimizers.SGD(lr=0.001)
    # model.compile(optimizer=sgd, loss='mean_squared_error')
    m_conv =  keras.models.Model(input, conv1)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')

    # x = x_train[0:1]
    # y = y_train[0:1]
    # plt.imshow(x[0])
    # plt.show()
    # E = []
    # for j in range(1):
    #     for i in range(30):
    #         # cc = m_conv.predict(x)
    #         # print(cc)
    #         rr = model.predict(x)
    #         # r = model.train_on_batch(x=x, y=y)
    #         r = model.train_on_batch(x=x, y=y)
    #         print(j, i, rr, r)
    #         E.append(r)
    # plt.plot(E)
    # plt.show()

    # shuffle is important !!!!!!!
    date_len = x_train.shape[0]
    from random import shuffle
    index = [i for i in range(date_len)]
    shuffle(index)
    x_train = x_train[index, :, :, :]
    y_train = y_train[index, :]

    # model.fit(x=x_train, y=y_train, epochs=1, shuffle=True, batch_size=1)
    for j in range(7):
        for i in range(x_train.shape[0]):
            r = model.train_on_batch(x_train[i:i+1], y_train[i:i+1])
            print('e:', j, i, '/', x_train.shape[0],r)

    # test
    cnt = 0
    for i in range(35 * 16):
        # plt.imshow(x_train[i])
        # plt.show()
        o = model.predict(np.expand_dims(x_train[i], axis=0))
        t = y_train[i]

        # convert (n, 1) to shape (n, )
        # o = o[..., 0]
        # t = t[..., 0]

        pred = np.argmax(o)
        true = np.argmax(t)
        print('output=', o)
        print('true=', t)
        print('predict=', pred, 'true=', true)
        if pred == true:
            cnt += 1
        print('\n')
    print('accruacy is ', cnt / (35 * 16))



import numpy as np
from matplotlib import pyplot as plt
from Layers import Input, Output, Dense, Relu, Flatten, Convolution2D, Softmax, AveragePooling2D
from Model import Model
import Loss
from util import Image_generator, int_to_one_hot
from train import data_prepare

if __name__ == '__main__':
    # x_train, y_train = data_prepare()
    #
    # # model config
    # input = Input(input_shape=(80, 80, 3))
    # av = AveragePooling2D(last_layer=input)
    # # avv = AveragePooling2D(last_layer=av)
    # conv1 = Convolution2D(last_layer=av, kernal_number=16, kernal_size=(3, 3))
    # r1 = Relu(last_layer=conv1)
    # av1 = AveragePooling2D(last_layer=r1)
    # f = Flatten(last_layer=av1)
    # d1 = Dense(last_layer=f, output_units=32)
    # d2 = Dense(last_layer=d1, output_units=3)
    # sm = Softmax(last_layer=d2)
    # output = Output(last_layer=sm)
    #
    # ce = Loss.Cross_entropy()
    # model = Model(input_layer=input, output_layer=output, loss=ce)
    #
    # model.load_weights(root_directory='weights')
    #
    # # test
    # cnt = 0
    # for i in range(35 * 16):
    #     # plt.imshow(x_train[i])
    #     # plt.show()
    #     o = model.predict(input=x_train[i])
    #     t = y_train[i]
    #
    #     # convert (n, 1) to shape (n, )
    #     o = o[..., 0]
    #     t = t[..., 0]
    #
    #     pred = np.argmax(o)
    #     true = np.argmax(t)
    #     print('test:', i)
    #     print('output=', o)
    #     print('true=', t)
    #     print('predict=', pred, 'true=', true)
    #     if pred == true:
    #         cnt += 1
    #     print('\n')
    # print('accruacy is ', cnt / (35 * 16))

    input = Input(input_shape=(8, 8, 2))
    av = AveragePooling2D(last_layer=input)
    output = Output(last_layer=av)

    x = np.ones((8, 8, 2))
    input.FP(x)
    gradient = np.ones((4, 4, 2))
    output.BP(gradient, 1)
    print(av.last_layer_gradient)
    print(av.last_layer_gradient.shape)
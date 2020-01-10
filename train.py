import numpy as np
from matplotlib import pyplot as plt
from Layers import Input, Output, Dense, Relu, Flatten, Convolution2D, Softmax, AveragePooling2D
from Model import Model
import Loss
from util import Image_generator, int_to_one_hot

if __name__ == '__main__':
    # date preprocess
    g = Image_generator()
    x_train = np.zeros(shape=(48, 80, 80, 3))
    y_train = np.zeros(shape=(48, 3, 1))
    n_classes = 3

    temp_img = np.array(plt.imread('test_pictures/train_picture/1.jpg'), dtype=np.float64)
    temp_label = int_to_one_hot(x=0, n_classes=n_classes)
    x_train[0:16], y_train[0:16] = g.one_input_flow_batch(input=temp_img, label=temp_label, batch_size=16)

    temp_img = np.array(plt.imread('test_pictures/train_picture/2.jpg'), dtype=np.float64)
    temp_label = int_to_one_hot(x=1, n_classes=n_classes)
    x_train[16:32], y_train[16:32] = g.one_input_flow_batch(input=temp_img, label=temp_label, batch_size=16)

    temp_img = np.array(plt.imread('test_pictures/train_picture/3.jpg'), dtype=np.float64)
    temp_label = int_to_one_hot(x=2, n_classes=n_classes)
    x_train[32:48], y_train[32:48] = g.one_input_flow_batch(input=temp_img, label=temp_label, batch_size=16)

    x_train /= 255

    print(x_train.shape, y_train.shape)

    # for x in x_train:
    #     plt.imshow(x)
    #     plt.show()
    # print(y_train)
    # print(x_train)

    # end of data preprocess

    # model config
    input = Input(input_shape=(80, 80, 3))
    av = AveragePooling2D(last_layer=input)
    # avv = AveragePooling2D(last_layer=av)
    conv1 = Convolution2D(last_layer=av, kernal_number=8, kernal_size=(3, 3))
    r1 = Relu(last_layer=conv1)
    av1 = AveragePooling2D(last_layer=r1)
    f = Flatten(last_layer=av1)
    d1 = Dense(last_layer=f, output_units=32)
    # r1 = Relu(last_layer=d1)
    d2 = Dense(last_layer=d1, output_units=3)
    sm = Softmax(last_layer=d2)
    output = Output(last_layer=sm)

    ce = Loss.Cross_entropy()
    model = Model(input_layer=input, output_layer=output, loss=ce)

    # shuffle is bery bery important !!!!!!!
    from random import shuffle
    index = [i for i in range(48)]
    shuffle(index)
    x_train = x_train[index, :, :, :]
    y_train = y_train[index, :, :]

    # train
    E = model.train_SGD(x_train_batch=x_train, y_train_batch=y_train, epoch=15, step_pre_epoch=48, lr=0.01)
    # E = model.train_all_batch(x_train=x_train, y_train=y_train, epoch=100, lr=0.01)
    # E = model.train_MBGD(x_train_batch=x_train, y_train_batch=y_train, epoch=30, step_pre_epoch=20, batch_pre_epoch=7, lr=0.01)
    plt.plot(E)
    plt.show()
    model.save_weights(root_directory='weights')

    # test
    cnt = 0
    for i in range(48):
        # plt.imshow(x_train[i])
        # plt.show()
        o = model.predict(input=x_train[i])
        t = y_train[i]

        # convert (n, 1) to shape (n, )
        o = o[..., 0]
        t = t[..., 0]

        pred = np.argmax(o)
        true = np.argmax(t)
        print('output=', o)
        print('true=', t)
        print('predict=', pred, 'true=', true)
        if pred == true:
            cnt += 1
        print('\n')
    print('accruacy is ', cnt / 48)





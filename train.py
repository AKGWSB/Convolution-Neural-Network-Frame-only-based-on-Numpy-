import numpy as np
from matplotlib import pyplot as plt
from Layers import Input, Output, Dense, Relu, Flatten, Convolution2D, Softmax, AveragePooling2D
from Model import Model
import Loss
from util import Image_generator, int_to_one_hot

def data_prepare():
    # date preprocess
    g = Image_generator()
    x_train = np.zeros(shape=(35 * 16, 80, 80, 3))
    y_train = np.zeros(shape=(35 * 16, 3, 1))
    n_classes = 3

    # temp_img = np.array(plt.imread('test_pictures/train_picture/1.jpg'), dtype=np.float64)
    # temp_label = int_to_one_hot(x=0, n_classes=n_classes)
    # x_train[0:16], y_train[0:16] = g.one_input_flow_batch(input=temp_img, label=temp_label, batch_size=16)
    #
    # temp_img = np.array(plt.imread('test_pictures/train_picture/2.jpg'), dtype=np.float64)
    # temp_label = int_to_one_hot(x=1, n_classes=n_classes)
    # x_train[16:32], y_train[16:32] = g.one_input_flow_batch(input=temp_img, label=temp_label, batch_size=16)
    #
    # temp_img = np.array(plt.imread('test_pictures/train_picture/3.jpg'), dtype=np.float64)
    # temp_label = int_to_one_hot(x=2, n_classes=n_classes)
    # x_train[32:48], y_train[32:48] = g.one_input_flow_batch(input=temp_img, label=temp_label, batch_size=16)

    import os
    cnt = 0
    dir = 'train_picture/1'
    filelist = os.listdir(dir)
    for path in filelist:
        temp_img = np.array(plt.imread(dir + '/' + path), dtype=np.float64)
        temp_label = int_to_one_hot(x=0, n_classes=n_classes)
        x_train[cnt * 16:(cnt + 1) * 16], y_train[cnt * 16:(cnt + 1) * 16] = g.one_input_flow_batch(input=temp_img,
                                                                                                    label=temp_label,
                                                                                                    batch_size=16,
                                                                                                    is_brighter=False,
                                                                                                    is_darker=False,
                                                                                                    is_flip_X=False,
                                                                                                    is_flip_Y=False)
        cnt += 1

    dir = 'train_picture/2'
    filelist = os.listdir(dir)
    for path in filelist:
        temp_img = np.array(plt.imread(dir + '/' + path), dtype=np.float64)
        temp_label = int_to_one_hot(x=1, n_classes=n_classes)
        x_train[cnt * 16:(cnt + 1) * 16], y_train[cnt * 16:(cnt + 1) * 16] = g.one_input_flow_batch(input=temp_img,
                                                                                                    label=temp_label,
                                                                                                    batch_size=16)
        cnt += 1

    dir = 'train_picture/3'
    filelist = os.listdir(dir)
    for path in filelist:
        temp_img = np.array(plt.imread(dir + '/' + path), dtype=np.float64)
        temp_label = int_to_one_hot(x=2, n_classes=n_classes)
        x_train[cnt * 16:(cnt + 1) * 16], y_train[cnt * 16:(cnt + 1) * 16] = g.one_input_flow_batch(input=temp_img,
                                                                                                    label=temp_label,
                                                                                                    batch_size=16)
        cnt += 1

    x_train /= 255

    print(x_train.shape, y_train.shape)

    return x_train, y_train

if __name__ == '__main__':

    x_train, y_train = data_prepare()

    # model config
    input = Input(input_shape=(80, 80, 3))
    # av = AveragePooling2D(last_layer=input)
    conv1 = Convolution2D(last_layer=input, kernal_number=8, kernal_size=(3, 3))
    # r1 = Relu(last_layer=conv1)
    # av1 = AveragePooling2D(last_layer=r1)
    f = Flatten(last_layer=conv1)
    d1 = Dense(last_layer=f, output_units=32)
    d2 = Dense(last_layer=d1, output_units=3)
    sm = Softmax(last_layer=d2)
    output = Output(last_layer=sm)

    ce = Loss.Cross_entropy()
    mse = Loss.Mean_squared_error()
    model = Model(input_layer=input, output_layer=output, loss=ce)
    # model = Model(input_layer=input, output_layer=output, loss=mse)

    # x = x_train[0]
    # y = y_train[0]
    # plt.imshow(x)
    # plt.show()
    # E = []
    # for j in range(1):
    #     for i in range(30):
    #         # cc = conv1.get_o(x)
    #         # print(cc)
    #         rr = model.predict(x)
    #         r = model.train_once(input=x, target=y, lr=0.001)
    #         print(j, i, rr[..., 0], r)
    #         E.append(r)
    # plt.plot(E)
    # plt.show()


    # shuffle is important !!!!!!!
    date_len = x_train.shape[0]
    from random import shuffle
    index = [i for i in range(date_len)]
    shuffle(index)
    x_train = x_train[index, :, :, :]
    y_train = y_train[index, :, :]

    # train
    E = model.train_SGD(x_train_batch=x_train, y_train_batch=y_train, epoch=1, step_pre_epoch=x_train.shape[0], lr=0.001)
    plt.plot(E)
    plt.show()
    model.save_weights(root_directory='weights')

    # test
    cnt = 0
    for i in range(35*16):
        # plt.imshow(x_train[i])
        # plt.show()
        o = model.predict(input=x_train[i])
        t = y_train[i]

        # convert (n, 1) to shape (n, )
        o = o[..., 0]
        t = t[..., 0]

        pred = np.argmax(o)
        true = np.argmax(t)
        print('test:', i)
        print('output=', o)
        print('true=', t)
        print('predict=', pred, 'true=', true)
        if pred == true:
            cnt += 1
        print('\n')
    print('accruacy is ', cnt / (35*16))





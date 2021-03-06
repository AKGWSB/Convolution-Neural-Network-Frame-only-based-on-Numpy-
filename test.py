import numpy as np
from matplotlib import pyplot as plt
from Layers import Input, Output, Dense, Relu, Flatten, Convolution2D, Softmax, AveragePooling2D
from Model import Model
import Loss
from util import Image_generator, int_to_one_hot

'''
this .py is for test ， debug
这个文件是为了测试 bug 而建立的
'''

# -------- full connect network test --------
def full_connect_network_test():

    input_shape = (5, 2)
    output_shape = (3, 1)

    x = np.random.rand(input_shape[0], input_shape[1])
    y = np.zeros(shape=output_shape)
    y[0][0] = 1

    input = Input(input_shape=input_shape)
    f1 = Flatten(last_layer=input)
    d1 = Dense(output_units=5, last_layer=f1)
    d2 = Dense(output_units=11, last_layer=d1)
    r1 = Relu(last_layer=d2)
    d3 = Dense(output_units=output_shape[0], last_layer=r1)
    sm = Softmax(last_layer=d3)
    output = Output(last_layer=sm)

    mse = Loss.Mean_squared_error()
    ce = Loss.Cross_entropy()
    model = Model(input_layer=input, output_layer=output, loss=ce)

    E = []
    while True:
        error = model.train_once(input=x, target=y, lr=0.01)
        E.append(error)
        # print(error, error.shape)

        if (error < 0.1):
            break

    plt.plot(E)
    plt.show()
# ----- end of full connect network test -----



# --------- convolution layer test --------
def convolution_layer_test():
    img_ = plt.imread('test_pictures/' + "oqs.jpg")
    img = img_[..., 0] * 0.2 + img_[..., 1]*0.7 + img_[..., 2]*0.1
    plt.imshow(img, cmap='gray')
    plt.show()

    input = Input(input_shape=img.shape)
    conv = Convolution2D(last_layer=input, test_mod=True)
    output = Output(last_layer=conv)

    input.FP(x=img)
    o = output.output[..., 0]
    plt.imshow(o, cmap='gray')
    plt.show()
# ----- end of convolution layer test -----

def convolution_network_test():

    # x = plt.imread('test_pictures/' + "hl.jpg")
    x = np.random.rand(8, 9, 3)
    input_shape = x.shape
    output_shape = (3, 1)
    y = np.zeros(shape=output_shape)
    y[0][0] = 1

    input = Input(input_shape=input_shape)
    conv = Convolution2D(last_layer=input, kernal_number=4, kernal_size=(3, 3))
    flatten = Flatten(last_layer=conv)
    d1 = Dense(output_units=6, last_layer=flatten)
    d2 = Dense(output_units=3, last_layer=d1)
    sm = Softmax(last_layer=d2)
    output = Output(last_layer=sm)

    ce = Loss.Cross_entropy()
    model = Model(input_layer=input, output_layer=output, loss=ce)

    E = []
    while True:
        error = model.train_once(input=x, target=y, lr=0.1)
        E.append(error)

        if (error < 0.1):
            break

    plt.plot(E)
    plt.show()

def dense_mnist_test():
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    n_classes = 10
    x_train = np.expand_dims(x_train[0:100], -1) / 255
    y_train = np.expand_dims(np.eye(n_classes)[y_train[0:100]], -1) # one hot labeling
    # print(x_train.shape, y_train[0], y_train[0].shape)

    input = Input(input_shape=(28, 28, 1))
    # conv = Convolution2D(last_layer=input, kernal_number=4, kernal_size=(3, 3))
    # r1 = Relu(last_layer=conv)
    flatten = Flatten(last_layer=input)
    d1 = Dense(output_units=32, last_layer=flatten)
    r2 = Relu(last_layer=d1)
    d2 = Dense(output_units=10, last_layer=r2)
    r3 = Relu(last_layer=d2)
    sm = Softmax(last_layer=r3)
    output = Output(last_layer=sm)

    ce = Loss.Cross_entropy()
    model = Model(input_layer=input, output_layer=output, loss=ce)

    model.train_SGD(x_train_batch=x_train, y_train_batch=y_train, epoch=100, step_pre_epoch=100, lr=0.01)

    # test
    cnt = 0
    for i in range(100):
        o = model.predict(input=x_train[i])
        o = o[..., 0]
        t = y_train[i][..., 0]

        pl = np.argmax(o)
        pt = np.argmax(t)
        print('output=', o)
        print('true=', t)
        print('predict=', pl, 'true=', pt)
        if pl == pt:
            cnt += 1
        print('\n')
    print('accruacy is ', cnt / 100)

def dense_sin_test():
    # sin data prepare
    x_train = np.zeros(shape=(360, 1))
    y_train = np.zeros(shape=(360, 1))
    for i in range(360):
        degree = (i/180)*3.1415926
        x_train[i] = degree
        y_train[i] = np.sin(degree)
    # plt.plot(xp)
    # plt.show()

    x_train = np.expand_dims(x_train, -1)
    y_train = np.expand_dims(y_train, -1)
    # end of sin data prepare

    input = Input(input_shape=(1, 1))
    d1 = Dense(output_units=32, last_layer=input)
    r1 = Relu(last_layer=d1)
    d2 = Dense(output_units=64, last_layer=r1)
    r2 = Relu(last_layer=d2)
    d3 = Dense(output_units=32, last_layer=r2)
    r3 = Relu(last_layer=d3)
    d4 = Dense(output_units=1, last_layer=r3)
    output = Output(last_layer=d4)

    mse = Loss.Mean_squared_error()
    model = Model(input_layer=input, output_layer=output, loss=mse)

    # shuffle is bery bery important !!!!!!!
    date_len = x_train.shape[0]
    print(x_train.shape, y_train.shape)
    from random import shuffle
    index = [i for i in range(date_len)]
    shuffle(index)
    x_train_sf = x_train[index, :, :]
    y_train_sf = y_train[index, :, :]

    model.train_SGD(x_train_batch=x_train_sf, y_train_batch=y_train_sf, epoch=100, step_pre_epoch=800, lr=0.01)
    # model.train_all_batch(x_train=x_train, y_train=y_train, epoch=1000, lr=0.01)

    res = []
    for i in range(360):
        x = x_train[i]
        y = model.predict(input=x)
        res.append(np.sum(y))

    plt.clf()
    plt.plot(x_train[..., 0], res)
    plt.plot(x_train[..., 0], y_train[..., 0])
    plt.show()
        # path = 'plots/' + str(xx) + ',jpg'
        # plt.savefig(path)

# ----------------- convolution mnist test -----------------
def get_mnist_100():
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    n_classes = 10
    x_train = np.expand_dims(x_train[0:100], -1) / 255
    y_train = np.expand_dims(np.eye(n_classes)[y_train[0:100]], -1)  # one hot labeling

    return x_train, y_train

def get_convolution_mnist_model():
    input = Input(input_shape=(28, 28, 1))
    conv = Convolution2D(last_layer=input, kernal_number=4, kernal_size=(3, 3))
    r = Relu(last_layer=conv)
    ap = AveragePooling2D(last_layer=r, step=2)
    r1 = Relu(last_layer=ap)
    flatten = Flatten(last_layer=r1)
    d2 = Dense(output_units=10, last_layer=flatten)
    r3 = Relu(last_layer=d2)
    sm = Softmax(last_layer=r3)
    output = Output(last_layer=sm)

    ce = Loss.Cross_entropy()
    model = Model(input_layer=input, output_layer=output, loss=ce)

    return model

def convolution_mnist_test():
    x_train, y_train = get_mnist_100()
    model = get_convolution_mnist_model()
    E = model.train_SGD(x_train_batch=x_train, y_train_batch=y_train, epoch=15, step_pre_epoch=100, lr=0.01)
    plt.plot(E)
    plt.show()
    model.save_weights(root_directory='weights')

def load_weights_test():
    x_train, y_train = get_mnist_100()
    model = get_convolution_mnist_model()
    model.load_weights(root_directory='weights')

    # test
    cnt = 0
    for i in range(100):
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
    print('accruacy is ', cnt / 100)
# -----------end of convolution mnist test -----------------

def image_generator_test():
    g = Image_generator()
    x = np.array(plt.imread('test_pictures/' + "hl.jpg"), dtype=np.float64)
    y = int_to_one_hot(3, 5)

    x_train, y_train = g.one_input_flow_batch(input=x, label=y)
    for p in x_train:
        plt.imshow(p/255)
        plt.show()
    print(y_train, y_train.shape)

def cnn_test():
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

    model.load_weights(root_directory='weights')

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

if __name__ == '__main__':

    # full_connect_network_test()

    # convolution_layer_test()

    # convolution_network_test()

    # dense_mnist_test()

    dense_sin_test()

    # convolution_mnist_test()
    # load_weights_test()

    # image_generator_test()

    # cnn_test()







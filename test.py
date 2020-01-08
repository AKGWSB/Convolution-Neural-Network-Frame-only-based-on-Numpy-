import numpy as np
from matplotlib import pyplot as plt
from Layers import Input, Output, Dense, Relu, Flatten, Convolution2D, Softmax, AveragePooling2D
from Model import Model
import Loss
from util import Image_generator

'''
this .py is for test 
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

    model.train_SGD(x_train_batch=x_train, y_train_batch=y_train, epoch=100, step_pre_epoch=800, lr=0.01)
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
    result = g.one_input_flow_batch(input=x)
    for p in result:
        plt.imshow(p/255)
        plt.show()

if __name__ == '__main__':

    # full_connect_network_test()

    # convolution_layer_test()

    # convolution_network_test()

    # dense_mnist_test()

    # dense_sin_test()

    # convolution_mnist_test()
    # load_weights_test()

    image_generator_test()





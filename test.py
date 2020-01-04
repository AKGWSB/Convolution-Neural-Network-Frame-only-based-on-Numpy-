import numpy as np
from matplotlib import pyplot as plt
from Layers import Input, Output, Dense, Relu, Flatten, Convolution2D, Softmax
from Model import Model
import Loss

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

if __name__ == '__main__':

    # full_connect_network_test()
    # convolution_layer_test()
    convolution_network_test()





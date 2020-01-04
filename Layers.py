import numpy as np

class Input:
    '''
    Input layer
    '''

    # parameter
    # input_shape : the shape of input mat
    def __init__(self, input_shape):
        # lsit struct bound
        self.next_layer = None
        self.last_layer = None

        # IO shape config
        self.output_shape = input_shape

    # Forward propagation
    # parameter
    # x : the x_train , input to model
    def FP(self, x):
        self.input = x
        self.next_layer.FP(x=x)

    # end of back propagation
    def BP(self, gradient, lr):
        pass

class Output:
    '''
    Output layer
    '''

    # parameters
    # last_layer   : the last layer in your model
    def __init__(self, last_layer=None):
        # lsit struct bound
        self.next_layer = None
        self.last_layer = last_layer
        self.last_layer.next_layer = self

    # Forward propagation
    # when FP called in output, output_layer.output = last_layer's output
    def FP(self, x):
        self.output = x

    # Back propagation
    def BP(self, gradient, lr=0.0001):
        self.last_layer.BP(gradient=gradient, lr=lr)

class Dense:
    '''
    full connect layer, only support input shape (n, 1)
    ------------------- shape ----------------------
    input_shape                             ：m x 1
    output_shape                            ：n x 1
    weight_shape                            : m x n
    next_layer's gradient feed back shape   ：n x 1
    gradient to last_layer shape            ：m x 1
    ------------------- shape ----------------------
    （output_units = n）
    '''

    # parameters
    # output_units : output_units=n output shape is (m, 1)
    # last_layer   : the last layer in your model
    def __init__(self, output_units, last_layer=None):
        # model list struct bound
        self.next_layer = None
        self.last_layer = last_layer
        self.last_layer.next_layer = self

        # IO shape config
        self.input_shape = self.last_layer.output_shape
        self.output_shape = (output_units, 1)

        # parameters' shape config & init
        # kaiming init :
        self.weights = np.random.rand(self.input_shape[0], self.output_shape[0])
        self.weights /= (0.5 * np.sqrt(self.input_shape[0] * self.output_shape[0]))
        self.bias = np.zeros((self.output_shape[0], 1))

    # Forward propagation
    # parameters
    # x : last layer's output
    def FP(self, x):
        self.input = x
        self.output = self.weights.T.dot(self.input) + self.bias
        self.next_layer.FP(x=self.output)

    # Back propagation
    # parameters
    # gradient : last layer's gradient
    # lr       : learning rate
    def BP(self, gradient, lr):
        self.gradient = gradient

        last_layer_gradient = self.weights.dot(self.gradient)
        self.last_layer.BP(gradient=last_layer_gradient, lr=lr)

        grad_for_w = np.tile(self.input.T, self.output_shape)   # gradient for weights
        self.weights -= (grad_for_w * self.gradient).T * lr
        self.bias -= self.gradient * lr  # gradient for bias mat is 1

        # for debug
        # print('gradient shape = ', self.gradient.shape)
        # print('weights shape = ', self.weights.shape)
        # print('g_w shape = ', grad_for_w.shape)
        # print('input shape = ', self.input_shape)
        # print('output shape = ', self.output.shape)
        # print('\n')

class Relu:
    '''
    Relu layer
    -------------------------------- shape -----------------------------------
    input_shape                             ：any_shape
    output_shape                            ：the same as input_shape
    next_layer's gradient feed back shape   ：the same as output_shape
    gradient to last_layer shape            ：the same as input_shape
    -------------------------------- shape -----------------------------------
    '''

    def __init__(self, last_layer=None):
        # model list struct bound
        self.next_layer = None
        self.last_layer = last_layer
        self.last_layer.next_layer = self

        # IO shape config
        self.input_shape = self.last_layer.output_shape
        self.output_shape = self.input_shape

    def FP(self, x):
        self.input = x
        self.next_layer.FP(x=np.maximum(x, 0))

    def BP(self, gradient, lr):
        # input>0, self.gradient = 1 * gradient , else self.gradient = 0
        select_mat = np.zeros(shape=self.input.shape)
        select_mat = np.greater(self.input, select_mat)
        self.last_layer.BP(gradient=select_mat*gradient, lr=lr)

class Softmax:
    '''
    Softmax Layer
    -------------------------------- shape -----------------------------------
    input_shape                             ：any_shape
    output_shape                            ：the same as input_shape
    next_layer's gradient feed back shape   ：the same as output_shape
    gradient to last_layer shape            ：the same as input_shape
    -------------------------------- shape -----------------------------------
    '''

    def __init__(self, last_layer=None):
        # model list struct bound
        self.next_layer = None
        self.last_layer = last_layer
        self.last_layer.next_layer = self

        # IO shape config
        self.input_shape = self.last_layer.output_shape
        self.output_shape = self.input_shape

    def FP(self, x):
        self.input = x
        self.output = self.input / np.sum(self.input)
        self.next_layer.FP(x=self.output)

    def BP(self, gradient, lr):
        self.gradient = gradient
        self.last_layer.BP(gradient=self.gradient/np.sum(self.input), lr=lr)

class Flatten:
    '''
    Flatten layer input shape= (a, b, c), output shape= (a*b*c, 1)
    ------------------- shape ----------------------
    input_shape                             ：a x b x c
    output_shape                            ：a*b*c x 1
    next_layer's gradient feed back shape   ：= output_shape
    gradient to last_layer shape            ：= input_shape
    ------------------- shape ----------------------
    '''

    def __init__(self, last_layer=None):
        # model list struct bound
        self.next_layer = None
        self.last_layer = last_layer
        self.last_layer.next_layer = self

        # IO shape config
        self.input_shape = self.last_layer.output_shape
        self.output_shape = (np.prod(self.input_shape), 1)

    def FP(self, x):
        self.next_layer.FP(x=np.expand_dims(x.flatten(), axis=-1))

    def BP(self, gradient, lr):
        self.last_layer.BP(gradient=gradient.reshape(self.input_shape), lr=lr)

class Convolution2D:
    '''
    Convolution2D Layer
    未完工
    '''

    def __init__(self, last_layer=None, kernal_number=1, kernal_size=(3,3), test_mod=False):
        # model list struct bound
        self.next_layer = None
        self.last_layer = last_layer
        self.last_layer.next_layer = self

        # IO shape config
        self.kernal_number = kernal_number
        self.input_shape = self.last_layer.output_shape
        self.output_shape = (self.input_shape[0]-2*int(kernal_size[0]/2), self.input_shape[1]-2*int(kernal_size[1]/2), self.kernal_number)

        # parameters' shape config & init
        # (kernal_size[0], kernal_siez[1], last_layer's output's channel_number), kernal_number
        self.kernals = np.random.rand(kernal_size[0], kernal_size[1], self.input_shape[-1], kernal_number)
        self.kernals /= np.sum(self.kernals)

        # Laplace Operator test
        self.test_mod = test_mod
        if self.test_mod == True:
            # using Laplace Operator for convolution
            temp = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
            self.kernals = np.random.rand(3, 3, 1)
            self.kernals[..., 0] = temp

    def FP(self, x):
        self.input = x
        self.output = np.zeros(shape=self.output_shape)

        # width bias
        w1 = self.kernals.shape[0]
        w2 = self.kernals.shape[1]

        # Convolution2D
        for k in range(self.kernal_number):
            filter = self.kernals[..., k]
            for i in range(self.output_shape[0]):
                for j in range(self.output_shape[1]):
                    if self.test_mod == True:
                        # for img show, auto double limit to 0~255
                        self.output[i][j][k] = np.minimum(np.maximum(np.sum(filter * self.input[i:i+w1, j:j+w2]), 0), 255)
                    else:
                        # train, unlimited
                        self.output[i][j][k] = np.sum(filter * self.input[i:i + w1, j:j + w2])

        self.next_layer.FP(x=self.output)

    def BP(self, gradient, lr):
        self.gradient = gradient

        # weights flip 180 degree
        self.w_flip = np.flip(np.flip(self.kernals, 0), 1)
        # padding gradient
        gl1 = self.gradient.shape[0]
        gl2 = self.gradient.shape[1]
        b1 = int(self.kernals.shape[0]/2)
        b2 = int(self.kernals.shape[1]/2)
        ks1 = self.kernals.shape[0]
        ks2 = self.kernals.shape[1]
        self.gradient_pad = np.zeros(shape=(gl1+4*b1, gl2+4*b2, self.kernal_number))
        self.gradient_pad[2*b1:2*b1+gl1, 2*b2:2*b2+gl2] = self.gradient
        self.last_layer_gradient = np.zeros(shape=self.input_shape)
        # convolution for calculate last_layer's gradient
        for k in range(self.kernal_number):
            for i in range(self.input_shape[0]):
                for j in range(self.input_shape[1]):
                    for c in range(self.input_shape[2]):
                        self.last_layer_gradient[i][j][c] += np.sum(self.gradient_pad[i:i+ks1, j:j+ks2, c]*self.w_flip[..., c, k])

        self.last_layer.BP(gradient=self.last_layer_gradient, lr=lr)

        # convolution for updating weights
        w1 = self.output_shape[0]
        w2 = self.output_shape[1]
        self.grad_for_w = np.zeros(shape=self.kernals.shape)
        # for every kernals
        for k in range(self.kernal_number):
            for i in range(self.kernals.shape[0]):
                for j in range(self.kernals.shape[1]):
                    # for RGB channels
                    for c in range(self.kernals.shape[2]):
                        self.grad_for_w[i][j][c][k] = np.sum(self.input[i:i+w1, j:j+w2, c] * self.gradient[..., k])
        self.kernals -= self.grad_for_w * lr



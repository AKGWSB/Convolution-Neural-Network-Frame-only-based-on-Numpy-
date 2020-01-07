import numpy as np

'''
Author: Ruo Long Lee, Collage of Computer Science, Shen Zhen University
      : 李若龙 深大计软
'''

'''
工地英语警告↓
all layers' input / output shape must be a two or more dimension's np array
sample: input a vector of vec3(1, 1, 1), three numbers , the shape must be (3, 1)
      : even a single number , must have the shape of (1, 1) 
      : in other words , it's a type of two dimension's 'tensor', or blocks(??)
      : as channels of RGB image , the third shape of the array is channels
      
所有层的输入输出形状必须是至少两个维度的数组
示例：输入一个三个数字组成的向量，维度必须是 (3, 1)
    ：即使是单个数字，也必须拥有 (1, 1) 的大小
    ：换句话说就是经过层的‘流’至少是二维表示的，一种‘张量’，或者类似方块的堆叠
    ：彩色图像的通道数使用第三个维度表示
'''

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

        self.output_shape = self.last_layer.output_shape

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
        self.weights -= np.mean(self.weights)
        self.bias = np.random.rand(self.output_shape[0], 1)
        self.bias /= np.sum(self.bias)
        self.bias -= np.mean(self.bias)

    # Forward propagation
    # parameters
    # x : last layer's output
    def FP(self, x):
        self.input = x
        self.output = self.weights.T.dot(self.input) + self.bias
        # print('output.shape = ',self.output.shape)
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
        self.last_layer_gradient = select_mat*gradient
        self.last_layer.BP(gradient=self.last_layer_gradient, lr=lr)

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
        self.expi = np.exp(self.input)
        self.sum = np.sum(self.expi)
        self.output = self.expi / self.sum
        self.next_layer.FP(x=self.output)

    def BP(self, gradient, lr):
        self.gradient = gradient
        self.tp = self.expi/self.sum
        self.last_layer_gradient = np.zeros(shape=self.input_shape)

        for i in range(self.input_shape[0]):
            self.gradient_for_Ii = np.zeros(shape=self.input_shape)
            for j in range(self.input_shape[0]):
                if i == j:
                    self.gradient_for_Ii[j] = self.output[i]*(1 - self.output[i])
                else:
                    self.gradient_for_Ii[j] = -1 * self.output[i] * self.output[j]

            self.last_layer_gradient[i] = np.sum(self.gradient_for_Ii * self.gradient)

        self.last_layer.BP(gradient=self.last_layer_gradient, lr=lr)

        # bottom_diff = (top_diff - dot(top_diff, top_data)) * top_data

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



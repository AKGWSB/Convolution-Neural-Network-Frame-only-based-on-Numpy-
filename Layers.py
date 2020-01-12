import numpy as np
import matplotlib.pyplot as plt

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
        self.input = x.copy()
        self.next_layer.FP(x=x)

    # end of back propagation
    def BP(self, gradient, lr):
        pass

    # no parameters to save or load
    def save_weights(self, name, root_directory):
        self.next_layer.save_weights(name=name,root_directory=root_directory)

    def load_weights(self, name, root_directory):
        self.next_layer.load_weights(name=name, root_directory=root_directory)

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
        self.output = x.copy()

    # Back propagation
    def BP(self, gradient, lr=0.01):
        self.gradient = gradient.copy()
        self.last_layer.BP(gradient=self.gradient, lr=lr)

    # no parameters to save or load
    def save_weights(self, name, root_directory):
        pass

    def load_weights(self, name, root_directory):
        pass

class Dense:
    '''
    full connect layer, only support input shape (m, 1)
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
    # output_units : if output_units = m, then output shape is (m, 1)
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
        # glorot_uniform init :
        limit = np.sqrt(6/(self.input_shape[0]+self.output_shape[0]))
        self.weights = np.random.uniform(-1*limit, limit, size=(self.input_shape[0], self.output_shape[0]))
        # self.weights /= (0.5 * np.sqrt(self.input_shape[0] * self.output_shape[0]))

        # self.weights /= (0.5 * self.input_shape[0])
        # self.weights /= np.sum(self.weights)
        #self.weights -= np.mean(self.weights)  # problem

        self.bias = np.zeros((self.output_shape[0], 1))
        # self.bias = np.random.rand(self.output_shape[0], 1)
        # self.bias /= np.sum(self.bias)

    # Forward propagation
    # parameters
    # x : last layer's output
    def FP(self, x):
        self.input = x.copy()
        self.output = self.weights.T.dot(self.input) + self.bias
        # print('output.shape = ',self.output.shape)
        self.next_layer.FP(x=self.output)

    # Back propagation
    # parameters
    # gradient : last layer's gradient
    # lr       : learning rate
    def BP(self, gradient, lr):
        self.gradient = gradient.copy()
        # print('self g',self.gradient.shape,  self.gradient)
        last_layer_gradient = self.weights.dot(self.gradient)
        # print('dense pass g=', last_layer_gradient)
        self.last_layer.BP(gradient=last_layer_gradient, lr=lr)

        # print('ds input=', self.input[..., -1])
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

    def save_weights(self, name, root_directory):
        num = int(name) + 1
        path = root_directory + '/' + str(num)
        np.save(path, self.weights)
        print('save weights successfully, path is:', path)
        self.next_layer.save_weights(name=str(num), root_directory=root_directory)

    def load_weights(self, name, root_directory):
        num = int(name) + 1
        path = root_directory + '/' + str(num) + '.npy'
        self.weights = np.load(path)
        print('load weights successfully, path is:', path)
        self.next_layer.load_weights(name=str(num), root_directory=root_directory)


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
        self.input = x.copy()
        self.next_layer.FP(x=np.maximum(x, 0))

    def BP(self, gradient, lr):
        self.gradient = gradient.copy()
        # input>0, self.gradient = 1 * gradient , else self.gradient = 0
        select_mat = np.zeros(shape=self.input.shape)
        select_mat = np.greater(self.input, select_mat).astype(np.int32)
        self.last_layer_gradient = select_mat*self.gradient
        # print('relu pass g=', self.last_layer_gradient)
        self.last_layer.BP(gradient=self.last_layer_gradient, lr=lr)

    # no parameters to save or load
    def save_weights(self, name, root_directory):
        self.next_layer.save_weights(name=name, root_directory=root_directory)

    def load_weights(self, name, root_directory):
        self.next_layer.load_weights(name=name, root_directory=root_directory)

class Softmax:
    '''
    Softmax Layer
    -------------------------------- shape -----------------------------------
    input_shape                             ：(n, 1) any shape, but must be (x, 1) , two dimension
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
        self.input = x.copy()
        self.expi = np.exp(self.input)
        self.sum = np.sum(self.expi)
        self.output = self.expi / self.sum
        self.next_layer.FP(x=self.output)

    def BP(self, gradient, lr):
        self.gradient = gradient.copy()
        self.tp = self.expi/self.sum
        self.last_layer_gradient = np.zeros(shape=self.input_shape, dtype=np.float64)

        for i in range(self.input_shape[0]):
            self.gradient_for_Ii = np.zeros(shape=self.input_shape, dtype=np.float64)

            for j in range(self.input_shape[0]):
                if i == j:
                    self.gradient_for_Ii[j] = self.output[i]*(1 - self.output[i])
                else:
                    self.gradient_for_Ii[j] = -1 * self.output[i] * self.output[j]

            self.last_layer_gradient[i] = np.sum(self.gradient_for_Ii * self.gradient)

        # print('sm pass g', self.last_layer_gradient[..., 0])
        self.last_layer.BP(gradient=self.last_layer_gradient, lr=lr)

    # no parameters to save or load
    def save_weights(self, name, root_directory):
        self.next_layer.save_weights(name=name, root_directory=root_directory)

    def load_weights(self, name, root_directory):
        self.next_layer.load_weights(name=name, root_directory=root_directory)

class Flatten:
    '''
    Flatten layer input shape= (a, b, c), output shape= (a*b*c, 1)
    ------------------- shape ----------------------
    input_shape                             ：(a, b, c)
    output_shape                            ：(a*b*c, 1)
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
        self.input = x.copy()
        self.next_layer.FP(x=self.input.reshape(self.output_shape))

    def BP(self, gradient, lr):
        self.gradient = gradient.copy()
        # print('flatten self g=', self.gradient)
        self.last_layer_gradient = self.gradient.reshape(self.input_shape)
        # print('flatten pass g=', self.last_layer_gradient)
        self.last_layer.BP(gradient=self.last_layer_gradient, lr=lr)

    # no parameters to save or load
    def save_weights(self, name, root_directory):
        self.next_layer.save_weights(name=name, root_directory=root_directory)

    def load_weights(self, name, root_directory):
        self.next_layer.load_weights(name=name, root_directory=root_directory)

class Convolution2D:
    '''
    Convolution2D Layer
    ------------------- shape ----------------------
    input_shape                             ：(m, n, c) , c is channels number
    kernal_shape                            : (k1, k2, filter_number)
    output_shape                            ：(m-2*k1/2, n-2*k2/2, filter_number)
    next_layer's gradient feed back shape   ：output_shape
    gradient to last_layer shape            ：input_shape
    ------------------- shape ----------------------
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
        # self.output = np.zeros(shape=self.output_shape, dtype=np.float64)

        # parameters' shape config & init
        # (kernal_size[0], kernal_siez[1], last_layer's output's channel_number), kernal_number
        # self.kernals = np.random.rand(kernal_size[0], kernal_size[1], self.input_shape[-1], kernal_number)
        # self.kernals = np.ones((kernal_size[0], kernal_size[1], self.input_shape[-1], kernal_number))
        # self.kernals /= np.sum(self.kernals)  # problem
        # self.kernals /= (self.kernals.shape[0] * self.kernals.shape[1] * self.kernals.shape[2] * 0.5)
        # self.bias = np.zeros(shape=self.output_shape)

        limit = np.sqrt(6/(np.prod((kernal_size[0], kernal_size[1], self.input_shape[-1], kernal_number))))
        self.kernals = np.random.uniform(-1*limit, limit, size=(kernal_size[0], kernal_size[1], self.input_shape[-1], kernal_number))
        # self.kernals = np.random.rand(kernal_size[0], kernal_size[1], self.input_shape[-1], kernal_number)

        # # Laplace Operator test
        # self.test_mod = test_mod
        # if self.test_mod == True:
        #     # using Laplace Operator for convolution
        #     temp = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
        #     self.kernals = np.random.rand(3, 3, 1)
        #     self.kernals[..., 0] = temp

    def FP(self, x):
        self.input = x.copy()
        self.output = np.zeros(shape=self.output_shape, dtype=np.float64)

        # width bias
        w1 = self.kernals.shape[0]
        w2 = self.kernals.shape[1]

        # Convolution2D
        for k in range(self.kernal_number):
            filter = self.kernals[..., k]
            for i in range(self.output_shape[0]):
                for j in range(self.output_shape[1]):
                    # if self.test_mod == True:
                    #     # for img show, auto double limit to 0~255
                    #     self.output[i][j][k] = np.minimum(np.maximum(np.sum(filter * self.input[i:i+w1, j:j+w2]), 0), 255)
                    # else:
                        # train, unlimited
                        self.output[i][j][k] = np.sum(filter * self.input[i:i + w1, j:j + w2])
            # plt.imshow(np.maximum(np.minimum(self.output[..., k], 1), 0), cmap='gray')
            # plt.show()

        # self.output += self.bias
        # print('conv output=', self.output)
        self.next_layer.FP(x=self.output)

    def get_o(self, x):
        self.input = x.copy()
        self.output_ = np.zeros(shape=self.output_shape, dtype=np.float64)

        # width bias
        w1 = self.kernals.shape[0]
        w2 = self.kernals.shape[1]

        # Convolution2D
        for k in range(self.kernal_number):
            filter = self.kernals[..., k]
            for i in range(self.output_shape[0]):
                for j in range(self.output_shape[1]):
                        self.output_[i][j][k] = np.sum(filter * self.input[i:i + w1, j:j + w2])
        return self.output_

    def BP(self, gradient, lr):
        self.gradient = gradient.copy()

        # # weights flip 180 degree
        # self.w_flip = np.flip(np.flip(self.kernals.copy(), 0), 1)
        # # padding gradient
        # gl1 = self.gradient.shape[0]        # 回传梯度的形状
        # gl2 = self.gradient.shape[1]
        # b1 = int(self.kernals.shape[0]/2)   # 卷积卷掉的像素个数
        # b2 = int(self.kernals.shape[1]/2)
        # ks1 = self.kernals.shape[0]         # 卷积核形状
        # ks2 = self.kernals.shape[1]
        # self.gradient_pad = np.zeros(shape=(gl1+4*b1, gl2+4*b2, self.kernal_number))
        # self.gradient_pad[2*b1:2*b1+gl1, 2*b2:2*b2+gl2] = self.gradient.copy()
        # self.last_layer_gradient = np.zeros(shape=self.input_shape)
        # # convolution for calculate last_layer's gradient
        # for k in range(self.kernal_number):
        #     for i in range(self.input_shape[0]):
        #         for j in range(self.input_shape[1]):
        #             for c in range(self.input_shape[2]):
        #                 self.last_layer_gradient[i][j][c] += np.sum(self.gradient_pad[i:i+ks1, j:j+ks2, c]*self.w_flip[..., c, k])
        #
        # self.last_layer.BP(gradient=self.last_layer_gradient, lr=lr)

        # convolution for updating weights
        w1 = self.output_shape[0]
        w2 = self.output_shape[1]
        self.gfw = np.zeros(self.kernals.shape) # gradient for weights
        for k in range(self.kernal_number):
            gk = self.gradient[..., k]
            for c in range(self.kernals.shape[2]):
                Ich = self.input[..., c]
                for i in range(self.kernals.shape[0]):
                    for j in range(self.kernals.shape[1]):
                        # self.kernals[i][j][c][k] -= np.sum(Ich[i:i+w1, j:j+w2] * gk) * lr
                        self.gfw[i][j][c][k] = np.sum(Ich[i:i+w1, j:j+w2] * gk) * lr
        self.kernals -= self.gfw

    def save_weights(self, name, root_directory):
        num = int(name) + 1
        path = root_directory + '/' + str(num)
        np.save(path, self.kernals)
        print('save weights successfully, path is:', path)
        self.next_layer.save_weights(name=str(num), root_directory=root_directory)

    def load_weights(self, name, root_directory):
        num = int(name) + 1
        path = root_directory + '/' + str(num) + '.npy'
        self.kernals = np.load(path)
        print('load weights successfully, path is:', path)
        self.next_layer.load_weights(name=str(num), root_directory=root_directory)


class AveragePooling2D:
    '''
    AveragePooling2D layer
    ------------------- shape ----------------------
    input_shape                             ：(m, n, c) , c is channels number
    output_shape                            ：= input_shape
    next_layer's gradient feed back shape   ：= input_shape
    gradient to last_layer shape            ：= input_shape
    ------------------- shape ----------------------
    '''

    # parameters:
    # step      : the pooling window is step x step shape
    def __init__(self, last_layer=None, step=2):
        # model list struct bound
        self.next_layer = None
        self.last_layer = last_layer
        self.last_layer.next_layer = self

        # IO shape config
        self.input_shape = self.last_layer.output_shape
        self.output_shape =self.last_layer.output_shape
        self.output_shape = list(self.input_shape)
        self.output_shape[0]  = int(self.input_shape[0] // 2)
        self.output_shape[1]  = int(self.input_shape[1] // 2)
        self.output_shape = tuple(self.output_shape)

        # print('av config --  shape=', self.output_shape)

        self.step = step

    def FP(self, x):
        self.input = x.copy()
        self.output = np.zeros(shape=self.output_shape, dtype=np.float64)
        for i in range(self.output_shape[0]):
            for j in range(self.output_shape[1]):
                for c in range(self.output_shape[2]):
                    self.output[i][j][c] = np.sum(self.input[2*i:2*i+self.step, 2*j:2*j+self.step, c]) / (self.step*self.step)

        # print('av output shape=', self.output.shape)
        # if self.output_shape[2] == 3:
        #     plt.imshow(self.output)
        #     plt.show()
        self.next_layer.FP(x=self.output)

    def BP(self, gradient, lr):
        self.gradient = gradient.copy()
        # print('av g', self.gradient)
        self.last_layer_gradient = np.zeros(shape=self.input_shape, dtype=np.float64)
        for i in range(self.output_shape[0]):
            for j in range(self.output_shape[1]):
                for c in range(self.output_shape[2]):
                    self.last_layer_gradient[2*i:2*i+self.step, 2*j:2*j+self.step, c:c+1] = self.gradient[i][j][c] / (self.step*self.step)
        # print('av pass g=', self.last_layer_gradient)
        self.last_layer.BP(gradient=self.last_layer_gradient, lr=lr)

    # no parameters to save or load
    def save_weights(self, name, root_directory):
        self.next_layer.save_weights(name=name, root_directory=root_directory)

    def load_weights(self, name, root_directory):
        self.next_layer.load_weights(name=name, root_directory=root_directory)

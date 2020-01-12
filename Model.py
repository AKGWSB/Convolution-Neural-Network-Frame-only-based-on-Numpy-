import numpy as np
import Loss

'''
Author: Ruo Long Lee, Collage of Computer Science, Shen Zhen University
      : 李若龙 深大计软
'''

'''
工地英语警告↓
class of a model, need an input_layer (class), an output_layer (class), and a loss function (class) to config
model have some function like: train or predict, save/load weights

模型类，需要一个输入层，输出层，和一个损失函数的实例化对象去进行初始化
拥有一些常见的方法，比如训练，测试，保存模型等
'''

class Model:

    # parameters:
    # input_layer  : the input_layer  of your sequential model
    # output_layer : the output_layer of your sequential model
    def __init__(self, input_layer=None, output_layer=None, loss=None):
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.loss = loss

    # parameters:
    # name           : the sequential model's weights' name, range 1 ~ n    (dont't need to config this parameter)
    # root_directory : the directory to save weights
    def save_weights(self, root_directory):
        self.input_layer.save_weights(name=0, root_directory=root_directory)

    # parameters:
    # name           : the sequential model's weights' name, range 1 ~ n    (dont't need to config this parameter)
    # root_directory : the directory to save weights
    def load_weights(self, root_directory):
        self.input_layer.load_weights(name=0, root_directory=root_directory)

    # parameters:
    # input     : a single input sample, the shape of input must = model.input_layer.input_shape
    # return    : the output of model
    def predict(self, input):
        self.input_layer.FP(x=input)
        output = self.output_layer.output
        return output

    # parameters:
    # input     : a single input sample, the shape of input must = input_shape
    # target    : the expected output of model, shape = output_shape
    def train_once(self, input, target, lr):
        self.input_layer.FP(x=input)
        output = self.output_layer.output
        error = self.loss.get_error(output=output, target=target)
        gradient = self.loss.get_gradient()
        self.output_layer.BP(gradient=gradient, lr=lr)

        print('output=', output[..., 0], 'target=', target[..., 0])

        return error

    # some problem here function ↓
    def train_all_batch(self, x_train, y_train, epoch, lr):
        batch_size = x_train.shape[0]
        # print(batch_size)
        E = []
        for ep in range(epoch):
            gradient = np.zeros(shape=self.output_layer.output_shape)
            error = 0

            for i in range(batch_size):
                self.input_layer.FP(x=x_train[i])
                o = self.output_layer.output
                e = self.loss.get_error(output=o, target=y_train[i])
                g = self.loss.get_gradient()

                error += e
                gradient += g

            error /= batch_size
            gradient /= batch_size

            E.append(error)
            print('epoch:', ep, 'error=', error)

            self.output_layer.BP(gradient=gradient, lr=lr)

        return E

    # --- Stochastic gradient descent ---
    # the input (x_train ) must be (batch_size, input_shape )
    # the output (y_train) must be (batch_size, output_shape)
    # the first axis (0) of input_batch / output_batch is the size of batch
    # return error pre epoch
    def train_SGD(self, x_train_batch, y_train_batch, epoch, step_pre_epoch, lr):
        E = []
        train_size = x_train_batch.shape[0]
        for xx in range(epoch):
            e = 0
            for i in range(step_pre_epoch):
                ridx = np.sum(np.random.randint(0, train_size, (1,)))  # random index, select a sample from batch
                error = self.train_once(input=x_train_batch[ridx], target=y_train_batch[ridx], lr=lr)
                print('epoch', xx+1, '/', epoch, ' step', i+1, '/', step_pre_epoch, ' error:', error)
                # E.append(error)
                e += error
            E.append(e/step_pre_epoch)
        return E

    def train_MBGD(self, x_train_batch, y_train_batch, epoch, step_pre_epoch, batch_pre_epoch, lr):
        E = []
        train_size = x_train_batch.shape[0]
        for xx in range(epoch):
            for i in range(step_pre_epoch):
                # ridx = np.random.randint(0, train_size, size=(batch_pre_epoch, ))
                ridx = np.random.choice(train_size, (batch_pre_epoch, 1))
                e = 0
                graidient = np.zeros(self.output_layer.output_shape)
                for idxx in ridx:
                    idx = np.sum(idxx)
                    x = x_train_batch[idx]
                    y = y_train_batch[idx]
                    # print(x.shape, y.shape)
                    self.input_layer.FP(x=x)
                    output = self.output_layer.output
                    error = self.loss.get_error(output=output, target=y)
                    e += error
                    g = self.loss.get_gradient()
                    graidient += g
                e /= batch_pre_epoch
                graidient /= batch_pre_epoch
                self.output_layer.BP(gradient=graidient, lr=lr)
                print('epoch', xx+1, '/', epoch, 'error=', e)
                E.append(e)




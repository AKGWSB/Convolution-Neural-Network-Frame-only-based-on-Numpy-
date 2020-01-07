import numpy as np
import Loss

'''
Author: Ruo Long Lee, Collage of Computer Science, Shen Zhen University
      : 李若龙 深大计软
'''

class Model:

    def __init__(self, input_layer=None, output_layer=None, loss=None):
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.loss = loss

    def predict(self, input):
        self.input_layer.FP(x=input)
        output = self.output_layer.output
        return output

    def train_once(self, input, target, lr):
        self.input_layer.FP(x=input)
        output = self.output_layer.output

        error = self.loss.get_error(output=output, target=target)
        gradient = self.loss.get_gradient()

        self.output_layer.BP(gradient=gradient, lr=lr)

        # print('error: ', error)   # one hot label
        # print(output, target)

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

    # the input (x_train ) must be (batch_size, input_shape )
    # the output (y_train) must be (batch_size, output_shape)
    # the first axis (0) of input / output batch is the size of batch
    def train_SGD(self, x_train_batch, y_train_batch, epoch, step_pre_epoch, lr):
        E = []
        train_size = x_train_batch.shape[0]
        for xx in range(epoch):
            for i in range(step_pre_epoch):
                ridx = np.sum(np.random.randint(0, train_size, (1,)))  # random index, select a sample from batch
                error = self.train_once(input=x_train_batch[ridx], target=y_train_batch[ridx], lr=lr)
                print('epoch ', xx+1, 'error= ', error)
                E.append(error)
        return E


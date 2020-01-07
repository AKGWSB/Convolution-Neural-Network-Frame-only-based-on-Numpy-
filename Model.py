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

        print('error: ', error)   # one hot label

        return error

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


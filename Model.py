import numpy as np
import Loss

class Model:

    def __init__(self, input_layer=None, output_layer=None, loss=None):
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.loss = loss

    def train_once(self, input, target, lr):
        self.input_layer.FP(x=input)
        output = self.output_layer.output

        error = self.loss.get_error(output=output, target=target)
        gradient = self.loss.get_gradient()

        self.output_layer.BP(gradient=gradient, lr=lr)

        print('error: ', error, output[..., 0])   # one hot label

        return error
import numpy as np

'''
Loss.py
every loss function has two stationary function:
get_error   : get the error between model's output and train target
get_gradient: get the model's parameters' gradient from loss function
'''

class Mean_squared_error:

    def __init__(self):
        pass

    def get_error(self, output, target):
        self.output = output
        self.target = target
        return 0.5 * np.sum((output - target) * (output - target))

    def get_gradient(self):
        return (self.output - self.target)

class Cross_entropy:

    def __init__(self):
        pass

    def get_error(self, output, target):
        self.output = output
        self.target = target
        # print(output)
        self.error = -1*np.sum(self.target*np.log(self.output))
        return self.error

    def get_gradient(self):
        return (-1*self.target/self.output)

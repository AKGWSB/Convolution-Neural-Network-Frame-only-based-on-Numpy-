import numpy as np

'''
Author: Ruo Long Lee, Collage of Computer Science, Shen Zhen University
      : 李若龙 深大计软
'''

'''
工地英语警告↓
Loss.py
every loss function has two stationary function:
get_error    : get the error between model's output and train target
get_gradient : get the model's parameters' gradient from loss function

每个损失函数必须有两个固定名字的方法：
get_error    ：正向传播一次，由模型的输出，得到一个误差
get_gradient ：得到反向传播需要的梯度，即损失函数对最后输出层输出的梯度
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
        self.error = -1 * np.sum(self.target * np.log(self.output))

        # print('output=', self.output[..., -1], 'target=', self.target[..., -1])

        return self.error

    def get_gradient(self):
        return -1 * self.target / self.output

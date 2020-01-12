# Convolution-Neural-Network-Frame-only-based-on-Numpy-

'''
Author: Ruo Long Lee, Collage of Computer Science, Shen Zhen University
      : 李若龙 深大计软
'''

Convolution Neural Network Frame only based on Numpy 

纯Numpy写的卷积神经网络框架, 非常垃圾, 没有优化, bug多

Layers.py contains some class of Layer like Convolution2D or Dense

Layers.py包含一些层的类

Loss.py define some kind of loss function

Loss.py 定义了一些损失函数

Model.py contains a class of model, which need an input_layer, an output_layer, and a loss function to confige it 

Model.py 包含模型类，需要用一个输入层的对象，一个输出层的对象，以及一个损失函数对象去实例化model类

test.py is a test file for debug

test.py 用来改bug的测试文件

train.py is a demo, using some pictures i captured to train my model

train.py 是一个演示，使用自己拍摄的数据集训练模型

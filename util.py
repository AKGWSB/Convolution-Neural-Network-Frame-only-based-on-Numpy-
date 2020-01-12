import numpy as np

'''
Author: Ruo Long Lee, Collage of Computer Science, Shen Zhen University
      : 李若龙 深大计软
'''

'''
工地英语警告↓
util contains some useful functions like convert to one hot encoding
or a class of image data generator

帮助函数：转换为独热编码
图像增强的类
'''

def int_to_one_hot(x, n_classes):
    return np.expand_dims(np.eye(n_classes)[x], -1)

class Image_generator:

    '''
    1 image generator to 16
    randomly flip, brightness change
    '''

    def __init__(self):
        pass

    # param input : a single image
    # param label : a single target label
    # batch_size  : the number of output image
    # param input : 单个输入图片
    # param label : 该图片对应的目标值
    # batch_size  : 输出图片的数量
    def one_input_flow_batch(self,
                       input,
                       label,
                       batch_size=16,
                       is_flip_X=True,
                       is_flip_Y=True,
                       is_darker=True,
                       is_brighter=True
                       ):
        out_shape = [batch_size]
        label_shape = [batch_size]
        for x in input.shape:
            out_shape.append(x)
        out_shape = tuple(out_shape)

        for x in label.shape:
            label_shape.append(x)
        label_shape = tuple(label_shape)

        result = np.zeros(shape=out_shape)
        label_ = np.zeros(shape=label_shape)
        label_ += label

        # for test
        # fxcnt = 0
        # fycnt = 0
        # dcnt = 0
        # bcnt = 0
        for i in range(batch_size):
            temp = input.copy()

            r = np.random.randint(0, 2)
            if r == 1 and is_flip_X==True:
                temp = np.flip(temp, axis=0)
                # fxcnt+=1

            r = np.random.randint(0, 2)
            if r == 1 and is_flip_Y == True:
                temp = np.flip(temp, axis=1)
                # fycnt+=1

            r = np.random.randint(0, 2)
            if r == 1 and is_darker == True:
                temp *= 0.8
                # dcnt+=1

            r = np.random.randint(0, 2)
            if r == 1 and is_brighter == True:
                temp *= 1.2
                temp = np.minimum(temp, 255)
                # bcnt+=1

            result[i] = temp

        # print(fxcnt, fycnt, dcnt, bcnt)
        return result, label_

    def multi_input_flow_batch(self,
                               input_batch,
                               is_flip_X=True,
                               is_flip_Y=True,
                               is_darker=True,
                               is_brighter=True
                               ):
        pass
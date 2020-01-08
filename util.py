import numpy as np

class Image_generator:

    def __init__(self):
        pass

    def one_input_flow_batch(self,
                       input,
                       batch_size=16,
                       is_flip_X=True,
                       is_flip_Y=True,
                       is_darker=True,
                       is_brighter=True
                       ):
        out_shape = [batch_size]
        for x in input.shape:
            out_shape.append(x)
        out_shape = tuple(out_shape)
        # print(out_shape)
        result = np.zeros(shape=out_shape)

        fxcnt = 0
        fycnt = 0
        dcnt = 0
        bcnt = 0
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
                # bcnt+=1

            result[i] = temp

        # print(fxcnt, fycnt, dcnt, bcnt)
        return result


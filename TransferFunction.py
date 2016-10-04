import math

'''
The sigmoud transfer function
          1
S(t) = -------
        1+e^t
'''
def Sigmoid(input):
    return 1 / (1 + math.e ** -input)

'''
The derivative of the Sigmoid function
'''
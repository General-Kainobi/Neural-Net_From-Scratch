import numpy as np
import matplotlib as mpt
import random
import time



class network(object):
    def __init__(self, sizes):#sizes is a list containing the number of nuerons per layer
        self.nlayers=len(sizes)
        self.sizes = sizes
        #inizializzo randomicamente sia i bias che i pesi per neuron
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]] # skip first layer(input)
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, input):
        #Calculates the output of the network
        next=input
        for bi, we in (self.biases, self.weights):
            next = sigmoid(np.dot(we,next)+bi) # maps the weighted passforward sum into [0,1]
        return next



#sigmoid transformation for R-->[0,1]    
def sigmoid(out):
    return 1.0/(1.0+np.exp(-out))

#per la backpropagation
def deriv_sigmoid(out):
    sigmoid(out)*(1-sigmoid(out))
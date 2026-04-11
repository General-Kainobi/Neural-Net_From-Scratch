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
    
    def SGD(self, training_data, epoch, mini_batch_size, eta, test_data=None):
        """Train the neural net using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` given by the mnist_loader representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""

        #personal comment for future ref: The minibatch approach is mainly just for computational efficiency
        #basically sending each image or image batch to a different gpu or different cpu core-->Not very efficient given computational 
        #power for matrix mults and vector operations is much better GPU vs CPU
        #interesting read on https://d2l.ai/chapter_optimization/minibatch-sgd.html

        if test_data: ntest=len(test_data)
        n=len(training_data)
        for i in range(epoch):#iterate through epochs of training (different predic-actual passes)
            time_start=time.time()
            random.shuffle(training_data)#good practice from what i've seen
            mini_batches= [
                training_data[m:m+mini_batch_size]
                for m in range(0,n,mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch,eta)
            time_end=time.time()
            if test_data:
                print("Epoch {0}: {1} / {2}, took {3:.2f} seconds".format(i, self.evaluate(test_data), ntest, time_end-time_start))
            else:
                print("Epoch {0} complete in {1:.2f} seconds".format(i, time_end-time_start))


    
    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y) #updating weights and biases for the mini batch produced
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)] 
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]




#sigmoid transformation for R-->[0,1]    
def sigmoid(out):
    return 1.0/(1.0+np.exp(-out))

#per la backpropagation
def deriv_sigmoid(out):
    sigmoid(out)*(1-sigmoid(out))
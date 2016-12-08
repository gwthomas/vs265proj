import lasagne
import numpy as np

import theano_stuff, util
from visual_system import VisualSystem

if __name__ == '__main__':
    Xtrain, Ytrain, Xtest, Ytest = util.load_cifar('cifar-10-batches-py', reshape=False)#True)
    Ytrain = util.one_hot(Ytrain, 10)

    k = 3 # Number of recurrences
    batchsize = 100
    net = VisualSystem(k, batchsize)
    net.compile()
    net.opt_setup(theano_stuff.cross_entropy_error, lasagne.updates.adam)
    i = 0
    while True:
        net.train(Xtrain, Ytrain, itrs=50, batchsize=batchsize)
        # For technical reasons we can only evaluate exactly batchsize samples at a time with this network
        batchidx = np.random.permutation(len(Xtest))[:batchsize]
        print('Accuracy:', net.evaluate(Xtest[batchidx], Ytest[batchidx], lambda y: np.argmax(y, axis=1)))
        print(i)
        i += 1

import lasagne
import numpy as np

import theano_stuff, util
from convnet import ConvolutionalNetwork
from mlp import MultilayerPerceptron

if __name__ == '__main__':
    Xtrain, Ytrain, Xtest, Ytest = util.load_cifar('cifar-10-batches-py', reshape=True)
    Ytrain = util.one_hot(Ytrain, 10)

    net = ConvolutionalNetwork(
            input_shape=(3,32,32),
            num_out=10,
            filters=[[100,3], [50,3]],
            poolings=[2,2],
            hidden_sizes=[200,100],
            output_nl=lasagne.nonlinearities.softmax
    )
    net.opt_setup(theano_stuff.cross_entropy_error, lasagne.updates.adam)

    while True:
        net.train(Xtrain, Ytrain, itrs=50, batchsize=100)
        print('Accuracy:', net.evaluate(Xtest, Ytest, lambda y: np.argmax(y, axis=1)))

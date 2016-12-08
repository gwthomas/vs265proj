import numpy as np
import theano
import theano.tensor as T
import theano.tensor.nnet as nn

from theano_stuff import TheanoFunction

def init_weights(*shape):
    return theano.shared((np.random.randn(*shape) * 0.01).astype(theano.config.floatX))

def mlp_forward(input, sizes, weights=None, nonlinearity=nn.relu):
    output = input.T
    ret_weights = weights if weights else []
    for i in range(1,len(sizes)):
        if weights:
            W, b = weights[i-1]
        else:
            W = init_weights(sizes[i], sizes[i-1])
            b = init_weights(sizes[i])
        output = nonlinearity(T.dot(W, output) + b.dimshuffle(0, 'x'))
        ret_weights.append((W, b))
    return output.T, ret_weights

def combine(*inputs):
    return T.concatenate(inputs, axis=1)

def flatten_tuplist(tuples):
    ret = []
    for tuple in tuples:
        ret.extend(tuple)
    return ret

class VisualSystem(TheanoFunction):
    def __init__(self, k, batchsize):
        self._input_var = T.fmatrix('input')

        n_retina_out = 3*32*32 # @retina team: change this to the dimension of whatever your part outputs
        n_lgn_out = 100
        n_v1_out = 100
        n_v2_out = 100
        n_final_out = 10

        lgn_sizes = [n_retina_out+n_lgn_out, 100, n_lgn_out]
        v1_sizes = [n_lgn_out+n_v2_out, 100, n_v1_out]
        v2_sizes = [n_v1_out, 100, n_v2_out]

        lgn_in = combine(self._input_var, T.zeros((batchsize, n_v1_out)))
        lgn_out, lgn_weights = mlp_forward(lgn_in, lgn_sizes)
        v1_in = combine(lgn_out, T.zeros((batchsize, n_v2_out)))
        v1_out, v1_weights = mlp_forward(v1_in, v1_sizes)
        v2_in = v1_out
        v2_out, v2_weights = mlp_forward(v2_in, v2_sizes)

        for i in range(k):
            lgn_in = combine(self._input_var, v1_out)
            lgn_out, _ = mlp_forward(lgn_in, lgn_sizes, lgn_weights)
            v1_in = combine(lgn_out, v2_out)
            v1_out, _ = mlp_forward(v1_in, v1_sizes, v1_weights)
            v2_in = v1_out
            v2_out, _ = mlp_forward(v2_in, v2_sizes, v2_weights)

        final_out, final_weights = mlp_forward(v2_out, [n_v2_out, n_final_out], nonlinearity=nn.softmax)
        self._output_var = final_out
        self._param_vars = flatten_tuplist(lgn_weights + v1_weights + v2_weights + final_weights)

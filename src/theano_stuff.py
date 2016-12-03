import lasagne
import numpy as np
import theano
import theano.tensor as T

def mean_squared_error(yhat, y):
    return T.mean((yhat - y)**2)

def cross_entropy_error(yhat, y):
    return T.mean(lasagne.objectives.categorical_crossentropy(yhat, y))

class TheanoFunction(object):
    # Override this. Init should set up member variables _input_var, _output_var,
    # and _param_vars, then call super init
    def __init__(self):
        self._fn = theano.function(
            inputs=[self._input_var],
            outputs=self._output_var,
            allow_input_downcast=True
        )

    def opt_setup(self, loss_fn, update_fn):
        target_var = T.fmatrix('target')
        loss_var = loss_fn(self._output_var, target_var)
        update_info = update_fn(loss_var, self._param_vars)
        self._update = theano.function(
                inputs=[self._input_var, target_var], outputs=loss_var,
                updates=update_info,
                allow_input_downcast=True
        )

    def get_input_var(self):
        return self._input_var

    def get_output_var(self):
        return self._output_var

    def get_param_vars(self):
        return self._param_vars

    def get_params(self):
        return [np.array(param_var.eval()) for param_var in self.get_param_vars()]

    def set_params(self, params):
        assert len(params) == len(self._param_vars)
        for param_var, new_value in zip(self.get_param_vars(), params):
            param_var.set_value(new_value)

    def save_params(self, filename):
        with open(filename, 'wb') as f:
            np.savez(f, *self.get_params())

    def load_params(self, filename):
        with np.load(filename) as data:
            self.set_params([data['arr_'+str(i)] for i in range(len(data.files))])

    def __call__(self, *args):
        return self._fn(*args)

    def train(self, Xtrain, Ytrain, itrs=100, batchsize=100, filename='params.npz'):
        n = len(Xtrain)
        for itr in range(itrs):
            batchidx = np.random.permutation(n)[:batchsize]
            loss = self._update(Xtrain[batchidx], Ytrain[batchidx])
            print('Itr %i: loss = %f' % (itr, loss))
            self.save_params(filename)

    def evaluate(self, Xtest, Ytest, predict):
        output = self._fn(Xtest)
        predictions = predict(output)
        return np.mean(predictions == Ytest)

from theano_stuff import TheanoFunction
import theano.tensor as T
import lasagne
import lasagne.layers as L
import lasagne.nonlinearities as NL

class MultilayerPerceptron(TheanoFunction):
    def __init__(self, sizes,
            input=None,
            nl=NL.rectify,
            output_nl=None  # Change to softmax to get probabilities
    ):
        self._input_var, l_prev = None, None
        if input is True:
            self._input_var = T.fmatrix('input')
        elif isinstance(input, theano.tensor.var.TensorVariable):
            self._input_var = input
        elif isinstance(input, L.Layer):
            l_prev = input
        else:
            print('Invalid input:', input)

        if not l_prev:
            l_prev = L.InputLayer(
                shape=(None, sizes[0]),
                input_var=self._input_var
            )

        for size in sizes[1:-1]:
            l_prev = L.DenseLayer(l_prev,
                num_units=size,
                nonlinearity=nl
            )

        l_output = L.DenseLayer(l_prev,
            num_units=sizes[-1],
            nonlinearity=output_nl,
            name="output"
        )

        self._output_layer = l_output
        self._output_var = L.get_output(l_output)
        self._param_vars = L.get_all_params(l_output, trainable=True)

    def forward(self, input_var):
        return L.get_output(self._output_var, inputs=input_var)

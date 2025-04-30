import numpy as np
from activations import *
from random import shuffle

# represent a layer of the model
class Layer:
    def __init__(self, input_size, output_size, alpha, batch_size, bias=False, activation_func=None):
        self.input_size = input_size
        self.output_size = output_size
        self.alpha = alpha
        self.batch_size = batch_size
        self.activation_func = activation_func
        self.bias = bias


        # Create full connection: each input is connected to each output neuron
        # Hence, weights shape = (input_size, output_size)
        # This is a dense layer: every output neuron uses all input features
        self.weights = np.random.rand(self.input_size, self.output_size)

        if bias:
            bias_row = np.random.rand(1, self.output_size)
            self.weights = np.vstack((self.weights, bias_row))  # Add bias weights: extra row for bias terms

        self.update_matrix = None
        self.current_inputs = None
        self.pre_activated_outputs = None

    def get_batch_size(self):
        return self.batch_size
    
    def set_alpha(self, new_alpha):
        self.alpha = new_alpha

    def __call__(self, Layer_inputs):
        """
        layer_inputs -> (batch_size, input_size+1)
        self.weights -> (input_size+1, output_size)  +1 for bias
        """
        if self.bias:
            bias_column = np.ones((self.batch_size, 1))
            Layer_inputs = np.hstack((Layer_inputs, bias_column))  # Add bias inputs: extra column of 1s for bias application

        self.current_inputs = np.copy(Layer_inputs)
        layer_output = Layer_inputs @ self.weights

        if self.activation_func is not None:
            self.pre_activated_output = np.copy(layer_output)
            layer_output = self.activation_func(layer_output)

        return layer_output
    
    def back(self, ret):
        if self.activation_func is not None:
            if isinstance(self.activation_func, Softmax):
                ret = self.activation_func.derivative(self.pre_activated_output, ret)
            else:
                ret = self.activation_func.derivative(self.pre_activated_output, ret)

        self.update_matrix = self.current_inputs.T @ ret
        new_ret = ret @ self.weights.T

        if self.bias:
            new_ret = new_ret[:, :-1]

        return new_ret

    def update(self):
        self.weights -= self.alpha * self.update_matrix
        self.update_matrix = None
        self.current_inputs = None
        self.pre_activated_outputs = None

# represent a model 
# LayerList: Container class for stacking custom Layer objects
# - Acts like a neural network: accepts inputs and applies each layer sequentially
# - Supports dynamic layer addition using append()
# - Allows global learning rate updates with set_alpha()
# - Forward pass is handled by __call__(), passing data through all layers
class LayerList:
    def __init__(self, *Layers):
        self.model = list(Layers)

    def append(self, *Layers):
        for layer in Layers:
            self.model.append(layer)

    def set_alpha(self, new_alpha):
        for layer in self.model:
            layer.set_alpha(new_alpha)

    def __call__(self, model_input):
        intermediate_results = model_input
        for layer in self.model:
            intermediate_results = layer(intermediate_results)

        return intermediate_results
    
    def back(self, error):
        for layer in self.model[::-1]:
            error = layer.back(error)

    def step(self):
        for layer in self.model:
            layer.update()
    

    @staticmethod
    def batch(input_data, expected, batch_size):
        num_data = input_data.shape[0]
        indices = [i for i in range(num_data)]
        shuffle(indices)

        batched_inputs, batched_expected = [], []
        
        for i in range(num_data // batch_size):
            batch_inp, batch_exp = [], []
            
            for j in range(batch_size):
                batch_inp.append(input_data[indices[batch_size*i+j]])
                batch_exp.append(expected[indices[batch_size*i+j]])
            batched_inputs.append(np.array(batch_inp))
            batched_expected.append(np.array(batch_exp))

        return np.array(batched_inputs), np.array(batched_expected) 

    def fit(self, input_data, expected, epochs, alpha, batch_size, loss_deriv_funct):
        self.set_alpha(alpha)
        
        prev_update = 1
        for e in range(epochs):
            batched_input, batched_expected = LayerList.batch(input_data, expected, batch_size)
            
            for i in range(len(batched_input)):
                model_output = self(batched_input[i])
                self.back(loss_deriv_funct(model_output, batched_expected[i]))
                self.step()

            if e == 10 * prev_update:
                alpha /= 10
                self.set_alpha(alpha)
                prev_update = e
    
    def predict(self,inputs):
        predictions = []

        for inp in inputs:
            # inp shape -> (1, num_features)
            predictions.append(self(np.expand_dims(inp, axis=0)))

        return predictions

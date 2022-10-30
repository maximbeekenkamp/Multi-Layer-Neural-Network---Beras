import numpy as np

from .core import Diffable


class LeakyReLU(Diffable):
    def __init__(self, alpha=0.3):
        super().__init__()
        self.alpha = alpha
        self.inputs = None
        self.outputs = None

    def forward(self, inputs):
        # Given an input array `x`, computes LeakyReLU(x)
        self.inputs = inputs
        self.outputs = np.where(self.inputs < 0, (self.inputs * self.alpha), self.inputs)
        return self.outputs

    def input_gradients(self):
        # Computes and returns the gradients
        grad = np.where(self.inputs < 0, self.alpha, 1)
        return grad

    def compose_to_input(self, J):
        return self.input_gradients() * J


class ReLU(LeakyReLU):
    def __init__(self):
        super().__init__(alpha=0)


class Softmax(Diffable):
    def __init__(self):
        super().__init__()
        self.inputs = None
        self.outputs = None

    def forward(self, inputs):
        """Softmax forward pass!"""
        self.inputs = inputs
        x = inputs - np.max(inputs, axis = -1, keepdims=True)
        self.outputs = (np.exp(x)) / np.sum(np.exp(x), axis=-1, keepdims=True)
        return self.outputs

    def input_gradients(self):
        """Softmax backprop!"""
        inputs_shape_0 = self.inputs.shape[0]
        inputs_shape_1 = self.inputs.shape[1]
        grads = np.zeros((inputs_shape_0, inputs_shape_1, inputs_shape_1))
        diag_vec = np.arange(inputs_shape_1)
        for b in range(inputs_shape_0):
            diag_mat = grads[b]
            np.fill_diagonal(diag_mat, diag_vec)
            grads[b] = diag_mat - np.dot(self.outputs[b], np.transpose(self.outputs[b]))
        return grads

class Sigmoid(Diffable):

    def __init__(self):
        super().__init__()
        self.inputs = None
        self.outputs = None

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = 1/(1 + np.exp(-self.inputs))
        return self.outputs

    def input_gradients(self):
        return (self.outputs) * (1 - self.outputs)
        
    def compose_to_input(self, J):
        return self.input_gradients() * J

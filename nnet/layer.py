import numpy as np


class Layer(object):
    def __init__(self, input_dim, output_dim):
        self.weights = np.random.rand(input_dim, output_dim).T

    @staticmethod
    def act(x):
        vect_x = np.array(x)
        return 1 / (1 + np.exp(-vect_x))

    def forward(self, vect_in):
        weighted_sums = np.dot(self.weights, vect_in)
        return self.act(weighted_sums)

    def backward(self, grad):
        raise NotImplementedError

    def teach(self, vect_in, expected, learn_rate=0.1):
        vect_out = self.forward(vect_in)
        vect_err = vect_out - expected
        adjust = learn_rate * np.outer(vect_err, vect_in)
        self.weights -= adjust





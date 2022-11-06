import numpy as np
from nnet.layer import Layer
from pytest import approx, fixture


def bits_nums(n, numbits):
    return np.unpackbits(np.array([n], dtype=np.uint8))[-numbits:]


@fixture
def net_input():
    inp = [bits_nums(n, 3) for n in range(8)]
    return inp


@fixture
def net_result(net_input):
    inp = [np.any(a[-2:]) for a in net_input]
    return np.array(inp, dtype=float)


def set_test_weights(l):
    l.weights = [np.array([0.1, 0.1, 0.1]), np.array([0.5, 0.5, 0.5])]


def test_training_data(net_input, net_result):
    assert len(net_input) == 8
    assert len(net_result) == 8
    for i in range(8):
        assert net_input[i][-2:].any() == net_result[i]

def test_layer(net_input, net_result):
    l = Layer(3, 2)
    set_test_weights(l)
    out = l.forward(np.array([1.0, 1.0, 1.0]))
    assert out == approx([l.act(0.3), l.act(1.5)])

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
    l.weights[0] = 0.1
    l.weights[1] = 0.5


def test_training_data(net_input, net_result):
    assert len(net_input) == 8
    assert len(net_result) == 8
    for i in range(8):
        assert net_input[i][-2:].any() == net_result[i]

def test_layer(net_input, net_result):
    l2 = Layer(3, 2)
    set_test_weights(l2)
    out = l2.forward(np.array([1.0, 1.0, 1.0]))
    assert out == approx([l2.act(0.4), l2.act(2.0)])


def test_2_1_layer_training():
    inp_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    out_data = np.array([0, 0, 0, 1])

    l = Layer(2, 1)
    for i in range(5000):
        for j in range(4):
            l.teach(inp_data[j], out_data[j])
    out1 = l.forward(np.array([0.0, 0.0]))
    out2 = l.forward(np.array([0.0, 1.0]))
    out3 = l.forward(np.array([1.0, 0.0]))
    out4 = l.forward(np.array([1.0, 1.0]))
    assert out4 == approx(1.0, abs=0.05)


def test_layer_training(net_input, net_result):
    l = Layer(3, 1)
    for i in range(4000):
        for j in range(8):
            l.teach(net_input[j], net_result[j])
    out = l.forward(np.array([1.0, 1.0, 1.0]))
    assert out == approx(1.0, abs=0.05)

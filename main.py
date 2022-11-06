import numpy as np

def bits_nums(n, numbits):
    return np.unpackbits(np.array([n], dtype=np.uint8))[-numbits:]



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    inp = [format(n, "03b") for n in range(8)]
    print(inp)
    x = np.array([1, 2, 3])

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

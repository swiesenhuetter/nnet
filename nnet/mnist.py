import numpy as np
from PIL import Image
import sys


def load_data():
    data_set = np.loadtxt("..\\.data\\mnist_test.csv", skiprows=1, dtype=np.uint8, delimiter=',')
    return data_set


def get_image(data_set, index):
    img_data = data_set[index, 1:].reshape((28, 28))
    label = data_set[index, 0]
    img = Image.fromarray(img_data, 'L')
    return label, img


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python mnist.py <index>")
        sys.exit(1)

    index = int(sys.argv[1])

    dataset = load_data()
    label, img = get_image(dataset, index)
    print(f"Label: {label}")
    img.show()

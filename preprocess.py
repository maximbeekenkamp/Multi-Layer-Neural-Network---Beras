import gzip
import pickle

import numpy as np


def get_data_MNIST(subset, data_path="../data"):
    """
    :param subset: string indicating whether we want the training or testing data
        (only accepted values are 'train' and 'test')
    :param data_path: directory containing the training and testing inputs and labels
    :return: NumPy array of inputs (float32) and labels (uint8)
    """
    subset = subset.lower().strip()
    assert subset in ("test", "train"), f"unknown data subset {subset} requested"
    inputs_file_path, labels_file_path, num_examples = {
        "train": ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", 60000),
        "test": ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz", 10000),
    }[subset]
    inputs_file_path = f"{data_path}/mnist/{inputs_file_path}"
    labels_file_path = f"{data_path}/mnist/{labels_file_path}"

    with open(inputs_file_path, "rb") as f, gzip.GzipFile(fileobj=f) as bytstream_input:
        image_buf = bytstream_input.read(16)
        image_buf = bytstream_input.read(784 * num_examples)
        image_buf = np.frombuffer(image_buf, dtype=np.uint8)
    image = image_buf
    image = image / 255.0
    image = np.float32(np.reshape(image, (num_examples, 784)))

    with open(labels_file_path, "rb") as f, gzip.GzipFile(fileobj=f) as bytstream_label:
        label_buf = bytstream_label.read(8)
        label_buf = bytstream_label.read(num_examples)
        label_buf = np.frombuffer(label_buf, dtype=np.uint8)

    label = label_buf

    return image, label


def shuffle_data(image_full, label_full, seed):
    """
    Shuffles the full dataset with the given random seed.

    :param: the dataset before shuffling
    :return: the dataset after shuffling
    """
    rng = np.random.default_rng(seed)
    shuffled_index = rng.permutation(np.arange(len(image_full)))
    image_full = image_full[shuffled_index]
    label_full = label_full[shuffled_index]
    return image_full, label_full


def get_specific_class(image_full, label_full, specific_class=0, num=None):
    """
    The MNIST dataset includes all ten digits, but they are not sorted,
        and it does not have the same number of images for each digits.
    Also, for KNN, we only need a small subset of the dataset.
    So, we need a function that selects the images and labels for a specific digit.

    The same for the CIFAR dataset. We only need a small subset of CIFAR.

    :param image_full: the image array returned by the get_data function
    :param label_full: the label array returned by the get_data function
    :param specific_class: the specific class you want
    :param num: number of the images and labels to return
    :return image: Numpy array of inputs (float32)
    :return label: Numpy array of labels
                   (either uint8 or string, whichever type it was originally)
    """

    mask = label_full == specific_class

    image_full = image_full[mask]
    label_full = label_full[mask]

    image = image_full[:num]
    label = label_full[:num]

    return image, label


def get_subset(image_full, label_full, class_list=list(range(10)), num=100):
    """
    The MNIST dataset includes all ten digits, but they are not sorted,
        and it does not have the same number of images for each digits.
    Also, for KNN, we only need a small subset of the dataset.
    So, we need a function that selects the images and labels for a list of specific digits.

    The same for the CIFAR dataset. We only need a small subset of CIFAR.

    :param image: the image array returned by the get_data function
    :param label: the label array returned by the get_data function
    :param class_list: the list of specific classes you want
    :param num: number of the images and labels to return for each class
    :return image: Numpy array of inputs (float32)
    :return label: Numpy array of labels
                   (either uint8 or string, whichever type it was originally)
    """

    image_list = []
    label_list = []

    for specific_class in class_list:

        image_element, label_element = get_specific_class(
            image_full, label_full, specific_class, num
        )
        image_list.append(image_element)
        label_list.append(label_element)

    image = np.concatenate(image_list)
    label = np.concatenate(label_list)

    return image, label

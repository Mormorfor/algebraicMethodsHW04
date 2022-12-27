import pickle
from PIL import Image
import numpy as np
from numpy.linalg import linalg

def getGrayScale(data):
    vec_list = []
    for img in data:
        img_as_mat = np.transpose(np.reshape(img, (3, 32, 32)), (1, 2, 0))
        image = Image.fromarray(img_as_mat)
        image = image.convert("L")
        gray_array = np.array(image)
        gray_array = np.reshape(gray_array, (1, -1))
        vec_list.append(gray_array)
    X = np.asarray(vec_list)
    return X[:, 0, :]

def find_k_neighbors(distances, labels, ks):
    sorted_dist = np.argsort(distances)
    sorted_dist = list(sorted_dist)
    k_labels_dict = dict()
    for k in ks:
        k_closest = sorted_dist[:k]
        k_labels = []
        for point in k_closest:
            k_labels.append(labels[point])
        new_label = max(set(k_labels), key=k_labels.count)
        k_labels_dict[k] = new_label
    return k_labels_dict


def find_errors(predictions, ks, ss, test_l):
    errors_in_run = dict()
    for k in ks:
        for s in ss:
            errors = 0
            for i in range(len(test_l)):
                if predictions[s, k, i] != test_l[i]:
                    errors += 1
            errors_in_run[s, k] = errors/len(test_l)
    return errors_in_run


def test_data():
    path = "./cifar-10-python/cifar-10-batches-py/test_batch"
    batch = unpickle(path)
    data = batch[b'data']
    labels = batch[b'labels']
    data = getGrayScale(data)
    return data.T, labels


def train_data():
    path1 = "./cifar-10-python/cifar-10-batches-py/data_batch_1"
    batch1 = unpickle(path1)
    all_data = batch1[b'data']
    all_labels = batch1[b'labels']

    path2 = "./cifar-10-python/cifar-10-batches-py/data_batch_2"
    batch2 = unpickle(path2)
    data2 = batch2[b'data']
    labels2 = batch2[b'labels']

    path3 = "./cifar-10-python/cifar-10-batches-py/data_batch_3"
    batch3 = unpickle(path3)
    data3 = batch3[b'data']
    labels3 = batch3[b'labels']

    path4 = "./cifar-10-python/cifar-10-batches-py/data_batch_4"
    batch4 = unpickle(path4)
    data4 = batch4[b'data']
    labels4 = batch4[b'labels']

    path5 = "./cifar-10-python/cifar-10-batches-py/data_batch_5"
    batch5 = unpickle(path5)
    data5 = batch5[b'data']
    labels5 = batch5[b'labels']


    all_data = np.concatenate((all_data, data2), axis=0)
    all_data = np.concatenate((all_data, data3), axis=0)
    all_data = np.concatenate((all_data, data4), axis=0)
    all_data = np.concatenate((all_data, data5), axis=0)
    all_labels = all_labels + labels2
    all_labels = all_labels + labels3
    all_labels = all_labels + labels4
    all_labels = all_labels + labels5

    all_data = getGrayScale(all_data)
    all_data = norm(all_data)
    return all_data.T, all_labels

def norm(data):
    mean = np.mean(data, axis=1)
    normalized = np.apply_along_axis(lambda x: x-mean, 0, data)
    return normalized

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict



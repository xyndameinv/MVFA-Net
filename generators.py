import os
import copy
from random import shuffle

import numpy as np
from keras.utils import np_utils

import pickle
from augment import augment_data

def pickle_dump(item, out_file):
    with open(out_file, "wb") as opened_file:
        pickle.dump(item, opened_file)
def pickle_load(in_file):
    with open(in_file, "rb") as opened_file:
        return pickle.load(opened_file)


def get_training_and_validation_generators(data_file, batch_size, training_keys_file, validation_keys_file,
                                           data_split=0.9, overwrite=False, labels=None, augment=False,n_labels=1,
                                           augment_flip=True, augment_distortion_factor=0.25,validation_batch_size=None):
    """
    Creates the training and validation generators that can be used when training the model.
    :param validation_batch_size: Batch size for the validation data.
    :param augment_flip: if True and augment is True, then the data will be randomly flipped along the x, y and z axis
    :param augment_distortion_factor: if augment is True, this determines the standard deviation from the original
    that the data will be distorted (in a stretching or shrinking fashion). Set to None, False, or 0 to prevent the
    augmentation from distorting the data in this way.
    :param augment: If True, training data will be distorted on the fly so as to avoid over-fitting.
    :param labels: List or tuple containing the ordered label values in the image files. The length of the list or tuple
    should be equal to the n_labels value.
    Example: (10, 25, 50)
    The data generator would then return binary truth arrays representing the labels 10, 25, and 30 in that order.
    :param data_file: hdf5 file to load the data from.
    :param batch_size: Size of the batches that the training generator will provide.
    :param n_labels: Number of binary labels.
    :param training_keys_file: Pickle file where the index locations of the training data will be stored.  将要存储训练数据的索引位置
    :param validation_keys_file: Pickle file where the index locations of the validation data will be stored. 将存储验证数据的索引位置
    :param data_split: How the training and validation data will be split. 0 means all the data will be used for
    validation and none of it will be used for training. 1 means that all the data will be used for training and none
    will be used for validation. Default is 0.8 or 80%.
    :param overwrite: If set to True, previous files will be overwritten. The default mode is false, so that the
    training and validation splits won't be overwritten when rerunning model training.
    :return: Training data generator, validation data generator, number of training steps, number of validation steps
    """
    if not validation_batch_size:
        validation_batch_size = batch_size

    training_list, validation_list = get_validation_split(data_file,
                                                          data_split=data_split,
                                                          overwrite=overwrite,
                                                          training_file=training_keys_file,
                                                          validation_file=validation_keys_file)  #划分验证集和训练集 得到索引列表

    training_generator = data_generator(data_file, training_list,
                                        batch_size=batch_size,
                                        n_labels=n_labels,
                                        augment=augment,
                                        augment_flip=augment_flip,
                                        augment_distortion_factor=augment_distortion_factor)   #没看见返回值
    validation_generator = data_generator(data_file, validation_list,
                                          batch_size=validation_batch_size,
                                          n_labels=n_labels)

    # Set the number of training and testing samples per epoch correctly
    num_training_steps = get_number_of_steps(len(training_list), batch_size)
    print("Number of training patch: ", len(training_list))
    print("Number of training steps: ", num_training_steps)

    num_validation_steps = get_number_of_steps(len(validation_list),validation_batch_size)
    print("Number of validation patch: ", len(validation_list))
    print("Number of validation steps: ", num_validation_steps)

    return training_generator, validation_generator, num_training_steps, num_validation_steps


def get_number_of_steps(n_samples, batch_size):
    if n_samples <= batch_size:
        return n_samples
    elif np.remainder(n_samples, batch_size) == 0:
        return n_samples//batch_size
    else:
        return n_samples//batch_size + 1


def get_validation_split(data_file, training_file, validation_file, data_split=0.9, overwrite=False):
    """
    Splits the data into the training and validation indices list. 索引列表
    :param data_file: pytables hdf5 data file
    :param training_file:
    :param validation_file:
    :param data_split:
    :param overwrite:
    :return:
    """
    if overwrite or not os.path.exists(training_file):
        print("Creating validation split...")
        nb_samples = data_file.root.data.shape[0]
        sample_list = list(range(nb_samples))
        training_list, validation_list = split_list(sample_list, split=data_split)
        pickle_dump(training_list, training_file)
        pickle_dump(validation_list, validation_file)
        return training_list, validation_list
    else:
        print("Loading previous validation split...")
        return pickle_load(training_file), pickle_load(validation_file)


def split_list(input_list, split=0.9, shuffle_list=True):
    if shuffle_list:
        shuffle(input_list)
    n_training = int(len(input_list) * split)

    training = input_list[:n_training]
    testing = input_list[n_training:]
    return training, testing

def data_generator(data_file, index_list, batch_size=1, n_labels=1,augment=False, augment_flip=True,
                   augment_distortion_factor=0.25,shuffle_index_list=True):
    orig_index_list = index_list
    while True:
        x_list = list()
        y_list = list()
        index_list = copy.copy(orig_index_list)

        if shuffle_index_list:
            shuffle(index_list) #打乱顺序
        print("bacth size",batch_size)
        while len(index_list) > 0:
            index = index_list.pop()
            data, truth = data_file.root.data[index,...], data_file.root.truth[index,...]
            x_list.append(data)
            y_list.append(truth)
            add_data(x_list, y_list, data_file, index, augment=augment, augment_flip=augment_flip,
                     augment_distortion_factor=augment_distortion_factor)
            if len(x_list) == batch_size or (len(index_list) == 0 and len(x_list) > 0):
                x,y = convert_data(x_list, y_list, n_labels=n_labels)
                x = np.squeeze(x)[:,np.newaxis, ...]
                y = np.squeeze(y)[:,np.newaxis, ...]
                yield x,y
                x_list = list()
                y_list = list()

def add_data(x_list, y_list, data_file, index, augment=False, augment_flip=False, augment_distortion_factor=0.25):
    """
    Adds data from the data file to the given lists of feature and target data
    :param x_list: list of data to which data from the data_file will be appended.
    :param y_list: list of data to which the target data from the data_file will be appended.
    :param data_file: hdf5 data file.
    :param index: index of the data file from which to extract the data.
    :param augment: if True, data will be augmented according to the other augmentation parameters (augment_flip and
    augment_distortion_factor)
    :param augment_flip: if True and augment is True, then the data will be randomly flipped along the x, y and z axis
    :param augment_distortion_factor: if augment is True, this determines the standard deviation from the original
    that the data will be distorted (in a stretching or shrinking fashion). Set to None, False, or 0 to prevent the
    augmentation from distorting the data in this way.
    :return:
    """
    data, truth = data_file.root.data[index,...], data_file.root.truth[index,...]
    if augment:
        affine = data_file.root.affine[index]
        data, truth = augment_data(data, truth, affine, flip=augment_flip, scale_deviation=augment_distortion_factor)
    x_list.append(data)
    y_list.append(truth)

def convert_data(x_list, y_list, n_labels=1):
    x = np.asarray(x_list)
    y = np.asarray(y_list)
    print(x.shape)
    print(y.shape)
    if n_labels > 1:
        y = np_utils.to_categorical(y[...,0],n_labels)
#        y = get_multi_class_labels(y, n_labels=n_labels, labels=labels)
    return x, y

#def get_multi_class_labels(data, n_labels, labels=None):
#    """
#    Translates a label map into a set of binary labels.
#    :param data: numpy array containing the label map with shape: (n_samples, 1, ...).
#    :param n_labels: number of labels.
#    :param labels: integer values of the labels.
#    :return: binary numpy array of shape: (n_samples, n_labels, ...)
#    """
#    new_shape = [data.shape[0], n_labels] + list(data.shape[2:])
#    y = np.zeros(new_shape, np.int8)
#    for label_index in range(n_labels):
#        if labels is not None:
#            y[:, label_index][data[:, 0] == labels[label_index]] = 1
#        else:
#            y[:, label_index][data[:, 0] == (label_index + 1)] = 1
#    return y
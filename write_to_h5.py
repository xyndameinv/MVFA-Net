# -*- coding: utf-8 -*-
"""
Created on Mon May  6 12:47:25 2019

@author: Tan
"""
import tables
import os
import numpy as np
import glob
import nibabel as nib
#os.environ["HDF5_DISABLE_VERSION_CHECK"] = '1'
def create_data_file(out_file,n_samples, image_shape):
    hdf5_file = tables.open_file(out_file, mode='w')
    filters = tables.Filters(complevel=5, complib='blosc')
    data_shape = tuple([0]+ list(image_shape)+ [1])
    truth_shape = tuple([0] + list(image_shape)+[1])
    data_storage = hdf5_file.create_earray(hdf5_file.root, 'data', tables.Float32Atom(),shape=data_shape,
                                           filters=filters, expectedrows=n_samples)
    truth_storage = hdf5_file.create_earray(hdf5_file.root, 'truth', tables.UInt8Atom(),shape=truth_shape,
                                            filters=filters, expectedrows=n_samples)
    affine_storage = hdf5_file.create_earray(hdf5_file.root, 'affine', tables.Float32Atom(), shape=(0, 4, 4),
                                             filters=filters, expectedrows=n_samples)
    return hdf5_file, data_storage, truth_storage, affine_storage

def fetch_data_files(my_data_dir):
    training_data_files = list()
    image_dirs = glob.glob(os.path.join(my_data_dir,'img_128_8',"*"))
    mask_dirs = glob.glob(os.path.join(my_data_dir,'label_128_8',"*"))
    for dir in range(0,len(image_dirs)):
        subject_files = list()
        subject_files.extend([image_dirs[dir],mask_dirs[dir]])
        # print(subject_files)
        training_data_files.append(tuple(subject_files))
    return training_data_files

def read_image_files(image_files,image_shape=None):
    image_list = list()
    for index, image_file in enumerate(image_files):
        image_list.append(read_image(image_file, image_shape=image_shape))
    return image_list

def read_image(in_file, image_shape=None):
    # print("Reading: {0}".format(in_file))
    image = nib.load(os.path.abspath(in_file))
    return image

def add_data_to_storage(data_storage, truth_storage, affine_storage,subject_data, affine,truth_dtype):
    data = np.expand_dims(np.asarray(subject_data[0])[np.newaxis],axis=-1)
    # print(data.shape)
    data_storage.append(data)
    truth =np.expand_dims(np.asarray(subject_data[1], dtype=truth_dtype)[np.newaxis],axis=-1)
    # print(truth.shape)
    truth_storage.append(truth)
    affine_storage.append(np.asarray(affine)[np.newaxis])

def write_image_data_to_file(image_files, data_storage, truth_storage, image_shape,
                             affine_storage,truth_dtype=np.uint8):
    for set_of_files in image_files:
        images = read_image_files(set_of_files, image_shape=image_shape)
        subject_data = [image.get_fdata() for image in images]
        add_data_to_storage(data_storage, truth_storage, affine_storage,subject_data,
                            images[0].affine,truth_dtype)
    return data_storage, truth_storage

def write_data_to_file(my_data_dir,out_file, image_shape,truth_dtype=np.uint8, subject_ids=None):
    """
    Takes in a set of training images and writes those images to an hdf5 file.
    :param training_data_files: List of tuples containing the training data files. The modalities should be listed in
    the same order in each tuple. The last item in each tuple must be the labeled image. 
    Example: [('sub1-T1.nii.gz', 'sub1-T2.nii.gz', 'sub1-truth.nii.gz'), 
              ('sub2-T1.nii.gz', 'sub2-T2.nii.gz', 'sub2-truth.nii.gz')]
    :param out_file: Where the hdf5 file will be written to.
    :param image_shape: Shape of the images that will be saved to the hdf5 file.
    :param truth_dtype: Default is 8-bit unsigned integer. 
    :return: Location of the hdf5 file with the image data written to it. 
    """
    training_data_files = fetch_data_files(my_data_dir) #所有的训练块的文件位置以及对应的标签位置的list
    n_samples = len(training_data_files)

    try:
        hdf5_file, data_storage, truth_storage, affine_storage = create_data_file(out_file,n_samples=n_samples,
                                                                                  image_shape=image_shape)
    except Exception as e:
        # If something goes wrong, delete the incomplete data file
        os.remove(out_file)
        raise e
    write_image_data_to_file(training_data_files, data_storage, truth_storage, image_shape,
                             affine_storage=affine_storage,truth_dtype=truth_dtype)
    if subject_ids:
        hdf5_file.create_array(hdf5_file.root, 'subject_ids', obj=subject_ids)
    hdf5_file.close()
    return out_file,n_samples
#my_data_dir= r'E:\ty\data\train_test-patch(96-32)\train_patch'
#data_files,n_samples = write_data_to_file(my_data_dir,out_file=r'E:\ty\data\train_test-patch(96-32)\train_patch.h5',
#                                          image_shape=(96,96,32),truth_dtype=np.uint8, subject_ids=None)
#
if __name__=="__main__":
    my_data_dir_test= r'G:\cervial\data\128-8-from_crop'
    data_files_test,n_samples_test = write_data_to_file(my_data_dir_test,
                                        out_file=r'G:\cervial\data\128-8-from_crop\test_patch.h5',
                                        image_shape=(128,128,8),truth_dtype=np.uint8, subject_ids=None)
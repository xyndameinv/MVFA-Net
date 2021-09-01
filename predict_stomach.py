import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import nibabel as nib
import numpy as np
import glob
import tables
from shutil import copyfile
from training import load_old_model
#from keras.models import load_model
from evaluate import main
import time

def normalized_data(data):
    means = data.mean()
    stds = data.std()
    data -= means
    data /= stds
    return data

def fix_out_of_bound_patch_attempt(data, patch_shape, patch_index, ndim=3):
    """
    Pads the data and alters the patch index so that a patch will be correct.
    :param data:
    :param patch_shape:
    :param patch_index:
    :return: padded data, fixed patch index
    """
    image_shape = data.shape[-ndim:]
    pad_before = np.abs((patch_index < 0) * patch_index)
    pad_after = np.abs(((patch_index + patch_shape) > image_shape) * ((patch_index + patch_shape) - image_shape))
    pad_args = np.stack([pad_before, pad_after], axis=1)
    if pad_args.shape[0] < len(data.shape):
        pad_args = [[0, 0]] * (len(data.shape) - pad_args.shape[0]) + pad_args.tolist()
    data = np.pad(data, pad_args, mode="edge")
    patch_index += pad_before
    return data, patch_index
def get_set_of_patch_indices(start, stop, step):
    return np.asarray(np.mgrid[start[0]:stop[0]:step[0], start[1]:stop[1]:step[1],
                               start[2]:stop[2]:step[2]].reshape(3, -1).T, dtype=np.int)

def get_patch_from_3d_data(data, patch_shape, patch_index):
    """
    Returns a patch from a numpy array.
    :param data: numpy array from which to get the patch.
    :param patch_shape: shape/size of the patch.
    :param patch_index: corner index of the patch.
    :return: numpy array take from the data with the patch shape specified.
    """
    patch_index = np.asarray(patch_index, dtype=np.int16)
    patch_shape = np.asarray(patch_shape)
    image_shape = data.shape[-3:]
    if np.any(patch_index < 0) or np.any((patch_index + patch_shape) > image_shape):
        data, patch_index = fix_out_of_bound_patch_attempt(data, patch_shape, patch_index)
    return data[..., patch_index[0]:patch_index[0]+patch_shape[0], patch_index[1]:patch_index[1]+patch_shape[1],
                patch_index[2]:patch_index[2]+patch_shape[2]]

def compute_patch_indices(image_shape, patch_size, overlap, start=None):
    if isinstance(overlap, list):
        overlap = np.asarray([overlap]) #执行这个 [64,64,4]
    if start is None:
        n_patches = np.ceil(image_shape / (patch_size - overlap))
        overflow = (patch_size - overlap) * n_patches - image_shape + overlap
        start = -np.ceil(overflow/2)
    elif isinstance(start, int):
        start = np.asarray([start] * len(image_shape))
    stop = image_shape + start
    step = patch_size - overlap
    # print("start: ",start[0]); print("stop: ",stop[0][0]);print("step: ",step)
    return get_set_of_patch_indices(start[0], stop[0], step[0])

def predict(model, data):
        return model.predict(data)

def reconstruct_from_patches(patches, patch_indices, data_shape, default_value=0):
    """
    Reconstructs an array of the original shape from the lists of patches and corresponding patch indices. Overlapping
    patches are averaged.
    :param patches: List of numpy array patches.
    :param patch_indices: List of indices that corresponds to the list of patches.
    :param data_shape: Shape of the array from which the patches were extracted.
    :param default_value: The default value of the resulting data. if the patch coverage is complete, this value will
    be overwritten.
    :return: numpy array containing the data reconstructed by the patches.
    """
    data = np.ones(data_shape) * default_value  #data_shape=[1,64,64,32]
    image_shape = data_shape[-3:]
    count = np.zeros(data_shape, dtype=np.int)
    print("patches shape", np.asarray(patches).shape)
    print("patch_index shape",np.asarray(patch_indices).shape)  #mark
    for patch, index in zip(patches, patch_indices):
        #patches shape (405, c=1, 64, 64, 32)  patch shape (c=1, 64, 64, 32)
        #patch_index shape (405, 3)  index shape (3,)
        image_patch_shape = patch.shape[-3:]
        if np.any(index < 0):
            print("index",index)  #mark
            fix_patch = np.asarray((index < 0) * np.abs(index), dtype=np.int)
            print("fix_patch",fix_patch)
            patch = patch[..., fix_patch[0]:, fix_patch[1]:, fix_patch[2]:]
            index[index < 0] = 0
            print("after index",index)
            print("patch shape",patch.shape)
        if np.any((index + image_patch_shape) >= image_shape):
            fix_patch = np.asarray(image_patch_shape - (((index + image_patch_shape) >= image_shape)
                                                        * ((index + image_patch_shape) - image_shape)), dtype=np.int)
            print("fix_patch2",fix_patch)
            patch = patch[..., :fix_patch[0], :fix_patch[1], :fix_patch[2]]
            print('patch2 shape',patch.shape)
            print('\n')
        patch_index = np.zeros(data_shape, dtype=np.bool) #patch_index.shape=[1,x,y,z]
        patch_index[...,
                    index[0]:index[0]+patch.shape[-3],
                    index[1]:index[1]+patch.shape[-2],
                    index[2]:index[2]+patch.shape[-1]] = True
        patch_data = np.zeros(data_shape)
        patch_data[patch_index] = patch.flatten()

        new_data_index = np.logical_and(patch_index, np.logical_not(count > 0))
        data[new_data_index] = patch_data[new_data_index]

        averaged_data_index = np.logical_and(patch_index, count > 0)
        if np.any(averaged_data_index):
            data[averaged_data_index] = (data[averaged_data_index] * count[averaged_data_index] + patch_data[averaged_data_index]) / (count[averaged_data_index] + 1)
        count[patch_index] += 1
    return data

def patch_wise_prediction(model, data, overlap, batch_size=1):
    """
    :param batch_size:
    :param model:
    :param data:
    :param overlap:
    :return:
    """
    patch_shape = tuple([int(dim) for dim in model.input.shape[-3:]]) #patch_shape.shape=[64,64,32] #获得块的shape 不含通道 这个应该是128 128 8吧
    predictions = list()
    indices = compute_patch_indices(data.shape[-3:], patch_size=patch_shape, overlap=overlap)
    batch = list()
    i = 0
    while i < len(indices):
        while len(batch) < batch_size:
            patch = get_patch_from_3d_data(data, patch_shape=patch_shape, patch_index=indices[i])
            batch.append(patch)
            i += 1
        batch = np.asarray(batch)
        batch = batch[np.newaxis]  #batch.shape=[1,batch_size=1,64,64,32]

        #prediction = predict(model, np.asarray(batch))
        prediction = predict(model, batch)     #prediction.shape=[batch_size=1,1,64,64,32]
        batch = list()
        for predicted_patch in prediction:
            predictions.append(predicted_patch)
    output_shape = [int(model.output.shape[1])] + list(data.shape[-3:]) #output_shape=[1,64,64,32]
    print("predictions shape",np.asarray(predictions).shape)  #predictions shape (405, 1, 64, 64, 32)
    return reconstruct_from_patches(predictions, patch_indices=indices, data_shape=output_shape)  #重新构造原始的不滑块的数组

def multi_class_prediction(prediction, affine):
    prediction_images = []
    for i in range(prediction.shape[1]):
        prediction_images.append(nib.Nifti1Image(prediction[0, i], affine))
    return prediction_images

def get_prediction_labels(prediction, threshold=0.5, labels=None):
    n_samples = prediction.shape[0]
    label_arrays = []
    for sample_number in range(n_samples):
        label_data = np.argmax(prediction[sample_number], axis=0) + 1
        label_data[np.max(prediction[sample_number], axis=0) < threshold] = 0
        if labels:
            for value in np.unique(label_data).tolist()[1:]:
                label_data[label_data == value] = labels[value - 1]
        label_arrays.append(np.array(label_data, dtype=np.uint8))
    return label_arrays
def prediction_to_image(prediction, affine,hdr, label_map, threshold=0.5, labels=None):
    print('prediction.shape[1]',prediction.shape[1])
    if prediction.shape[1] == 1:
        data = prediction[0, 0]
       # print('data:',data)
        if label_map:
            label_map_data = np.zeros(prediction[0, 0].shape, np.int8)
            if labels:
                label = labels[0]
            else:
                label = 1
            label_map_data[data > threshold] = label
            data = label_map_data
            #print(data)
    elif prediction.shape[1] > 1:
        if label_map:
            label_map_data = get_prediction_labels(prediction, threshold=threshold, labels=labels)
            data = label_map_data[0]
        else:
            return multi_class_prediction(prediction, affine)
    else:
        raise RuntimeError("Invalid prediction array shape: {0}".format(prediction.shape))
    return nib.Nifti1Image(data, affine,hdr)
def run_validation_case(data,affine,hdr,output_dir, model, output_label_map, threshold=0.5, labels=None, overlap=[32,32,16]):
    """
    Runs a test case and writes predicted images to file.
    :param data_index: Index from of the list of test cases to get an image prediction from.
    :param output_dir: Where to write prediction images.
    :param output_label_map: If True, will write out a single image with one or more labels. Otherwise outputs
    the (sigmoid) prediction values from the model.
    :param threshold: If output_label_map is set to True, this threshold defines the value above which is
    considered a positive result and will be assigned a label.
    :param labels:
    :param training_modalities:
    :param data_file:
    :param model:
    """
    # affine = data_file.root.affine[data_index]
    # test_data = np.asarray([data_file.root.data[data_index]])
    # for i, modality in enumerate(training_modalities):
    #     image = nib.Nifti1Image(test_data[0, i], affine)
    #     image.to_filename(os.path.join(output_dir, "data_{0}.nii.gz".format(modality)))
    #
    # test_truth = nib.Nifti1Image(data_file.root.truth[data_index][0], affine)
    # test_truth.to_filename(os.path.join(output_dir, "truth.nii.gz"))
    test_data = data
    prediction = patch_wise_prediction(model=model, data=test_data, overlap=overlap)[np.newaxis]
    print(prediction.shape)
    label_map = output_label_map
    prediction_image = prediction_to_image(prediction, affine, hdr,label_map, threshold=threshold,
                                           labels=labels)  #把预测结果转为nii数据存储
    if isinstance(prediction_image, list):
        for i, image in enumerate(prediction_image):
            image.to_filename(os.path.join(output_dir, "prediction_{0}.nii.gz".format(i + 1)))  #预测结果为多个
    else:
        #print('haha')
        prediction_image.to_filename(os.path.join(output_dir, "prediction.nii.gz"))

if __name__ == "__main__":
    base_output_dir = r'.\prediction_we_0.50.53_2'
    model_file = r'.\prediction_we_0.50.53_1\model3-ep007-loss0.079-val_loss0.100.h5'
    base_img_name = r'..\data\lasttest_0.50.53'
    strattime=time.time()
    model = load_old_model(model_file) #加载训练好的模型
    print('loaded')
    strattime1=time.time()
    imgs_folder = os.listdir(base_img_name)
    for img_folder in imgs_folder:
        print(img_folder)  #mark
        output_dir = base_output_dir + '/' + img_folder + '/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        imgs_path = glob.glob(os.path.join(base_img_name, img_folder, "*.nii.gz"))
        #img_name = imgs_path[0]
        #label_name = imgs_path[1]
        if 'LABEL' in imgs_path[0] or 'LABEL' in imgs_path[1] or 'LABLE' in imgs_path[0] or 'LABLE' in imgs_path[1]:
            img_name = imgs_path[1]
            label_name = imgs_path[0]
        else:
            img_name = imgs_path[0]
            label_name = imgs_path[1]
        #copy file
        copyfile(img_name, output_dir + 'data.nii.gz')
        copyfile(label_name, output_dir + 'truth.nii.gz')
#prediction文件夹中放好测试数据文件夹 每个文件夹里有数据 标签 之后还会往里面放测试结果
        img = nib.load(img_name)  #加载要被测试的数据

        arr_img = img.get_fdata()
        arr_img = np.squeeze(arr_img)
        arr_img = normalized_data(arr_img)
            # print(arr_img.shape)
        hdr = nib.Nifti1Header()
        re_affine = img.affine

        label = nib.load(label_name)
        arr_label = label.get_fdata()

        arr_img = np.asarray(arr_img)
        arr_labe = np.asarray(arr_label)

        overlap = [64,64,4]
        output_label_map = True  # can not determine the use of the parameter
        run_validation_case(arr_img, re_affine, hdr, output_dir, model, output_label_map,
                                threshold=0.5, labels=(1,), overlap=overlap)
        #path = '.\prediction'
        #main(path=path) #做了各种指标计算
    endtime=time.time()
    print('tatal time:',endtime-strattime)
    print('tatal time1:',endtime-strattime1)
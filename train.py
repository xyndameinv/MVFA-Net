import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#设置Gpu
# multi_gpu = False
# if multi_gpu:
#     multi_gpu_num = 2
# else:
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#     multi_gpu_num = None
import numpy as np
import glob

from generators import get_training_and_validation_generators
from MVFA_Net import unet_model_3d
from training import load_old_model, train_model
import tables
from write_to_h5 import write_data_to_file

config = dict()
config['my_train_patch_dir'] = r'E:\cervial\data\128-8-from_crop'
config["data_file"] = r'..\data\128-8-from_crop_0.50.53\test_patch.h5'
config["pool_size"] = (2, 2, 1)  # pool size for the max pooling operations
config["patch_shape"] = (128, 128, 8)  # switch to None to train on the whole image
config["labels"] = (1,)  # the label numbers on the input image
config["n_labels"] = 1
config["nb_channels"] = 1
#config["all_modalities"] = ["hm"]
#config["training_modalities"] = config["all_modalities"]
if "patch_shape" in config and config["patch_shape"] is not None:
    config["input_shape"] = tuple([config["nb_channels"]] + list(config["patch_shape"]))
else:
    config["input_shape"] = tuple([config["nb_channels"]] + list(config["image_shape"]))
config["truth_channel"] = config["nb_channels"]
config["deconvolution"] = False  # if False, will use upsampling instead of deconvolution

config["train_batch_size"] = 4
config["val_batch_size"] = 4
config["n_epochs"] = 50  #25 cutoff the training after this many epochs
config["patience"] = 2  # learning rate will be reduced after this many epochs if the validation loss is not improving
config["early_stop"] = 4  # training will be stopped after this many epochs without the validation loss improving
config["initial_learning_rate"] = 0.0002
config["learning_rate_drop"] = 0.5 # factor by which the learning rate will be reduced
config["validation_split"] = 0.05  # portion of the data that will be used for validation
#config["validation_patch_overlap"] = 0  # if > 0, during training, validation patches will be overlapping

#config["model_file"] = os.path.abspath("cervical_segmentation_model.h5")
config["model_file"] = os.path.abspath('model2-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5')
config["training_file"] = os.path.abspath('training_ids_patch_0.pkl')  #存训练数据在datafile中的索引
config["validation_file"] = os.path.abspath('validation_patch_0.pkl')
#config["validation_file"] = os.path.abspath("validation_ids.pkl")

def main(overwrite=False):
    if os.path.exists('model-ep041-loss0.071-val_loss0.076.h5'):
        print("load old model")
        model = load_old_model('model-ep041-loss0.071-val_loss0.076.h5');
    else:
        # instantiate new model
        model = unet_model_3d(input_shape=config["input_shape"],
                              pool_size=config["pool_size"],
                              n_labels=config["n_labels"],
                              initial_learning_rate=config["initial_learning_rate"],
                              deconvolution=config["deconvolution"])

    if overwrite or not os.path.exists(config["data_file"]):
        config["data_file"],n_samples = write_data_to_file(config['my_train_patch_dir'], config["data_file"],
                                          image_shape=config["patch_shape"])

    data_file = tables.open_file(config["data_file"],'r')
    #print(data_file_opened.shape)
    print(type(data_file))

    # get training and testing generators
    training_generator, validation_generator, num_training_steps, num_validation_steps = get_training_and_validation_generators(
                                        data_file, batch_size=config["train_batch_size"], training_keys_file=config["training_file"],
                                            validation_keys_file=config["validation_file"],
                                          data_split=1.-config["validation_split"], overwrite=False, labels=None, augment=False,
                                           augment_flip=True, augment_distortion_factor=0.25,validation_batch_size=None)
    # run training
    # train_model(model=model,
    #             model_file=config["model_file"],
    #             train_img=train_x,
    #             train_mask=train_y,
    #             batch_size=config["train_batch_size"],
    #             epochs=config["n_epochs"],
    #             initial_learning_rate=config["initial_learning_rate"],
    #             learning_rate_drop=config["learning_rate_drop"],
    #             learning_rate_patience=config["patience"],
    #             early_stopping_patience=config["early_stop"])D
    train_model(model=model,
                model_file=config["model_file"],
                training_generator=training_generator,
                validation_generator=validation_generator,
                steps_per_epoch=num_training_steps,
                validation_steps=num_validation_steps,
                initial_learning_rate=config["initial_learning_rate"],
                learning_rate_drop=config["learning_rate_drop"],
                learning_rate_epochs=True,
                learning_rate_patience=config["patience"],
                early_stopping_patience=config["early_stop"],
                n_epochs=config["n_epochs"])


if __name__ == "__main__":
    main()

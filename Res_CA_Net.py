import numpy as np
import keras
import tensorflow as tf
from keras import backend as K
from keras.engine import Input, Model
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Activation,Reshape, \
                        BatchNormalization, PReLU, Deconvolution3D,Add,SpatialDropout3D,\
                            add,GlobalAveragePooling3D,AveragePooling3D,multiply,Lambda,Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
#from keras.utils import multi_gpu_model

from metrics import dice_coefficient_loss, get_label_dice_coefficient_function, dice_coefficient,weighted_dice_coefficient_loss

K.set_image_data_format("channels_first")

try:
    from keras.engine import merge
except ImportError:
    from keras.layers.merge import concatenate

def get_up_convolution(n_filters, pool_size, kernel_size=(2, 2, 2), strides=(2, 2, 2),
                       deconvolution=False):
    if deconvolution:
        return Deconvolution3D(filters=n_filters, kernel_size=kernel_size,
                               strides=strides)
    else:
        return UpSampling3D(size=pool_size)
		
def expand_dim_backend(x):
    x = K.expand_dims(x,-1)
    x = K.expand_dims(x,-1)
    x = K.expand_dims(x,-1)
    return x
	
def senet(layer, n_filter):
    seout = GlobalAveragePooling3D()(layer)
    seout = Dense(units=int(n_filter/2))(seout)
    seout = Activation("relu")(seout)
    seout = Dense(units=n_filter)(seout)
    seout = Activation("sigmoid")(seout)
    print("seout1 shape",seout.shape)
    # seout = Reshape([-1,1,1,n_filter])(seout)
    seout = Lambda(expand_dim_backend)(seout)
    print("seout shape",seout.shape)
    return seout
	
def resudial_block(input_layer, n_filters,kernel_1=(1, 1, 1),kernel_3=(3,3,3), padding='same', strides=(1, 1, 1)):
    layer = BatchNormalization(axis=1)(input_layer)
    layer = Activation("relu")(layer)
    layer = Conv3D(n_filters, kernel_1, padding=padding, strides=strides)(layer)

    layer = BatchNormalization(axis=1)(layer)
    layer = Activation("relu")(layer)
    layer = Conv3D(n_filters, kernel_3, padding=padding, strides=strides)(layer)

    seout = senet(layer,n_filters)
    seout = multiply([seout,layer])
    
    layer = BatchNormalization(axis=1)(seout)
    layer = Activation("relu")(layer)
    layer = Conv3D(n_filters, kernel_1, padding=padding, strides=strides)(layer)
    x_short = Conv3D(n_filters,kernel_1,padding=padding,strides=strides)(input_layer)

    layer_out = add([x_short, layer])

    return layer_out

def resudial_block_2(input_layer, n_filters,kernel_1=(1, 1, 1),kernel_3=(3,3,3), padding='same', strides=(1, 1, 1)):
    layer = BatchNormalization(axis=1)(input_layer)
    layer = Activation("relu")(layer)
    layer = Conv3D(n_filters, kernel_1, padding=padding, strides=strides)(layer)

    layer = BatchNormalization(axis=1)(layer)
    layer = Activation("relu")(layer)
    layer = Conv3D(n_filters, kernel_3, padding=padding, strides=strides)(layer)

    layer = BatchNormalization(axis=1)(layer)
    layer = Activation("relu")(layer)
    layer = Conv3D(n_filters, kernel_1, padding=padding, strides=strides)(layer)

    x_short = input_layer
    # x_short = Conv3D(n_filters,kernel_1,padding=padding,strides=strides)(input_layer)
    # print("layer shape",layer.shape)
    # print("input layer shape", input_layer.shape)
    layer = add([x_short, layer])
    return layer

def unet_model_3d(input_shape, pool_size=(2, 2, 1), n_labels=1, initial_learning_rate=0.0001, deconvolution=False,
                  depth=4, n_base_filters=32, include_label_wise_dice_coefficients=False, metrics=dice_coefficient,
                  batch_normalization=True, activation_name="sigmoid"):
    ############################
    #resUnet + dropout
    ###############################
    inputs = Input(input_shape)
    # current_layer = inputs    
    inputs_1 = Conv3D(n_base_filters,kernel_size=(3,3,3),strides=(1,1,1),padding="same")(inputs)
    inputs_1 = BatchNormalization(axis=1)(inputs_1)
    inputs_1 = Activation("relu")(inputs_1)
    layer1 = resudial_block(inputs_1,n_base_filters)
    #layer1 = resudial_block_2(layer1,n_base_filters)
    layer1_pool = MaxPooling3D(pool_size=(2,2,2))(layer1)

    layer2 = resudial_block(layer1_pool,n_base_filters*2)
    #layer2 = resudial_block_2(layer2,n_filters=n_base_filters*2)
    layer2_poo2 = MaxPooling3D(pool_size=pool_size)(layer2)

    layer3 = resudial_block(layer2_poo2,n_base_filters*4)
    #layer3 = resudial_block_2(layer3,n_base_filters*4)
    layer3_poo3 = MaxPooling3D(pool_size=pool_size)(layer3)

    layer3_poo3 = SpatialDropout3D(rate=0.1)(layer3_poo3)
    layer4 = Conv3D(n_base_filters*8, kernel_size=(3,3,3), padding="same", strides=(1,1,1))(layer3_poo3)
    layer4 = BatchNormalization(axis=1)(layer4)
    layer4 = Activation("relu")(layer4)
    layer4 = Conv3D(n_base_filters * 8, kernel_size=(3, 3, 3), padding="same", strides=(1, 1, 1))(layer4)
    layer4 = BatchNormalization(axis=1)(layer4)
    layer4 = Activation("relu")(layer4)
    layer4 = SpatialDropout3D(rate=0.1)(layer4)

    layer_up_3 = get_up_convolution(pool_size=pool_size, deconvolution=False,
                                            n_filters=n_base_filters*3)(layer4)
    concat3 = concatenate([layer_up_3, layer3], axis=1)
    layer33 = resudial_block(concat3,n_base_filters*4)
    #layer33 = resudial_block_2(layer33,n_base_filters*4)

    layer_up_2 = get_up_convolution(pool_size=pool_size, deconvolution=False,
                                    n_filters=n_base_filters * 2)(layer33)
    concat2 = concatenate([layer_up_2, layer2], axis=1)
    layer22 = resudial_block(concat2, n_base_filters * 2)
    #layer22 = resudial_block_2(layer22, n_base_filters * 2)

    layer_up_1 = get_up_convolution(pool_size=(2,2,2), deconvolution=False,
                                    n_filters=n_base_filters * 1)(layer22)
    concat1 = concatenate([layer_up_1, layer1], axis=1)
    layer11 = resudial_block(concat1, n_base_filters * 1)
    #layer11 = resudial_block_2(layer11, n_base_filters * 1)

    final_convolution = Conv3D(n_labels, (1, 1, 1))(layer11)
    print("final_convolution.shape:",final_convolution.shape)
    act = Activation(activation_name)(final_convolution)
    print("act.shape:", act.shape)
    model = Model(inputs=inputs, outputs=act)

    if not isinstance(metrics, list):
        metrics = [metrics]

    # def get_lr_metric(optimizer):
    #     def lr(y_true, y_pred):
    #         return optimizer.lr
    #     return lr
    # lr_metric = get_lr_metric(Adam(lr=initial_learning_rate))

    model.compile(optimizer=Adam(lr=initial_learning_rate), loss=dice_coefficient_loss, metrics=metrics)
    print(model.summary())
    return model



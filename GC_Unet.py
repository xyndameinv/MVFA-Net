import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.engine import Input, Model
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Activation, BatchNormalization, PReLU, Deconvolution3D,SpatialDropout3D,Reshape,multiply,add,Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam

#from keras.utils import multi_gpu_model

from metrics import dice_coefficient_loss, get_label_dice_coefficient_function, dice_coefficient,weighted_dice_coefficient_loss

K.set_image_data_format("channels_first")

try:
    from keras.engine import merge
except ImportError:
    from keras.layers.merge import concatenate



def compute_level_output_shape(n_filters, depth, pool_size, image_shape):
    """
    Each level has a particular output shape based on the number of filters used in that level and the depth or number 
    of max pooling operations that have been done on the data at that point.
    :param image_shape: shape of the 3d image.
    :param pool_size: the pool_size parameter used in the max pooling operation.
    :param n_filters: Number of filters used by the last node in a given level.
    :param depth: The number of levels down in the U-shaped model a given node is.
    :return: 5D vector of the shape of the output node 
    """
    output_image_shape = np.asarray(np.divide(image_shape, np.power(pool_size, depth)), dtype=np.int32).tolist()
    return tuple([None, n_filters] + output_image_shape)


def get_up_convolution(n_filters, pool_size, kernel_size=(2, 2, 2), strides=(2, 2, 2),
                       deconvolution=False):
    # if deconvolution:
    #     return Deconvolution3D(filters=n_filters, kernel_size=kernel_size,
    #                            strides=strides)
    # else:
        return UpSampling3D(size=pool_size)
def expand_dims1(input):
    x=tf.expand_dims(input, axis=1)
    return x

def expand_dims2(input):
    x=tf.expand_dims(input, axis=-1)
    return x

def matmul(input):
    """input must be a  list"""
    return tf.matmul(input[0],input[1])

def double_conv_block(input_layer, n_filters,kernel=(3,3,3), padding='same', strides=(1, 1, 1)):
    layer = Conv3D(n_filters, kernel, padding=padding, strides=strides)(input_layer)
    layer = BatchNormalization(axis=1)(layer)
    layer = Activation("relu")(layer)

    layer = Conv3D(n_filters, kernel, padding=padding, strides=strides)(layer)
    layer = BatchNormalization(axis=1)(layer)
    layer = Activation("relu")(layer)
    return layer
def gcblock(input_layer,n_filter):
    shape=input_layer.shape
    print(input_layer)
    print(shape)
    c=int(shape[1])
    print("维度：",c)
    input_x=input_layer
    input_x = Reshape((c,int(shape[2])*int(shape[3])*int(shape[4])))(input_x)  # [C,H*W*Z]
    print(input_x)
    input_x = Lambda(expand_dims1)(input_x)
    input_x = Lambda(expand_dims1)(input_x)#[1,1,C,H*W*Z]
    print(input_x)
    #context modeling
    context_mask=Conv3D(1,(1,1,1))(input_layer) #[1,W,H,Z]
    context_mask=Reshape((1,int(shape[2])*int(shape[3])*int(shape[4])))(context_mask) #[1,W*H*Z]
    print(context_mask)
    #context_mask=softmax(context_mask,axis=-1) #[1,W*H*Z]
    context_mask=Activation("softmax")(context_mask)#[1,W*H*Z]
    context_mask=Lambda(expand_dims1)(context_mask) #[1,1,W*H*Z]
    context_mask=Lambda(expand_dims2)(context_mask) #[1,1,W*H*Z,1]
    print(context_mask)
    context=Lambda(matmul)([input_x,context_mask]) #[1,1,c,1]
    print(context)
    context=Reshape((c,1,1,1))(context) #[c,1,1,1]
    #transform1
    context_transform1=Conv3D(int(n_filter/2),(1,1,1))(context)
    context_transform1 = BatchNormalization(axis=1)(context_transform1)
    context_transform1 = Activation("relu")(context_transform1)
    
    context_transform1=Conv3D(n_filter,(1,1,1))(context_transform1)
    context_transform1 = Activation("sigmoid")(context_transform1)
    context_transform1=multiply([context_transform1,input_layer])
    
    #transform2
    context_transform2=Conv3D(int(n_filter/2),(1,1,1))(context)
    context_transform2 = BatchNormalization(axis=1)(context_transform2)
    context_transform2 = Activation("relu")(context_transform2)
    
    context_transform2=Conv3D(n_filter,(1,1,1))(context_transform2)
    x=add([context_transform1,context_transform2])
    print(type(x).__name__)
    return x

def unet_model_3d(input_shape, pool_size=(2, 2, 2), n_labels=1, initial_learning_rate=0.0001, deconvolution=False,
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
    layer1 = double_conv_block(inputs_1,n_base_filters)
    layer1_pool = MaxPooling3D(pool_size=pool_size)(layer1)

    layer2 = double_conv_block(layer1_pool,n_base_filters*2)
    layer2_poo2 = MaxPooling3D(pool_size=pool_size)(layer2)

    layer3 = double_conv_block(layer2_poo2,n_base_filters*4)
    layer3_poo3 = MaxPooling3D(pool_size=pool_size)(layer3)

    layer3_poo3 = SpatialDropout3D(rate=0.1)(layer3_poo3)
    layer4 = Conv3D(n_base_filters*8, kernel_size=(3,3,3), padding="same", strides=(1,1,1))(layer3_poo3)
    layer4 = BatchNormalization(axis=1)(layer4)
    layer4 = Activation("relu")(layer4)
    layer4 = Conv3D(n_base_filters * 8, kernel_size=(3, 3, 3), padding="same", strides=(1, 1, 1))(layer4)
    layer4 = BatchNormalization(axis=1)(layer4)
    layer4 = Activation("relu")(layer4)
    layer4 = gcblock(layer4,n_base_filters * 8)
    layer4 = SpatialDropout3D(rate=0.1)(layer4)

    layer_up_3 = get_up_convolution(pool_size=pool_size, deconvolution=False,
                                            n_filters=n_base_filters*3)(layer4)
    concat3 = concatenate([layer_up_3, layer3], axis=1)
    layer33 = double_conv_block(concat3,n_base_filters*4)

    layer_up_2 = get_up_convolution(pool_size=pool_size, deconvolution=False,
                                    n_filters=n_base_filters * 2)(layer33)
    concat2 = concatenate([layer_up_2, layer2], axis=1)
    layer22 = double_conv_block(concat2, n_base_filters * 2)

    layer_up_1 = get_up_convolution(pool_size=pool_size, deconvolution=False,
                                    n_filters=n_base_filters * 1)(layer22)
    concat1 = concatenate([layer_up_1, layer1], axis=1)
    layer11 = double_conv_block(concat1, n_base_filters * 1)

    final_convolution = Conv3D(n_labels, (1, 1, 1))(layer11)
    print("final_convolution.shape:",final_convolution.shape)
    act = Activation(activation_name)(final_convolution)
    print("act.shape:", act.shape)
    model = Model(inputs=inputs, outputs=act)

    if not isinstance(metrics, list):
        metrics = [metrics]

    model.compile(optimizer=Adam(lr=initial_learning_rate), loss=dice_coefficient_loss, metrics=metrics)
    print(model.summary())
    return model
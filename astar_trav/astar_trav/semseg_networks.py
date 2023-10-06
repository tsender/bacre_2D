import tensorflow as tf
from typing import Tuple

HE_NORMAL = tf.initializers.he_normal()
LEAKY_RELU = tf.keras.layers.LeakyReLU(alpha=0.2)
KERNEL_REG_L2 = tf.keras.regularizers.L2(l2=0.001)

def create_unet_network_large(input_shape: Tuple[int], num_labels: int) -> tf.keras.Model:
    """Creates a U-net architecture

    Args:
        input_shape_shape: The shape of the input tensor
        num_labels: Number of labels

    Returns:
        Keras model
    """
    assert input_shape[0] >= 256
    assert input_shape[1] >= 256
    input = tf.keras.Input(shape=input_shape, dtype=tf.float32)

    # Encoder Block 1
    e1 = tf.keras.layers.Conv2D(64, (3,3), strides = 2, padding='same', activation='selu', kernel_initializer=HE_NORMAL, kernel_regularizer=KERNEL_REG_L2)(input) # (B, H/2, H/2, 64)
    
    # Encoder Block 2
    e2 = tf.keras.layers.Conv2D(128, (3,3), strides = 2, padding='same', activation='selu', kernel_initializer=HE_NORMAL, kernel_regularizer=KERNEL_REG_L2)(e1) # (B, H/4, H/4, 128)

    # Encoder Block 3
    e3 = tf.keras.layers.Conv2D(256, (3,3), strides = 2, padding='same', activation='selu', kernel_initializer=HE_NORMAL, kernel_regularizer=KERNEL_REG_L2)(e2) # (B, H/8, H/8, 256)
    
    # Encoder Block 4
    e4 = tf.keras.layers.Conv2D(512, (3,3), strides = 2, padding='same', activation='selu', kernel_initializer=HE_NORMAL, kernel_regularizer=KERNEL_REG_L2)(e3) # (B, H/16, H/16, 512)

    # Encoder Block 5
    e5 = tf.keras.layers.Conv2D(512, (3,3), strides = 2, padding='same', activation='selu', kernel_initializer=HE_NORMAL, kernel_regularizer=KERNEL_REG_L2)(e4) # (B, H/32, H/32, 512)

    # Decoder Block 1
    d1 = tf.keras.layers.UpSampling2D(size=(2,2))(e5)
    d1 = tf.keras.layers.Conv2D(512, (3,3), padding='same', activation='selu', kernel_initializer=HE_NORMAL, kernel_regularizer=KERNEL_REG_L2)(d1) # (B, H/16, H/16, 512)

    # Decoder Block 2
    d2 = tf.keras.layers.Concatenate()([d1, e4])
    d2 = tf.keras.layers.UpSampling2D(size=(2,2))(d2)
    d2 = tf.keras.layers.Conv2D(256, (3,3), padding='same', activation='selu', kernel_initializer=HE_NORMAL, kernel_regularizer=KERNEL_REG_L2)(d2) # (B, H/8, H/8, 256)

    # Decoder Block 3
    d3 = tf.keras.layers.Concatenate()([d2, e3])
    d3 = tf.keras.layers.UpSampling2D(size=(2,2))(d3)
    d3 = tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='selu', kernel_initializer=HE_NORMAL, kernel_regularizer=KERNEL_REG_L2)(d3) # (B, H/4, H/4, 128)

    # Decoder Block 4
    d4 = tf.keras.layers.Concatenate()([d3, e2])
    d4 = tf.keras.layers.UpSampling2D(size=(2,2))(d4)
    d4 = tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='selu', kernel_initializer=HE_NORMAL, kernel_regularizer=KERNEL_REG_L2)(d4) # (B, H/2, H/2, 64)

    # Decoder Block 5
    d5 = tf.keras.layers.Concatenate()([d4, e1])
    d5 = tf.keras.layers.UpSampling2D(size=(2,2))(d5)
    output = tf.keras.layers.Conv2D(num_labels, (3,3), padding='same', activation='softmax', kernel_initializer=HE_NORMAL, kernel_regularizer=KERNEL_REG_L2)(d5) # (B, H, H, num_labels)

    model = tf.keras.Model(input, output)
    return model

def create_unet_network_small(input_shape: Tuple[int], num_labels: int) ->tf.keras.Model:
    """Creates a smaller U-net architecture with about 2.3 million parameters

    Args:
        input_shape_shape: The shape of the input tensor
        num_labels: Number of labels

    Returns:
        Keras model
    """
    assert input_shape[0] >= 256
    assert input_shape[1] >= 256
    input = tf.keras.Input(shape=input_shape, dtype=tf.float32)

    # Encoder Block 1
    e1 = tf.keras.layers.Conv2D(32, (3,3), strides = 2, padding='same', activation=LEAKY_RELU, kernel_initializer=HE_NORMAL, kernel_regularizer=KERNEL_REG_L2)(input) # (B, H/2, H/2, 64)
    
    # Encoder Block 2
    e2 = tf.keras.layers.Conv2D(64, (3,3), strides = 2, padding='same', activation=LEAKY_RELU, kernel_initializer=HE_NORMAL, kernel_regularizer=KERNEL_REG_L2)(e1) # (B, H/4, H/4, 128)

    # Encoder Block 3
    e3 = tf.keras.layers.Conv2D(128, (3,3), strides = 2, padding='same', activation=LEAKY_RELU, kernel_initializer=HE_NORMAL, kernel_regularizer=KERNEL_REG_L2)(e2) # (B, H/8, H/8, 256)
    
    # Encoder Block 4
    e4 = tf.keras.layers.Conv2D(256, (3,3), strides = 2, padding='same', activation=LEAKY_RELU, kernel_initializer=HE_NORMAL, kernel_regularizer=KERNEL_REG_L2)(e3) # (B, H/16, H/16, 512)

    # Encoder Block 5
    e5 = tf.keras.layers.Conv2D(256, (3,3), strides = 2, padding='same', activation=LEAKY_RELU, kernel_initializer=HE_NORMAL, kernel_regularizer=KERNEL_REG_L2)(e4) # (B, H/32, H/32, 512)

    # Decoder Block 1
    d1 = tf.keras.layers.UpSampling2D(size=(2,2))(e5)
    d1 = tf.keras.layers.Conv2D(256, (3,3), padding='same', activation=LEAKY_RELU, kernel_initializer=HE_NORMAL, kernel_regularizer=KERNEL_REG_L2)(d1) # (B, H/16, H/16, 512)

    # Decoder Block 2
    d2 = tf.keras.layers.Concatenate()([d1, e4])
    d2 = tf.keras.layers.UpSampling2D(size=(2,2))(d2)
    d2 = tf.keras.layers.Conv2D(128, (3,3), padding='same', activation=LEAKY_RELU, kernel_initializer=HE_NORMAL, kernel_regularizer=KERNEL_REG_L2)(d2) # (B, H/8, H/8, 256)

    # Decoder Block 3
    d3 = tf.keras.layers.Concatenate()([d2, e3])
    d3 = tf.keras.layers.UpSampling2D(size=(2,2))(d3)
    d3 = tf.keras.layers.Conv2D(64, (3,3), padding='same', activation=LEAKY_RELU, kernel_initializer=HE_NORMAL, kernel_regularizer=KERNEL_REG_L2)(d3) # (B, H/4, H/4, 128)

    # Decoder Block 4
    d4 = tf.keras.layers.Concatenate()([d3, e2])
    d4 = tf.keras.layers.UpSampling2D(size=(2,2))(d4)
    d4 = tf.keras.layers.Conv2D(32, (3,3), padding='same', activation=LEAKY_RELU, kernel_initializer=HE_NORMAL, kernel_regularizer=KERNEL_REG_L2)(d4) # (B, H/2, H/2, 64)

    # Decoder Block 5
    d5 = tf.keras.layers.Concatenate()([d4, e1])
    d5 = tf.keras.layers.UpSampling2D(size=(2,2))(d5)
    output = tf.keras.layers.Conv2D(num_labels, (3,3), padding='same', activation='softmax', kernel_initializer=HE_NORMAL, kernel_regularizer=KERNEL_REG_L2)(d5) # (B, H, H, num_labels)

    model = tf.keras.Model(input, output)
    return model

def create_unet_network_small_v2(input_shape: Tuple[int], num_labels: int) ->tf.keras.Model:
    """Creates a smaller U-net architecture with about 587k parameters

    Args:
        input_shape_shape: The shape of the input tensor
        num_labels: Number of labels

    Returns:
        Keras model
    """
    assert input_shape[0] >= 256
    assert input_shape[1] >= 256
    input = tf.keras.Input(shape=input_shape, dtype=tf.float32)

    # Encoder Block 1
    e1 = tf.keras.layers.Conv2D(16, (3,3), strides = 2, padding='same', activation=LEAKY_RELU, kernel_initializer=HE_NORMAL, kernel_regularizer=KERNEL_REG_L2)(input) # (B, H/2, H/2, 64)
    
    # Encoder Block 2
    e2 = tf.keras.layers.Conv2D(32, (3,3), strides = 2, padding='same', activation=LEAKY_RELU, kernel_initializer=HE_NORMAL, kernel_regularizer=KERNEL_REG_L2)(e1) # (B, H/4, H/4, 128)

    # Encoder Block 3
    e3 = tf.keras.layers.Conv2D(64, (3,3), strides = 2, padding='same', activation=LEAKY_RELU, kernel_initializer=HE_NORMAL, kernel_regularizer=KERNEL_REG_L2)(e2) # (B, H/8, H/8, 256)
    
    # Encoder Block 4
    e4 = tf.keras.layers.Conv2D(128, (3,3), strides = 2, padding='same', activation=LEAKY_RELU, kernel_initializer=HE_NORMAL, kernel_regularizer=KERNEL_REG_L2)(e3) # (B, H/16, H/16, 512)

    # Encoder Block 5
    e5 = tf.keras.layers.Conv2D(128, (3,3), strides = 2, padding='same', activation=LEAKY_RELU, kernel_initializer=HE_NORMAL, kernel_regularizer=KERNEL_REG_L2)(e4) # (B, H/32, H/32, 512)

    # Decoder Block 1
    d1 = tf.keras.layers.UpSampling2D(size=(2,2))(e5)
    d1 = tf.keras.layers.Conv2D(128, (3,3), padding='same', activation=LEAKY_RELU, kernel_initializer=HE_NORMAL, kernel_regularizer=KERNEL_REG_L2)(d1) # (B, H/16, H/16, 512)

    # Decoder Block 2
    d2 = tf.keras.layers.Concatenate()([d1, e4])
    d2 = tf.keras.layers.UpSampling2D(size=(2,2))(d2)
    d2 = tf.keras.layers.Conv2D(64, (3,3), padding='same', activation=LEAKY_RELU, kernel_initializer=HE_NORMAL, kernel_regularizer=KERNEL_REG_L2)(d2) # (B, H/8, H/8, 256)

    # Decoder Block 3
    d3 = tf.keras.layers.Concatenate()([d2, e3])
    d3 = tf.keras.layers.UpSampling2D(size=(2,2))(d3)
    d3 = tf.keras.layers.Conv2D(32, (3,3), padding='same', activation=LEAKY_RELU, kernel_initializer=HE_NORMAL, kernel_regularizer=KERNEL_REG_L2)(d3) # (B, H/4, H/4, 128)

    # Decoder Block 4
    d4 = tf.keras.layers.Concatenate()([d3, e2])
    d4 = tf.keras.layers.UpSampling2D(size=(2,2))(d4)
    d4 = tf.keras.layers.Conv2D(16, (3,3), padding='same', activation=LEAKY_RELU, kernel_initializer=HE_NORMAL, kernel_regularizer=KERNEL_REG_L2)(d4) # (B, H/2, H/2, 64)

    # Decoder Block 5
    d5 = tf.keras.layers.Concatenate()([d4, e1])
    d5 = tf.keras.layers.UpSampling2D(size=(2,2))(d5)
    output = tf.keras.layers.Conv2D(num_labels, (3,3), padding='same', activation='softmax', kernel_initializer=HE_NORMAL, kernel_regularizer=KERNEL_REG_L2)(d5) # (B, H, H, num_labels)

    model = tf.keras.Model(input, output)
    return model
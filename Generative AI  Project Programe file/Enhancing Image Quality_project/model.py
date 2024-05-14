import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, PReLU, Add
from tensorflow.keras.models import Model


def residual_block(x):
    """
    Residual block as used in SRGAN.
    """
    filters = 64
    kernel_size = 3
    strides = 1
    padding = 'same'

    # First convolutional layer
    y = Conv2D(filters, kernel_size, strides=strides, padding=padding)(x)
    y = BatchNormalization()(y)
    y = PReLU()(y)

    # Second convolutional layer
    y = Conv2D(filters, kernel_size, strides=strides, padding=padding)(y)
    y = BatchNormalization()(y)

    # Adding the input (identity) to the output of the residual block
    y = Add()([x, y])
    return y


def generator():
    """
    Generator network as used in SRGAN.
    """
    # Input layer
    inputs = Input(shape=(None, None, 3))

    # Feature extraction
    x = Conv2D(64, 9, padding='same')(inputs)
    x = PReLU()(x)

    # Adding residual blocks
    for _ in range(16):
        x = residual_block(x)

    # Post-residual block convolutional layers
    x = Conv2D(64, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([inputs, x])

    # Upsampling
    x = Conv2D(256, 3, padding='same')(x)
    x = tf.nn.depth_to_space(x, 2)
    x = PReLU()(x)

    # Final convolutional layer
    outputs = Conv2D(3, 9, padding='same', activation='tanh')(x)

    return Model(inputs, outputs)


def discriminator():
    """
    Discriminator network as used in SRGAN.
    """
    inputs = Input(shape=(None, None, 3))

    x = Conv2D(64, 3, padding='same')(inputs)
    x = PReLU()(x)

    x = Conv2D(64, 3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)

    x = Conv2D(128, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)

    x = Conv2D(128, 3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)

    x = Conv2D(256, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)

    x = Conv2D(256, 3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    return Model(inputs, outputs)


# Testing the generator and discriminator
if __name__ == "__main__":
    generator_model = generator()
    discriminator_model = discriminator()

    generator_model.summary()
    discriminator_model.summary()

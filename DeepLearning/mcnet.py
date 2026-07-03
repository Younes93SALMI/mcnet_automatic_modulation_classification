# mcnet.py

import tensorflow as tf
from tensorflow.keras import layers, Model


def conv_relu(x, filters, kernel_size, strides=(1, 1), padding="same", name=None):
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        use_bias=True,
        name=f"{name}_conv" if name else None,
    )(x)
    x = layers.ReLU(name=f"{name}_relu" if name else None)(x)
    return x


def pre_block(x):
    # Branch A: 3x1 conv + avg pool
    a = conv_relu(x, 32, (3, 1), padding="same", name="pre_a")
    a = layers.AveragePooling2D(
        pool_size=(1, 3),
        strides=(1, 2),
        padding="same",
        name="pre_pool_a",
    )(a)

    # Branch B: 1x3 conv with stride
    b = conv_relu(
        x,
        32,
        (1, 3),
        strides=(1, 2),
        padding="same",
        name="pre_b",
    )

    return layers.Concatenate(axis=-1, name="pre_concat")([b, a])


def m_block(x, filters_1x1, filters_asym, filters_reduce, downsample=False, name="mblock"):
    z = conv_relu(
        x,
        filters_1x1,
        (1, 1),
        padding="same",
        name=f"{name}_reduce",
    )

    stride = (1, 2) if downsample else (1, 1)

    b1 = conv_relu(
        z,
        filters_asym,
        (1, 3),
        strides=stride,
        padding="same",
        name=f"{name}_1x3",
    )

    b2 = conv_relu(
        z,
        filters_asym,
        (3, 1),
        padding="same",
        name=f"{name}_3x1",
    )

    if downsample:
        b2 = layers.AveragePooling2D(
            pool_size=(1, 3),
            strides=(1, 2),
            padding="same",
            name=f"{name}_pool",
        )(b2)

    b3 = conv_relu(
        z,
        filters_reduce,
        (1, 1),
        strides=stride,
        padding="same",
        name=f"{name}_1x1",
    )

    return layers.Concatenate(axis=-1, name=f"{name}_concat")([b1, b2, b3])


def build_mcnet(input_shape=(2, 1024, 1), num_classes=24):
    inputs = layers.Input(shape=input_shape, name="input")

    # Initial convolution
    x = conv_relu(
        inputs,
        64,
        (3, 7),
        strides=(1, 2),
        padding="same",
        name="conv1",
    )

    x = layers.MaxPooling2D(
        pool_size=(1, 3),
        strides=(1, 2),
        padding="same",
        name="pool1",
    )(x)

    # Pre-block
    x = pre_block(x)

    # Skip path for block A
    jump_a = conv_relu(
        x,
        128,
        (1, 1),
        strides=(1, 2),
        padding="same",
        name="jump_a",
    )

    jump_a = layers.MaxPooling2D(
        pool_size=(1, 3),
        strides=(1, 2),
        padding="same",
        name="jump_pool_a",
    )(jump_a)

    # M-block A
    x_main = layers.MaxPooling2D(
        pool_size=(1, 3),
        strides=(1, 2),
        padding="same",
        name="post_pooling",
    )(x)

    x_main = m_block(
        x_main,
        filters_1x1=32,
        filters_asym=48,
        filters_reduce=32,
        downsample=True,
        name="mblockA",
    )

    x = layers.Add(name="add_mixA")([x_main, jump_a])

    # M-block B
    x_b = m_block(
        x,
        filters_1x1=32,
        filters_asym=48,
        filters_reduce=32,
        downsample=False,
        name="mblockB",
    )

    x = layers.Add(name="add_mixB")([x_b, x])

    # M-block C with skip downsampling
    jump_c = layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(1, 2),
        padding="same",
        name="jump_pool_c",
    )(x)

    x_c = m_block(
        x,
        filters_1x1=32,
        filters_asym=48,
        filters_reduce=32,
        downsample=True,
        name="mblockC",
    )

    x = layers.Add(name="add_mixC")([x_c, jump_c])

    # M-block D
    x_d = m_block(
        x,
        filters_1x1=32,
        filters_asym=48,
        filters_reduce=32,
        downsample=False,
        name="mblockD",
    )

    x = layers.Add(name="add_mixD")([x_d, x])

    # M-block E with skip downsampling
    jump_e = layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(1, 2),
        padding="same",
        name="jump_pool_e",
    )(x)

    x_e = m_block(
        x,
        filters_1x1=32,
        filters_asym=48,
        filters_reduce=32,
        downsample=True,
        name="mblockE",
    )

    x = layers.Add(name="add_mixE")([x_e, jump_e])

    # M-block F
    identity = x

    x_f = m_block(
        x,
        filters_1x1=32,
        filters_asym=96,
        filters_reduce=64,
        downsample=False,
        name="mblockF",
    )

    # Final depth concatenation
    x = layers.Concatenate(axis=-1, name="concat_all")([x_f, identity])

    # Classification head
    x = layers.AveragePooling2D(
        pool_size=(2, 8),
        strides=(1, 1),
        padding="valid",
        name="global_pool",
    )(x)

    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(num_classes, name="fc")(x)
    x = layers.Dropout(0.5, name="dropout")(x)
    outputs = layers.Softmax(name="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs, name="MCNet")

    return model

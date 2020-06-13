from functools import partial

import tensorflow as tf
from tensorflow.keras import layers
from keras_applications import mobilenet_v3


def build_model(model_type, **kwargs):
    if model_type == 'mobilenet_v3_small':
        return build_mobilenet_v3_small(**kwargs)
    raise ValueError("unknown model {}".format(model_type))


def build_mobilenet_v3_small(
    num_classes=6,
    input_shape=(224, 224, 3),
    dropout_rate=0.5,
    weights=None,
    minimalistic=False,
):
    backbone = mobilenet_v3.MobileNetV3Small(
        include_top=False,
        dropout_rate=0.5,
        backend=tf.keras.backend,
        layers=tf.keras.layers,
        models=tf.keras.models,
        utils=tf.keras.utils,
        weights=weights,
        minimalistic=minimalistic,
    )
    classifier = tf.keras.models.Sequential([
        layers.GlobalAveragePooling2D(),
        layers.Reshape((1, 1, 576)),
        layers.Dropout(dropout_rate),
        layers.Conv2D(num_classes,
                      kernel_size=1,
                      padding='same',
                      name='Logits'),
        layers.Flatten(),
        layers.Softmax(name='Predictions/Softmax')
    ])
    inputs = layers.Input(input_shape)
    x = backbone(inputs)
    x = classifier(x)
    model = tf.keras.Model(inputs, x, name="ar_mobilenet_v3_small")
    model.build(input_shape)

    preprocess_fn = partial(
        mobilenet_v3.preprocess_input,
        backend=tf.keras.backend,
        layers=tf.keras.layers,
        models=tf.keras.models,
        utils=tf.keras.utils,
    )
    return model, preprocess_fn

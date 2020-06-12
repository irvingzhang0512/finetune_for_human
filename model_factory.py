import keras
from keras_applications import mobilenet_v3
from keras import layers


def build_model(model_type, **kwargs):
    if model_type == 'mobilenet_v3_small':
        return build_mobilenet_v3_small(**kwargs)
    raise ValueError("unknown model {}".format(model_type))


def build_mobilenet_v3_small(
    num_classes=6,
    input_shape=(224, 224, 3),
    dropout_rate=0.5,
    last_conv_ch=1024,
    weights=None,
):
    backbone = mobilenet_v3.MobileNetV3Small(
        include_top=False,
        dropout_rate=0.5,
        backend=keras.backend,
        layers=keras.layers,
        models=keras.models,
        utils=keras.utils,
        weights=weights,
    )
    classifier = keras.models.Sequential([
        layers.GlobalAveragePooling2D(),
        layers.Reshape((1, 1, 576)),
        layers.Conv2D(last_conv_ch,
                      kernel_size=1,
                      padding='same',
                      name='Conv_2'),
        layers.Activation('relu'),
        layers.Dropout(dropout_rate),
        layers.Conv2D(num_classes,
                      kernel_size=1,
                      padding='same',
                      name='Logits'),
        layers.Flatten(),
    ])
    inputs = layers.Input(input_shape)
    x = backbone(inputs)
    x = classifier(x)
    model = keras.Model(inputs, x, name="ar_mobilenet_v3_small")
    model.build(input_shape)
    return model, mobilenet_v3.preprocess_input
